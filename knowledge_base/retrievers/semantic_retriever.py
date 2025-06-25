"""
量子智能化功能点估算系统 - 语义检索器

实现多种高级检索策略，包括多查询检索、上下文压缩、混合检索等
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever

from knowledge_base.vector_stores.mongodb_atlas import MongoDBAtlasVectorManager, create_langchain_vector_store
from knowledge_base.embeddings.embedding_models import get_default_embedding_model
from models.common_models import EstimationStandard, RetrievalResult, KnowledgeChunk, ValidationResult
from config.settings import get_settings

logger = logging.getLogger(__name__)


class EnhancedSemanticRetriever(BaseRetriever):
    """增强的语义检索器"""
    
    def __init__(
        self,
        vector_manager: MongoDBAtlasVectorManager,
        embeddings: Embeddings,
        llm: Optional[BaseLanguageModel] = None,
        source_type: Optional[str] = None,
        use_multi_query: bool = True,
        use_compression: bool = True,
        k: int = 5
    ):
        super().__init__()
        self.vector_manager = vector_manager
        self.embeddings = embeddings
        self.llm = llm
        self.source_type = source_type
        self.use_multi_query = use_multi_query
        self.use_compression = use_compression
        self.k = k
        self.settings = get_settings()
        
        # 创建基础向量检索器
        if source_type:
            self.vector_store = create_langchain_vector_store(
                source_type, embeddings, vector_manager
            )
            self.base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        else:
            self.base_retriever = None
        
        # 创建增强检索器
        self._setup_enhanced_retrievers()
    
    def _setup_enhanced_retrievers(self):
        """设置增强检索器"""
        if not self.base_retriever:
            return
        
        # 多查询检索器
        if self.use_multi_query and self.llm:
            try:
                self.multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=self.base_retriever,
                    llm=self.llm,
                    include_original=True
                )
            except Exception as e:
                logger.warning(f"⚠️ 多查询检索器初始化失败: {str(e)}")
                self.multi_query_retriever = self.base_retriever
        else:
            self.multi_query_retriever = self.base_retriever
        
        # 上下文压缩检索器
        if self.use_compression and self.llm:
            try:
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.multi_query_retriever
                )
            except Exception as e:
                logger.warning(f"⚠️ 上下文压缩检索器初始化失败: {str(e)}")
                self.compression_retriever = self.multi_query_retriever
        else:
            self.compression_retriever = self.multi_query_retriever
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """获取相关文档（同步版本）"""
        # 由于基础接口是同步的，这里需要处理异步逻辑
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.aget_relevant_documents(query))
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            return asyncio.run(self.aget_relevant_documents(query))
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """异步获取相关文档"""
        if self.vector_manager and not self.base_retriever:
            # 直接使用向量管理器进行搜索
            results = await self.vector_manager.similarity_search(
                query=query,
                embeddings=self.embeddings,
                source_type=self.source_type,
                k=self.k
            )
            return [doc for doc, score in results]
        
        # 使用增强检索器
        if self.compression_retriever:
            return await self.compression_retriever.aget_relevant_documents(query)
        elif self.multi_query_retriever:
            return await self.multi_query_retriever.aget_relevant_documents(query)
        else:
            return await self.base_retriever.aget_relevant_documents(query)
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索"""
        k = k or self.k
        
        if self.vector_manager:
            return await self.vector_manager.similarity_search(
                query=query,
                embeddings=self.embeddings,
                source_type=self.source_type,
                k=k
            )
        else:
            # 使用基础检索器，但无法获取分数
            docs = await self.aget_relevant_documents(query)
            return [(doc, 1.0) for doc in docs[:k]]


class MultiSourceRetriever:
    """多源检索器 - 支持跨NESMA、COSMIC、通用知识库检索"""
    
    def __init__(
        self,
        vector_manager: MongoDBAtlasVectorManager,
        embeddings: Embeddings,
        llm: Optional[BaseLanguageModel] = None
    ):
        self.vector_manager = vector_manager
        self.embeddings = embeddings
        self.llm = llm
        self.settings = get_settings()
        
        # 为每个知识源创建检索器
        self.retrievers = {}
        for source_type in ["nesma", "cosmic", "common"]:
            self.retrievers[source_type] = EnhancedSemanticRetriever(
                vector_manager=vector_manager,
                embeddings=embeddings,
                llm=llm,
                source_type=source_type,
                k=5
            )
    
    async def retrieve_by_source(
        self,
        query: str,
        source_type: str,
        k: int = 5
    ) -> RetrievalResult:
        """按指定知识源检索"""
        import time
        start_time = time.time()
        
        if source_type not in self.retrievers:
            raise ValueError(f"不支持的知识源: {source_type}")
        
        retriever = self.retrievers[source_type]
        
        # 执行检索
        try:
            docs_with_scores = await retriever.similarity_search_with_score(query, k)
            
            # 转换为KnowledgeChunk
            chunks = []
            for doc, score in docs_with_scores:
                chunk = KnowledgeChunk(
                    chunk_id=f"{source_type}_{hash(doc.page_content)%1000000}",
                    content=doc.page_content,
                    source_document=doc.metadata.get("source", "unknown"),
                    source_type=EstimationStandard(source_type.upper()),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=score,
                    metadata=doc.metadata
                )
                chunks.append(chunk)
            
            retrieval_time = int((time.time() - start_time) * 1000)
            
            return RetrievalResult(
                query=query,
                source_type=EstimationStandard(source_type.upper()),
                retrieved_chunks=chunks,
                total_chunks=len(chunks),
                retrieval_time_ms=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"❌ {source_type} 检索失败: {str(e)}")
            return RetrievalResult(
                query=query,
                source_type=EstimationStandard(source_type.upper()),
                retrieved_chunks=[],
                total_chunks=0,
                retrieval_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def retrieve_multi_source(
        self,
        query: str,
        sources: List[str] = ["nesma", "cosmic", "common"],
        k_per_source: int = 3
    ) -> Dict[str, RetrievalResult]:
        """多源并行检索"""
        tasks = []
        for source in sources:
            if source in self.retrievers:
                task = self.retrieve_by_source(query, source, k_per_source)
                tasks.append((source, task))
        
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (source, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"❌ {source} 检索异常: {str(result)}")
                    results[source] = RetrievalResult(
                        query=query,
                        source_type=EstimationStandard(source.upper()),
                        retrieved_chunks=[],
                        total_chunks=0,
                        retrieval_time_ms=0
                    )
                else:
                    results[source] = result
        
        return results
    
    async def adaptive_retrieve(
        self,
        query: str,
        preferred_source: Optional[str] = None,
        fallback_sources: Optional[List[str]] = None,
        min_chunks: int = 3
    ) -> RetrievalResult:
        """自适应检索 - 根据结果质量动态选择知识源"""
        
        # 确定检索顺序
        if preferred_source:
            sources_to_try = [preferred_source]
            if fallback_sources:
                sources_to_try.extend(fallback_sources)
            else:
                # 默认备用源
                all_sources = ["nesma", "cosmic", "common"]
                sources_to_try.extend([s for s in all_sources if s != preferred_source])
        else:
            sources_to_try = ["nesma", "cosmic", "common"]
        
        # 逐个尝试检索
        for source in sources_to_try:
            if source not in self.retrievers:
                continue
            
            result = await self.retrieve_by_source(query, source, 5)
            
            # 检查结果质量
            if self._is_result_sufficient(result, min_chunks):
                logger.info(f"✅ 使用 {source} 知识源检索成功，获得 {len(result.retrieved_chunks)} 个高质量结果")
                return result
            else:
                logger.info(f"⚠️ {source} 知识源结果不足，尝试下一个源")
        
        # 如果所有单一源都不足，尝试合并检索
        logger.info("📊 执行多源合并检索")
        multi_results = await self.retrieve_multi_source(query, sources_to_try, 2)
        
        # 合并所有结果
        all_chunks = []
        total_time = 0
        for result in multi_results.values():
            all_chunks.extend(result.retrieved_chunks)
            total_time += result.retrieval_time_ms
        
        # 按相关性排序
        all_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return RetrievalResult(
            query=query,
            source_type=EstimationStandard.BOTH,
            retrieved_chunks=all_chunks[:10],  # 最多返回10个结果
            total_chunks=len(all_chunks),
            retrieval_time_ms=total_time
        )
    
    def _is_result_sufficient(self, result: RetrievalResult, min_chunks: int) -> bool:
        """判断检索结果是否充足"""
        if len(result.retrieved_chunks) < min_chunks:
            return False
        
        # 检查相关性阈值
        high_quality_chunks = [
            chunk for chunk in result.retrieved_chunks 
            if chunk.relevance_score > 0.7
        ]
        
        return len(high_quality_chunks) >= min_chunks


class HybridRetriever:
    """混合检索器 - 结合向量检索和关键词检索"""
    
    def __init__(
        self,
        semantic_retriever: EnhancedSemanticRetriever,
        documents: Optional[List[Document]] = None
    ):
        self.semantic_retriever = semantic_retriever
        
        # 创建BM25关键词检索器
        if documents:
            self.keyword_retriever = BM25Retriever.from_documents(documents)
        else:
            self.keyword_retriever = None
    
    async def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """混合检索：结合语义和关键词搜索"""
        
        # 语义检索
        semantic_results = await self.semantic_retriever.similarity_search_with_score(
            query, k=k
        )
        
        # 关键词检索
        keyword_results = []
        if self.keyword_retriever:
            try:
                keyword_docs = await self.keyword_retriever.aget_relevant_documents(query)
                # BM25没有直接分数，使用固定分数
                keyword_results = [(doc, 0.8) for doc in keyword_docs[:k]]
            except Exception as e:
                logger.warning(f"⚠️ 关键词检索失败: {str(e)}")
        
        # 合并和重新排序
        all_results = {}
        
        # 添加语义检索结果
        for doc, score in semantic_results:
            doc_key = hash(doc.page_content)
            if doc_key not in all_results:
                all_results[doc_key] = (doc, 0.0)
            all_results[doc_key] = (
                all_results[doc_key][0],
                all_results[doc_key][1] + score * semantic_weight
            )
        
        # 添加关键词检索结果
        for doc, score in keyword_results:
            doc_key = hash(doc.page_content)
            if doc_key not in all_results:
                all_results[doc_key] = (doc, 0.0)
            all_results[doc_key] = (
                all_results[doc_key][0],
                all_results[doc_key][1] + score * keyword_weight
            )
        
        # 按综合分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]


async def create_knowledge_retrievers(
    vector_manager: MongoDBAtlasVectorManager,
    embeddings: Optional[Embeddings] = None,
    llm: Optional[BaseLanguageModel] = None
) -> Dict[str, Any]:
    """创建知识检索器集合"""
    
    if embeddings is None:
        embeddings = get_default_embedding_model()
    
    # 创建多源检索器
    multi_source_retriever = MultiSourceRetriever(
        vector_manager=vector_manager,
        embeddings=embeddings,
        llm=llm
    )
    
    # 为每个标准创建专用检索器
    retrievers = {
        "multi_source": multi_source_retriever,
        "nesma": EnhancedSemanticRetriever(
            vector_manager=vector_manager,
            embeddings=embeddings,
            llm=llm,
            source_type="nesma",
            k=5
        ),
        "cosmic": EnhancedSemanticRetriever(
            vector_manager=vector_manager,
            embeddings=embeddings,
            llm=llm,
            source_type="cosmic",
            k=5
        ),
        "common": EnhancedSemanticRetriever(
            vector_manager=vector_manager,
            embeddings=embeddings,
            llm=llm,
            source_type="common",
            k=5
        )
    }
    
    return retrievers


if __name__ == "__main__":
    async def main():
        # 测试语义检索器
        from knowledge_base.vector_stores.mongodb_atlas import MongoDBAtlasVectorManager
        
        # 初始化向量管理器
        vector_manager = MongoDBAtlasVectorManager()
        await vector_manager.initialize()
        
        # 创建检索器
        retrievers = await create_knowledge_retrievers(vector_manager)
        
        # 测试查询
        test_queries = [
            "NESMA功能点分类规则",
            "COSMIC数据移动类型",
            "功能复杂度计算方法"
        ]
        
        multi_retriever = retrievers["multi_source"]
        
        for query in test_queries:
            print(f"\n🔍 测试查询: {query}")
            
            # 自适应检索
            result = await multi_retriever.adaptive_retrieve(query)
            print(f"📊 检索结果: {len(result.retrieved_chunks)} 个文档块")
            print(f"⏱️ 检索耗时: {result.retrieval_time_ms}ms")
            
            if result.retrieved_chunks:
                best_chunk = result.retrieved_chunks[0]
                print(f"🎯 最佳匹配 (分数: {best_chunk.relevance_score:.3f})")
                print(f"   内容预览: {best_chunk.content_preview}")
        
        await vector_manager.close()
    
    asyncio.run(main()) 