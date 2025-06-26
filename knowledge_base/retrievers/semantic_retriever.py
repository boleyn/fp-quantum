"""
量子智能化功能点估算系统 - 语义检索器

基于PgVector实现高质量的语义检索
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from knowledge_base.vector_stores.pgvector_store import PgVectorStore
from models.common_models import EstimationStandard, KnowledgeQuery, KnowledgeResult

logger = logging.getLogger(__name__)


class EnhancedSemanticRetriever(BaseRetriever):
    """增强语义检索器 - 基于PgVector"""
    
    # 声明类字段（Pydantic 需要）
    vector_store: Any = None
    embeddings: Any = None 
    llm: Any = None
    source_type: Optional[str] = None
    k: int = 5
    config: Dict[str, Any] = None
    base_retriever: Any = None
    multi_query_retriever: Any = None
    compression_retriever: Any = None
    
    def __init__(
        self,
        vector_store: PgVectorStore,
        embeddings: Embeddings,
        llm: BaseLanguageModel,
        source_type: Optional[str] = None,
        k: int = 5
    ):
        super().__init__()
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.source_type = source_type
        self.k = k
        
        # 检索配置
        self.config = {
            "score_threshold": 0.7,
            "diversity_threshold": 0.8,
            "use_multi_query": True,
            "use_compression": True
        }
        
        # 设置增强检索器
        self._setup_enhanced_retrievers()

    def _setup_enhanced_retrievers(self):
        """设置增强检索器"""
        # 创建基础检索器 - 直接使用LangChain原生的PGVector实例
        self.base_retriever = self.vector_store.as_retriever(
            source_type=self.source_type,
            search_kwargs={"k": self.k, "filter": {"source_type": self.source_type} if self.source_type else None}
        )
        
        # 多查询检索器
        if self.config["use_multi_query"] and self.llm:
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
        if self.config["use_compression"] and self.llm:
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
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.aget_relevant_documents(query, run_manager=run_manager))
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            return asyncio.run(self.aget_relevant_documents(query, run_manager=run_manager))
    
    async def aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """异步获取相关文档"""
        try:
            # 使用增强检索器
            if self.compression_retriever:
                return await self.compression_retriever.aget_relevant_documents(query, run_manager=run_manager)
            elif self.multi_query_retriever:
                return await self.multi_query_retriever.aget_relevant_documents(query, run_manager=run_manager)
            else:
                return await self.base_retriever.aget_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            logger.error(f"❌ 检索文档失败: {str(e)}")
            return []
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索"""
        k = k or self.k
        
        try:
            # 直接使用PgVector的相似度搜索
            return await self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict or ({"source_type": self.source_type} if self.source_type else None)
            )
        except Exception as e:
            logger.error(f"❌ 相似度搜索失败: {str(e)}")
            return []


class PgVectorMultiSourceRetriever:
    """多源检索器 - 基于PgVector支持跨NESMA、COSMIC、通用知识库检索"""
    
    def __init__(
        self,
        vector_store: PgVectorStore,
        embeddings: Embeddings,
        llm: BaseLanguageModel
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        
        # 为每个知识源创建检索器
        self.retrievers = {}
        for source_type in ["NESMA", "COSMIC", "COMMON"]:
            self.retrievers[source_type] = EnhancedSemanticRetriever(
                vector_store=vector_store,
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
    ) -> KnowledgeResult:
        """按指定知识源检索"""
        import time
        start_time = time.time()
        
        source_type = source_type.upper()
        if source_type not in self.retrievers:
            raise ValueError(f"不支持的知识源: {source_type}")
        
        retriever = self.retrievers[source_type]
        
        # 执行检索
        try:
            docs_with_scores = await retriever.similarity_search_with_score(query, k)
            
            # 转换为检索结果
            retrieved_chunks = []
            for doc, score in docs_with_scores:
                chunk_data = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                    "source": doc.metadata.get("source", "unknown")
                }
                retrieved_chunks.append(chunk_data)
            
            retrieval_time = int((time.time() - start_time) * 1000)
            
            result = KnowledgeResult(
                query=query,
                source_type=EstimationStandard(source_type),
                retrieved_chunks=retrieved_chunks,
                total_chunks=len(retrieved_chunks),
                max_score=max([chunk["score"] for chunk in retrieved_chunks]) if retrieved_chunks else 0.0,
                min_score=min([chunk["score"] for chunk in retrieved_chunks]) if retrieved_chunks else 0.0,
                avg_score=sum([chunk["score"] for chunk in retrieved_chunks]) / len(retrieved_chunks) if retrieved_chunks else 0.0,
                processing_time_ms=retrieval_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 检索失败 {source_type}: {str(e)}")
            return KnowledgeResult(
                query=query,
                source_type=EstimationStandard(source_type),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def retrieve_multi_source(
        self,
        query: str,
        sources: List[str] = ["NESMA", "COSMIC", "COMMON"],
        k_per_source: int = 3
    ) -> Dict[str, KnowledgeResult]:
        """多源检索"""
        results = {}
        
        # 并行检索所有源
        tasks = []
        for source in sources:
            task = self.retrieve_by_source(query, source, k_per_source)
            tasks.append((source, task))
        
        # 等待所有检索完成
        for source, task in tasks:
            try:
                result = await task
                results[source] = result
            except Exception as e:
                logger.error(f"❌ 多源检索失败 {source}: {str(e)}")
                results[source] = KnowledgeResult(
                    query=query,
                    source_type=EstimationStandard(source)
                )
        
        return results
    
    async def adaptive_retrieve(
        self,
        query: str,
        preferred_source: Optional[str] = None,
        fallback_sources: Optional[List[str]] = None,
        min_chunks: int = 3
    ) -> KnowledgeResult:
        """自适应检索 - 优先使用指定源，不足时使用备用源"""
        
        if preferred_source:
            # 优先检索指定源
            result = await self.retrieve_by_source(query, preferred_source, 5)
            if self._is_result_sufficient(result, min_chunks):
                return result
        
        # 如果首选源不足，尝试备用源
        fallback_sources = fallback_sources or ["NESMA", "COSMIC", "COMMON"]
        if preferred_source and preferred_source in fallback_sources:
            fallback_sources.remove(preferred_source)
        
        # 尝试备用源
        for source in fallback_sources:
            try:
                result = await self.retrieve_by_source(query, source, 5)
                if self._is_result_sufficient(result, min_chunks):
                    return result
            except Exception as e:
                logger.warning(f"⚠️ 备用源检索失败 {source}: {str(e)}")
                continue
        
        # 如果所有单源都不足，进行多源合并检索
        multi_results = await self.retrieve_multi_source(query, fallback_sources, 2)
        
        # 合并所有结果
        all_chunks = []
        for result in multi_results.values():
            all_chunks.extend(result.retrieved_chunks)
        
        # 按分数排序并取top-k
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        final_chunks = all_chunks[:min_chunks * 2]  # 取足够的数量
        
        return KnowledgeResult(
            query=query,
            source_type=EstimationStandard.BOTH,
            retrieved_chunks=final_chunks,
            total_chunks=len(final_chunks),
            max_score=max([chunk["score"] for chunk in final_chunks]) if final_chunks else 0.0,
            min_score=min([chunk["score"] for chunk in final_chunks]) if final_chunks else 0.0,
            avg_score=sum([chunk["score"] for chunk in final_chunks]) / len(final_chunks) if final_chunks else 0.0
        )
    
    def _is_result_sufficient(self, result: KnowledgeResult, min_chunks: int) -> bool:
        """检查结果是否足够"""
        return (
            result.total_chunks >= min_chunks and
            result.avg_score >= 0.7
        )


async def create_pgvector_retrievers(
    vector_store: PgVectorStore,
    embeddings: Embeddings,
    llm: BaseLanguageModel
) -> Dict[str, Any]:
    """创建基于PgVector的知识检索器"""
    
    logger.info("🔍 创建PgVector知识检索器...")
    
    retrievers = {
        # 单源检索器
        "nesma": EnhancedSemanticRetriever(
            vector_store=vector_store,
            embeddings=embeddings,
            llm=llm,
            source_type="NESMA",
            k=5
        ),
        "cosmic": EnhancedSemanticRetriever(
            vector_store=vector_store,
            embeddings=embeddings,
            llm=llm,
            source_type="COSMIC",
            k=5
        ),
        "common": EnhancedSemanticRetriever(
            vector_store=vector_store,
            embeddings=embeddings,
            llm=llm,
            source_type="COMMON",
            k=5
        ),
        
        # 多源检索器
        "multi_source": PgVectorMultiSourceRetriever(
            vector_store=vector_store,
            embeddings=embeddings,
            llm=llm
        )
    }
    
    logger.info(f"✅ PgVector知识检索器创建完成: {list(retrievers.keys())}")
    return retrievers


if __name__ == "__main__":
    # 测试语义检索器
    import os
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    
    async def main():
        # 测试PgVector语义检索器
        from knowledge_base.vector_stores.pgvector_store import PgVectorStore
        
        # 初始化组件
        vector_store = PgVectorStore()
        await vector_store.initialize()
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # 创建检索器
        retrievers = await create_pgvector_retrievers(vector_store, embeddings, llm)
        multi_retriever = retrievers["multi_source"]
        
        # 测试查询
        test_queries = [
            "什么是NESMA功能点估算",
            "COSMIC数据移动的定义",
            "内部逻辑文件的分类规则"
        ]
        
        for query in test_queries:
            print(f"\n🔍 测试查询: {query}")
            
            # 单源检索测试
            nesma_result = await multi_retriever.retrieve_by_source(query, "NESMA", 3)
            print(f"NESMA检索结果: {nesma_result.total_chunks}个块, 平均分数: {nesma_result.avg_score:.3f}")
            
            # 多源检索测试
            multi_results = await multi_retriever.retrieve_multi_source(query, k_per_source=2)
            for source, result in multi_results.items():
                print(f"{source}检索结果: {result.total_chunks}个块, 平均分数: {result.avg_score:.3f}")
        
        await vector_store.close()
    
    asyncio.run(main()) 