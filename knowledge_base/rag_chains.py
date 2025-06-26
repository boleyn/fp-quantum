"""
量子智能化功能点估算系统 - RAG链

构建完整的RAG管道，整合文档加载、向量存储、检索和生成
统一使用PgVector向量存储
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import asyncio

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入本地模块
from .loaders.pdf_loader import EnhancedPDFLoader, BatchPDFProcessor
from .loaders.web_loader import load_web_knowledge_base
from .loaders.custom_loaders import FunctionPointDocumentLoader
# 统一使用PgVector向量存储
from .vector_stores.pgvector_store import PgVectorStore, create_pgvector_store
from .vector_stores.hybrid_search import HybridSearchStrategy, NESMAHybridSearch, COSMICHybridSearch
from .retrievers.semantic_retriever import EnhancedSemanticRetriever
from .retrievers.keyword_retriever import TFIDFRetriever, NESMAKeywordRetriever, COSMICKeywordRetriever
from .retrievers.multi_query_retriever import NESMAMultiQueryRetriever, COSMICMultiQueryRetriever
from .embeddings.embedding_models import get_embedding_model

logger = logging.getLogger(__name__)


class RAGChainBuilder:
    """RAG链构建器 - 基于PgVector"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        llm: BaseLanguageModel,
        use_hybrid_search: bool = True
    ):
        self.embeddings = embeddings
        self.llm = llm
        self.use_hybrid_search = use_hybrid_search
        
        # 存储组件
        self.documents: Dict[str, List[Document]] = {}
        self.vector_store: Optional[PgVectorStore] = None
        self.retrievers: Dict[str, Any] = {}
        self.chains: Dict[str, Any] = {}
        
    async def build_knowledge_base(
        self,
        document_paths: Dict[str, Union[str, Path, List[str]]],
        include_web_sources: bool = True
    ) -> Dict[str, int]:
        """构建知识库"""
        
        logger.info("🚀 开始构建基于PgVector的知识库...")
        
        # 1. 加载文档
        all_documents = await self._load_all_documents(document_paths, include_web_sources)
        
        # 2. 创建PgVector存储
        vector_store = await self._create_pgvector_store(all_documents)
        
        # 3. 创建检索器
        retrievers = await self._create_retrievers(vector_store, all_documents)
        
        # 4. 构建RAG链
        chains = await self._build_rag_chains(retrievers)
        
        # 统计信息
        stats = {}
        for doc_type, docs in all_documents.items():
            stats[doc_type] = len(docs)
        
        logger.info(f"✅ PgVector知识库构建完成: {stats}")
        return stats
    
    async def _load_all_documents(
        self,
        document_paths: Dict[str, Union[str, Path, List[str]]],
        include_web_sources: bool
    ) -> Dict[str, List[Document]]:
        """加载所有文档"""
        
        all_documents = {}
        
        # 加载本地文档
        for doc_type, paths in document_paths.items():
            logger.info(f"📚 加载 {doc_type} 文档...")
            
            if isinstance(paths, (str, Path)):
                paths = [paths]
            
            type_documents = []
            
            for path in paths:
                path = Path(path)
                
                if path.is_dir():
                    # 目录：使用自定义加载器
                    loader = FunctionPointDocumentLoader()
                    docs = loader.load_directory(path)
                    type_documents.extend(docs)
                    
                elif path.suffix.lower() == '.pdf':
                    # PDF文件：使用增强PDF加载器
                    pdf_loader = EnhancedPDFLoader(str(path))
                    docs = await pdf_loader.aload()
                    type_documents.extend(docs)
                    
                else:
                    # 其他文件：使用自定义加载器
                    loader = FunctionPointDocumentLoader()
                    docs = loader.load_file(path)
                    type_documents.extend(docs)
            
            # 为文档添加类型标记
            for doc in type_documents:
                doc.metadata['source_type'] = doc_type.upper()
            
            all_documents[doc_type] = type_documents
            logger.info(f"✅ {doc_type} 文档加载完成: {len(type_documents)} 个")
        
        # 加载网页资源
        if include_web_sources:
            logger.info("🌐 加载网页资源...")
            try:
                web_documents = await load_web_knowledge_base()
                all_documents.update(web_documents)
                
                web_total = sum(len(docs) for docs in web_documents.values())
                logger.info(f"✅ 网页资源加载完成: {web_total} 个文档")
                
            except Exception as e:
                logger.error(f"❌ 网页资源加载失败: {e}")
        
        return all_documents
    
    async def _create_pgvector_store(
        self,
        all_documents: Dict[str, List[Document]]
    ) -> PgVectorStore:
        """创建PgVector存储"""
        
        logger.info("��️ 创建PgVector存储...")
        
        vector_store = create_pgvector_store(
            documents_by_type=all_documents,
            embeddings=self.embeddings
        )
        
        self.vector_store = vector_store
        logger.info(f"✅ PgVector存储创建完成: {vector_store.collection_name}")
        return vector_store
    
    async def _create_retrievers(
        self,
        vector_store: PgVectorStore,
        all_documents: Dict[str, List[Document]]
    ) -> Dict[str, Any]:
        """创建检索器"""
        
        logger.info("🔍 创建检索器...")
        
        retrievers = {}
        
        # 为每种文档类型创建检索器
        for doc_type, documents in all_documents.items():
            type_documents = documents
            
            if self.use_hybrid_search:
                # 创建混合检索器
                if doc_type.upper() == "NESMA":
                    hybrid_retriever = NESMAHybridSearch(vector_store, self.embeddings)
                elif doc_type.upper() == "COSMIC":
                    hybrid_retriever = COSMICHybridSearch(vector_store, self.embeddings)
                else:
                    hybrid_retriever = HybridSearchStrategy(vector_store, self.embeddings)
                
                # 构建BM25索引
                hybrid_retriever.build_bm25_index(type_documents)
                retrievers[f"{doc_type}_hybrid"] = hybrid_retriever
            
            # 创建语义检索器
            semantic_retriever = EnhancedSemanticRetriever(
                vector_store=vector_store,
                embeddings=self.embeddings
            )
            retrievers[f"{doc_type}_semantic"] = semantic_retriever
            
            # 创建关键词检索器
            if doc_type.upper() == "NESMA":
                keyword_retriever = NESMAKeywordRetriever(type_documents)
            elif doc_type.upper() == "COSMIC":
                keyword_retriever = COSMICKeywordRetriever(type_documents)
            else:
                keyword_retriever = TFIDFRetriever(type_documents)
            
            retrievers[f"{doc_type}_keyword"] = keyword_retriever
            
            # 创建多查询检索器
            if doc_type.upper() == "NESMA":
                multi_query_retriever = NESMAMultiQueryRetriever(
                    vector_store, self.llm, self.embeddings
                )
            elif doc_type.upper() == "COSMIC":
                multi_query_retriever = COSMICMultiQueryRetriever(
                    vector_store, self.llm, self.embeddings
                )
            else:
                from .retrievers.multi_query_retriever import EnhancedMultiQueryRetriever
                multi_query_retriever = EnhancedMultiQueryRetriever(
                    vector_store, self.llm, self.embeddings
                )
            
            retrievers[f"{doc_type}_multi_query"] = multi_query_retriever
            
            # 创建集成检索器（组合多种检索方法）
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            
            # 如果有关键词检索器，创建集成检索器
            try:
                from langchain.retrievers import EnsembleRetriever
                
                # 注意：这里需要适配不同检索器的接口
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[base_retriever],  # 目前只包含向量检索器
                    weights=[1.0]
                )
                retrievers[f"{doc_type}_ensemble"] = ensemble_retriever
                
            except Exception as e:
                logger.warning(f"创建集成检索器失败 {doc_type}: {e}")
        
        self.retrievers = retrievers
        logger.info(f"✅ 检索器创建完成: {len(retrievers)} 个")
        return retrievers
    
    async def _build_rag_chains(
        self,
        retrievers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建RAG链"""
        
        logger.info("⛓️ 构建RAG链...")
        
        chains = {}
        
        # NESMA专用RAG链
        if "nesma_hybrid" in retrievers or "nesma_semantic" in retrievers:
            nesma_chain = self._create_standard_rag_chain(
                retriever=retrievers.get("nesma_hybrid") or retrievers.get("nesma_semantic"),
                standard="NESMA"
            )
            chains["nesma"] = nesma_chain
        
        # COSMIC专用RAG链
        if "cosmic_hybrid" in retrievers or "cosmic_semantic" in retrievers:
            cosmic_chain = self._create_standard_rag_chain(
                retriever=retrievers.get("cosmic_hybrid") or retrievers.get("cosmic_semantic"),
                standard="COSMIC"
            )
            chains["cosmic"] = cosmic_chain
        
        # 通用RAG链
        if "common_hybrid" in retrievers or "common_semantic" in retrievers:
            common_chain = self._create_standard_rag_chain(
                retriever=retrievers.get("common_hybrid") or retrievers.get("common_semantic"),
                standard="COMMON"
            )
            chains["common"] = common_chain
        
        self.chains = chains
        logger.info(f"✅ RAG链构建完成: {list(chains.keys())}")
        return chains
    
    def _create_standard_rag_chain(self, retriever: Any, standard: str):
        """创建标准RAG链"""
        
        # 创建检索提示模板
        if standard == "NESMA":
            system_prompt = """
            你是NESMA功能点分析专家。请基于提供的NESMA知识库文档回答用户问题。
            
            回答要求：
            1. 严格基于NESMA官方标准和规则
            2. 引用具体的NESMA分类规则和计算方法
            3. 提供准确的功能类型识别指导
            4. 包含DET、RET等复杂度计算要素
            5. 如果信息不足，明确说明限制
            
            相关文档：
            {context}
            
            用户问题：{question}
            """
            
        elif standard == "COSMIC":
            system_prompt = """
            你是COSMIC功能点分析专家。请基于提供的COSMIC知识库文档回答用户问题。
            
            回答要求：
            1. 严格基于COSMIC官方标准和方法
            2. 重点关注数据移动识别和分类
            3. 提供功能用户和边界分析指导
            4. 解释Entry、Exit、Read、Write四种数据移动
            5. 包含CFP计算的具体步骤
            
            相关文档：
            {context}
            
            用户问题：{question}
            """
            
        else:
            system_prompt = """
            你是功能点估算专家。请基于提供的知识库文档回答用户问题。
            
            回答要求：
            1. 基于提供的标准文档和最佳实践
            2. 提供准确、实用的分析指导
            3. 如涉及具体标准，明确指出是NESMA还是COSMIC
            4. 如果信息不足，明确说明并建议进一步查询
            
            相关文档：
            {context}
            
            用户问题：{question}
            """
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # 构建RAG链
        def format_docs(docs):
            """格式化文档"""
            if not docs:
                return "未找到相关文档。"
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知来源')
                content = doc.page_content.strip()
                formatted.append(f"文档 {i} (来源: {source}):\n{content}")
            
            return "\n\n".join(formatted)
        
        # 处理混合检索器和标准检索器的差异
        if hasattr(retriever, 'hybrid_search'):
            # 混合检索器
            def retrieve_context(query):
                results = retriever.hybrid_search(query["question"], k=5)
                docs = [doc for doc, score in results]
                return format_docs(docs)
        elif hasattr(retriever, 'retrieve_documents'):
            # 多查询检索器
            async def retrieve_context(query):
                docs = await retriever.retrieve_documents(query["question"], k=5)
                return format_docs(docs)
        else:
            # 标准检索器
            def retrieve_context(query):
                docs = retriever.search(query["question"], k=5)
                if docs and isinstance(docs[0], tuple):
                    # 如果返回的是(doc, score)元组
                    docs = [doc for doc, score in docs]
                return format_docs(docs)
        
        # 构建RAG链
        rag_chain = (
            {"context": retrieve_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def get_chain(self, standard: str) -> Optional[Any]:
        """获取指定标准的RAG链"""
        return self.chains.get(standard.lower())
    
    def get_retriever(self, retriever_name: str) -> Optional[Any]:
        """获取指定的检索器"""
        return self.retrievers.get(retriever_name)
    
    async def query(
        self,
        question: str,
        standard: Optional[str] = None,
        use_multi_standard: bool = False
    ) -> Dict[str, Any]:
        """执行查询"""
        
        if use_multi_standard or standard is None:
            # 查询多个标准
            results = {}
            
            for std in ["nesma", "cosmic", "common"]:
                chain = self.get_chain(std)
                if chain:
                    try:
                        if asyncio.iscoroutinefunction(chain.invoke):
                            answer = await chain.ainvoke({"question": question})
                        else:
                            answer = chain.invoke({"question": question})
                        results[std.upper()] = answer
                    except Exception as e:
                        logger.error(f"查询{std}标准失败: {e}")
                        results[std.upper()] = f"查询失败: {e}"
            
            return results
        
        else:
            # 查询单个标准
            chain = self.get_chain(standard.lower())
            if not chain:
                return {"error": f"未找到{standard}标准的RAG链"}
            
            try:
                if asyncio.iscoroutinefunction(chain.invoke):
                    answer = await chain.ainvoke({"question": question})
                else:
                    answer = chain.invoke({"question": question})
                
                return {standard.upper(): answer}
                
            except Exception as e:
                logger.error(f"查询失败: {e}")
                return {"error": str(e)}


class RAGChainFactory:
    """RAG链工厂"""
    
    @staticmethod
    async def create_complete_rag_system(
        document_paths: Dict[str, Union[str, Path, List[str]]],
        embedding_model_name: str = "bge_m3",
        llm: Optional[BaseLanguageModel] = None,
        use_hybrid_search: bool = True
    ) -> RAGChainBuilder:
        """创建完整的RAG系统"""
        
        # 获取嵌入模型
        embeddings = get_embedding_model(embedding_model_name)
        
        # 如果没有提供LLM，使用默认配置
        if llm is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                temperature=0.1,
                max_tokens=4000
            )
        
        # 创建RAG构建器
        rag_builder = RAGChainBuilder(
            embeddings=embeddings,
            llm=llm,
            use_hybrid_search=use_hybrid_search
        )
        
        # 构建知识库
        await rag_builder.build_knowledge_base(
            document_paths=document_paths,
            include_web_sources=True
        )
        
        return rag_builder


# 预定义的文档路径配置
DEFAULT_DOCUMENT_PATHS = {
    "nesma": "knowledge_base/documents/nesma",
    "cosmic": "knowledge_base/documents/cosmic", 
    "common": "knowledge_base/documents/common"
}


async def setup_default_rag_system() -> RAGChainBuilder:
    """设置默认的RAG系统"""
    
    return await RAGChainFactory.create_complete_rag_system(
        document_paths=DEFAULT_DOCUMENT_PATHS,
        embedding_model_name="bge_m3",
        use_hybrid_search=True
    )


if __name__ == "__main__":
    async def main():
        # 测试RAG系统
        print("🚀 测试RAG链系统...")
        
        try:
            # 创建RAG系统
            rag_system = await setup_default_rag_system()
            
            # 测试查询
            test_questions = [
                "什么是ILF功能类型？",
                "如何识别Entry数据移动？",
                "DET和RET的计算方法是什么？"
            ]
            
            for question in test_questions:
                print(f"\n❓ 问题: {question}")
                
                try:
                    results = await rag_system.query(
                        question=question,
                        use_multi_standard=True
                    )
                    
                    for standard, answer in results.items():
                        print(f"📝 {standard} 回答:")
                        print(f"   {answer[:200]}...")
                        
                except Exception as e:
                    print(f"❌ 查询失败: {e}")
            
        except Exception as e:
            print(f"❌ RAG系统初始化失败: {e}")
    
    # 运行测试
    asyncio.run(main()) 