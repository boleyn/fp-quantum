"""
量子智能化功能点估算系统 - PgVector向量存储

基于LangChain PostgreSQL PGVector集成的向量存储解决方案
专门用于知识库向量化数据的存储和检索
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import DistanceStrategy

from config.settings import get_settings

logger = logging.getLogger(__name__)


class PgVectorStore:
    """PgVector向量存储管理器 - 基于LangChain PostgreSQL集成"""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_stores: Dict[str, PGVector] = {}
        
        # 支持的源类型
        self.source_types = ["nesma", "cosmic", "common", "mixed"]
        
        # 构建PostgreSQL连接字符串（使用psycopg3）
        self.connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """构建PostgreSQL连接字符串"""
        db_settings = self.settings.database
        logger.info(db_settings.postgres_host)
        # 使用psycopg3连接字符串格式
        connection_string = f"postgresql+psycopg://{db_settings.postgres_user}:{db_settings.postgres_password}@{db_settings.postgres_host}:{db_settings.postgres_port}/{db_settings.postgres_database}"
        
        logger.info(f"🔗 PgVector连接字符串: {connection_string.replace(db_settings.postgres_password, '***')}")
        return connection_string
    
    async def initialize(self, embeddings: Embeddings):
        """初始化PgVector连接和向量存储"""
        try:
            # 为每个源类型创建LangChain PGVector实例
            await self._initialize_langchain_stores(embeddings)
            
            logger.info("✅ PgVector向量存储初始化完成")
            
        except Exception as e:
            logger.error(f"❌ PgVector初始化失败: {e}")
            raise
    
    async def _initialize_langchain_stores(self, embeddings: Embeddings):
        """初始化LangChain PGVector存储"""
        for source_type in self.source_types:
            try:
                # 为每个源类型创建独立的collection
                collection_name = f"fp_quantum_{source_type}"
                
                # 创建PGVector实例
                vector_store = PGVector(
                    embeddings=embeddings,
                    collection_name=collection_name,
                    connection=self.connection_string,
                    distance_strategy=DistanceStrategy.COSINE,
                    use_jsonb=True,  # 使用JSONB存储元数据
                    create_extension=True,  # 自动创建pgvector扩展
                    pre_delete_collection=False  # 不删除现有collection
                )
                
                self.vector_stores[source_type] = vector_store
                logger.info(f"✅ {source_type} LangChain PGVector存储已创建")
                
            except Exception as e:
                logger.error(f"❌ 创建 {source_type} LangChain存储失败: {e}")
                raise
    
    async def add_documents(
        self,
        documents: List[Document],
        source_type: str = "common"
    ) -> List[str]:
        """添加文档到向量存储"""
        if source_type not in self.source_types:
            raise ValueError(f"不支持的源类型: {source_type}")
        
        if not documents:
            return []
        
        try:
            # 获取对应的向量存储
            vector_store = self.vector_stores[source_type]
            
            # 为文档添加元数据
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata.update({
                    "source_type": source_type.upper(),
                    "standard": source_type.upper(),
                    "created_at": datetime.utcnow().isoformat()
                })
            
            # 批量添加文档
            ids = await vector_store.aadd_documents(documents)
            
            logger.info(f"✅ 成功添加 {len(ids)} 个文档到 {source_type} 向量存储")
            return ids
            
        except Exception as e:
            logger.error(f"❌ 添加文档到 {source_type} 失败: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        source_type: Optional[str] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        try:
            results = []
            
            # 确定搜索范围
            if source_type and source_type in self.vector_stores:
                search_stores = [source_type]
            else:
                search_stores = list(self.vector_stores.keys())
            
            # 在指定的存储中搜索
            for store_type in search_stores:
                vector_store = self.vector_stores[store_type]
                
                try:
                    # 使用异步相似度搜索并获取分数
                    store_results = await vector_store.asimilarity_search_with_score(
                        query=query,
                        k=k,
                        filter=filter_metadata
                    )
                    
                    # 添加源类型到结果
                    for doc, score in store_results:
                        if not doc.metadata:
                            doc.metadata = {}
                        doc.metadata["search_source"] = store_type
                        results.append((doc, score))
                        
                except Exception as e:
                    logger.warning(f"⚠️ 在 {store_type} 中搜索失败: {e}")
                    continue
            
            # 按分数排序并返回top-k结果
            results.sort(key=lambda x: x[1], reverse=True)
            final_results = results[:k]
            
            logger.debug(f"🔍 相似度搜索返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 相似度搜索失败: {e}")
            return []
    
    async def similarity_search_simple(
        self,
        query: str,
        source_type: Optional[str] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """简单相似度搜索（不返回分数）"""
        try:
            results = []
            
            # 确定搜索范围
            if source_type and source_type in self.vector_stores:
                search_stores = [source_type]
            else:
                search_stores = list(self.vector_stores.keys())
            
            # 在指定的存储中搜索
            for store_type in search_stores:
                vector_store = self.vector_stores[store_type]
                
                try:
                    # 使用异步相似度搜索
                    store_results = await vector_store.asimilarity_search(
                        query=query,
                        k=k,
                        filter=filter_metadata
                    )
                    
                    # 添加源类型到结果
                    for doc in store_results:
                        if not doc.metadata:
                            doc.metadata = {}
                        doc.metadata["search_source"] = store_type
                        results.append(doc)
                        
                except Exception as e:
                    logger.warning(f"⚠️ 在 {store_type} 中搜索失败: {e}")
                    continue
            
            # 返回前k个结果
            final_results = results[:k]
            
            logger.debug(f"🔍 简单相似度搜索返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 简单相似度搜索失败: {e}")
            return []
    
    async def get_collection_stats(self, source_type: str) -> Dict[str, Any]:
        """获取collection统计信息"""
        if source_type not in self.vector_stores:
            raise ValueError(f"不支持的源类型: {source_type}")
        
        try:
            vector_store = self.vector_stores[source_type]
            
            # 获取基本统计信息
            stats = {
                "source_type": source_type,
                "collection_name": f"fp_quantum_{source_type}",
                "status": "active"
            }
            
            logger.debug(f"📊 {source_type} collection统计: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ 获取 {source_type} 统计信息失败: {e}")
            return {}
    
    async def delete_documents(
        self,
        source_type: str,
        ids: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """删除文档"""
        if source_type not in self.vector_stores:
            raise ValueError(f"不支持的源类型: {source_type}")
        
        try:
            vector_store = self.vector_stores[source_type]
            
            if ids:
                # 按ID删除
                await vector_store.adelete(ids=ids)
                deleted_count = len(ids)
            else:
                # 按筛选条件删除（需要特殊实现）
                logger.warning("⚠️ 按筛选条件删除暂未实现")
                deleted_count = 0
            
            logger.info(f"✅ 从 {source_type} 删除了 {deleted_count} 个文档")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ 从 {source_type} 删除文档失败: {e}")
            return 0
    
    async def close(self):
        """关闭连接"""
        try:
            # PGVector使用的是LangChain的连接管理，通常自动关闭
            logger.info("✅ PgVector连接已关闭")
        except Exception as e:
            logger.error(f"❌ 关闭PgVector连接失败: {e}")
    
    def as_retriever(self, source_type: Optional[str] = None, **kwargs):
        """创建检索器接口 - 直接使用LangChain原生的as_retriever方法"""
        
        # 如果指定了源类型，使用对应的LangChain PGVector实例
        if source_type and source_type in self.vector_stores:
            vector_store = self.vector_stores[source_type]
            return vector_store.as_retriever(**kwargs)
        
        # 如果没有指定源类型，使用默认的第一个存储
        if self.vector_stores:
            default_store = list(self.vector_stores.values())[0]
            return default_store.as_retriever(**kwargs)
        
        raise ValueError("没有可用的向量存储")
    
    async def similarity_search_with_score(
        self,
        query: str,
        source_type: Optional[str] = None,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索（别名方法）"""
        return await self.similarity_search(
            query=query,
            source_type=source_type,
            k=k,
            filter_metadata=filter
        )

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查是否有可用的向量存储
            if not self.vector_stores:
                logger.warning("⚠️ 未初始化向量存储")
                return False
            
            # 测试一个简单的搜索来检查连接
            test_store = list(self.vector_stores.values())[0]
            await test_store.asimilarity_search("test", k=1)
            
            logger.info("✅ PgVector健康检查通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ PgVector健康检查失败: {e}")
            return False


async def create_pgvector_store(embeddings: Embeddings) -> PgVectorStore:
    """创建并初始化PgVector存储"""
    store = PgVectorStore()
    await store.initialize(embeddings)
    return store


def get_langchain_pgvector(
    source_type: str,
    embeddings: Embeddings,
    pgvector_store: PgVectorStore
) -> PGVector:
    """获取指定源类型的LangChain PGVector实例"""
    if source_type not in pgvector_store.vector_stores:
        raise ValueError(f"不支持的源类型: {source_type}")
    
    return pgvector_store.vector_stores[source_type]


# 测试代码
if __name__ == "__main__":
    async def main():
        # 测试PgVector存储
        from langchain_openai import OpenAIEmbeddings
        
        # 创建测试embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key="test-key"
        )
        
        try:
            # 创建存储
            store = await create_pgvector_store(embeddings)
            
            # 创建测试文档
            test_docs = [
                Document(
                    page_content="这是一个NESMA功能点估算的测试文档",
                    metadata={"category": "test", "type": "nesma"}
                ),
                Document(
                    page_content="COSMIC数据移动分析规则",
                    metadata={"category": "test", "type": "cosmic"}
                )
            ]
            
            # 添加文档
            ids = await store.add_documents(test_docs, "nesma")
            print(f"添加了 {len(ids)} 个文档")
            
            # 搜索测试
            results = await store.similarity_search("NESMA功能点", "nesma", k=2)
            print(f"搜索到 {len(results)} 个结果")
            
            # 关闭存储
            await store.close()
            
        except Exception as e:
            print(f"测试失败: {e}")
    
    asyncio.run(main()) 