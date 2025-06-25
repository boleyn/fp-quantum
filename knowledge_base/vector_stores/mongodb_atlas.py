"""
量子智能化功能点估算系统 - MongoDB Atlas向量存储

MongoDB Atlas Vector Search集成，支持生产环境的向量存储和检索
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging

from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.embeddings import Embeddings
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
import pymongo

from config.settings import get_settings
from models.common_models import EstimationStandard

logger = logging.getLogger(__name__)


class MongoDBAtlasVectorManager:
    """MongoDB Atlas向量存储管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.collections: Dict[str, AsyncIOMotorCollection] = {}
        
        # 集合名称映射
        self.collection_names = {
            "nesma": "nesma_knowledge_vectors",
            "cosmic": "cosmic_knowledge_vectors", 
            "common": "common_knowledge_vectors",
            "mixed": "mixed_knowledge_vectors"
        }
    
    async def initialize(self):
        """初始化MongoDB Atlas连接"""
        try:
            # 创建异步客户端
            mongodb_url = self.settings.database.mongodb_url
            self.client = AsyncIOMotorClient(
                mongodb_url,
                maxPoolSize=self.settings.database.mongodb_max_pool_size,
                minPoolSize=self.settings.database.mongodb_min_pool_size,
                maxIdleTimeMS=self.settings.database.mongodb_max_idle_time_ms,
            )
            
            # 选择数据库
            if self.settings.database.mongodb_atlas_uri:
                db_name = self.settings.database.mongodb_atlas_db
            else:
                db_name = self.settings.database.mongodb_db
            
            self.database = self.client[db_name]
            
            # 初始化集合
            for source_type, collection_name in self.collection_names.items():
                self.collections[source_type] = self.database[collection_name]
            
            # 验证连接
            await self.client.admin.command('ismaster')
            logger.info(f"✅ MongoDB Atlas连接成功: {db_name}")
            
            # 创建向量搜索索引
            await self._ensure_vector_indexes()
            
        except Exception as e:
            logger.error(f"❌ MongoDB Atlas连接失败: {str(e)}")
            raise
    
    async def _ensure_vector_indexes(self):
        """确保向量搜索索引存在"""
        index_definition = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": self.settings.llm.embedding_dimensions,
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "source_type"
                    },
                    {
                        "type": "filter", 
                        "path": "standard"
                    },
                    {
                        "type": "filter",
                        "path": "document_type"
                    }
                ]
            }
        }
        
        for source_type, collection in self.collections.items():
            try:
                # 检查索引是否存在
                existing_indexes = await collection.list_indexes().to_list(length=None)
                vector_index_exists = any(
                    idx.get("name") == "vector_index" for idx in existing_indexes
                )
                
                if not vector_index_exists:
                    # 使用同步客户端创建索引（MongoDB Atlas要求）
                    sync_client = pymongo.MongoClient(self.settings.database.mongodb_url)
                    sync_db = sync_client[self.database.name]
                    sync_collection = sync_db[collection.name]
                    
                    # 创建向量搜索索引
                    sync_collection.create_search_index(index_definition)
                    logger.info(f"✅ 为集合 {collection.name} 创建向量索引")
                    
                    sync_client.close()
                
            except Exception as e:
                logger.warning(f"⚠️ 为集合 {collection.name} 创建向量索引失败: {str(e)}")
    
    async def add_documents(
        self,
        documents: List[Document],
        embeddings: Embeddings,
        source_type: str = "common"
    ) -> List[str]:
        """添加文档到向量存储"""
        if source_type not in self.collections:
            raise ValueError(f"不支持的源类型: {source_type}")
        
        collection = self.collections[source_type]
        
        # 生成嵌入向量
        texts = [doc.page_content for doc in documents]
        embeddings_list = await embeddings.aembed_documents(texts)
        
        # 准备插入数据
        docs_to_insert = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings_list)):
            doc_dict = {
                "text": doc.page_content,
                "embedding": embedding,
                "metadata": doc.metadata,
                "source_type": source_type.upper(),
                "standard": doc.metadata.get("standard", source_type.upper()),
                "document_type": doc.metadata.get("document_type", "general"),
                "chunk_index": i,
                "source": doc.metadata.get("source", "unknown")
            }
            docs_to_insert.append(doc_dict)
        
        # 批量插入
        result = await collection.insert_many(docs_to_insert)
        inserted_ids = [str(id) for id in result.inserted_ids]
        
        logger.info(f"✅ 成功添加 {len(inserted_ids)} 个文档到 {source_type} 集合")
        return inserted_ids
    
    async def similarity_search(
        self,
        query: str,
        embeddings: Embeddings,
        source_type: Optional[str] = None,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        # 生成查询向量
        query_embedding = await embeddings.aembed_query(query)
        
        # 如果指定了source_type，只在对应集合中搜索
        if source_type and source_type in self.collections:
            collections_to_search = [self.collections[source_type]]
        else:
            # 在所有集合中搜索
            collections_to_search = list(self.collections.values())
        
        all_results = []
        
        for collection in collections_to_search:
            try:
                # 构建搜索管道
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": k * 2,  # 获取更多候选项
                            "limit": k,
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "metadata": 1,
                            "source_type": 1,
                            "standard": 1,
                            "document_type": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                # 添加过滤条件
                if filter_dict:
                    match_stage = {"$match": filter_dict}
                    pipeline.insert(1, match_stage)
                
                # 执行搜索
                cursor = collection.aggregate(pipeline)
                results = await cursor.to_list(length=k)
                
                # 转换为Document和分数的元组
                for result in results:
                    doc = Document(
                        page_content=result["text"],
                        metadata=result["metadata"]
                    )
                    score = result["score"]
                    all_results.append((doc, score))
                    
            except Exception as e:
                logger.warning(f"⚠️ 在集合 {collection.name} 中搜索失败: {str(e)}")
                continue
        
        # 按分数排序并返回top-k结果
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
    
    async def delete_documents(
        self,
        source_type: str,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """删除文档"""
        if source_type not in self.collections:
            raise ValueError(f"不支持的源类型: {source_type}")
        
        collection = self.collections[source_type]
        
        if filter_dict:
            result = await collection.delete_many(filter_dict)
        else:
            result = await collection.delete_many({})
        
        logger.info(f"✅ 从 {source_type} 集合删除了 {result.deleted_count} 个文档")
        return result.deleted_count
    
    async def get_collection_stats(self, source_type: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        if source_type not in self.collections:
            raise ValueError(f"不支持的源类型: {source_type}")
        
        collection = self.collections[source_type]
        
        # 获取文档数量
        total_docs = await collection.count_documents({})
        
        # 获取按标准分组的统计
        pipeline = [
            {
                "$group": {
                    "_id": "$standard",
                    "count": {"$sum": 1},
                    "doc_types": {"$addToSet": "$document_type"}
                }
            }
        ]
        
        cursor = collection.aggregate(pipeline)
        standard_stats = await cursor.to_list(length=None)
        
        stats = {
            "total_documents": total_docs,
            "collection_name": collection.name,
            "standards": {
                stat["_id"]: {
                    "count": stat["count"],
                    "document_types": stat["doc_types"]
                }
                for stat in standard_stats
            }
        }
        
        return stats
    
    async def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB Atlas连接已关闭")


def create_langchain_vector_store(
    source_type: str,
    embeddings: Embeddings,
    mongodb_manager: MongoDBAtlasVectorManager
) -> MongoDBAtlasVectorSearch:
    """创建LangChain MongoDB Atlas向量存储实例"""
    if source_type not in mongodb_manager.collection_names:
        raise ValueError(f"不支持的源类型: {source_type}")
    
    collection_name = mongodb_manager.collection_names[source_type]
    
    # 使用同步客户端创建LangChain向量存储
    sync_client = pymongo.MongoClient(mongodb_manager.settings.database.mongodb_url)
    
    if mongodb_manager.settings.database.mongodb_atlas_uri:
        db_name = mongodb_manager.settings.database.mongodb_atlas_db
    else:
        db_name = mongodb_manager.settings.database.mongodb_db
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=sync_client[db_name][collection_name],
        embedding=embeddings,
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )
    
    return vector_store


async def setup_mongodb_vector_stores(
    documents_by_type: Dict[str, List[Document]],
    embeddings: Embeddings
) -> MongoDBAtlasVectorManager:
    """设置MongoDB Atlas向量存储"""
    manager = MongoDBAtlasVectorManager()
    await manager.initialize()
    
    for source_type, documents in documents_by_type.items():
        if documents:
            try:
                await manager.add_documents(documents, embeddings, source_type)
                logger.info(f"✅ 成功设置 {source_type} 向量存储")
            except Exception as e:
                logger.error(f"❌ 设置 {source_type} 向量存储失败: {str(e)}")
    
    return manager


if __name__ == "__main__":
    async def main():
        # 测试MongoDB Atlas向量存储
        from knowledge_base.embeddings.embedding_models import get_default_embedding_model
        
        manager = MongoDBAtlasVectorManager()
        await manager.initialize()
        
        # 获取所有集合的统计信息
        for source_type in manager.collection_names.keys():
            try:
                stats = await manager.get_collection_stats(source_type)
                print(f"\n📊 {source_type.upper()} 集合统计:")
                print(f"  总文档数: {stats['total_documents']}")
                print(f"  集合名称: {stats['collection_name']}")
                print(f"  标准分布: {stats['standards']}")
            except Exception as e:
                print(f"❌ 获取 {source_type} 统计失败: {str(e)}")
        
        await manager.close()
    
    asyncio.run(main()) 