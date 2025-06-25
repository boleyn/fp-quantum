"""
量子智能化功能点估算系统 - Chroma向量存储

开发环境使用的轻量级向量存储解决方案
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Chroma向量存储管理器"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_prefix: str = "fp_quantum"
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_prefix = collection_prefix
        self.collections: Dict[str, Chroma] = {}
        
        # 确保持久化目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
    def get_collection(
        self,
        collection_name: str,
        embeddings: Embeddings,
        create_if_not_exists: bool = True
    ) -> Optional[Chroma]:
        """获取或创建集合"""
        
        full_collection_name = f"{self.collection_prefix}_{collection_name}"
        
        if full_collection_name in self.collections:
            return self.collections[full_collection_name]
        
        try:
            collection_path = self.persist_directory / full_collection_name
            
            if create_if_not_exists or collection_path.exists():
                vector_store = Chroma(
                    collection_name=full_collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(collection_path)
                )
                
                self.collections[full_collection_name] = vector_store
                logger.info(f"✅ Chroma集合已准备: {full_collection_name}")
                return vector_store
            else:
                logger.warning(f"Chroma集合不存在: {full_collection_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 获取Chroma集合失败 {full_collection_name}: {e}")
            return None
    
    def create_collection_from_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embeddings: Embeddings,
        batch_size: int = 100
    ) -> Optional[Chroma]:
        """从文档创建集合"""
        
        if not documents:
            logger.warning(f"文档列表为空，无法创建集合: {collection_name}")
            return None
        
        try:
            full_collection_name = f"{self.collection_prefix}_{collection_name}"
            collection_path = self.persist_directory / full_collection_name
            
            # 删除现有集合（如果存在）
            if collection_path.exists():
                import shutil
                shutil.rmtree(collection_path)
                logger.info(f"删除现有集合: {full_collection_name}")
            
            # 批量处理文档
            vector_store = None
            total_docs = len(documents)
            
            logger.info(f"📚 开始创建Chroma集合: {full_collection_name} ({total_docs} 文档)")
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                
                logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 文档)")
                
                if vector_store is None:
                    # 创建新的向量存储
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=full_collection_name,
                        persist_directory=str(collection_path)
                    )
                else:
                    # 添加文档到现有存储
                    vector_store.add_documents(batch)
            
            # 持久化
            if hasattr(vector_store, 'persist'):
                vector_store.persist()
            
            self.collections[full_collection_name] = vector_store
            
            logger.info(f"✅ Chroma集合创建完成: {full_collection_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ 创建Chroma集合失败 {collection_name}: {e}")
            return None
    
    def search_similar_documents(
        self,
        collection_name: str,
        query: str,
        embeddings: Embeddings,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """搜索相似文档"""
        
        vector_store = self.get_collection(collection_name, embeddings, create_if_not_exists=False)
        if not vector_store:
            logger.warning(f"集合不存在: {collection_name}")
            return []
        
        try:
            if filter_metadata:
                # 使用过滤器搜索
                results = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                # 基本相似度搜索
                results = vector_store.similarity_search(query=query, k=k)
            
            logger.debug(f"从 {collection_name} 检索到 {len(results)} 个相似文档")
            return results
            
        except Exception as e:
            logger.error(f"❌ 搜索相似文档失败 {collection_name}: {e}")
            return []
    
    def search_with_scores(
        self,
        collection_name: str,
        query: str,
        embeddings: Embeddings,
        k: int = 5,
        score_threshold: float = 0.5
    ) -> List[tuple]:
        """搜索相似文档并返回相似度分数"""
        
        vector_store = self.get_collection(collection_name, embeddings, create_if_not_exists=False)
        if not vector_store:
            logger.warning(f"集合不存在: {collection_name}")
            return []
        
        try:
            # 搜索带分数的结果
            results_with_scores = vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # 过滤低分结果
            filtered_results = [
                (doc, score) for doc, score in results_with_scores
                if score >= score_threshold
            ]
            
            logger.debug(
                f"从 {collection_name} 检索到 {len(filtered_results)} 个文档 "
                f"(阈值: {score_threshold})"
            )
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ 搜索带分数文档失败 {collection_name}: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str, embeddings: Embeddings) -> Dict[str, Any]:
        """获取集合统计信息"""
        
        vector_store = self.get_collection(collection_name, embeddings, create_if_not_exists=False)
        if not vector_store:
            return {"exists": False}
        
        try:
            # 获取集合信息
            collection = vector_store._collection
            stats = {
                "exists": True,
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata or {}
            }
            
            # 尝试获取额外统计信息
            try:
                # 获取所有文档ID
                all_docs = collection.get()
                if all_docs:
                    stats["document_ids"] = len(all_docs["ids"]) if all_docs.get("ids") else 0
                    
                    # 分析元数据
                    if all_docs.get("metadatas"):
                        metadata_analysis = self._analyze_metadata(all_docs["metadatas"])
                        stats["metadata_analysis"] = metadata_analysis
            except Exception as e:
                logger.debug(f"无法获取详细统计信息: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 获取集合统计失败 {collection_name}: {e}")
            return {"exists": False, "error": str(e)}
    
    def _analyze_metadata(self, metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析元数据"""
        
        analysis = {
            "total_documents": len(metadatas),
            "metadata_keys": set(),
            "source_types": {},
            "loader_types": {}
        }
        
        for metadata in metadatas:
            if metadata:
                # 收集元数据键
                analysis["metadata_keys"].update(metadata.keys())
                
                # 统计来源类型
                source_type = metadata.get("source_type", "unknown")
                analysis["source_types"][source_type] = \
                    analysis["source_types"].get(source_type, 0) + 1
                
                # 统计加载器类型
                loader_type = metadata.get("loader_type", "unknown")
                analysis["loader_types"][loader_type] = \
                    analysis["loader_types"].get(loader_type, 0) + 1
        
        # 转换集合为列表
        analysis["metadata_keys"] = list(analysis["metadata_keys"])
        
        return analysis
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        
        try:
            full_collection_name = f"{self.collection_prefix}_{collection_name}"
            collection_path = self.persist_directory / full_collection_name
            
            # 从内存中移除
            if full_collection_name in self.collections:
                del self.collections[full_collection_name]
            
            # 删除磁盘文件
            if collection_path.exists():
                import shutil
                shutil.rmtree(collection_path)
                logger.info(f"✅ 已删除Chroma集合: {full_collection_name}")
                return True
            else:
                logger.warning(f"集合不存在: {full_collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 删除集合失败 {collection_name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        
        try:
            collections = []
            
            # 从持久化目录扫描
            if self.persist_directory.exists():
                for item in self.persist_directory.iterdir():
                    if item.is_dir() and item.name.startswith(self.collection_prefix):
                        # 移除前缀
                        collection_name = item.name[len(self.collection_prefix) + 1:]
                        collections.append(collection_name)
            
            return sorted(collections)
            
        except Exception as e:
            logger.error(f"❌ 列出集合失败: {e}")
            return []
    
    def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """备份集合"""
        
        try:
            full_collection_name = f"{self.collection_prefix}_{collection_name}"
            source_path = self.persist_directory / full_collection_name
            backup_path = Path(backup_path)
            
            if not source_path.exists():
                logger.warning(f"集合不存在，无法备份: {collection_name}")
                return False
            
            # 创建备份目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制集合目录
            import shutil
            shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
            
            logger.info(f"✅ 集合备份完成: {collection_name} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 备份集合失败 {collection_name}: {e}")
            return False
    
    def restore_collection(self, collection_name: str, backup_path: str) -> bool:
        """恢复集合"""
        
        try:
            full_collection_name = f"{self.collection_prefix}_{collection_name}"
            target_path = self.persist_directory / full_collection_name
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                logger.warning(f"备份不存在，无法恢复: {backup_path}")
                return False
            
            # 删除现有集合
            if target_path.exists():
                import shutil
                shutil.rmtree(target_path)
            
            # 从备份恢复
            import shutil
            shutil.copytree(backup_path, target_path)
            
            # 从内存中移除（强制重新加载）
            if full_collection_name in self.collections:
                del self.collections[full_collection_name]
            
            logger.info(f"✅ 集合恢复完成: {backup_path} -> {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 恢复集合失败 {collection_name}: {e}")
            return False


def create_chroma_knowledge_base(
    documents_by_type: Dict[str, List[Document]],
    embeddings: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_prefix: str = "fp_quantum"
) -> ChromaVectorStore:
    """创建Chroma知识库"""
    
    logger.info("🚀 开始创建Chroma知识库...")
    
    # 创建Chroma存储管理器
    chroma_store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_prefix=collection_prefix
    )
    
    # 为每种类型的文档创建集合
    total_docs = 0
    created_collections = []
    
    for doc_type, documents in documents_by_type.items():
        if documents:
            collection = chroma_store.create_collection_from_documents(
                collection_name=doc_type,
                documents=documents,
                embeddings=embeddings
            )
            
            if collection:
                created_collections.append(doc_type)
                total_docs += len(documents)
                logger.info(f"✅ 集合 '{doc_type}' 创建完成: {len(documents)} 文档")
            else:
                logger.error(f"❌ 集合 '{doc_type}' 创建失败")
    
    logger.info(f"🎉 Chroma知识库创建完成!")
    logger.info(f"   总文档数: {total_docs}")
    logger.info(f"   集合数: {len(created_collections)}")
    logger.info(f"   集合列表: {created_collections}")
    
    return chroma_store


if __name__ == "__main__":
    # 测试Chroma向量存储
    
    from langchain_openai import OpenAIEmbeddings
    
    # 创建示例文档
    test_documents = [
        Document(
            page_content="NESMA功能点分析是一种标准化的软件规模估算方法",
            metadata={"source_type": "NESMA", "category": "introduction"}
        ),
        Document(
            page_content="COSMIC功能点分析基于数据移动的概念进行估算",
            metadata={"source_type": "COSMIC", "category": "introduction"}
        ),
        Document(
            page_content="内部逻辑文件(ILF)是应用程序内部维护的数据",
            metadata={"source_type": "NESMA", "category": "data_function"}
        )
    ]
    
    # 创建嵌入模型（需要配置API密钥）
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 创建Chroma存储
        chroma_store = ChromaVectorStore(persist_directory="./test_chroma")
        
        # 测试创建集合
        collection = chroma_store.create_collection_from_documents(
            collection_name="test",
            documents=test_documents,
            embeddings=embeddings
        )
        
        if collection:
            print("✅ Chroma集合创建成功")
            
            # 测试搜索
            results = chroma_store.search_similar_documents(
                collection_name="test",
                query="什么是NESMA",
                embeddings=embeddings,
                k=2
            )
            
            print(f"搜索结果: {len(results)} 个文档")
            for i, doc in enumerate(results):
                print(f"  {i+1}. {doc.page_content[:50]}...")
        
        # 测试统计信息
        stats = chroma_store.get_collection_stats("test", embeddings)
        print(f"集合统计: {stats}")
        
    except Exception as e:
        print(f"测试失败 (可能需要配置OpenAI API密钥): {e}") 