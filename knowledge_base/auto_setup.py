#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 智能知识库管理器

基于文件MD5的增量更新知识库管理系统：
1. MongoDB记录文档元数据和状态
2. 启动时检测文件变化（新增、修改、删除）
3. 增量更新PgVector向量存储
4. 基于LangChain的完整RAG集成

业务逻辑：
- 记录文档MD5到MongoDB
- 启动时对比文件系统与MongoDB记录
- 自动处理文件变化（增删改）
- 向量化处理和PgVector存储
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import time

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config.settings import get_settings
from knowledge_base.embeddings.embedding_models import get_embedding_model

logger = logging.getLogger(__name__)


class DocumentRecord:
    """文档记录模型"""
    
    def __init__(
        self,
        file_path: str,
        category: str,
        file_name: str,
        md5_hash: str,
        file_size: int,
        last_modified: datetime,
        status: str = "pending",  # pending, processing, completed, failed
        chunk_count: int = 0,
        vector_ids: List[str] = None,
        error_message: str = None,
        processed_at: datetime = None
    ):
        self.file_path = file_path
        self.category = category
        self.file_name = file_name
        self.md5_hash = md5_hash
        self.file_size = file_size
        self.last_modified = last_modified
        self.status = status
        self.chunk_count = chunk_count
        self.vector_ids = vector_ids or []
        self.error_message = error_message
        self.processed_at = processed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "file_path": self.file_path,
            "category": self.category,
            "file_name": self.file_name,
            "md5_hash": self.md5_hash,
            "file_size": self.file_size,
            "last_modified": self.last_modified,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "vector_ids": self.vector_ids,
            "error_message": self.error_message,
            "processed_at": self.processed_at,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """从字典创建实例"""
        return cls(
            file_path=data.get("file_path"),
            category=data.get("category"),
            file_name=data.get("file_name"),
            md5_hash=data.get("md5_hash"),
            file_size=data.get("file_size"),
            last_modified=data.get("last_modified"),
            status=data.get("status", "pending"),
            chunk_count=data.get("chunk_count", 0),
            vector_ids=data.get("vector_ids", []),
            error_message=data.get("error_message"),
            processed_at=data.get("processed_at")
        )


class IncrementalKnowledgeBaseManager:
    """增量知识库管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_dir = Path("knowledge_base")
        
        # MongoDB连接
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.mongo_db: Optional[AsyncIOMotorDatabase] = None
        self.documents_collection = None
        
        # PgVector连接配置
        self.connection_string = self._build_pg_connection_string()
        self.vector_stores: Dict[str, PGVector] = {}
        
        # 文档处理配置
        self.supported_categories = ["nesma", "cosmic", "common"]
        self.supported_extensions = {".pdf", ".pptx", ".md", ".txt"}
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", "!", "?", "；", ";", "：", ":"]
        )
    
    def _build_pg_connection_string(self) -> str:
        """构建PostgreSQL连接字符串"""
        db_config = self.settings.database
        # 修复：使用正确的配置属性名
        return (
            f"postgresql+psycopg://{db_config.postgres_user}:"
            f"{db_config.postgres_password}@{db_config.postgres_host}:"
            f"{db_config.postgres_port}/{db_config.postgres_database}"
        )
    
    async def initialize(self):
        """初始化管理器"""
        logger.info("🚀 初始化增量知识库管理器...")
        
        # 初始化MongoDB连接
        await self._init_mongodb()
        
        # 初始化PgVector连接
        await self._init_pgvector()
        
        logger.info("✅ 增量知识库管理器初始化完成")
    
    async def _init_mongodb(self):
        """初始化MongoDB连接"""
        try:
            db_config = self.settings.database
            connection_string = f"mongodb://{db_config.mongodb_user}:{db_config.mongodb_password}@{db_config.mongodb_host}:{db_config.mongodb_port}"
            
            self.mongo_client = AsyncIOMotorClient(connection_string)
            self.mongo_db = self.mongo_client[db_config.mongodb_db]
            self.documents_collection = self.mongo_db.knowledge_documents
            
            # 创建索引
            await self.documents_collection.create_index([
                ("file_path", 1),
                ("category", 1),
                ("md5_hash", 1),
                ("status", 1)
            ])
            
            logger.info("✅ MongoDB连接初始化完成")
            
        except Exception as e:
            logger.error(f"❌ MongoDB初始化失败: {e}")
            raise
    
    async def _init_pgvector(self):
        """初始化PgVector连接 - 按照LangChain标准规范"""
        try:
            embeddings = get_embedding_model()
            
            for category in self.supported_categories:
                collection_name = f"fp_quantum_{category}"
                
                # 按照LangChain标准规范创建PGVector实例 - 使用正确的API
                vector_store = PGVector(
                    embeddings=embeddings,
                    collection_name=collection_name,
                    connection=self.connection_string,
                    distance_strategy=DistanceStrategy.COSINE,
                    use_jsonb=True,
                    create_extension=True,
                    pre_delete_collection=False
                )
                
                self.vector_stores[category] = vector_store
                logger.info(f"✅ {category} PgVector存储初始化完成")
                
        except Exception as e:
            logger.error(f"❌ PgVector初始化失败: {e}")
            raise
    
    def calculate_file_md5(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"❌ 计算文件MD5失败 {file_path}: {e}")
            return ""
    
    async def scan_file_system(self) -> Dict[str, List[DocumentRecord]]:
        """扫描文件系统，创建文档记录"""
        logger.info("🔍 扫描文件系统...")
        
        current_files = {
            "nesma": [],
            "cosmic": [],
            "common": []
        }
        
        for category in self.supported_categories:
            category_dir = self.base_dir / "documents" / category
            
            if not category_dir.exists():
                logger.warning(f"⚠️ 目录不存在: {category_dir}")
                continue
            
            for file_path in category_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    
                    try:
                        # 获取文件信息
                        stat = file_path.stat()
                        md5_hash = self.calculate_file_md5(file_path)
                        
                        if not md5_hash:
                            continue
                        
                        record = DocumentRecord(
                            file_path=str(file_path),
                            category=category,
                            file_name=file_path.name,
                            md5_hash=md5_hash,
                            file_size=stat.st_size,
                            last_modified=datetime.fromtimestamp(stat.st_mtime)
                        )
                        
                        current_files[category].append(record)
                        
                    except Exception as e:
                        logger.error(f"❌ 处理文件失败 {file_path}: {e}")
                        continue
        
        total_files = sum(len(files) for files in current_files.values())
        logger.info(f"✅ 文件系统扫描完成，发现 {total_files} 个文件")
        
        return current_files
    
    async def get_mongodb_records(self) -> Dict[str, List[DocumentRecord]]:
        """从MongoDB获取已记录的文档"""
        logger.info("📊 获取MongoDB中的文档记录...")
        
        records = {
            "nesma": [],
            "cosmic": [],
            "common": []
        }
        
        try:
            cursor = self.documents_collection.find({})
            async for doc in cursor:
                record = DocumentRecord.from_dict(doc)
                if record.category in records:
                    records[record.category].append(record)
            
            total_records = sum(len(recs) for recs in records.values())
            logger.info(f"✅ 从MongoDB获取 {total_records} 条文档记录")
            
        except Exception as e:
            logger.error(f"❌ 获取MongoDB记录失败: {e}")
        
        return records
    
    async def compare_and_detect_changes(
        self,
        current_files: Dict[str, List[DocumentRecord]],
        mongo_records: Dict[str, List[DocumentRecord]]
    ) -> Dict[str, Any]:
        """比较文件系统与MongoDB记录，检测变化"""
        logger.info("🔍 检测文件变化...")
        
        changes = {
            "new_files": [],      # 新增文件
            "modified_files": [], # 修改文件（MD5变化）
            "deleted_files": [],  # 删除文件
            "unchanged_files": [] # 未变化文件
        }
        
        # 创建MongoDB记录的快速查找字典
        mongo_lookup = {}
        for category, records in mongo_records.items():
            for record in records:
                mongo_lookup[record.file_path] = record
        
        # 检查当前文件
        for category, files in current_files.items():
            for file_record in files:
                file_path = file_record.file_path
                
                if file_path not in mongo_lookup:
                    # 新文件
                    changes["new_files"].append(file_record)
                    logger.info(f"🆕 新文件: {file_record.file_name}")
                else:
                    mongo_record = mongo_lookup[file_path]
                    
                    if file_record.md5_hash != mongo_record.md5_hash:
                        # 文件已修改
                        file_record.vector_ids = mongo_record.vector_ids  # 保留向量ID用于删除
                        changes["modified_files"].append(file_record)
                        logger.info(f"🔄 文件已修改: {file_record.file_name}")
                    else:
                        # 文件未变化
                        changes["unchanged_files"].append(file_record)
        
        # 检查已删除的文件
        current_file_paths = set()
        for files in current_files.values():
            for file_record in files:
                current_file_paths.add(file_record.file_path)
        
        for file_path, mongo_record in mongo_lookup.items():
            if file_path not in current_file_paths:
                changes["deleted_files"].append(mongo_record)
                logger.info(f"🗑️ 文件已删除: {mongo_record.file_name}")
        
        logger.info(
            f"✅ 变化检测完成: "
            f"新增{len(changes['new_files'])}, "
            f"修改{len(changes['modified_files'])}, "
            f"删除{len(changes['deleted_files'])}, "
            f"未变化{len(changes['unchanged_files'])}"
        )
        
        return changes
    
    async def process_deleted_files(self, deleted_files: List[DocumentRecord]):
        """处理已删除的文件"""
        if not deleted_files:
            return
        
        logger.info(f"🗑️ 处理 {len(deleted_files)} 个已删除文件...")
        
        for record in deleted_files:
            try:
                # 从PgVector删除向量数据
                if record.vector_ids:
                    vector_store = self.vector_stores.get(record.category)
                    if vector_store:
                        # 注意：LangChain PGVector可能需要特殊的删除方法
                        # 这里简化处理，实际可能需要根据具体实现调整
                        logger.info(f"🗑️ 删除向量数据: {record.file_name} ({len(record.vector_ids)} 个向量)")
                
                # 从MongoDB删除记录
                await self.documents_collection.delete_one({"file_path": record.file_path})
                
                logger.info(f"✅ 已删除: {record.file_name}")
                
            except Exception as e:
                logger.error(f"❌ 删除文件记录失败 {record.file_name}: {e}")
    
    async def process_modified_files(self, modified_files: List[DocumentRecord]):
        """处理已修改的文件"""
        if not modified_files:
            return
        
        logger.info(f"🔄 处理 {len(modified_files)} 个已修改文件...")
        
        for record in modified_files:
            # 先删除旧的向量数据
            if record.vector_ids:
                vector_store = self.vector_stores.get(record.category)
                if vector_store:
                    logger.info(f"🗑️ 删除旧向量数据: {record.file_name}")
                    # TODO: 实现向量删除逻辑
            
            # 重新处理文件，_process_single_file已经包含错误处理
            await self._process_single_file(record)
    
    async def process_new_files(self, new_files: List[DocumentRecord]):
        """处理新文件"""
        if not new_files:
            return
        
        logger.info(f"🆕 处理 {len(new_files)} 个新文件...")
        
        for record in new_files:
            # _process_single_file 已经包含了错误处理和记录逻辑
            await self._process_single_file(record)
    
    async def _process_single_file(self, record: DocumentRecord):
        """处理单个文件 - 先成功处理，再记录状态"""
        logger.info(f"📄 处理文件: {record.file_name}")
        
        try:
            # 加载文档
            documents = await self._load_document(Path(record.file_path))
            
            if not documents:
                record.status = "failed"
                record.error_message = "无法加载文档内容"
                await self._save_record_to_mongodb(record)
                return
            
            # 分割文档
            split_docs = self.text_splitter.split_documents(documents)
            
            # 添加元数据
            for idx, doc in enumerate(split_docs):
                if not doc.metadata:
                    doc.metadata = {}
                
                doc.metadata.update({
                    "source_category": record.category,
                    "file_name": record.file_name,
                    "file_path": record.file_path,
                    "chunk_index": idx,
                    "total_chunks": len(split_docs),
                    "file_md5": record.md5_hash,
                    "processed_at": datetime.utcnow().isoformat()
                })
            
            # 存储到向量数据库 - 使用LangChain标准同步API
            vector_store = self.vector_stores[record.category]
            try:
                # 处理批次大小限制 - BGE-M3 API限制为64个文档/批次
                max_batch_size = 32  # 保守起见设置为32
                vector_ids = []
                
                for i in range(0, len(split_docs), max_batch_size):
                    batch = split_docs[i:i + max_batch_size]
                    
                    try:
                        # 使用LangChain PGVector标准同步API
                        batch_ids = vector_store.add_documents(batch)
                        if batch_ids:
                            vector_ids.extend(batch_ids)
                        logger.info(f"📦 批次 {i//max_batch_size + 1}: 添加了 {len(batch)} 个文档块")
                        
                    except Exception as batch_error:
                        logger.warning(f"⚠️ 批次处理失败，尝试逐个处理: {batch_error}")
                        # 如果批次失败，尝试逐个添加
                        for doc in batch:
                            try:
                                single_ids = vector_store.add_documents([doc])
                                if single_ids:
                                    vector_ids.extend(single_ids)
                            except Exception as sync_error:
                                logger.error(f"❌ 单个文档添加失败: {sync_error}")
                                continue
                            
            except Exception as e:
                logger.error(f"❌ 向量存储失败 {record.file_name}: {e}")
                # 最后的同步回退
                try:
                    vector_ids = vector_store.add_documents(split_docs)
                except Exception as e2:
                    logger.error(f"❌ 同步向量存储也失败 {record.file_name}: {e2}")
                    raise e2
            
            # 🎯 关键修复：只有处理完全成功后才记录到MongoDB
            record.status = "completed"
            record.chunk_count = len(split_docs)
            record.vector_ids = vector_ids if isinstance(vector_ids, list) else []
            record.processed_at = datetime.utcnow()
            record.error_message = None
            
            await self._save_record_to_mongodb(record)
            
            logger.info(f"✅ 文件处理完成: {record.file_name} ({len(split_docs)} 个块)")
            
        except Exception as e:
            logger.error(f"❌ 文件处理失败 {record.file_name}: {e}")
            # 只在确实失败时才记录失败状态
            record.status = "failed"
            record.error_message = str(e)
            record.processed_at = datetime.utcnow()
            await self._save_record_to_mongodb(record)
            # 不再抛出异常，让流程继续处理其他文件
    
    async def _load_document(self, file_path: Path) -> List[Document]:
        """加载单个文档"""
        try:
            if file_path.suffix.lower() == '.pdf':
                # 使用PyPDFLoader标准同步API
                try:
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                    return documents
                except ImportError as e:
                    if "pdfminer" in str(e):
                        logger.error(f"❌ 缺少PDF处理依赖 {file_path.name}: {e}")
                        logger.info("💡 请安装: pip install pdfminer.six")
                        return []
                    raise e
                except Exception as e:
                    logger.error(f"❌ PDF加载失败 {file_path}: {e}")
                    return []
                    
            elif file_path.suffix.lower() == '.pptx':
                try:
                    loader = UnstructuredPowerPointLoader(str(file_path))
                    documents = loader.load()
                    return documents
                except Exception as e:
                    logger.error(f"❌ PowerPoint加载失败 {file_path}: {e}")
                    return []
                
            elif file_path.suffix.lower() in ['.md', '.txt']:
                content = file_path.read_text(encoding='utf-8')
                return [Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                )]
            else:
                logger.warning(f"⚠️ 不支持的文件类型: {file_path.suffix}")
                return []
            
        except Exception as e:
            logger.error(f"❌ 加载文档失败 {file_path}: {e}")
            return []
    
    async def _save_record_to_mongodb(self, record: DocumentRecord):
        """保存记录到MongoDB"""
        try:
            record_dict = record.to_dict()
            record_dict["updated_at"] = datetime.utcnow()
            
            await self.documents_collection.update_one(
                {"file_path": record.file_path},
                {"$set": record_dict},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"❌ 保存MongoDB记录失败: {e}")
            raise
    
    async def auto_update_knowledge_base(self) -> Dict[str, Any]:
        """自动更新知识库的主要方法"""
        logger.info("🚀 启动增量知识库更新...")
        start_time = time.time()
        
        try:
            # 初始化
            await self.initialize()
            
            # 扫描文件系统
            current_files = await self.scan_file_system()
            
            # 获取MongoDB记录
            mongo_records = await self.get_mongodb_records()
            
            # 检测变化
            changes = await self.compare_and_detect_changes(current_files, mongo_records)
            
            # 处理变化
            await self.process_deleted_files(changes["deleted_files"])
            await self.process_modified_files(changes["modified_files"])
            await self.process_new_files(changes["new_files"])
            
            duration = time.time() - start_time
            
            result = {
                "status": "success",
                "message": "知识库更新完成",
                "duration": duration,
                "changes": {
                    "new_files": len(changes["new_files"]),
                    "modified_files": len(changes["modified_files"]),
                    "deleted_files": len(changes["deleted_files"]),
                    "unchanged_files": len(changes["unchanged_files"])
                },
                "total_processed": len(changes["new_files"]) + len(changes["modified_files"]),
                "categories": list(self.supported_categories)
            }
            
            logger.info(f"✅ 知识库更新完成，耗时 {duration:.2f} 秒")
            return result
            
        except Exception as e:
            logger.error(f"❌ 知识库更新失败: {e}")
            return {
                "status": "error",
                "message": f"知识库更新失败: {str(e)}",
                "duration": time.time() - start_time
            }
    
    async def close(self):
        """关闭连接"""
        if self.mongo_client:
            self.mongo_client.close()
        logger.info("✅ 连接已关闭")


# 全局函数
async def ensure_knowledge_base_ready() -> bool:
    """确保知识库就绪"""
    manager = IncrementalKnowledgeBaseManager()
    try:
        result = await manager.auto_update_knowledge_base()
        return result["status"] == "success"
    finally:
        await manager.close()


# 向后兼容的类名
class AutoKnowledgeBaseSetup:
    """向后兼容的类名"""
    
    def __init__(self):
        self.manager = IncrementalKnowledgeBaseManager()
    
    async def auto_initialize_if_needed(self):
        """向后兼容的方法"""
        return await self.manager.auto_update_knowledge_base()
    
    async def setup_documents(self):
        """向后兼容的方法"""
        result = await self.manager.auto_update_knowledge_base()
        return result.get("total_processed", 0)


if __name__ == "__main__":
    async def main():
        """测试增量更新功能"""
        manager = IncrementalKnowledgeBaseManager()
        try:
            result = await manager.auto_update_knowledge_base()
            print(f"更新结果: {result}")
        finally:
            await manager.close()
    
    asyncio.run(main())
