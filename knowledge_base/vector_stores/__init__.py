"""
量子智能化功能点估算系统 - 向量存储模块

提供多种向量存储后端的统一接口
"""

from .mongodb_atlas import (
    MongoDBAtlasVectorStore,
    setup_mongodb_vector
)

__all__ = [
    "MongoDBAtlasVectorStore",
    "setup_mongodb_vector"
]

__version__ = "1.0.0" 