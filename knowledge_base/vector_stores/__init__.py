"""
量子智能化功能点估算系统 - 向量存储模块

统一使用PgVector向量存储解决方案
"""

from .pgvector_store import PgVectorStore, create_pgvector_store
from .hybrid_search import HybridSearchStrategy, AdaptiveHybridSearch

# 主要导出
__all__ = [
    "PgVectorStore",
    "create_pgvector_store", 
    "HybridSearchStrategy",
    "AdaptiveHybridSearch"
]

# 默认向量存储工厂
async def create_vector_store(provider: str = "pgvector", **kwargs):
    """创建向量存储实例"""
    
    if provider.lower() == "pgvector":
        return await create_pgvector_store(**kwargs)
    elif provider.lower() == "hybrid":
        return HybridSearchStrategy(**kwargs)
    else:
        raise ValueError(f"不支持的向量存储提供商: {provider}。仅支持 'pgvector' 和 'hybrid'")

# 向量存储配置
VECTOR_STORE_CONFIG = {
    "pgvector": {
        "name": "PgVector向量存储",
        "description": "基于PostgreSQL + pgvector扩展的企业级向量存储",
        "use_cases": ["生产环境", "开发环境", "大规模数据", "高性能检索"],
        "features": ["ACID事务", "SQL查询", "高可用", "水平扩展", "统一架构"]
    },
    "hybrid": {
        "name": "混合搜索策略",
        "description": "结合向量检索和关键词搜索的混合解决方案", 
        "use_cases": ["复杂查询", "多模态检索", "结果优化"],
        "features": ["语义+关键词", "权重调整", "结果融合", "多策略"]
    }
} 