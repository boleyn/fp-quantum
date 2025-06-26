"""
量子智能化功能点估算系统 - 知识库模块

提供完整的RAG（检索增强生成）解决方案，包括：
- 文档加载和处理
- 向量存储和索引
- 多种检索策略
- RAG链构建和查询

功能特性：
1. 支持多种文档格式（PDF、Word、Markdown、网页等）
2. 基于PgVector的向量存储后端
3. 混合检索策略（语义+关键词+BM25）
4. NESMA和COSMIC专用检索优化
5. 异步处理和批量操作
6. 自动查询扩展和结果重排序
"""

# 核心组件导入
from .rag_chains import (
    RAGChainBuilder,
    RAGChainFactory,
    setup_default_rag_system,
    DEFAULT_DOCUMENT_PATHS
)

# 文档加载器
from .loaders.pdf_loader import (
    EnhancedPDFLoader,
    BatchPDFProcessor,
    pdf_loader_factory
)
from .loaders.custom_loaders import (
    FunctionPointDocumentLoader,
    JSONDocumentLoader,
    CSVDocumentLoader,
    YAMLDocumentLoader
)
from .loaders.web_loader import (
    EnhancedWebLoader,
    load_web_knowledge_base,
    WEB_SOURCES
)

# 向量存储相关导入 - 统一使用PgVector
from .vector_stores.pgvector_store import (
    PgVectorStore,
    create_pgvector_store,
)

# 检索器
from .retrievers.semantic_retriever import (
    EnhancedSemanticRetriever,
    PgVectorMultiSourceRetriever,
    create_pgvector_retrievers
)
from .retrievers.keyword_retriever import (
    TFIDFRetriever,
    BooleanRetriever,
    NESMAKeywordRetriever,
    COSMICKeywordRetriever,
    KeywordRetrieverFactory
)

# 嵌入模型
from .embeddings.embedding_models import (
    get_embedding_model,
    get_default_embedding_model,
    get_embedding_manager,
    EmbeddingModelManager,
    test_embedding_model
)

# 自动设置
from .auto_setup import AutoKnowledgeBaseSetup

# 版本信息
__version__ = "1.0.0"
__author__ = "量子智能化功能点估算团队"

# 主要导出
__all__ = [
    # 文档加载器
    "EnhancedPDFLoader",
    "BatchPDFProcessor", 
    "load_web_knowledge_base",
    "FunctionPointDocumentLoader",
    
    # 向量存储 - 仅PgVector
    "PgVectorStore",
    "create_pgvector_store",
    
    # 搜索和检索
    "HybridSearchStrategy",
    "AdaptiveHybridSearch",
    "EnhancedSemanticRetriever",
    "PgVectorMultiSourceRetriever",
    "create_pgvector_retrievers",
    "TFIDFRetriever",
    "NESMAKeywordRetriever",
    "COSMICKeywordRetriever",
    "NESMAMultiQueryRetriever",
    "COSMICMultiQueryRetriever",
    
    # 嵌入模型
    "get_embedding_model",
    
    # RAG链
    "RAGChainBuilder",
    "NESMARagChain",
    "COSMIRagChain",
    
    # 自动设置
    "auto_setup_knowledge_base",
]


# 便捷函数
async def quick_setup_rag(
    document_paths: dict = None,
    embedding_model: str = "bge_m3",
    vector_store: str = "pgvector",
    include_web: bool = False
):
    """
    快速设置RAG系统的便捷函数
    
    Args:
        document_paths: 文档路径字典，默认使用DEFAULT_DOCUMENT_PATHS
        embedding_model: 嵌入模型名称
        vector_store: 向量存储类型 (仅支持pgvector)
        include_web: 是否包含网页资源
        
    Returns:
        RAGChainBuilder: 配置好的RAG系统
    """
    
    if document_paths is None:
        document_paths = DEFAULT_DOCUMENT_PATHS
    
    return await RAGChainFactory.create_complete_rag_system(
        document_paths=document_paths,
        embedding_model_name=embedding_model,
        vector_store_type=vector_store,
        include_web_sources=include_web
    )


def get_available_models():
    """获取可用的嵌入模型列表"""
    manager = get_embedding_manager()
    return manager.get_available_models()


def get_supported_formats():
    """获取支持的文档格式"""
    return [
        "PDF (.pdf)",
        "Word (.docx, .doc)", 
        "Markdown (.md)",
        "文本文件 (.txt)",
        "JSON (.json)",
        "CSV (.csv)",
        "YAML (.yml, .yaml)",
        "网页 (HTML)"
    ]


# 配置信息
KNOWLEDGE_BASE_CONFIG = {
    "default_chunk_size": 1000,
    "default_chunk_overlap": 200,
    "default_embedding_model": "bge_m3",
    "default_vector_store": "pgvector",
    "supported_languages": ["zh", "en"],
    "max_file_size": "50MB",
    "batch_size": 100
}


# 模块级别的日志配置
import logging

logger = logging.getLogger(__name__)
logger.info(f"知识库模块已加载 (版本: {__version__})")
logger.info(f"支持的文档格式: {len(get_supported_formats())} 种")
logger.info(f"可用的嵌入模型: {len(get_available_models())} 个")


# 导入时的健康检查
def _health_check():
    """模块健康检查"""
    
    issues = []
    
    # 检查必要的依赖
    try:
        import langchain_core
    except ImportError:
        issues.append("langchain_core 未安装")
    
    try:
        import langchain_postgres
    except ImportError:
        issues.append("langchain_postgres 未安装 (PgVector向量存储将不可用)")
    
    try:
        import psycopg
    except ImportError:
        issues.append("psycopg 未安装 (PostgreSQL连接将不可用)")
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        issues.append("rank-bm25 未安装 (BM25检索将不可用)")
    
    if issues:
        logger.warning(f"知识库模块依赖检查发现问题: {issues}")
        logger.warning("部分功能可能受限，请安装相应依赖")
    else:
        logger.info("✅ 知识库模块依赖检查通过")

# 执行健康检查
_health_check()

# 自动设置知识库的便捷函数
async def auto_setup_knowledge_base():
    """自动设置知识库的便捷函数"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        setup = AutoKnowledgeBaseSetup()
        processed_docs = await setup.setup_documents()
        logger.info(f"✅ 知识库自动设置成功，处理了 {len(processed_docs)} 个文档块")
        return len(processed_docs) > 0
    except Exception as e:
        logger.error(f"知识库自动设置失败: {e}")
        return False
