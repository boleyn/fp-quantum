"""
量子智能化功能点估算系统 - 配置管理模块

基于Pydantic Settings的配置管理，支持环境变量自动加载和验证
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    
    # PostgreSQL配置 (PG Vector)
    postgres_host: str = Field(default="10.21.8.6", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_database: str = Field(default="fp_quantum", env="POSTGRES_DATABASE")
    postgres_db: str = Field(default="fp_quantum", env="POSTGRES_DB")  # 向后兼容
    postgres_user: str = Field(default="vector_user", env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    postgres_pool_size: int = Field(default=5, env="POSTGRES_POOL_SIZE")
    postgres_max_overflow: int = Field(default=10, env="POSTGRES_MAX_OVERFLOW")
    
    # MongoDB配置
    mongodb_host: str = Field(default="10.21.8.6", env="MONGODB_HOST")
    mongodb_port: int = Field(default=27017, env="MONGODB_PORT")
    mongodb_db: str = Field(default="fp_quantum", env="MONGODB_DB")
    mongodb_user: str = Field(default="fp_user", env="MONGODB_USER")
    mongodb_password: str = Field(env="MONGODB_PASSWORD")
    mongodb_max_pool_size: int = Field(default=100, env="MONGODB_MAX_POOL_SIZE")
    mongodb_min_pool_size: int = Field(default=10, env="MONGODB_MIN_POOL_SIZE")
    mongodb_max_idle_time_ms: int = Field(default=30000, env="MONGODB_MAX_IDLE_TIME_MS")
    
    # MongoDB Atlas配置 (生产环境)
    mongodb_atlas_uri: Optional[str] = Field(default=None, env="MONGODB_ATLAS_URI")
    mongodb_atlas_db: str = Field(default="fp_quantum_prod", env="MONGODB_ATLAS_DB")
    
    @property
    def postgres_url(self) -> str:
        """PostgreSQL连接URL"""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
    
    @property
    def postgres_async_url(self) -> str:
        """PostgreSQL异步连接URL"""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
    
    @property
    def mongodb_url(self) -> str:
        """MongoDB连接URL"""
        if self.mongodb_atlas_uri:
            return self.mongodb_atlas_uri
        return f"mongodb://{self.mongodb_user}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_db}"


class LLMConfig(BaseSettings):
    """LLM模型配置"""
    
    # DeepSeek API配置
    deepseek_api_key: str = Field(env="DEEPSEEK_API_KEY")
    deepseek_api_base: str = Field(default="https://api.deepseek.com/v1", env="DEEPSEEK_API_BASE")
    
    # BGE-M3 API配置
    bge_m3_api_key: str = Field(env="BGE_M3_API_KEY")
    bge_m3_api_base: str = Field(default="https://api.bge-provider.com/v1", env="BGE_M3_API_BASE")
    
    # OpenAI API配置 (备用)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # 编排者模型配置 (DeepSeek-R1)
    orchestrator_model: str = Field(default="deepseek-r1-250528", env="ORCHESTRATOR_MODEL")
    orchestrator_temperature: float = Field(default=0.1, env="ORCHESTRATOR_TEMPERATURE")
    orchestrator_max_tokens: int = Field(default=8000, env="ORCHESTRATOR_MAX_TOKENS")
    orchestrator_thinking_budget: int = Field(default=30000, env="ORCHESTRATOR_THINKING_BUDGET")
    orchestrator_timeout: int = Field(default=60, env="ORCHESTRATOR_TIMEOUT")
    
    # 执行者模型配置 (DeepSeek-V3)
    worker_model: str = Field(default="deepseek-v3-250324", env="WORKER_MODEL")
    worker_temperature: float = Field(default=0.1, env="WORKER_TEMPERATURE")
    worker_max_tokens: int = Field(default=128000, env="WORKER_MAX_TOKENS")
    worker_parallel_limit: int = Field(default=5, env="WORKER_PARALLEL_LIMIT")
    worker_timeout: int = Field(default=30, env="WORKER_TIMEOUT")
    
    # 向量模型配置 (BGE-M3)
    embedding_model: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL")
    embedding_max_tokens: int = Field(default=8000, env="EMBEDDING_MAX_TOKENS")
    embedding_dimensions: int = Field(default=1024, env="EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    embedding_timeout: int = Field(default=30, env="EMBEDDING_TIMEOUT")


class VectorStoreConfig(BaseSettings):
    """向量存储配置 - 统一使用PgVector"""
    
    # 向量存储提供商 - 只支持PgVector
    provider: str = Field(default="pgvector", env="VECTOR_STORE_PROVIDER")
    table_name: str = Field(default="knowledge_embeddings", env="VECTOR_TABLE_NAME")
    vector_dimension: int = Field(default=1024, env="VECTOR_DIMENSION")
    index_type: str = Field(default="ivfflat", env="VECTOR_INDEX_TYPE")  # ivfflat, hnsw
    index_lists: int = Field(default=100, env="VECTOR_INDEX_LISTS")
    
    @validator('provider')
    def validate_provider(cls, v):
        if v != 'pgvector':
            raise ValueError('仅支持 pgvector 作为向量存储提供商')
        return v


class KnowledgeBaseConfig(BaseSettings):
    """知识库配置"""
    
    # 文档处理配置
    document_processor: str = Field(default="langchain", env="DOCUMENT_PROCESSOR")
    document_output_format: str = Field(default="markdown", env="DOCUMENT_OUTPUT_FORMAT")
    document_extract_images: bool = Field(default=True, env="DOCUMENT_EXTRACT_IMAGES")
    document_extract_tables: bool = Field(default=True, env="DOCUMENT_EXTRACT_TABLES")
    document_ocr_enabled: bool = Field(default=True, env="DOCUMENT_OCR_ENABLED")
    
    # 知识库路径
    nesma_docs_path: Path = Field(default=Path("./knowledge_base/documents/nesma"), env="NESMA_DOCS_PATH")
    cosmic_docs_path: Path = Field(default=Path("./knowledge_base/documents/cosmic"), env="COSMIC_DOCS_PATH")
    common_docs_path: Path = Field(default=Path("./knowledge_base/documents/common"), env="COMMON_DOCS_PATH")
    
    # 文档分块配置
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    @validator('nesma_docs_path', 'cosmic_docs_path', 'common_docs_path')
    def validate_paths(cls, v):
        return Path(v)


class WorkflowConfig(BaseSettings):
    """工作流配置"""
    
    # 工作流设置
    max_retries: int = Field(default=3, env="WORKFLOW_MAX_RETRIES")
    timeout_seconds: int = Field(default=300, env="WORKFLOW_TIMEOUT_SECONDS")
    enable_validation: bool = Field(default=True, env="WORKFLOW_ENABLE_VALIDATION")
    enable_logging: bool = Field(default=True, env="WORKFLOW_ENABLE_LOGGING")
    
    # 重试配置
    retry_min_wait: int = Field(default=1, env="RETRY_MIN_WAIT")
    retry_max_wait: int = Field(default=60, env="RETRY_MAX_WAIT")
    retry_multiplier: int = Field(default=2, env="RETRY_MULTIPLIER")


class APIConfig(BaseSettings):
    """API服务配置"""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    
    # CORS配置
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")


class LoggingConfig(BaseSettings):
    """日志配置"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")  # json, text
    file_path: Optional[Path] = Field(default=Path("./logs/fp_quantum.log"), env="LOG_FILE_PATH")
    max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    @validator('file_path')
    def validate_log_path(cls, v):
        if v is None:
            return None
        return Path(v)


class Settings(BaseSettings):
    """主配置类"""
    
    # 应用基础配置
    app_name: str = "量子智能化功能点估算系统"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # 子配置模块
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f'environment must be one of {allowed_envs}')
        return v
    
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self.environment == "production"


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings


def load_config_from_file(config_file: str) -> Settings:
    """从指定文件加载配置"""
    return Settings(_env_file=config_file)


# 导出配置类和实例
__all__ = [
    "Settings",
    "DatabaseConfig", 
    "LLMConfig",
    "VectorStoreConfig",
    "KnowledgeBaseConfig", 
    "WorkflowConfig",
    "APIConfig",
    "LoggingConfig",
    "settings",
    "get_settings",
    "load_config_from_file",
] 