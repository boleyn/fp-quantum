"""
量子智能化功能点估算系统 - 文档加载器模块

提供多种格式文档的加载和处理功能
"""

from .pdf_loader import (
    EnhancedPDFLoader,
    BatchPDFProcessor,
    pdf_loader_factory
)

__all__ = [
    "EnhancedPDFLoader",
    "BatchPDFProcessor", 
    "pdf_loader_factory"
]

__version__ = "1.0.0" 