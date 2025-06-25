"""
量子智能化功能点估算系统 - PDF文档加载器

支持多种PDF文档加载策略，针对NESMA/COSMIC标准文档优化
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    PDFMinerLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings


class PDFLoaderStrategy:
    """PDF加载策略枚举"""
    PYPDF = "pypdf"                    # 基础PDF加载
    UNSTRUCTURED = "unstructured"      # 结构化PDF加载
    PDFMINER = "pdfminer"             # 高精度文本提取


class EnhancedPDFLoader:
    """增强的PDF文档加载器"""
    
    def __init__(self, strategy: str = PDFLoaderStrategy.UNSTRUCTURED):
        self.strategy = strategy
        self.settings = get_settings()
        
        # 文本分割器配置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.knowledge_base.chunk_size,
            chunk_overlap=self.settings.knowledge_base.chunk_overlap,
            separators=[
                "\n\n",  # 段落分隔
                "\n",    # 行分隔
                "。",    # 中文句号
                ".",     # 英文句号
                "!",     # 感叹号
                "?",     # 问号
                "；",    # 中文分号
                ";",     # 英文分号
                "，",    # 中文逗号
                ",",     # 英文逗号
                " ",     # 空格
                ""       # 字符
            ]
        )
    
    def load_pdf(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """加载单个PDF文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
        
        # 根据策略选择加载器
        if self.strategy == PDFLoaderStrategy.PYPDF:
            loader = PyPDFLoader(str(file_path))
        elif self.strategy == PDFLoaderStrategy.UNSTRUCTURED:
            loader = UnstructuredPDFLoader(
                str(file_path),
                mode="elements",
                strategy="hi_res"
            )
        elif self.strategy == PDFLoaderStrategy.PDFMINER:
            loader = PDFMinerLoader(str(file_path))
        else:
            raise ValueError(f"不支持的PDF加载策略: {self.strategy}")
        
        # 加载文档
        documents = loader.load()
        
        # 添加元数据
        base_metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "pdf",
            "loader_strategy": self.strategy,
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        # 为每个文档添加元数据
        for doc in documents:
            doc.metadata.update(base_metadata)
        
        return documents
    
    def load_directory(
        self, 
        directory_path: str, 
        pattern: str = "*.pdf",
        metadata_extractor: Optional[callable] = None
    ) -> List[Document]:
        """加载目录中的所有PDF文件"""
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        all_documents = []
        pdf_files = list(directory_path.glob(pattern))
        
        if not pdf_files:
            print(f"⚠️ 在目录 {directory_path} 中未找到PDF文件")
            return all_documents
        
        for pdf_file in pdf_files:
            try:
                # 提取文件元数据
                file_metadata = {}
                if metadata_extractor:
                    file_metadata = metadata_extractor(pdf_file)
                
                # 加载文档
                documents = self.load_pdf(str(pdf_file), file_metadata)
                all_documents.extend(documents)
                
                print(f"✅ 成功加载: {pdf_file.name} ({len(documents)} 个文档块)")
                
            except Exception as e:
                print(f"❌ 加载失败: {pdf_file.name} - {str(e)}")
                continue
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为较小的块"""
        return self.text_splitter.split_documents(documents)
    
    def load_and_split(
        self, 
        file_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """加载并分割PDF文档"""
        documents = self.load_pdf(file_path, metadata)
        return self.split_documents(documents)


class NESMAPDFLoader(EnhancedPDFLoader):
    """NESMA标准文档专用加载器"""
    
    def __init__(self):
        super().__init__(strategy=PDFLoaderStrategy.UNSTRUCTURED)
    
    def extract_nesma_metadata(self, file_path: Path) -> Dict[str, Any]:
        """提取NESMA文档的元数据"""
        metadata = {
            "source_type": "NESMA",
            "standard": "NESMA",
            "version": self._extract_version_from_filename(file_path.name),
            "document_type": self._classify_nesma_document(file_path.name)
        }
        return metadata
    
    def _extract_version_from_filename(self, filename: str) -> str:
        """从文件名提取版本信息"""
        # 寻找版本模式，如 v2.3, version_2.3, 2.3等
        import re
        patterns = [
            r'v?(\d+\.\d+)',
            r'version[_\s]?(\d+\.\d+)',
            r'nesma[_\s]?(\d+\.\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _classify_nesma_document(self, filename: str) -> str:
        """分类NESMA文档类型"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['manual', '手册', 'guide', '指南']):
            return "manual"
        elif any(keyword in filename_lower for keyword in ['specification', '规范', 'spec']):
            return "specification"
        elif any(keyword in filename_lower for keyword in ['example', '示例', 'case']):
            return "example"
        elif any(keyword in filename_lower for keyword in ['rule', '规则', 'standard']):
            return "rules"
        else:
            return "general"


class COSMICPDFLoader(EnhancedPDFLoader):
    """COSMIC标准文档专用加载器"""
    
    def __init__(self):
        super().__init__(strategy=PDFLoaderStrategy.UNSTRUCTURED)
    
    def extract_cosmic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """提取COSMIC文档的元数据"""
        metadata = {
            "source_type": "COSMIC",
            "standard": "COSMIC",
            "version": self._extract_version_from_filename(file_path.name),
            "document_type": self._classify_cosmic_document(file_path.name)
        }
        return metadata
    
    def _extract_version_from_filename(self, filename: str) -> str:
        """从文件名提取版本信息"""
        import re
        patterns = [
            r'v?(\d+\.\d+)',
            r'version[_\s]?(\d+\.\d+)',
            r'cosmic[_\s]?(\d+\.\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _classify_cosmic_document(self, filename: str) -> str:
        """分类COSMIC文档类型"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['manual', '手册', 'handbook']):
            return "manual"
        elif any(keyword in filename_lower for keyword in ['method', '方法', 'methodology']):
            return "methodology"
        elif any(keyword in filename_lower for keyword in ['guideline', '指南', 'guide']):
            return "guideline"
        elif any(keyword in filename_lower for keyword in ['example', '示例', 'case']):
            return "example"
        else:
            return "general"


def create_pdf_loader(source_type: str = "general") -> EnhancedPDFLoader:
    """工厂函数：创建PDF加载器"""
    if source_type.upper() == "NESMA":
        return NESMAPDFLoader()
    elif source_type.upper() == "COSMIC":
        return COSMICPDFLoader()
    else:
        return EnhancedPDFLoader()


async def load_knowledge_base_pdfs(
    nesma_path: Optional[str] = None,
    cosmic_path: Optional[str] = None,
    common_path: Optional[str] = None
) -> Dict[str, List[Document]]:
    """加载知识库中的所有PDF文档"""
    settings = get_settings()
    results = {}
    
    # 加载NESMA文档
    if nesma_path or settings.knowledge_base.nesma_docs_path.exists():
        path = nesma_path or str(settings.knowledge_base.nesma_docs_path)
        loader = NESMAPDFLoader()
        documents = loader.load_directory(
            path, 
            metadata_extractor=loader.extract_nesma_metadata
        )
        if documents:
            # 分割文档
            split_docs = loader.split_documents(documents)
            results["nesma"] = split_docs
            print(f"📚 NESMA文档加载完成: {len(documents)} 原始文档 -> {len(split_docs)} 分割块")
    
    # 加载COSMIC文档
    if cosmic_path or settings.knowledge_base.cosmic_docs_path.exists():
        path = cosmic_path or str(settings.knowledge_base.cosmic_docs_path)
        loader = COSMICPDFLoader()
        documents = loader.load_directory(
            path,
            metadata_extractor=loader.extract_cosmic_metadata
        )
        if documents:
            # 分割文档
            split_docs = loader.split_documents(documents)
            results["cosmic"] = split_docs
            print(f"🌌 COSMIC文档加载完成: {len(documents)} 原始文档 -> {len(split_docs)} 分割块")
    
    # 加载通用文档
    if common_path or settings.knowledge_base.common_docs_path.exists():
        path = common_path or str(settings.knowledge_base.common_docs_path)
        loader = EnhancedPDFLoader()
        documents = loader.load_directory(path)
        if documents:
            # 添加通用标识
            for doc in documents:
                doc.metadata["source_type"] = "COMMON"
                doc.metadata["standard"] = "COMMON"
            
            # 分割文档
            split_docs = loader.split_documents(documents)
            results["common"] = split_docs
            print(f"📖 通用文档加载完成: {len(documents)} 原始文档 -> {len(split_docs)} 分割块")
    
    return results


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # 测试PDF加载器
        print("🔍 测试PDF加载器...")
        
        # 加载知识库文档
        documents = await load_knowledge_base_pdfs()
        
        total_docs = sum(len(docs) for docs in documents.values())
        print(f"\n📊 总计加载 {total_docs} 个文档块:")
        
        for source_type, docs in documents.items():
            print(f"  {source_type.upper()}: {len(docs)} 个文档块")
            
            if docs:
                # 显示第一个文档的示例
                sample_doc = docs[0]
                print(f"    示例内容: {sample_doc.page_content[:200]}...")
                print(f"    元数据: {sample_doc.metadata}")
    
    asyncio.run(main()) 