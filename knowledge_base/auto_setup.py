#!/usr/bin/env python3
"""
自动生成的知识库设置脚本
生成时间: 2025-06-26 12:08:14
"""

import asyncio
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AutoKnowledgeBaseSetup:
    def __init__(self):
        self.base_dir = Path("knowledge_base")
        self.processing_order = [{'category': 'nesma', 'file': {'filename': 'NESMA_FPA_Method_v2.3.pdf', 'size': '0.5MB', 'metadata': {'title': 'NESMA功能点分析方法v2.3', 'type': 'official_standard', 'language': '英文', 'priority': 'high', 'description': 'NESMA官方功能点估算标准文档'}}, 'processing_order': 1}, {'category': 'cosmic', 'file': {'filename': 'COSMIC度量手册V5.0-part-1-原则、定义与规则.pdf', 'size': '0.3MB', 'metadata': {'title': 'COSMIC度量手册-原则与规则', 'type': 'official_standard', 'language': '中文', 'priority': 'high', 'description': 'COSMIC v5.0核心理论和定义'}}, 'processing_order': 1}, {'category': 'cosmic', 'file': {'filename': 'COSMIC度量手册V5.0-part-2-指南.pdf', 'size': '0.4MB', 'metadata': {'title': 'COSMIC度量手册-实施指南', 'type': 'implementation_guide', 'language': '中文', 'priority': 'high', 'description': 'COSMIC v5.0实施操作指南'}}, 'processing_order': 1}, {'category': 'cosmic', 'file': {'filename': 'COSMIC度量手册V5.0-part-3-案例.pdf', 'size': '0.7MB', 'metadata': {'title': 'COSMIC度量手册-案例集', 'type': 'case_studies', 'language': '中文', 'priority': 'medium', 'description': 'COSMIC v5.0实际应用案例'}}, 'processing_order': 2}, {'category': 'cosmic', 'file': {'filename': 'COSMIC早期软件规模度量指南-实践级-Early-Software-Sizing（Practitioners.pdf', 'size': '0.7MB', 'metadata': {'title': 'COSMIC早期估算-实践级', 'type': 'early_sizing_guide', 'language': '中文', 'priority': 'medium', 'description': '早期阶段功能点估算实践指南'}}, 'processing_order': 2}, {'category': 'cosmic', 'file': {'filename': 'COSMIC早期软件规模度量指南-–-专家级V2-Early-Software-Sizing（Experts.pdf', 'size': '1.3MB', 'metadata': {'title': 'COSMIC早期估算-专家级', 'type': 'advanced_guide', 'language': '中文', 'priority': 'medium', 'description': '高级早期功能点估算指南'}}, 'processing_order': 2}, {'category': 'common', 'file': {'filename': 'NESMA_FPA_Method_v2.3.pdf', 'size': '0.5MB', 'metadata': {'title': 'NESMA参考文档', 'type': 'reference', 'language': '英文', 'priority': 'medium', 'description': '通用NESMA参考文档'}}, 'processing_order': 2}, {'category': 'common', 'file': {'filename': '工作量拆分讲解V2.pptx', 'size': '11.3MB', 'metadata': {'title': '工作量拆分培训', 'type': 'training_material', 'language': '中文', 'priority': 'medium', 'description': '功能点工作量拆分培训材料'}}, 'processing_order': 2}]
    
    async def setup_documents(self):
        """按计划处理文档"""
        
        # 中英文优化的分词器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", "!", "?", "；", ";"]
        )
        
        processed_docs = []
        
        for item in self.processing_order:
            category = item["category"]
            file_info = item["file"]
            
            file_path = self.base_dir / "documents" / category / file_info["filename"]
            
            if not file_path.exists():
                continue
            
            print(f"🔄 处理文档: {file_info['metadata']['title']}")
            
            # 选择合适的加载器
            if file_path.suffix.lower() == '.pdf':
                # 使用PyPDFLoader，更简单可靠，无需额外依赖
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file_path))
            else:
                continue
            
            # 加载和分块
            docs = await loader.aload()
            split_docs = text_splitter.split_documents(docs)
            
            # 添加元数据
            for doc in split_docs:
                doc.metadata.update({
                    "source_category": category,
                    "document_type": file_info["metadata"]["type"],
                    "language": file_info["metadata"]["language"],
                    "priority": file_info["metadata"]["priority"],
                    "title": file_info["metadata"]["title"]
                })
            
            processed_docs.extend(split_docs)
            print(f"✅ 完成: {len(split_docs)} 个文档块")
        
        return processed_docs

if __name__ == "__main__":
    setup = AutoKnowledgeBaseSetup()
    asyncio.run(setup.setup_documents())
