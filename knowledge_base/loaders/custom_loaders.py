"""
量子智能化功能点估算系统 - 自定义文档加载器

支持各种特殊格式的功能点估算文档加载
"""

import json
import csv
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import logging
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class JSONDocumentLoader:
    """JSON文档加载器"""
    
    def __init__(self, text_key: str = "content", metadata_keys: Optional[List[str]] = None):
        self.text_key = text_key
        self.metadata_keys = metadata_keys or []
        
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """加载JSON文档"""
        
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            if isinstance(data, list):
                # 处理JSON数组
                for i, item in enumerate(data):
                    doc = self._create_document_from_dict(item, f"{file_path.name}#{i}")
                    if doc:
                        documents.append(doc)
            elif isinstance(data, dict):
                # 处理单个JSON对象
                doc = self._create_document_from_dict(data, file_path.name)
                if doc:
                    documents.append(doc)
                    
            logger.info(f"从 {file_path} 加载了 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return []
    
    def _create_document_from_dict(self, data: Dict[str, Any], source: str) -> Optional[Document]:
        """从字典创建文档"""
        
        # 获取文本内容
        if self.text_key in data:
            content = str(data[self.text_key])
        else:
            # 如果没有指定的文本键，使用整个JSON作为内容
            content = json.dumps(data, ensure_ascii=False, indent=2)
        
        # 构建元数据
        metadata = {
            'source': source,
            'loader_type': 'json',
            'loaded_at': datetime.now().isoformat()
        }
        
        # 添加指定的元数据键
        for key in self.metadata_keys:
            if key in data:
                metadata[key] = data[key]
        
        return Document(page_content=content, metadata=metadata)


class CSVDocumentLoader:
    """CSV文档加载器"""
    
    def __init__(
        self, 
        content_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ):
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.delimiter = delimiter
        self.encoding = encoding
        
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """加载CSV文档"""
        
        file_path = Path(file_path)
        documents = []
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                
                for i, row in enumerate(reader):
                    # 组合内容列
                    content_parts = []
                    for col in self.content_columns:
                        if col in row and row[col]:
                            content_parts.append(f"{col}: {row[col]}")
                    
                    if content_parts:
                        content = '\n'.join(content_parts)
                        
                        # 构建元数据
                        metadata = {
                            'source': f"{file_path.name}#row{i+1}",
                            'loader_type': 'csv',
                            'row_number': i + 1,
                            'loaded_at': datetime.now().isoformat()
                        }
                        
                        # 添加元数据列
                        for col in self.metadata_columns:
                            if col in row:
                                metadata[col] = row[col]
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                        
            logger.info(f"从 {file_path} 加载了 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"加载CSV文件失败 {file_path}: {e}")
            return []


class YAMLDocumentLoader:
    """YAML文档加载器"""
    
    def __init__(self, content_key: str = "content", metadata_keys: Optional[List[str]] = None):
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """加载YAML文档"""
        
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            documents = []
            
            if isinstance(data, list):
                # 处理YAML数组
                for i, item in enumerate(data):
                    doc = self._create_document_from_dict(item, f"{file_path.name}#{i}")
                    if doc:
                        documents.append(doc)
            elif isinstance(data, dict):
                # 处理单个YAML对象
                doc = self._create_document_from_dict(data, file_path.name)
                if doc:
                    documents.append(doc)
                    
            logger.info(f"从 {file_path} 加载了 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"加载YAML文件失败 {file_path}: {e}")
            return []
    
    def _create_document_from_dict(self, data: Dict[str, Any], source: str) -> Optional[Document]:
        """从字典创建文档"""
        
        # 获取文本内容
        if self.content_key in data:
            content = str(data[self.content_key])
        else:
            # 如果没有指定的文本键，使用整个YAML作为内容
            content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        
        # 构建元数据
        metadata = {
            'source': source,
            'loader_type': 'yaml',
            'loaded_at': datetime.now().isoformat()
        }
        
        # 添加指定的元数据键
        for key in self.metadata_keys:
            if key in data:
                metadata[key] = data[key]
        
        return Document(page_content=content, metadata=metadata)


class MarkdownDocumentLoader:
    """Markdown文档加载器"""
    
    def __init__(self, split_on_headers: bool = True):
        self.split_on_headers = split_on_headers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """加载Markdown文档"""
        
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents = []
            
            if self.split_on_headers:
                # 按标题分割
                sections = self._split_by_headers(content)
                
                for i, (title, section_content) in enumerate(sections):
                    metadata = {
                        'source': file_path.name,
                        'loader_type': 'markdown',
                        'section_title': title,
                        'section_index': i,
                        'total_sections': len(sections),
                        'loaded_at': datetime.now().isoformat()
                    }
                    
                    # 进一步分割长内容
                    if len(section_content) > 1000:
                        chunks = self.text_splitter.split_text(section_content)
                        for j, chunk in enumerate(chunks):
                            chunk_metadata = metadata.copy()
                            chunk_metadata.update({
                                'chunk_index': j,
                                'total_chunks': len(chunks)
                            })
                            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                    else:
                        documents.append(Document(page_content=section_content, metadata=metadata))
            else:
                # 不按标题分割，直接分块
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'source': file_path.name,
                        'loader_type': 'markdown',
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'loaded_at': datetime.now().isoformat()
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
                    
            logger.info(f"从 {file_path} 加载了 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"加载Markdown文件失败 {file_path}: {e}")
            return []
    
    def _split_by_headers(self, content: str) -> List[tuple]:
        """按标题分割内容"""
        
        import re
        
        # 查找所有标题
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        sections = []
        current_title = "前言"
        current_content = []
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # 保存上一个部分
                if current_content:
                    sections.append((current_title, '\n'.join(current_content).strip()))
                
                # 开始新部分
                current_title = match.group(2)
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个部分
        if current_content:
            sections.append((current_title, '\n'.join(current_content).strip()))
        
        return sections


class FunctionPointDocumentLoader:
    """功能点估算专用文档加载器"""
    
    def __init__(self):
        self.loaders = {
            '.json': JSONDocumentLoader(
                text_key='content',
                metadata_keys=['title', 'source_type', 'category', 'version']
            ),
            '.csv': CSVDocumentLoader(
                content_columns=['name', 'description', 'details'],
                metadata_columns=['type', 'complexity', 'category']
            ),
            '.yaml': YAMLDocumentLoader(
                content_key='content',
                metadata_keys=['title', 'source_type', 'category']
            ),
            '.yml': YAMLDocumentLoader(
                content_key='content', 
                metadata_keys=['title', 'source_type', 'category']
            ),
            '.md': MarkdownDocumentLoader(split_on_headers=True),
            '.txt': self._load_text_file
        }
        
    def load_directory(self, directory: Union[str, Path]) -> List[Document]:
        """加载目录中的所有支持的文档"""
        
        directory = Path(directory)
        all_documents = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                try:
                    documents = self.load_file(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {e}")
        
        logger.info(f"从目录 {directory} 加载了 {len(all_documents)} 个文档")
        return all_documents
        
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """加载单个文件"""
        
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix in self.loaders:
            loader = self.loaders[suffix]
            if callable(loader):
                return loader(file_path)
            else:
                return loader.load(file_path)
        else:
            logger.warning(f"不支持的文件格式: {suffix}")
            return []
    
    def _load_text_file(self, file_path: Path) -> List[Document]:
        """加载纯文本文件"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            chunks = text_splitter.split_text(content)
            documents = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': file_path.name,
                    'loader_type': 'text',
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'loaded_at': datetime.now().isoformat()
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {e}")
            return []


class DocumentValidator:
    """文档验证器"""
    
    def __init__(self):
        self.required_metadata = ['source', 'loader_type', 'loaded_at']
        
    def validate_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """验证文档集合"""
        
        validation_result = {
            'total_documents': len(documents),
            'valid_documents': 0,
            'invalid_documents': 0,
            'validation_errors': [],
            'statistics': {
                'avg_content_length': 0,
                'min_content_length': float('inf'),
                'max_content_length': 0,
                'loader_types': {},
                'sources': set()
            }
        }
        
        total_length = 0
        
        for i, doc in enumerate(documents):
            try:
                # 验证基本结构
                if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                    validation_result['validation_errors'].append(
                        f"文档 {i}: 缺少基本属性"
                    )
                    validation_result['invalid_documents'] += 1
                    continue
                
                # 验证内容
                if not doc.page_content or len(doc.page_content.strip()) == 0:
                    validation_result['validation_errors'].append(
                        f"文档 {i}: 内容为空"
                    )
                    validation_result['invalid_documents'] += 1
                    continue
                
                # 验证元数据
                missing_metadata = []
                for key in self.required_metadata:
                    if key not in doc.metadata:
                        missing_metadata.append(key)
                
                if missing_metadata:
                    validation_result['validation_errors'].append(
                        f"文档 {i}: 缺少元数据 {missing_metadata}"
                    )
                    validation_result['invalid_documents'] += 1
                    continue
                
                # 文档有效
                validation_result['valid_documents'] += 1
                
                # 统计信息
                content_length = len(doc.page_content)
                total_length += content_length
                
                validation_result['statistics']['min_content_length'] = min(
                    validation_result['statistics']['min_content_length'],
                    content_length
                )
                validation_result['statistics']['max_content_length'] = max(
                    validation_result['statistics']['max_content_length'],
                    content_length
                )
                
                loader_type = doc.metadata.get('loader_type', 'unknown')
                validation_result['statistics']['loader_types'][loader_type] = \
                    validation_result['statistics']['loader_types'].get(loader_type, 0) + 1
                
                validation_result['statistics']['sources'].add(
                    doc.metadata.get('source', 'unknown')
                )
                
            except Exception as e:
                validation_result['validation_errors'].append(
                    f"文档 {i}: 验证异常 {e}"
                )
                validation_result['invalid_documents'] += 1
        
        # 计算平均长度
        if validation_result['valid_documents'] > 0:
            validation_result['statistics']['avg_content_length'] = \
                total_length / validation_result['valid_documents']
        
        # 转换集合为列表
        validation_result['statistics']['sources'] = \
            list(validation_result['statistics']['sources'])
        
        # 处理最小长度的边界情况
        if validation_result['statistics']['min_content_length'] == float('inf'):
            validation_result['statistics']['min_content_length'] = 0
        
        return validation_result


if __name__ == "__main__":
    # 测试自定义加载器
    
    # 创建功能点文档加载器
    fp_loader = FunctionPointDocumentLoader()
    
    # 测试数据目录（如果存在）
    test_dir = Path("knowledge_base/documents")
    if test_dir.exists():
        documents = fp_loader.load_directory(test_dir)
        print(f"加载了 {len(documents)} 个文档")
        
        # 验证文档
        validator = DocumentValidator()
        validation_result = validator.validate_documents(documents)
        
        print(f"验证结果:")
        print(f"  有效文档: {validation_result['valid_documents']}")
        print(f"  无效文档: {validation_result['invalid_documents']}")
        print(f"  平均长度: {validation_result['statistics']['avg_content_length']:.0f}")
        print(f"  加载器类型: {validation_result['statistics']['loader_types']}")
    else:
        print("测试目录不存在，跳过测试") 