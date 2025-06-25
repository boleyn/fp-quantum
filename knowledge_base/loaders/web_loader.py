"""
量子智能化功能点估算系统 - 网页内容加载器

从网页抓取和处理NESMA、COSMIC相关内容
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class EnhancedWebLoader:
    """增强的网页加载器"""
    
    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_url(self, url: str) -> Document:
        """加载单个URL"""
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # 解析HTML
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # 提取标题
                        title = soup.find('title')
                        title_text = title.get_text().strip() if title else url
                        
                        # 提取主要内容
                        text_content = self._extract_main_content(soup)
                        
                        # 创建文档
                        metadata = {
                            'source': url,
                            'title': title_text,
                            'loaded_at': datetime.now().isoformat(),
                            'content_length': len(text_content)
                        }
                        
                        return Document(page_content=text_content, metadata=metadata)
                    
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"加载URL失败 {url} (尝试 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    async def load_urls(self, urls: List[str]) -> List[Document]:
        """批量加载URL"""
        
        tasks = [self.load_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Document):
                documents.append(result)
            else:
                logger.error(f"加载URL失败 {urls[i]}: {result}")
        
        return documents
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """提取主要内容"""
        
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # 尝试找到主要内容区域
        main_content = None
        
        # 常见的主要内容标签
        for selector in ['main', 'article', '.content', '#content', '.main']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # 如果没找到，使用body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # 提取文本
        text = main_content.get_text(separator='\n', strip=True)
        
        # 清理文本
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 合并多个空行
        text = re.sub(r'[ \t]+', ' ', text)  # 合并多个空格
        
        return text.strip()


class NESMAWebLoader:
    """NESMA官方资源网页加载器"""
    
    def __init__(self):
        self.nesma_urls = [
            "https://www.nesma.org/",
            "https://www.nesma.org/function-point-analysis/",
            # 可以添加更多NESMA相关URL
        ]
        
    async def load_nesma_resources(self) -> List[Document]:
        """加载NESMA官方资源"""
        
        async with EnhancedWebLoader() as loader:
            documents = await loader.load_urls(self.nesma_urls)
            
            # 为NESMA文档添加特定标记
            for doc in documents:
                doc.metadata['source_type'] = 'NESMA'
                doc.metadata['category'] = 'official_resource'
            
            return documents


class COSMICWebLoader:
    """COSMIC官方资源网页加载器"""
    
    def __init__(self):
        self.cosmic_urls = [
            "https://cosmic-sizing.org/",
            "https://cosmic-sizing.org/cosmic-method/",
            # 可以添加更多COSMIC相关URL
        ]
        
    async def load_cosmic_resources(self) -> List[Document]:
        """加载COSMIC官方资源"""
        
        async with EnhancedWebLoader() as loader:
            documents = await loader.load_urls(self.cosmic_urls)
            
            # 为COSMIC文档添加特定标记
            for doc in documents:
                doc.metadata['source_type'] = 'COSMIC'
                doc.metadata['category'] = 'official_resource'
            
            return documents


class WebContentProcessor:
    """网页内容处理器"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", "!", "?"]
        )
        
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """处理网页文档"""
        
        processed_docs = []
        
        for doc in documents:
            # 分割文档
            splits = self.text_splitter.split_documents([doc])
            
            # 为每个分片添加序号
            for i, split in enumerate(splits):
                split.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(splits),
                    'processed_at': datetime.now().isoformat()
                })
                
            processed_docs.extend(splits)
        
        return processed_docs
    
    def filter_by_keywords(
        self, 
        documents: List[Document], 
        keywords: List[str]
    ) -> List[Document]:
        """根据关键词过滤文档"""
        
        filtered_docs = []
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # 检查是否包含任何关键词
            if any(keyword.lower() in content_lower for keyword in keywords):
                doc.metadata['matched_keywords'] = [
                    kw for kw in keywords 
                    if kw.lower() in content_lower
                ]
                filtered_docs.append(doc)
        
        return filtered_docs


# 预定义的关键词集合
NESMA_KEYWORDS = [
    "function point", "functional point", "NESMA", "ILF", "EIF", "EI", "EO", "EQ",
    "DET", "RET", "complexity", "UFP", "unadjusted function points"
]

COSMIC_KEYWORDS = [
    "COSMIC", "CFP", "data movement", "entry", "exit", "read", "write",
    "functional user", "boundary", "functional process", "data group"
]


async def load_web_knowledge_base() -> Dict[str, List[Document]]:
    """加载网页知识库"""
    
    logger.info("🌐 开始加载网页知识库...")
    
    try:
        # 加载NESMA资源
        nesma_loader = NESMAWebLoader()
        nesma_docs = await nesma_loader.load_nesma_resources()
        
        # 加载COSMIC资源
        cosmic_loader = COSMICWebLoader()
        cosmic_docs = await cosmic_loader.load_cosmic_resources()
        
        # 处理文档
        processor = WebContentProcessor()
        
        # 过滤和处理NESMA文档
        nesma_filtered = processor.filter_by_keywords(nesma_docs, NESMA_KEYWORDS)
        nesma_processed = processor.process_documents(nesma_filtered)
        
        # 过滤和处理COSMIC文档
        cosmic_filtered = processor.filter_by_keywords(cosmic_docs, COSMIC_KEYWORDS)
        cosmic_processed = processor.process_documents(cosmic_filtered)
        
        logger.info(f"✅ 网页知识库加载完成:")
        logger.info(f"   NESMA文档: {len(nesma_processed)} 个片段")
        logger.info(f"   COSMIC文档: {len(cosmic_processed)} 个片段")
        
        return {
            'nesma': nesma_processed,
            'cosmic': cosmic_processed,
            'all': nesma_processed + cosmic_processed
        }
        
    except Exception as e:
        logger.error(f"❌ 网页知识库加载失败: {e}")
        return {'nesma': [], 'cosmic': [], 'all': []}


if __name__ == "__main__":
    async def main():
        # 测试网页加载器
        result = await load_web_knowledge_base()
        
        print(f"加载的文档总数: {len(result['all'])}")
        print(f"NESMA文档: {len(result['nesma'])}")
        print(f"COSMIC文档: {len(result['cosmic'])}")
        
        # 显示第一个文档示例
        if result['all']:
            first_doc = result['all'][0]
            print(f"\n示例文档:")
            print(f"标题: {first_doc.metadata.get('title', 'N/A')}")
            print(f"来源: {first_doc.metadata.get('source', 'N/A')}")
            print(f"内容长度: {len(first_doc.page_content)}")
    
    asyncio.run(main()) 