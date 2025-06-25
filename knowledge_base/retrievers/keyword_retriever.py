"""
量子智能化功能点估算系统 - 关键词检索器

基于传统IR技术的关键词检索实现
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import math

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """TF-IDF关键词检索器"""
    
    def __init__(
        self,
        documents: List[Document],
        language: str = "mixed"  # mixed, chinese, english
    ):
        self.documents = documents
        self.language = language
        
        # 预处理文档
        self.processed_docs = []
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        self.idf_scores: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        
        # 构建索引
        self._build_index()
        
    def _build_index(self):
        """构建TF-IDF索引"""
        
        logger.info(f"📚 构建TF-IDF索引，文档数: {len(self.documents)}")
        
        # 第一遍：计算文档频率
        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.doc_frequencies[token] += 1
                
            self.processed_docs.append(tokens)
        
        # 计算IDF分数
        total_docs = len(self.documents)
        for term, doc_freq in self.doc_frequencies.items():
            self.idf_scores[term] = math.log(total_docs / doc_freq)
        
        # 第二遍：计算TF-IDF向量
        for tokens in self.processed_docs:
            tf_counts = Counter(tokens)
            doc_length = len(tokens)
            
            doc_vector = {}
            for term, count in tf_counts.items():
                tf = count / doc_length if doc_length > 0 else 0
                idf = self.idf_scores[term]
                doc_vector[term] = tf * idf
            
            self.doc_vectors.append(doc_vector)
        
        logger.info(f"✅ TF-IDF索引构建完成，词汇表大小: {len(self.idf_scores)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        
        tokens = []
        
        if self.language in ["mixed", "english"]:
            # 英文单词提取
            english_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            tokens.extend(english_words)
        
        if self.language in ["mixed", "chinese"]:
            # 中文字符提取
            chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
            # 简单的中文分词（每个字符作为一个词）
            for chars in chinese_chars:
                tokens.extend(list(chars))
        
        # 数字和特殊术语
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)
        
        # 功能点相关术语
        fp_terms = re.findall(r'\b(?:ILF|EIF|EI|EO|EQ|DET|RET|CFP|Entry|Exit|Read|Write)\b', text)
        tokens.extend([term.lower() for term in fp_terms])
        
        # 过滤停用词
        stop_words = {
            '的', '是', '在', '有', '和', '与', '或', '但', '而', '这', '那', '一个', '一种',
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'
        }
        
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        return filtered_tokens
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """搜索相关文档"""
        
        # 查询分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning("查询为空，返回空结果")
            return []
        
        # 计算查询向量
        query_tf = Counter(query_tokens)
        query_length = len(query_tokens)
        
        query_vector = {}
        for term, count in query_tf.items():
            if term in self.idf_scores:
                tf = count / query_length
                idf = self.idf_scores[term]
                query_vector[term] = tf * idf
        
        if not query_vector:
            logger.warning("查询中没有有效词汇，返回空结果")
            return []
        
        # 计算相似度
        similarities = []
        for i, (doc, doc_vector) in enumerate(zip(self.documents, self.doc_vectors)):
            # 应用元数据过滤
            if filter_metadata:
                if not all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    continue
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                similarities.append((doc, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"TF-IDF检索返回 {len(similarities[:k])} 个结果")
        return similarities[:k]
    
    def _cosine_similarity(
        self,
        vector1: Dict[str, float],
        vector2: Dict[str, float]
    ) -> float:
        """计算余弦相似度"""
        
        # 计算点积
        dot_product = 0.0
        common_terms = set(vector1.keys()) & set(vector2.keys())
        
        for term in common_terms:
            dot_product += vector1[term] * vector2[term]
        
        # 计算向量长度
        norm1 = math.sqrt(sum(value ** 2 for value in vector1.values()))
        norm2 = math.sqrt(sum(value ** 2 for value in vector2.values()))
        
        # 避免除零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class BooleanRetriever:
    """布尔检索器"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        
        # 构建倒排索引
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        self._build_inverted_index()
    
    def _build_inverted_index(self):
        """构建倒排索引"""
        
        logger.info(f"🔨 构建倒排索引，文档数: {len(self.documents)}")
        
        for doc_id, doc in enumerate(self.documents):
            tokens = self._tokenize(doc.page_content)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.inverted_index[token].add(doc_id)
        
        logger.info(f"✅ 倒排索引构建完成，词汇数: {len(self.inverted_index)}")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        
        # 英文和数字
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # 中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(chinese_chars)
        
        # 功能点术语
        fp_terms = re.findall(r'\b(?:ILF|EIF|EI|EO|EQ|DET|RET|CFP|Entry|Exit|Read|Write)\b', text)
        tokens.extend([term.lower() for term in fp_terms])
        
        return tokens
    
    def search(
        self,
        query: str,
        operator: str = "AND",  # AND, OR
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """布尔搜索"""
        
        # 解析查询
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # 获取每个词的文档集合
        term_doc_sets = []
        for term in query_terms:
            if term in self.inverted_index:
                term_doc_sets.append(self.inverted_index[term])
            else:
                term_doc_sets.append(set())
        
        # 根据操作符合并结果
        if operator.upper() == "AND":
            if term_doc_sets:
                result_doc_ids = term_doc_sets[0].copy()
                for doc_set in term_doc_sets[1:]:
                    result_doc_ids &= doc_set
            else:
                result_doc_ids = set()
        elif operator.upper() == "OR":
            result_doc_ids = set()
            for doc_set in term_doc_sets:
                result_doc_ids |= doc_set
        else:
            raise ValueError(f"不支持的操作符: {operator}")
        
        # 获取文档
        results = []
        for doc_id in result_doc_ids:
            doc = self.documents[doc_id]
            
            # 应用元数据过滤
            if filter_metadata:
                if not all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    continue
            
            results.append(doc)
        
        logger.debug(f"布尔检索({operator})返回 {len(results)} 个结果")
        return results


class FuzzyMatchRetriever:
    """模糊匹配检索器"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
    
    def search(
        self,
        query: str,
        similarity_threshold: float = 0.6,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """模糊匹配搜索"""
        
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            # 应用元数据过滤
            if filter_metadata:
                if not all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    continue
            
            # 计算相似度
            content_lower = doc.page_content.lower()
            similarity = self._string_similarity(query_lower, content_lower)
            
            if similarity >= similarity_threshold:
                results.append((doc, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"模糊匹配返回 {len(results[:k])} 个结果")
        return results[:k]
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """字符串相似度计算（简化版Levenshtein距离）"""
        
        # 使用Jaccard相似度近似
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


class KeywordRetrieverFactory:
    """关键词检索器工厂"""
    
    @staticmethod
    def create_retriever(
        retriever_type: str,
        documents: List[Document],
        **kwargs
    ):
        """创建检索器"""
        
        if retriever_type.lower() == "tfidf":
            return TFIDFRetriever(documents, **kwargs)
        elif retriever_type.lower() == "boolean":
            return BooleanRetriever(documents)
        elif retriever_type.lower() == "fuzzy":
            return FuzzyMatchRetriever(documents)
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")


class NESMAKeywordRetriever:
    """NESMA专用关键词检索器"""
    
    def __init__(self, documents: List[Document]):
        # 过滤NESMA文档
        nesma_docs = [
            doc for doc in documents
            if doc.metadata.get('source_type') == 'NESMA'
        ]
        
        self.tfidf_retriever = TFIDFRetriever(nesma_docs)
        self.boolean_retriever = BooleanRetriever(nesma_docs)
        
        # NESMA特定术语权重
        self.term_weights = {
            'ilf': 2.0, 'eif': 2.0, 'ei': 2.0, 'eo': 2.0, 'eq': 2.0,
            'det': 1.5, 'ret': 1.5,
            'complexity': 1.5, 'function': 1.3
        }
    
    def search_function_rules(
        self,
        function_type: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索功能分类规则"""
        
        query = f"{function_type} 功能 分类 规则"
        
        # TF-IDF搜索
        results = self.tfidf_retriever.search(
            query=query,
            k=k,
            filter_metadata={'category': 'classification'}
        )
        
        # 应用NESMA术语权重提升
        boosted_results = []
        for doc, score in results:
            boost_factor = 1.0
            content_lower = doc.page_content.lower()
            
            for term, weight in self.term_weights.items():
                if term in content_lower:
                    boost_factor *= weight
            
            boosted_score = min(score * boost_factor, 1.0)  # 限制最大分数
            boosted_results.append((doc, boosted_score))
        
        # 重新排序
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_results
    
    def search_complexity_rules(
        self,
        function_type: str,
        det_count: Optional[int] = None,
        ret_count: Optional[int] = None,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """搜索复杂度计算规则"""
        
        query_parts = [f"{function_type}", "复杂度", "DET", "RET"]
        
        if det_count is not None:
            query_parts.append(f"DET {det_count}")
        if ret_count is not None:
            query_parts.append(f"RET {ret_count}")
        
        query = " ".join(query_parts)
        
        return self.tfidf_retriever.search(
            query=query,
            k=k,
            filter_metadata={
                'category': 'complexity',
                'function_type': function_type
            }
        )


class COSMICKeywordRetriever:
    """COSMIC专用关键词检索器"""
    
    def __init__(self, documents: List[Document]):
        # 过滤COSMIC文档
        cosmic_docs = [
            doc for doc in documents
            if doc.metadata.get('source_type') == 'COSMIC'
        ]
        
        self.tfidf_retriever = TFIDFRetriever(cosmic_docs)
        self.boolean_retriever = BooleanRetriever(cosmic_docs)
        
        # COSMIC特定术语权重
        self.term_weights = {
            'entry': 2.0, 'exit': 2.0, 'read': 2.0, 'write': 2.0,
            'movement': 1.8, 'functional': 1.5, 'boundary': 1.5,
            'cfp': 1.8, 'data': 1.3
        }
    
    def search_data_movement_rules(
        self,
        movement_type: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索数据移动规则"""
        
        query = f"{movement_type} 数据移动 规则 识别"
        
        results = self.tfidf_retriever.search(
            query=query,
            k=k,
            filter_metadata={'category': 'data_movement'}
        )
        
        # 应用COSMIC术语权重提升
        boosted_results = []
        for doc, score in results:
            boost_factor = 1.0
            content_lower = doc.page_content.lower()
            
            for term, weight in self.term_weights.items():
                if term in content_lower:
                    boost_factor *= weight
            
            boosted_score = min(score * boost_factor, 1.0)
            boosted_results.append((doc, boosted_score))
        
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results
    
    def search_boundary_rules(
        self,
        boundary_description: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索边界分析规则"""
        
        query = f"{boundary_description} 软件边界 持久存储 功能用户"
        
        return self.tfidf_retriever.search(
            query=query,
            k=k,
            filter_metadata={'category': 'boundary'}
        )


if __name__ == "__main__":
    # 测试关键词检索器
    
    # 创建示例文档
    test_documents = [
        Document(
            page_content="ILF是内部逻辑文件，由应用程序内部维护的数据组成",
            metadata={"source_type": "NESMA", "category": "classification", "function_type": "ILF"}
        ),
        Document(
            page_content="Entry数据移动表示数据从功能用户进入软件边界",
            metadata={"source_type": "COSMIC", "category": "data_movement", "movement_type": "Entry"}
        ),
        Document(
            page_content="DET计算包括用户可识别的数据元素，RET是记录元素类型的数量",
            metadata={"source_type": "NESMA", "category": "complexity", "function_type": "ILF"}
        )
    ]
    
    # 测试TF-IDF检索器
    print("🔍 测试TF-IDF检索器")
    tfidf = TFIDFRetriever(test_documents)
    
    results = tfidf.search("ILF 内部逻辑文件", k=2)
    print(f"检索结果: {len(results)} 个文档")
    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. 分数: {score:.3f}, 内容: {doc.page_content[:30]}...")
    
    # 测试NESMA专用检索器
    print("\n🎯 测试NESMA专用检索器")
    nesma_retriever = NESMAKeywordRetriever(test_documents)
    
    results = nesma_retriever.search_function_rules("ILF", k=2)
    print(f"NESMA功能规则: {len(results)} 个文档")
    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. 分数: {score:.3f}, 内容: {doc.page_content[:30]}...") 