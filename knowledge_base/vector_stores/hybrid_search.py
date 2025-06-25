"""
量子智能化功能点估算系统 - 混合检索策略

结合语义搜索和关键词搜索的混合检索方案
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# 安装时可能需要：pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 未安装，关键词搜索功能将受限")


class HybridSearchStrategy:
    """混合搜索策略基类"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Embeddings,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # 验证权重
        if abs(semantic_weight + keyword_weight - 1.0) > 0.001:
            raise ValueError("语义权重和关键词权重之和必须等于1.0")
        
        # BM25索引（延迟初始化）
        self._bm25_index = None
        self._bm25_documents: List[Document] = []
        self._bm25_corpus: List[List[str]] = []
        
    def build_bm25_index(self, documents: List[Document]):
        """构建BM25索引"""
        
        if not BM25_AVAILABLE:
            logger.warning("BM25不可用，跳过关键词索引构建")
            return
        
        logger.info(f"🔨 构建BM25索引，文档数: {len(documents)}")
        
        self._bm25_documents = documents
        self._bm25_corpus = []
        
        for doc in documents:
            # 分词处理
            tokens = self._tokenize(doc.page_content)
            self._bm25_corpus.append(tokens)
        
        # 创建BM25索引
        self._bm25_index = BM25Okapi(self._bm25_corpus)
        
        logger.info("✅ BM25索引构建完成")
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        
        # 简单的中英文分词
        # 对于生产环境，建议使用jieba等专业分词工具
        
        # 英文单词分割
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 中文字符分割
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        
        # 合并结果
        tokens = words + chinese_chars
        
        # 过滤停用词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '这', '那', '一个', '一种'}
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def semantic_search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """语义搜索"""
        
        try:
            if filter_metadata:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            logger.debug(f"语义搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"❌ 语义搜索失败: {e}")
            return []
    
    def keyword_search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """关键词搜索（BM25）"""
        
        if not BM25_AVAILABLE or not self._bm25_index:
            logger.warning("BM25索引不可用，回退到简单文本匹配")
            return self._simple_text_search(query, k, filter_metadata)
        
        try:
            # 分词查询
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                logger.warning("查询词为空，返回空结果")
                return []
            
            # BM25评分
            scores = self._bm25_index.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]
            
            results = []
            for idx in top_indices:
                doc = self._bm25_documents[idx]
                score = float(scores[idx])
                
                # 应用元数据过滤
                if filter_metadata:
                    if not all(
                        doc.metadata.get(key) == value
                        for key, value in filter_metadata.items()
                    ):
                        continue
                
                results.append((doc, score))
            
            logger.debug(f"关键词搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"❌ 关键词搜索失败: {e}")
            return []
    
    def _simple_text_search(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """简单文本匹配搜索（BM25不可用时的后备方案）"""
        
        query_lower = query.lower()
        results = []
        
        for doc in self._bm25_documents:
            # 应用元数据过滤
            if filter_metadata:
                if not all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    continue
            
            # 计算简单的文本匹配分数
            content_lower = doc.page_content.lower()
            score = 0.0
            
            # 基于关键词出现次数计算分数
            for token in self._tokenize(query):
                if token in content_lower:
                    score += content_lower.count(token)
            
            if score > 0:
                results.append((doc, score))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[Tuple[Document, float]]:
        """混合搜索"""
        
        # 获取更多候选结果进行融合
        candidate_k = min(k * 3, 50)
        
        # 语义搜索
        semantic_results = self.semantic_search(
            query=query,
            k=candidate_k,
            filter_metadata=filter_metadata
        )
        
        # 关键词搜索
        keyword_results = self.keyword_search(
            query=query,
            k=candidate_k,
            filter_metadata=filter_metadata
        )
        
        # 融合结果
        fused_results = self._fuse_results(
            semantic_results,
            keyword_results,
            query
        )
        
        # 重排序
        if rerank and len(fused_results) > k:
            fused_results = self._rerank_results(fused_results, query)
        
        # 返回top-k
        return fused_results[:k]
    
    def _fuse_results(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        query: str
    ) -> List[Tuple[Document, float]]:
        """融合搜索结果"""
        
        # 归一化分数
        semantic_normalized = self._normalize_scores(semantic_results)
        keyword_normalized = self._normalize_scores(keyword_results)
        
        # 合并结果，按文档内容去重
        doc_scores: Dict[str, Tuple[Document, float, float]] = {}
        
        # 处理语义搜索结果
        for doc, score in semantic_normalized:
            doc_id = self._get_document_id(doc)
            doc_scores[doc_id] = (doc, score, 0.0)
        
        # 处理关键词搜索结果
        for doc, score in keyword_normalized:
            doc_id = self._get_document_id(doc)
            if doc_id in doc_scores:
                # 文档已存在，更新关键词分数
                existing_doc, sem_score, _ = doc_scores[doc_id]
                doc_scores[doc_id] = (existing_doc, sem_score, score)
            else:
                # 新文档，仅有关键词分数
                doc_scores[doc_id] = (doc, 0.0, score)
        
        # 计算混合分数
        hybrid_results = []
        for doc_id, (doc, sem_score, kw_score) in doc_scores.items():
            hybrid_score = (
                self.semantic_weight * sem_score +
                self.keyword_weight * kw_score
            )
            hybrid_results.append((doc, hybrid_score))
        
        # 按混合分数排序
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"混合搜索融合了 {len(hybrid_results)} 个唯一结果")
        return hybrid_results
    
    def _normalize_scores(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """归一化分数到[0,1]范围"""
        
        if not results:
            return []
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # 所有分数相同，返回相同的归一化分数
            return [(doc, 1.0) for doc, _ in results]
        
        # Min-Max归一化
        normalized_results = []
        for doc, score in results:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_results.append((doc, normalized_score))
        
        return normalized_results
    
    def _get_document_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        
        # 基于内容和主要元数据生成ID
        content_hash = hash(doc.page_content[:500])  # 使用前500字符避免过长
        source = doc.metadata.get('source', 'unknown')
        chunk_index = doc.metadata.get('chunk_index', 0)
        
        return f"{source}_{chunk_index}_{content_hash}"
    
    def _rerank_results(
        self,
        results: List[Tuple[Document, float]],
        query: str
    ) -> List[Tuple[Document, float]]:
        """重排序结果"""
        
        # 简单的重排序策略：根据查询词在文档中的出现情况调整分数
        query_terms = set(self._tokenize(query))
        
        reranked_results = []
        for doc, score in results:
            doc_terms = set(self._tokenize(doc.page_content))
            
            # 计算查询词覆盖率
            if query_terms:
                coverage = len(query_terms.intersection(doc_terms)) / len(query_terms)
            else:
                coverage = 0.0
            
            # 调整分数（加权组合）
            boosted_score = score * (1 + coverage * 0.2)  # 最多提升20%
            
            reranked_results.append((doc, boosted_score))
        
        # 重新排序
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results


class NESMAHybridSearch(HybridSearchStrategy):
    """NESMA专用混合搜索"""
    
    def __init__(self, vector_store: VectorStore, embeddings: Embeddings):
        super().__init__(
            vector_store=vector_store,
            embeddings=embeddings,
            semantic_weight=0.6,  # NESMA更依赖规则匹配
            keyword_weight=0.4
        )
        
        # NESMA特定关键词
        self.nesma_keywords = {
            'function_types': ['ILF', 'EIF', 'EI', 'EO', 'EQ'],
            'complexity_terms': ['DET', 'RET', 'Low', 'Average', 'High'],
            'data_terms': ['data element', 'record element', 'logical file'],
            'process_terms': ['input', 'output', 'inquiry', 'maintain']
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """NESMA特定分词"""
        
        # 基础分词
        tokens = super()._tokenize(text)
        
        # 添加NESMA术语识别
        text_upper = text.upper()
        for category, keywords in self.nesma_keywords.items():
            for keyword in keywords:
                if keyword in text_upper:
                    tokens.append(keyword.lower())
        
        return tokens
    
    def search_function_classification_rules(
        self,
        function_description: str,
        function_type: Optional[str] = None,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索功能分类规则"""
        
        # 构建查询
        query_parts = [function_description]
        if function_type:
            query_parts.append(f"功能类型 {function_type}")
        
        query = " ".join(query_parts)
        
        # 过滤条件
        filter_metadata = {
            'source_type': 'NESMA',
            'category': 'classification'
        }
        
        if function_type:
            filter_metadata['function_type'] = function_type
        
        return self.hybrid_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
    
    def search_complexity_calculation_rules(
        self,
        function_type: str,
        det_count: Optional[int] = None,
        ret_count: Optional[int] = None,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """搜索复杂度计算规则"""
        
        query_parts = [f"{function_type} 复杂度计算"]
        
        if det_count is not None:
            query_parts.append(f"DET {det_count}")
        if ret_count is not None:
            query_parts.append(f"RET {ret_count}")
        
        query = " ".join(query_parts)
        
        filter_metadata = {
            'source_type': 'NESMA',
            'category': 'complexity',
            'function_type': function_type
        }
        
        return self.hybrid_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )


class COSMICHybridSearch(HybridSearchStrategy):
    """COSMIC专用混合搜索"""
    
    def __init__(self, vector_store: VectorStore, embeddings: Embeddings):
        super().__init__(
            vector_store=vector_store,
            embeddings=embeddings,
            semantic_weight=0.7,  # COSMIC更依赖语义理解
            keyword_weight=0.3
        )
        
        # COSMIC特定关键词
        self.cosmic_keywords = {
            'data_movements': ['Entry', 'Exit', 'Read', 'Write'],
            'boundary_terms': ['软件边界', 'persistent storage', 'functional user'],
            'process_terms': ['功能过程', 'functional process', 'data group'],
            'measurement_terms': ['CFP', 'COSMIC Function Point', 'data movement']
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """COSMIC特定分词"""
        
        # 基础分词
        tokens = super()._tokenize(text)
        
        # 添加COSMIC术语识别
        for category, keywords in self.cosmic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    tokens.append(keyword.lower().replace(' ', '_'))
        
        return tokens
    
    def search_data_movement_rules(
        self,
        process_description: str,
        movement_type: Optional[str] = None,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索数据移动识别规则"""
        
        query_parts = [process_description, "数据移动"]
        if movement_type:
            query_parts.append(movement_type)
        
        query = " ".join(query_parts)
        
        filter_metadata = {
            'source_type': 'COSMIC',
            'category': 'data_movement'
        }
        
        if movement_type:
            filter_metadata['movement_type'] = movement_type
        
        return self.hybrid_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
    
    def search_functional_user_rules(
        self,
        user_description: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """搜索功能用户识别规则"""
        
        query = f"{user_description} 功能用户 边界"
        
        filter_metadata = {
            'source_type': 'COSMIC',
            'category': 'functional_user'
        }
        
        return self.hybrid_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
    
    def search_boundary_analysis_rules(
        self,
        boundary_description: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """搜索边界分析规则"""
        
        query = f"{boundary_description} 软件边界 持久存储"
        
        filter_metadata = {
            'source_type': 'COSMIC',
            'category': 'boundary'
        }
        
        return self.hybrid_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )


class AdaptiveHybridSearch:
    """自适应混合搜索"""
    
    def __init__(
        self,
        nesma_search: NESMAHybridSearch,
        cosmic_search: COSMICHybridSearch
    ):
        self.nesma_search = nesma_search
        self.cosmic_search = cosmic_search
        
    def intelligent_search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        k: int = 10
    ) -> Dict[str, List[Tuple[Document, float]]]:
        """智能搜索 - 根据查询内容自动选择搜索策略"""
        
        # 分析查询意图
        query_intent = self._analyze_query_intent(query, context)
        
        results = {}
        
        if query_intent.get('standard') in ['NESMA', 'BOTH']:
            # 执行NESMA搜索
            nesma_results = self._execute_nesma_search(query, query_intent, k)
            if nesma_results:
                results['NESMA'] = nesma_results
        
        if query_intent.get('standard') in ['COSMIC', 'BOTH']:
            # 执行COSMIC搜索
            cosmic_results = self._execute_cosmic_search(query, query_intent, k)
            if cosmic_results:
                results['COSMIC'] = cosmic_results
        
        # 如果没有指定标准，两个都搜索
        if not results:
            results['NESMA'] = self.nesma_search.hybrid_search(query, k)
            results['COSMIC'] = self.cosmic_search.hybrid_search(query, k)
        
        return results
    
    def _analyze_query_intent(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """分析查询意图"""
        
        intent = {
            'standard': 'BOTH',  # 默认搜索两个标准
            'category': 'general',
            'specificity': 'medium'
        }
        
        query_lower = query.lower()
        
        # 识别标准特定术语
        nesma_terms = ['nesma', 'ilf', 'eif', 'det', 'ret', 'ufp']
        cosmic_terms = ['cosmic', 'cfp', 'entry', 'exit', 'read', 'write', 'data movement']
        
        nesma_count = sum(1 for term in nesma_terms if term in query_lower)
        cosmic_count = sum(1 for term in cosmic_terms if term in query_lower)
        
        if nesma_count > cosmic_count:
            intent['standard'] = 'NESMA'
        elif cosmic_count > nesma_count:
            intent['standard'] = 'COSMIC'
        
        # 识别查询类别
        if any(term in query_lower for term in ['分类', 'classification', '类型']):
            intent['category'] = 'classification'
        elif any(term in query_lower for term in ['复杂度', 'complexity']):
            intent['category'] = 'complexity'
        elif any(term in query_lower for term in ['边界', 'boundary']):
            intent['category'] = 'boundary'
        elif any(term in query_lower for term in ['计算', 'calculation']):
            intent['category'] = 'calculation'
        
        # 考虑上下文信息
        if context:
            if context.get('current_standard'):
                intent['standard'] = context['current_standard']
            if context.get('function_type'):
                intent['function_type'] = context['function_type']
        
        return intent
    
    def _execute_nesma_search(
        self,
        query: str,
        intent: Dict[str, Any],
        k: int
    ) -> List[Tuple[Document, float]]:
        """执行NESMA特定搜索"""
        
        category = intent.get('category', 'general')
        
        if category == 'classification':
            return self.nesma_search.search_function_classification_rules(
                function_description=query,
                k=k
            )
        elif category == 'complexity':
            function_type = intent.get('function_type')
            if function_type:
                return self.nesma_search.search_complexity_calculation_rules(
                    function_type=function_type,
                    k=k
                )
        
        # 默认混合搜索
        return self.nesma_search.hybrid_search(query, k)
    
    def _execute_cosmic_search(
        self,
        query: str,
        intent: Dict[str, Any],
        k: int
    ) -> List[Tuple[Document, float]]:
        """执行COSMIC特定搜索"""
        
        category = intent.get('category', 'general')
        
        if category == 'data_movement' or 'data movement' in query.lower():
            return self.cosmic_search.search_data_movement_rules(
                process_description=query,
                k=k
            )
        elif category == 'boundary' or '边界' in query:
            return self.cosmic_search.search_boundary_analysis_rules(
                boundary_description=query,
                k=k
            )
        elif 'functional user' in query.lower() or '功能用户' in query:
            return self.cosmic_search.search_functional_user_rules(
                user_description=query,
                k=k
            )
        
        # 默认混合搜索
        return self.cosmic_search.hybrid_search(query, k)


if __name__ == "__main__":
    # 测试混合搜索（需要实际的向量存储和嵌入模型）
    print("混合搜索策略模块已加载")
    print(f"BM25可用: {BM25_AVAILABLE}")
    print("使用说明:")
    print("1. 首先创建向量存储和嵌入模型")
    print("2. 实例化HybridSearchStrategy或其子类")
    print("3. 调用build_bm25_index()构建关键词索引")
    print("4. 使用hybrid_search()进行混合搜索") 