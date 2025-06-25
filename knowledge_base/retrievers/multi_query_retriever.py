"""
量子智能化功能点估算系统 - 多查询检索器

通过生成多个查询变体提高检索质量
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import asyncio
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

logger = logging.getLogger(__name__)


class EnhancedMultiQueryRetriever:
    """增强多查询检索器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        num_queries: int = 3,
        include_original: bool = True
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.num_queries = num_queries
        self.include_original = include_original
        
        # 创建基础多查询检索器
        self.base_retriever = vector_store.as_retriever()
        
        # 多查询检索器
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=llm,
            include_original=include_original
        )
        
        # 上下文压缩检索器
        compressor = LLMChainExtractor.from_llm(llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.multi_query_retriever
        )
    
    async def retrieve_documents(
        self,
        query: str,
        k: int = 10,
        use_compression: bool = True,
        deduplicate: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """检索文档"""
        
        try:
            logger.debug(f"执行多查询检索: {query}")
            
            if use_compression:
                # 使用压缩检索器
                documents = await self.compression_retriever.aget_relevant_documents(query)
            else:
                # 使用基础多查询检索器
                documents = await self.multi_query_retriever.aget_relevant_documents(query)
            
            # 去重
            if deduplicate:
                documents = self._deduplicate_documents(documents)
            
            # 分数过滤（如果支持）
            if score_threshold is not None:
                documents = self._filter_by_score(documents, score_threshold)
            
            # 限制返回数量
            documents = documents[:k]
            
            logger.info(f"多查询检索返回 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"❌ 多查询检索失败: {e}")
            return []
    
    def generate_query_variations(self, original_query: str) -> List[str]:
        """生成查询变体"""
        
        # 使用LLM生成查询变体
        prompt = f"""
        基于以下原始查询，生成{self.num_queries}个不同的查询变体，这些变体应该：
        1. 保持相同的语义含义
        2. 使用不同的措辞和表达方式
        3. 可能包含同义词或相关术语
        4. 适用于功能点估算知识库检索
        
        原始查询: {original_query}
        
        请返回{self.num_queries}个查询变体，每行一个：
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # 解析响应
            lines = response.content.strip().split('\n')
            variations = []
            
            for line in lines:
                line = line.strip()
                # 移除序号标记
                if line and not line.startswith('#'):
                    # 移除可能的序号前缀
                    import re
                    clean_line = re.sub(r'^\d+\.?\s*', '', line)
                    if clean_line:
                        variations.append(clean_line)
            
            # 确保数量
            variations = variations[:self.num_queries]
            
            # 添加原始查询（如果需要）
            if self.include_original and original_query not in variations:
                variations.insert(0, original_query)
            
            logger.debug(f"生成了 {len(variations)} 个查询变体")
            return variations
            
        except Exception as e:
            logger.error(f"❌ 生成查询变体失败: {e}")
            return [original_query]  # 回退到原始查询
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """去重文档"""
        
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            # 使用内容前500字符作为去重标识
            content_key = doc.page_content[:500]
            content_hash = hash(content_key)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        logger.debug(f"去重后文档数量: {len(unique_docs)} (原始: {len(documents)})")
        return unique_docs
    
    def _filter_by_score(
        self,
        documents: List[Document],
        threshold: float
    ) -> List[Document]:
        """根据分数过滤文档"""
        
        # 这里假设文档对象可能包含分数信息
        # 在实际实现中，可能需要调整
        filtered_docs = []
        
        for doc in documents:
            score = getattr(doc, 'score', None)
            if score is None or score >= threshold:
                filtered_docs.append(doc)
        
        return filtered_docs


class NESMAMultiQueryRetriever(EnhancedMultiQueryRetriever):
    """NESMA专用多查询检索器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseLanguageModel,
        embeddings: Embeddings
    ):
        super().__init__(vector_store, llm, embeddings)
        
        # NESMA特定的查询模板
        self.nesma_query_templates = [
            "如何识别和分类{function_type}类型的功能？",
            "{function_type}的详细定义和特征是什么？",
            "在NESMA标准中，{function_type}的分类规则有哪些？",
            "什么情况下功能应该被分类为{function_type}？",
            "{function_type}与其他功能类型的区别在哪里？"
        ]
        
        self.complexity_query_templates = [
            "如何计算{function_type}的复杂度？",
            "{function_type}的DET和RET计算方法是什么？",
            "复杂度为{complexity}的{function_type}有什么特征？",
            "影响{function_type}复杂度的因素有哪些？"
        ]
    
    def generate_function_classification_queries(
        self,
        function_description: str,
        function_type: Optional[str] = None
    ) -> List[str]:
        """生成功能分类查询变体"""
        
        queries = [function_description]
        
        if function_type:
            # 使用模板生成查询
            for template in self.nesma_query_templates:
                query = template.format(function_type=function_type)
                queries.append(query)
        
        # 添加通用查询变体
        general_queries = [
            f"{function_description} NESMA分类",
            f"如何根据NESMA标准分析：{function_description}",
            f"功能分类规则 {function_description}",
            f"{function_description} 功能类型识别"
        ]
        
        queries.extend(general_queries)
        return queries[:self.num_queries + 1]
    
    def generate_complexity_calculation_queries(
        self,
        function_type: str,
        complexity_context: Optional[str] = None
    ) -> List[str]:
        """生成复杂度计算查询变体"""
        
        queries = []
        
        # 使用复杂度模板
        for template in self.complexity_query_templates:
            if "{complexity}" in template and complexity_context:
                query = template.format(
                    function_type=function_type,
                    complexity=complexity_context
                )
            else:
                query = template.format(function_type=function_type)
            queries.append(query)
        
        # 添加具体查询
        specific_queries = [
            f"{function_type} DET RET 计算规则",
            f"{function_type} 复杂度矩阵",
            f"NESMA {function_type} 复杂度确定方法"
        ]
        
        queries.extend(specific_queries)
        return queries[:self.num_queries]


class COSMICMultiQueryRetriever(EnhancedMultiQueryRetriever):
    """COSMIC专用多查询检索器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseLanguageModel,
        embeddings: Embeddings
    ):
        super().__init__(vector_store, llm, embeddings)
        
        # COSMIC特定的查询模板
        self.data_movement_templates = [
            "如何识别{movement_type}类型的数据移动？",
            "{movement_type}数据移动的定义和特征是什么？",
            "在什么情况下数据移动应该被分类为{movement_type}？",
            "{movement_type}与其他数据移动类型的区别？"
        ]
        
        self.boundary_templates = [
            "如何定义软件边界？",
            "功能用户的识别方法是什么？",
            "持久存储边界的确定原则有哪些？",
            "边界分析的关键要素是什么？"
        ]
    
    def generate_data_movement_queries(
        self,
        process_description: str,
        movement_type: Optional[str] = None
    ) -> List[str]:
        """生成数据移动查询变体"""
        
        queries = [process_description]
        
        if movement_type:
            # 使用模板生成查询
            for template in self.data_movement_templates:
                query = template.format(movement_type=movement_type)
                queries.append(query)
        
        # 添加通用查询变体
        general_queries = [
            f"{process_description} 数据移动识别",
            f"COSMIC数据移动分析：{process_description}",
            f"功能过程分析 {process_description}",
            f"{process_description} Entry Exit Read Write"
        ]
        
        queries.extend(general_queries)
        return queries[:self.num_queries + 1]
    
    def generate_boundary_analysis_queries(
        self,
        boundary_description: str
    ) -> List[str]:
        """生成边界分析查询变体"""
        
        queries = [boundary_description]
        
        # 使用边界模板
        queries.extend(self.boundary_templates)
        
        # 添加具体查询
        specific_queries = [
            f"{boundary_description} 软件边界定义",
            f"功能用户识别 {boundary_description}",
            f"{boundary_description} 持久存储边界",
            f"COSMIC边界分析 {boundary_description}"
        ]
        
        queries.extend(specific_queries)
        return queries[:self.num_queries + 1]


class AdaptiveQueryExpander:
    """自适应查询扩展器"""
    
    def __init__(
        self,
        nesma_retriever: NESMAMultiQueryRetriever,
        cosmic_retriever: COSMICMultiQueryRetriever
    ):
        self.nesma_retriever = nesma_retriever
        self.cosmic_retriever = cosmic_retriever
        
        # 查询意图识别模式
        self.nesma_patterns = [
            r'\b(?:ILF|EIF|EI|EO|EQ)\b',
            r'\b(?:DET|RET)\b',
            r'NESMA',
            r'功能分类',
            r'复杂度计算'
        ]
        
        self.cosmic_patterns = [
            r'\b(?:Entry|Exit|Read|Write)\b',
            r'数据移动',
            r'COSMIC',
            r'功能用户',
            r'软件边界',
            r'CFP'
        ]
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        
        import re
        
        intent = {
            'standards': set(),
            'categories': set(),
            'confidence': 0.0
        }
        
        query_lower = query.lower()
        
        # 检查NESMA模式
        nesma_matches = 0
        for pattern in self.nesma_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                nesma_matches += 1
        
        # 检查COSMIC模式
        cosmic_matches = 0
        for pattern in self.cosmic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                cosmic_matches += 1
        
        # 确定标准
        if nesma_matches > 0:
            intent['standards'].add('NESMA')
        if cosmic_matches > 0:
            intent['standards'].add('COSMIC')
        
        # 如果没有明确指向，两个都包含
        if not intent['standards']:
            intent['standards'] = {'NESMA', 'COSMIC'}
        
        # 确定类别
        if any(term in query_lower for term in ['分类', 'classification']):
            intent['categories'].add('classification')
        if any(term in query_lower for term in ['复杂度', 'complexity']):
            intent['categories'].add('complexity')
        if any(term in query_lower for term in ['边界', 'boundary']):
            intent['categories'].add('boundary')
        if any(term in query_lower for term in ['数据移动', 'data movement']):
            intent['categories'].add('data_movement')
        
        # 计算置信度
        total_matches = nesma_matches + cosmic_matches
        if total_matches > 0:
            intent['confidence'] = min(total_matches / 3.0, 1.0)
        
        return intent
    
    async def expand_and_retrieve(
        self,
        query: str,
        k: int = 10,
        auto_detect: bool = True
    ) -> Dict[str, List[Document]]:
        """扩展查询并检索"""
        
        results = {}
        
        if auto_detect:
            # 自动检测查询意图
            intent = self.analyze_query_intent(query)
            standards = intent['standards']
        else:
            # 检索两个标准
            standards = {'NESMA', 'COSMIC'}
        
        # 执行检索
        if 'NESMA' in standards:
            try:
                nesma_docs = await self.nesma_retriever.retrieve_documents(
                    query=query,
                    k=k
                )
                results['NESMA'] = nesma_docs
            except Exception as e:
                logger.error(f"NESMA检索失败: {e}")
                results['NESMA'] = []
        
        if 'COSMIC' in standards:
            try:
                cosmic_docs = await self.cosmic_retriever.retrieve_documents(
                    query=query,
                    k=k
                )
                results['COSMIC'] = cosmic_docs
            except Exception as e:
                logger.error(f"COSMIC检索失败: {e}")
                results['COSMIC'] = []
        
        logger.info(f"扩展查询检索完成: {list(results.keys())}")
        return results


if __name__ == "__main__":
    # 测试多查询检索器
    print("多查询检索器模块已加载")
    print("功能特性:")
    print("1. 自动生成查询变体")
    print("2. 多查询并行检索")
    print("3. 结果去重和压缩")
    print("4. NESMA/COSMIC专用查询模板")
    print("5. 自适应查询意图识别")
    
    # 示例用法（需要实际的向量存储和语言模型）
    print("\n示例查询变体生成:")
    
    # 模拟查询变体生成
    original_query = "如何识别ILF类型的功能"
    
    nesma_templates = [
        "如何识别和分类ILF类型的功能？",
        "ILF的详细定义和特征是什么？", 
        "在NESMA标准中，ILF的分类规则有哪些？",
        "什么情况下功能应该被分类为ILF？"
    ]
    
    print(f"原始查询: {original_query}")
    print("生成的查询变体:")
    for i, template in enumerate(nesma_templates, 1):
        print(f"  {i}. {template}") 