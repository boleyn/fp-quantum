"""
量子智能化功能点估算系统 - 规则检索智能体

基于PgVector实现智能知识检索和验证
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.base.base_agent import BaseAgent
from models.common_models import EstimationStandard, KnowledgeQuery, KnowledgeResult, ValidationResult, ConfidenceLevel
from knowledge_base.vector_stores.pgvector_store import PgVectorStore
from knowledge_base.embeddings.embedding_models import get_embedding_model
from knowledge_base.auto_setup import ensure_knowledge_base_ready
import logging

logger = logging.getLogger(__name__)

# 结构化输出模型
class RelevanceValidationResult(BaseModel):
    """相关性验证结果模型"""
    relevant: bool = Field(description="内容是否相关")
    partially_relevant: bool = Field(description="内容是否部分相关", default=False)
    reasoning: str = Field(description="详细说明")


class RuleRetrieverAgent(BaseAgent):
    """规则检索智能体 - 基于PgVector"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        vector_store: PgVectorStore,
        agent_name: str = "RuleRetriever"
    ):
        super().__init__(agent_name, llm)  # 修复：参数顺序应该是agent_id, llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        
        # 检索配置
        self.retrieval_config = {
            "default_k": 5,
            "relevance_threshold": 0.7,
            "max_retries": 3
        }

    async def initialize(self):
        """初始化智能体"""
        await super().initialize()
        
        # 检查知识库状态
        logger.info("🔍 检查知识库状态...")
        try:
            kb_ready = await ensure_knowledge_base_ready()
            if kb_ready:
                logger.info("✅ 知识库已就绪")
            else:
                logger.warning("⚠️ 知识库初始化失败，但继续运行")
        except Exception as e:
            logger.error(f"❌ 知识库检查失败: {e}")
        
        logger.info("✅ 规则检索智能体初始化完成")
    
    async def retrieve_rules(
        self,
        query: str,
        standard: Optional[EstimationStandard] = None,
        use_cache: bool = True,
        min_chunks: int = 3,
        max_retries: int = 2
    ) -> KnowledgeResult:
        """检索相关规则"""
        
        start_time = datetime.now()
        
        # 🔥 增加详细的调试日志
        logger.info(f"🔍 开始检索规则:")
        logger.info(f"  - 查询: {query}")
        logger.info(f"  - 标准: {standard}")
        logger.info(f"  - 使用缓存: {use_cache}")
        logger.info(f"  - 最小块数: {min_chunks}")
        
        # 生成缓存键
        cache_key = f"{query}_{standard}_{min_chunks}"
        
        # 检查缓存
        if use_cache and hasattr(self, 'retrieval_cache') and cache_key in self.retrieval_cache:
            cached_result = self.retrieval_cache[cache_key]
            logger.info(f"📋 使用缓存结果: {query}")
            return cached_result
        
        # 初始化缓存和历史记录
        if not hasattr(self, 'retrieval_cache'):
            self.retrieval_cache = {}
        if not hasattr(self, 'retrieval_history'):
            self.retrieval_history = []
        
        try:
            # 🔥 优化：首先尝试从简单知识库检索，增加详细日志
            logger.info("📚 尝试从简单知识库检索...")
            retrieval_result = await self._retrieve_from_simple_kb(query, standard)
            
            logger.info(f"📊 简单知识库检索结果: {retrieval_result.total_chunks} 个块")
            if retrieval_result.retrieved_chunks:
                for i, chunk in enumerate(retrieval_result.retrieved_chunks):
                    logger.info(f"  块{i+1}: 相关性={chunk.get('relevance_score', 0):.3f}, 内容长度={len(chunk.get('content', ''))}")
            
            # 如果简单知识库没有结果，尝试向量检索
            if not retrieval_result.retrieved_chunks and hasattr(self, 'vector_store'):
                logger.info("🔍 简单知识库无结果，尝试向量检索...")
                retrieval_result = await self._retrieve_from_vector_store(query, standard, min_chunks)
                logger.info(f"📊 向量检索结果: {retrieval_result.total_chunks} 个块")
            
        except Exception as e:
            logger.error(f"❌ 检索失败: {str(e)}")
            import traceback
            logger.error(f"❌ 检索异常详情: {traceback.format_exc()}")
            # 返回空结果
            retrieval_result = KnowledgeResult(
                query=query,
                source_type=standard or EstimationStandard.BOTH,
                retrieved_chunks=[],
                total_chunks=0,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
        
        # 验证检索结果
        validation_result = None
        if retrieval_result and retrieval_result.retrieved_chunks:
            validation_result = await self._validate_retrieval_result(retrieval_result)
            # 不设置validation_result到对象上，因为KnowledgeResult模型没有这个字段
        
        # 缓存结果
        if use_cache and retrieval_result:
            self.retrieval_cache[cache_key] = retrieval_result
        
        # 记录检索历史
        if retrieval_result:
            self.retrieval_history.append(retrieval_result)
        
        # 🔥 记录最终结果统计
        logger.info(f"✅ 规则检索完成:")
        logger.info(f"  - 总块数: {retrieval_result.total_chunks}")
        logger.info(f"  - 检索用时: {getattr(retrieval_result, 'processing_time_ms', 0)}ms")
        if validation_result:
            logger.info(f"  - 验证分数: {validation_result.confidence_score:.3f}")
        
        return retrieval_result
    
    async def _retrieve_from_simple_kb(
        self,
        query: str,
        standard: Optional[EstimationStandard] = None
    ) -> KnowledgeResult:
        """从简单知识库JSON文件检索"""
        
        import json
        from pathlib import Path
        
        start_time = datetime.now()
        
        try:
            # 加载简单知识库
            kb_file = Path("knowledge_base/simple_kb.json")
            if not kb_file.exists():
                logger.warning("⚠️ 简单知识库文件不存在")
                return KnowledgeResult(
                    query=query,
                    source_type=standard or EstimationStandard.BOTH,
                    retrieved_chunks=[],
                    total_chunks=0,
                    processing_time_ms=0
                )
            
            with open(kb_file, 'r', encoding='utf-8') as f:
                knowledge_entries = json.load(f)
            
            logger.info(f"📚 加载了 {len(knowledge_entries)} 条知识库条目")
            
            # 🔥 记录搜索参数
            logger.info(f"🔍 搜索参数详情:")
            logger.info(f"  - 查询词: '{query}'")
            logger.info(f"  - 标准过滤: {standard}")
            
            # 简单的关键词匹配检索
            matched_chunks = []
            query_lower = query.lower()
            
            # 🔥 分析查询关键词
            query_keywords = query_lower.split()
            logger.info(f"  - 关键词列表: {query_keywords}")
            
            for i, entry in enumerate(knowledge_entries):
                content = entry.get('content', '')
                metadata = entry.get('metadata', {})
                
                # 检查标准过滤
                if standard == EstimationStandard.NESMA and metadata.get('type') != 'nesma_rules':
                    continue
                elif standard == EstimationStandard.COSMIC and metadata.get('type') != 'cosmic_rules':
                    continue
                
                # 简单的相关性评分
                relevance_score = 0.0
                match_details = []
                
                # 关键词匹配
                content_lower = content.lower()
                keyword_matches = []
                for keyword in query_keywords:
                    if keyword in content_lower:
                        keyword_matches.append(keyword)
                        relevance_score += 0.5 / len(query_keywords)
                
                if keyword_matches:
                    match_details.append(f"关键词匹配: {keyword_matches}")
                
                # 功能类型匹配
                if 'function_type' in metadata:
                    if metadata['function_type'].lower() in query_lower:
                        relevance_score += 0.4
                        match_details.append(f"功能类型匹配: {metadata['function_type']}")
                
                # 类型匹配
                if metadata.get('type') == 'nesma_rules' and 'nesma' in query_lower:
                    relevance_score += 0.3
                    match_details.append("NESMA类型匹配")
                
                # 🔥 记录匹配分析过程
                if relevance_score > 0:
                    logger.info(f"  条目{i+1} 匹配得分: {relevance_score:.3f}")
                    logger.info(f"    匹配详情: {match_details}")
                    logger.info(f"    内容预览: {content[:100]}...")
                    
                    matched_chunks.append({
                        'content': content,
                        'metadata': metadata,
                        'relevance_score': min(1.0, relevance_score),
                        'chunk_id': entry.get('id', f'kb_entry_{i}'),
                        'match_details': match_details
                    })
            
            # 按相关性排序
            matched_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # 限制结果数量
            matched_chunks = matched_chunks[:5]
            
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"🔍 简单知识库检索完成: 找到 {len(matched_chunks)} 个匹配结果")
            
            # 🔥 记录最终匹配结果详情
            for i, chunk in enumerate(matched_chunks):
                logger.info(f"  最终结果{i+1}: 分数={chunk['relevance_score']:.3f}, ID={chunk['chunk_id']}")
            
            return KnowledgeResult(
                query=query,
                source_type=standard or EstimationStandard.BOTH,
                retrieved_chunks=matched_chunks,
                total_chunks=len(matched_chunks),
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"❌ 简单知识库检索失败: {str(e)}")
            import traceback
            logger.error(f"❌ 异常详情: {traceback.format_exc()}")
            return KnowledgeResult(
                query=query,
                source_type=standard or EstimationStandard.BOTH,
                retrieved_chunks=[],
                total_chunks=0,
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def _retrieve_from_vector_store(
        self,
        query: str,
        standard: Optional[EstimationStandard] = None,
        min_chunks: int = 3
    ) -> KnowledgeResult:
        """从向量存储检索（LangChain vector as retriever）"""
        
        start_time = datetime.now()
        
        try:
            logger.info("🔍 开始向量存储检索...")
            logger.info(f"  - 查询: {query}")
            logger.info(f"  - 标准: {standard}")
            logger.info(f"  - 最小块数: {min_chunks}")
            
            # 检查向量存储是否可用
            if not self.vector_store:
                logger.warning("⚠️ 向量存储未初始化")
                return KnowledgeResult(
                    query=query,
                    source_type=standard or EstimationStandard.BOTH,
                    retrieved_chunks=[],
                    total_chunks=0,
                    processing_time_ms=0
                )
            
            # 确定搜索源类型
            source_type = None
            if standard == EstimationStandard.NESMA:
                source_type = "nesma"
            elif standard == EstimationStandard.COSMIC:
                source_type = "cosmic"
            
            logger.info(f"📋 向量检索源类型: {source_type}")
            
            # 使用向量存储的similarity_search_with_score方法
            results_with_scores = await self.vector_store.similarity_search_with_score(
                query=query,
                k=max(min_chunks, 5),
                filter={"source_type": standard.value} if standard else None
            )
            
            logger.info(f"📊 向量检索返回: {len(results_with_scores)} 个文档")
            
            # 转换为标准格式
            retrieved_chunks = []
            for i, (doc, score) in enumerate(results_with_scores):
                chunk = {
                    'content': doc.page_content,
                    'metadata': doc.metadata or {},
                    'relevance_score': float(1.0 - score),  # 转换距离为相似度分数
                    'chunk_id': doc.metadata.get('id', f'vector_chunk_{i}'),
                    'source': 'vector_store'
                }
                retrieved_chunks.append(chunk)
                
                logger.info(f"  向量块{i+1}: ID={chunk['chunk_id']}, 分数={chunk['relevance_score']:.3f}, 长度={len(chunk['content'])}")
            
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"✅ 向量检索完成: {len(retrieved_chunks)} 个结果，耗时 {processing_time_ms}ms")
            
            return KnowledgeResult(
                query=query,
                source_type=standard or EstimationStandard.BOTH,
                retrieved_chunks=retrieved_chunks,
                total_chunks=len(retrieved_chunks),
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"❌ 向量存储检索失败: {str(e)}")
            import traceback
            logger.error(f"❌ 异常详情: {traceback.format_exc()}")
            
            # 如果向量检索失败，尝试使用retriever接口
            try:
                logger.info("🔄 尝试使用retriever接口...")
                
                # 使用LangChain的vector as retriever进行检索
                search_kwargs = {"k": max(min_chunks, 5)}
                if standard:
                    search_kwargs["filter"] = {"source_type": standard.value}
                
                retriever = self.vector_store.as_retriever(
                    source_type=source_type,
                    search_kwargs=search_kwargs
                )
                
                # 执行异步检索
                documents = await retriever.ainvoke(query)
                
                logger.info(f"📊 Retriever接口返回: {len(documents)} 个文档")
                
                # 转换为标准格式
                retrieved_chunks = []
                for i, doc in enumerate(documents):
                    chunk = {
                        'content': doc.page_content,
                        'metadata': doc.metadata or {},
                        'relevance_score': 0.8,  # 默认相关性分数
                        'chunk_id': doc.metadata.get('id', f'retriever_chunk_{i}'),
                        'source': 'retriever'
                    }
                    retrieved_chunks.append(chunk)
                    
                    logger.info(f"  检索块{i+1}: ID={chunk['chunk_id']}, 长度={len(chunk['content'])}")
                
                processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                
                logger.info(f"✅ Retriever检索完成: {len(retrieved_chunks)} 个结果，耗时 {processing_time_ms}ms")
                
                return KnowledgeResult(
                    query=query,
                    source_type=standard or EstimationStandard.BOTH,
                    retrieved_chunks=retrieved_chunks,
                    total_chunks=len(retrieved_chunks),
                    processing_time_ms=processing_time_ms
                )
                
            except Exception as retriever_error:
                logger.error(f"❌ Retriever接口也失败: {str(retriever_error)}")
                
                return KnowledgeResult(
                    query=query,
                    source_type=standard or EstimationStandard.BOTH,
                    retrieved_chunks=[],
                    total_chunks=0,
                    processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
                )
    
    async def _validate_retrieval_result(self, retrieval_result: KnowledgeResult) -> ValidationResult:
        """验证检索结果的质量"""
        
        if not retrieval_result.retrieved_chunks:
                    return ValidationResult(
            is_valid=False,
            confidence_score=0.0,
            confidence_level=ConfidenceLevel.LOW,
            errors=["检索结果为空"],
            warnings=[],
            suggestions=["尝试不同的查询词", "检查知识库内容"],
            metadata={}
        )
        
        # 基础质量指标
        total_chunks = len(retrieval_result.retrieved_chunks)
        
        # 处理不同类型的chunk格式（字典或对象）
        def get_relevance_score(chunk):
            if isinstance(chunk, dict):
                return chunk.get('relevance_score', 0.0)
            else:
                return getattr(chunk, 'relevance_score', 0.0)
        
        high_relevance_chunks = [
            chunk for chunk in retrieval_result.retrieved_chunks 
            if get_relevance_score(chunk) > 0.8
        ]
        medium_relevance_chunks = [
            chunk for chunk in retrieval_result.retrieved_chunks 
            if 0.6 <= get_relevance_score(chunk) <= 0.8
        ]
        
        # 计算总体质量分数
        quality_score = 0.0
        if total_chunks > 0:
            quality_score = (
                len(high_relevance_chunks) * 1.0 + 
                len(medium_relevance_chunks) * 0.7
            ) / total_chunks
        
        # 使用LLM进行深度验证
        llm_validation = await self._llm_validate_relevance(retrieval_result)
        
        # 综合判断
        issues = []
        suggestions = []
        
        if total_chunks < 2:
            issues.append("检索结果数量较少")
            suggestions.append("扩大检索范围或调整查询策略")
        
        if len(high_relevance_chunks) == 0:
            issues.append("缺乏高相关性内容")
            suggestions.append("优化查询表达或检查知识库完整性")
        
        # 确定置信度
        if quality_score >= 0.8 and llm_validation.get("relevant", False):
            confidence_level = ConfidenceLevel.HIGH
            is_valid = True
        elif quality_score >= 0.6 and llm_validation.get("partially_relevant", False):
            confidence_level = ConfidenceLevel.MEDIUM
            is_valid = True
        else:
            confidence_level = ConfidenceLevel.LOW
            is_valid = len(retrieval_result.retrieved_chunks) > 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=quality_score,
            confidence_level=confidence_level,
            errors=issues,
            warnings=[],
            suggestions=suggestions,
            metadata={
                "total_chunks": total_chunks,
                "high_relevance_count": len(high_relevance_chunks),
                "medium_relevance_count": len(medium_relevance_chunks),
                "llm_validation": llm_validation
            }
        )
    
    async def _llm_validate_relevance(self, retrieval_result: KnowledgeResult) -> Dict[str, Any]:
        """使用LLM验证检索结果的相关性"""
        
        if not self.llm or not retrieval_result.retrieved_chunks:
            return {"relevant": False, "reasoning": "LLM未初始化或无检索结果"}
        
        # 定义验证工具
        @tool
        def validate_relevance(
            relevant: bool,
            partially_relevant: bool,
            reasoning: str
        ) -> dict:
            """验证检索内容的相关性
            
            Args:
                relevant: 内容是否与查询相关
                partially_relevant: 内容是否部分相关
                reasoning: 详细的验证说明
            """
            return {
                "relevant": relevant,
                "partially_relevant": partially_relevant,
                "reasoning": reasoning
            }
        
        # 创建带工具的LLM
        llm_with_tools = self.llm.bind_tools([validate_relevance])
        
        # 处理不同类型的chunk格式（字典或对象）
        def get_chunk_content(chunk):
            if isinstance(chunk, dict):
                return chunk.get('content', '')
            else:
                return getattr(chunk, 'content', '')
        
        # 获取前3个最相关的块
        top_chunks = retrieval_result.retrieved_chunks[:3]
        chunks_text = "\n\n---\n\n".join([
            f"文档{i+1}: {get_chunk_content(chunk)[:300]}..."
            for i, chunk in enumerate(top_chunks)
        ])
        
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是知识检索质量评估专家。请评估检索到的文档内容是否与用户查询相关。

评估标准：
1. 内容是否直接回答了查询问题
2. 信息是否准确和有用
3. 是否包含查询所需的关键信息

请使用validate_relevance工具返回评估结果。"""),
            ("human", """查询: {query}

检索到的文档内容:
{chunks_text}

请评估这些内容的相关性。""")
        ])
        
        try:
            response = await llm_with_tools.ainvoke(
                validation_prompt.format_messages(
                    query=retrieval_result.query,
                    chunks_text=chunks_text
                )
            )
            
            # 解析工具调用结果
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                return tool_call["args"]
            else:
                logger.warning("LLM未使用工具调用，返回默认结果")
                return {
                    "relevant": True,
                    "reasoning": "LLM未使用工具调用，默认为相关"
                }
            
        except Exception as e:
            logger.warning(f"⚠️ LLM验证失败: {str(e)}")
            return {
                "relevant": False,
                "reasoning": f"LLM验证出错: {str(e)}"
            }
    
    async def retrieve_by_context(
        self,
        context: Dict[str, Any],
        specific_queries: Optional[List[str]] = None
    ) -> Dict[str, KnowledgeResult]:
        """根据上下文进行多方面检索"""
        
        # 提取检索查询
        queries_to_search = []
        
        if specific_queries:
            queries_to_search.extend(specific_queries)
        
        # 根据上下文生成查询
        if "function_type" in context:
            function_type = context["function_type"]
            queries_to_search.append(f"{function_type} 功能分类规则")
            queries_to_search.append(f"{function_type} 定义和特征")
        
        if "complexity" in context:
            queries_to_search.append("复杂度计算方法")
            queries_to_search.append("数据元素计算规则")
        
        if "data_elements" in context:
            queries_to_search.append("数据元素识别规则")
            queries_to_search.append("记录元素类型判定")
        
        # 并行检索
        results = {}
        tasks = []
        
        for query in queries_to_search:
            task = self.retrieve_rules(
                query=query,
                standard=context.get("standard"),
                min_chunks=2
            )
            tasks.append((query, task))
        
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (query, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"❌ 上下文检索失败 '{query}': {str(result)}")
                else:
                    results[query] = result
        
        return results
    
    async def get_retrieval_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        
        if not self.retrieval_history:
            return {
                "total_retrievals": 0,
                "average_chunks": 0,
                "average_time_ms": 0,
                "success_rate": 0.0
            }
        
        total_retrievals = len(self.retrieval_history)
        total_chunks = sum(len(r.retrieved_chunks) for r in self.retrieval_history)
        total_time = sum(r.processing_time_ms for r in self.retrieval_history)
        successful_retrievals = sum(
            1 for r in self.retrieval_history 
            if r.retrieved_chunks and len(r.retrieved_chunks) > 0
        )
        
        return {
            "total_retrievals": total_retrievals,
            "average_chunks": total_chunks / total_retrievals,
            "average_time_ms": total_time / total_retrievals,
            "success_rate": successful_retrievals / total_retrievals,
            "cache_hits": len(self.retrieval_cache),
            "standards_distribution": self._get_standards_distribution()
        }
    
    def _get_standards_distribution(self) -> Dict[str, int]:
        """获取标准分布统计"""
        distribution = {}
        for result in self.retrieval_history:
            standard = result.source_type.value
            distribution[standard] = distribution.get(standard, 0) + 1
        return distribution
    
    async def clear_cache(self):
        """清除检索缓存"""
        self.retrieval_cache.clear()
        logger.info("🗑️ 检索缓存已清除")
    
    async def export_retrieval_history(self) -> List[Dict[str, Any]]:
        """导出检索历史"""
        return [
            {
                "query": result.query,
                "source_type": result.source_type.value,
                "chunks_count": len(result.retrieved_chunks),
                "retrieval_time_ms": result.processing_time_ms,
                "timestamp": datetime.now().isoformat(),
                "validation_score": (
                    result.validation_result.validation_score 
                    if result.validation_result else None
                )
            }
            for result in self.retrieval_history
        ]

    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "规则检索与验证",
            "知识库搜索",
            "上下文化检索",
            "检索结果质量评估",
            "多标准知识源管理"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行规则检索任务"""
        if task_name == "retrieve_rules":
            result = await self.retrieve_rules(
                query=inputs["query"],
                standard=inputs.get("standard"),
                use_cache=inputs.get("use_cache", True),
                min_chunks=inputs.get("min_chunks", 3),
                max_retries=inputs.get("max_retries", 2)
            )
            # 将KnowledgeResult对象转换为字典
            return {
                "query": result.query,
                "source_type": result.source_type.value,
                "retrieved_chunks": result.retrieved_chunks,
                "total_chunks": result.total_chunks,
                "retrieval_time_ms": result.processing_time_ms,
                "validation_result": result.validation_result.__dict__ if result.validation_result else None
            }
        elif task_name == "retrieve_by_context":
            return await self.retrieve_by_context(
                context=inputs["context"],
                specific_queries=inputs.get("specific_queries")
            )
        elif task_name == "clear_cache":
            await self.clear_cache()
            return {"success": True, "message": "检索缓存已清除"}
        elif task_name == "get_statistics":
            return await self.get_retrieval_statistics()
        elif task_name == "export_history":
            return {"history": await self.export_retrieval_history()}
        else:
            raise ValueError(f"未知任务: {task_name}")


async def create_rule_retriever_agent(
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    vector_store: PgVectorStore
) -> RuleRetrieverAgent:
    """创建规则检索智能体"""
    
    agent = RuleRetrieverAgent(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store
    )
    
    await agent.initialize()
    return agent


if __name__ == "__main__":
    async def main():
        # 测试规则检索智能体
        from knowledge_base.vector_stores.pgvector_store import PgVectorStore
        
        # 初始化向量管理器
        vector_store = PgVectorStore()
        await vector_store.initialize()
        
        # 创建规则检索智能体
        agent = await create_rule_retriever_agent(vector_store.llm, vector_store.embeddings, vector_store)
        
        # 测试查询
        test_queries = [
            "NESMA 内部逻辑文件分类标准",
            "COSMIC 数据移动识别规则",
            "功能复杂度计算公式"
        ]
        
        for query in test_queries:
            print(f"\n🔍 测试查询: {query}")
            
            # 执行检索
            result = await agent.retrieve_rules(
                query=query,
                standard=EstimationStandard.BOTH,
                min_chunks=2
            )
            
            print(f"📊 检索结果: {len(result.retrieved_chunks)} 个文档块")
            print(f"⏱️ 检索耗时: {result.processing_time_ms}ms")
            
            if result.validation_result:
                print(f"✅ 验证结果: {result.validation_result.confidence_level.value}")
                print(f"📈 质量分数: {result.validation_result.validation_score:.3f}")
            
            if result.retrieved_chunks:
                best_chunk = result.retrieved_chunks[0]
                # 处理不同类型的chunk格式（字典或对象）
                def get_chunk_content(chunk):
                    if isinstance(chunk, dict):
                        return chunk.get('content', '')
                    else:
                        return getattr(chunk, 'content', '')
                
                def get_relevance_score(chunk):
                    if isinstance(chunk, dict):
                        return chunk.get('relevance_score', 0.0)
                    else:
                        return getattr(chunk, 'relevance_score', 0.0)
                
                print(f"🎯 最佳匹配 (分数: {get_relevance_score(best_chunk):.3f})")
                print(f"   内容预览: {get_chunk_content(best_chunk)[:200]}...")
        
        # 获取统计信息
        stats = await agent.get_retrieval_statistics()
        print(f"\n📊 检索统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        await vector_store.close()
    
    asyncio.run(main()) 