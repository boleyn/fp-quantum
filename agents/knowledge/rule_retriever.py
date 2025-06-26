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

from agents.base.base_agent import BaseAgent
from models.common_models import EstimationStandard, KnowledgeQuery, KnowledgeResult, ValidationResult, ConfidenceLevel
from knowledge_base.vector_stores.pgvector_store import PgVectorStore
from knowledge_base.embeddings.embedding_models import get_embedding_model

logger = logging.getLogger(__name__)


class RuleRetrieverAgent(BaseAgent):
    """规则检索智能体 - 基于PgVector"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        vector_store: PgVectorStore,
        agent_name: str = "RuleRetriever"
    ):
        super().__init__(llm, agent_name)
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
        
        # 生成缓存键
        cache_key = f"{query}_{standard}_{min_chunks}"
        
        # 检查缓存
        if use_cache and cache_key in self.retrieval_cache:
            cached_result = self.retrieval_cache[cache_key]
            logger.info(f"📋 使用缓存结果: {query}")
            return cached_result
        
        # 确定检索策略
        if standard == EstimationStandard.NESMA:
            preferred_source = "nesma"
            fallback_sources = ["common"]
        elif standard == EstimationStandard.COSMIC:
            preferred_source = "cosmic"
            fallback_sources = ["common"]
        else:
            preferred_source = None
            fallback_sources = None
        
        # 执行检索
        retrieval_result = None
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # 首次尝试：自适应检索
                    retrieval_result = await self.multi_source_retriever.adaptive_retrieve(
                        query=query,
                        preferred_source=preferred_source,
                        fallback_sources=fallback_sources,
                        min_chunks=min_chunks
                    )
                else:
                    # 重试：降低要求
                    retrieval_result = await self.multi_source_retriever.adaptive_retrieve(
                        query=query,
                        preferred_source=None,
                        fallback_sources=None,
                        min_chunks=max(1, min_chunks - attempt)
                    )
                
                # 检查结果质量
                if retrieval_result.retrieved_chunks:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ 检索尝试 {attempt + 1} 失败: {str(e)}")
                if attempt == max_retries:
                    # 最后一次尝试失败，返回空结果
                    retrieval_result = KnowledgeResult(
                        query=query,
                        source_type=standard or EstimationStandard.BOTH,
                        retrieved_chunks=[],
                        total_chunks=0,
                        retrieval_time_ms=0
                    )
        
        # 验证检索结果
        if retrieval_result and retrieval_result.retrieved_chunks:
            validation_result = await self._validate_retrieval_result(retrieval_result)
            retrieval_result.validation_result = validation_result
        
        # 缓存结果
        if use_cache and retrieval_result:
            self.retrieval_cache[cache_key] = retrieval_result
        
        # 记录检索历史
        if retrieval_result:
            self.retrieval_history.append(retrieval_result)
        
        return retrieval_result
    
    async def _validate_retrieval_result(self, retrieval_result: KnowledgeResult) -> ValidationResult:
        """验证检索结果的质量"""
        
        if not retrieval_result.retrieved_chunks:
            return ValidationResult(
                is_valid=False,
                confidence_level=ConfidenceLevel.LOW,
                issues=["检索结果为空"],
                suggestions=["尝试不同的查询词", "检查知识库内容"]
            )
        
        # 基础质量指标
        total_chunks = len(retrieval_result.retrieved_chunks)
        high_relevance_chunks = [
            chunk for chunk in retrieval_result.retrieved_chunks 
            if chunk.relevance_score > 0.8
        ]
        medium_relevance_chunks = [
            chunk for chunk in retrieval_result.retrieved_chunks 
            if 0.6 <= chunk.relevance_score <= 0.8
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
            confidence_level=confidence_level,
            validation_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            details={
                "total_chunks": total_chunks,
                "high_relevance_count": len(high_relevance_chunks),
                "medium_relevance_count": len(medium_relevance_chunks),
                "llm_validation": llm_validation
            }
        )
    
    async def _llm_validate_relevance(self, retrieval_result: KnowledgeResult) -> Dict[str, Any]:
        """使用LLM验证检索结果的相关性"""
        
        if not self.llm or not retrieval_result.retrieved_chunks:
            return {"relevant": False, "reasoning": "无法进行LLM验证"}
        
        # 准备验证内容
        query = retrieval_result.query
        top_chunks = retrieval_result.retrieved_chunks[:3]  # 只验证前3个结果
        
        chunks_text = "\n\n---\n\n".join([
            f"文档{i+1}: {chunk.content[:300]}..."
            for i, chunk in enumerate(top_chunks)
        ])
        
        validation_prompt = f"""
        请分析以下检索结果与查询的相关性：
        
        查询: {query}
        
        检索到的文档内容:
        {chunks_text}
        
        请评估：
        1. 这些文档内容是否与查询相关？
        2. 相关性程度如何？
        3. 是否包含可以回答查询的信息？
        
        请用JSON格式回答：
        {{
            "relevant": true/false,
            "partially_relevant": true/false,
            "relevance_score": 0-1之间的分数,
            "reasoning": "详细说明"
        }}
        """
        
        try:
            response = await self.llm.ainvoke(validation_prompt)
            
            # 解析LLM响应
            import json
            result = json.loads(response.content)
            
            return result
            
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
        total_time = sum(r.retrieval_time_ms for r in self.retrieval_history)
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
                "retrieval_time_ms": result.retrieval_time_ms,
                "timestamp": datetime.now().isoformat(),
                "validation_score": (
                    result.validation_result.validation_score 
                    if result.validation_result else None
                )
            }
            for result in self.retrieval_history
        ]


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
            print(f"⏱️ 检索耗时: {result.retrieval_time_ms}ms")
            
            if result.validation_result:
                print(f"✅ 验证结果: {result.validation_result.confidence_level.value}")
                print(f"📈 质量分数: {result.validation_result.validation_score:.3f}")
            
            if result.retrieved_chunks:
                best_chunk = result.retrieved_chunks[0]
                print(f"🎯 最佳匹配 (分数: {best_chunk.relevance_score:.3f})")
                print(f"   内容预览: {best_chunk.content[:200]}...")
        
        # 获取统计信息
        stats = await agent.get_retrieval_statistics()
        print(f"\n📊 检索统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        await vector_store.close()
    
    asyncio.run(main()) 