"""
量子智能化功能点估算系统 - 质量验证智能体

负责验证检索到的知识和分析结果的质量
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from agents.base.base_agent import SpecializedAgent
from models.common_models import ValidationResult, ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ValidatorAgent(SpecializedAgent):
    """质量验证智能体"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(
            agent_id="validator",
            specialty="quality_assurance",
            llm=llm
        )
        
        self.settings = get_settings()
        
        # 验证规则和标准
        self.validation_criteria = self._load_validation_criteria()
        self.quality_thresholds = self._load_quality_thresholds()
        
    def _load_validation_criteria(self) -> Dict[str, Any]:
        """加载验证标准"""
        return {
            "相关性验证": {
                "标准": [
                    "内容与查询主题匹配",
                    "涵盖查询的关键概念",
                    "提供直接相关的信息",
                    "避免无关或偏离主题的内容"
                ],
                "评分要素": [
                    "主题匹配度",
                    "概念覆盖度",
                    "信息精确度",
                    "上下文适配度"
                ]
            },
            "充分性验证": {
                "标准": [
                    "信息量足够回答问题",
                    "涵盖问题的主要方面",
                    "提供足够的细节和例子",
                    "包含必要的操作指导"
                ],
                "评分要素": [
                    "信息完整度",
                    "细节丰富度",
                    "覆盖广度",
                    "实用性"
                ]
            },
            "一致性验证": {
                "标准": [
                    "不同源之间信息一致",
                    "无自相矛盾的陈述",
                    "术语使用一致",
                    "规则应用一致"
                ],
                "评分要素": [
                    "内部一致性",
                    "跨源一致性",
                    "术语一致性",
                    "逻辑一致性"
                ]
            },
            "准确性验证": {
                "标准": [
                    "事实信息准确",
                    "引用来源可靠",
                    "数据计算正确",
                    "规则解释准确"
                ],
                "评分要素": [
                    "事实准确度",
                    "来源可靠度",
                    "计算正确性",
                    "专业水准"
                ]
            }
        }
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """加载质量阈值"""
        return {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "知识检索结果验证",
            "分析结果质量评估",
            "相关性和充分性检查",
            "一致性和准确性验证",
            "改进建议生成"
        ]
    
    async def validate_retrieved_knowledge(
        self,
        query: str,
        retrieved_documents: List[Document],
        knowledge_source: str = "unknown"
    ) -> ValidationResult:
        """验证检索到的知识"""
        
        logger.info(f"🔍 开始验证检索知识，文档数: {len(retrieved_documents)}")
        
        start_time = time.time()
        
        try:
            # 1. 相关性验证
            relevance_score = await self._validate_relevance(query, retrieved_documents)
            
            # 2. 充分性验证
            sufficiency_score = await self._validate_sufficiency(query, retrieved_documents)
            
            # 3. 一致性验证
            consistency_score = await self._validate_consistency(retrieved_documents)
            
            # 4. 准确性验证
            accuracy_score = await self._validate_accuracy(retrieved_documents, knowledge_source)
            
            # 5. 计算综合质量分数
            overall_quality = self._calculate_overall_quality(
                relevance_score, sufficiency_score, consistency_score, accuracy_score
            )
            
            # 6. 生成验证报告
            validation_report = await self._generate_validation_report(
                query, retrieved_documents, {
                    "relevance": relevance_score,
                    "sufficiency": sufficiency_score,
                    "consistency": consistency_score,
                    "accuracy": accuracy_score,
                    "overall": overall_quality
                }
            )
            
            processing_time = time.time() - start_time
            
            # 判断是否通过验证
            is_valid = overall_quality >= self.quality_thresholds["acceptable"]
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence_score=overall_quality,
                validation_details={
                    "relevance_score": relevance_score,
                    "sufficiency_score": sufficiency_score,
                    "consistency_score": consistency_score,
                    "accuracy_score": accuracy_score,
                    "overall_quality": overall_quality,
                    "quality_level": self._determine_quality_level(overall_quality),
                    "document_count": len(retrieved_documents),
                    "processing_time": processing_time
                },
                issues=validation_report.get("issues", []),
                suggestions=validation_report.get("suggestions", [])
            )
            
            logger.info(f"✅ 知识验证完成，质量分数: {overall_quality:.3f}，耗时 {processing_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 知识验证失败: {str(e)}")
            raise
    
    async def _validate_relevance(
        self,
        query: str,
        documents: List[Document]
    ) -> float:
        """验证相关性"""
        
        if not documents:
            return 0.0
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是知识质量评估专家，需要评估检索文档与查询的相关性。

相关性评估标准：
1. 文档内容是否直接回答查询问题
2. 文档是否包含查询的关键概念
3. 文档信息是否针对查询场景
4. 文档是否提供有用的相关信息

评分范围：0.0-1.0（1.0表示完全相关）"""),
            ("human", """查询：{query}

检索到的文档：
{documents}

请评估这些文档与查询的相关性，返回0.0-1.0的分数，并说明理由。""")
        ])
        
        documents_text = self._format_documents_for_validation(documents)
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                query=query,
                documents=documents_text
            )
        )
        
        # 解析相关性分数
        relevance_score = self._extract_score_from_response(response.content)
        
        # 补充基于关键词匹配的相关性评估
        keyword_relevance = self._calculate_keyword_relevance(query, documents)
        
        # 综合评估
        final_relevance = (relevance_score * 0.7 + keyword_relevance * 0.3)
        
        return min(1.0, max(0.0, final_relevance))
    
    async def _validate_sufficiency(
        self,
        query: str,
        documents: List[Document]
    ) -> float:
        """验证充分性"""
        
        if not documents:
            return 0.0
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是知识完整性评估专家，需要评估检索文档是否充分回答查询。

充分性评估标准：
1. 信息量是否足够完整
2. 是否涵盖问题的主要方面
3. 是否提供足够的细节和示例
4. 是否包含实用的操作指导

评分范围：0.0-1.0（1.0表示完全充分）"""),
            ("human", """查询：{query}

检索到的文档：
{documents}

请评估这些文档是否充分回答了查询，返回0.0-1.0的分数，并说明理由。""")
        ])
        
        documents_text = self._format_documents_for_validation(documents)
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                query=query,
                documents=documents_text
            )
        )
        
        # 解析充分性分数
        sufficiency_score = self._extract_score_from_response(response.content)
        
        # 基于文档数量和长度的补充评估
        quantity_score = self._calculate_quantity_sufficiency(documents)
        
        # 综合评估
        final_sufficiency = (sufficiency_score * 0.8 + quantity_score * 0.2)
        
        return min(1.0, max(0.0, final_sufficiency))
    
    async def _validate_consistency(self, documents: List[Document]) -> float:
        """验证一致性"""
        
        if len(documents) < 2:
            return 1.0  # 单个文档默认一致
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是信息一致性评估专家，需要检查多个文档之间的一致性。

一致性评估标准：
1. 不同文档间信息是否一致
2. 是否存在相互矛盾的陈述
3. 术语和概念使用是否一致
4. 规则和标准应用是否一致

评分范围：0.0-1.0（1.0表示完全一致）"""),
            ("human", """检索到的文档：
{documents}

请评估这些文档之间的一致性，返回0.0-1.0的分数，并指出任何不一致之处。""")
        ])
        
        documents_text = self._format_documents_for_validation(documents)
        
        response = await self.llm.ainvoke(
            prompt.format_messages(documents=documents_text)
        )
        
        # 解析一致性分数
        consistency_score = self._extract_score_from_response(response.content)
        
        # 基于重复信息的补充评估
        redundancy_penalty = self._calculate_redundancy_penalty(documents)
        
        # 综合评估
        final_consistency = consistency_score * (1 - redundancy_penalty * 0.1)
        
        return min(1.0, max(0.0, final_consistency))
    
    async def _validate_accuracy(
        self,
        documents: List[Document],
        knowledge_source: str
    ) -> float:
        """验证准确性"""
        
        if not documents:
            return 0.0
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是信息准确性评估专家，特别擅长{knowledge_source}标准。

准确性评估标准：
1. 事实信息是否准确
2. 引用和来源是否可靠
3. 技术细节是否正确
4. 标准规则是否准确解释

评分范围：0.0-1.0（1.0表示完全准确）"""),
            ("human", """文档内容：
{documents}

知识来源：{knowledge_source}

请评估这些文档的准确性，返回0.0-1.0的分数，并指出任何准确性问题。""")
        ])
        
        documents_text = self._format_documents_for_validation(documents)
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                documents=documents_text,
                knowledge_source=knowledge_source
            )
        )
        
        # 解析准确性分数
        accuracy_score = self._extract_score_from_response(response.content)
        
        # 基于来源可靠性的调整
        source_reliability = self._assess_source_reliability(documents, knowledge_source)
        
        # 综合评估
        final_accuracy = accuracy_score * source_reliability
        
        return min(1.0, max(0.0, final_accuracy))
    
    def _calculate_keyword_relevance(
        self,
        query: str,
        documents: List[Document]
    ) -> float:
        """计算关键词相关性"""
        
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        
        total_relevance = 0.0
        
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            
            # 计算关键词重叠率
            overlap = query_words & doc_words
            relevance = len(overlap) / len(query_words)
            
            total_relevance += relevance
        
        return total_relevance / len(documents) if documents else 0.0
    
    def _calculate_quantity_sufficiency(self, documents: List[Document]) -> float:
        """计算数量充分性"""
        
        if not documents:
            return 0.0
        
        # 基于文档数量
        doc_count_score = min(1.0, len(documents) / 5)  # 5个文档为满分
        
        # 基于内容长度
        total_length = sum(len(doc.page_content) for doc in documents)
        length_score = min(1.0, total_length / 2000)  # 2000字符为满分
        
        return (doc_count_score + length_score) / 2
    
    def _calculate_redundancy_penalty(self, documents: List[Document]) -> float:
        """计算冗余惩罚"""
        
        if len(documents) < 2:
            return 0.0
        
        redundancy_score = 0.0
        
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents):
                if i >= j:
                    continue
                
                # 简单的重复内容检测
                words1 = set(doc1.page_content.lower().split())
                words2 = set(doc2.page_content.lower().split())
                
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > 0.8:  # 80%重叠认为是冗余
                        redundancy_score += overlap
        
        # 归一化
        max_comparisons = len(documents) * (len(documents) - 1) / 2
        return redundancy_score / max_comparisons if max_comparisons > 0 else 0.0
    
    def _assess_source_reliability(
        self,
        documents: List[Document],
        knowledge_source: str
    ) -> float:
        """评估来源可靠性"""
        
        base_reliability = 0.8  # 基础可靠性
        
        # 基于知识源类型调整
        source_reliability_map = {
            "NESMA": 0.95,  # 官方标准
            "COSMIC": 0.95,  # 官方标准
            "academic": 0.85,  # 学术来源
            "industry": 0.75,  # 行业报告
            "blog": 0.6,  # 博客文章
            "unknown": 0.5  # 未知来源
        }
        
        source_type = knowledge_source.upper()
        if source_type in source_reliability_map:
            base_reliability = source_reliability_map[source_type]
        
        # 基于文档元数据调整
        for doc in documents:
            metadata = doc.metadata
            
            # 检查来源信息
            if metadata.get("author"):
                base_reliability += 0.05
            
            if metadata.get("publication_date"):
                base_reliability += 0.05
            
            if metadata.get("official_source"):
                base_reliability += 0.1
        
        return min(1.0, base_reliability)
    
    def _calculate_overall_quality(
        self,
        relevance: float,
        sufficiency: float,
        consistency: float,
        accuracy: float
    ) -> float:
        """计算综合质量分数"""
        
        # 加权平均
        weights = {
            "relevance": 0.3,
            "sufficiency": 0.25,
            "consistency": 0.2,
            "accuracy": 0.25
        }
        
        overall = (
            relevance * weights["relevance"] +
            sufficiency * weights["sufficiency"] +
            consistency * weights["consistency"] +
            accuracy * weights["accuracy"]
        )
        
        return overall
    
    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        
        if score >= self.quality_thresholds["excellent"]:
            return "优秀"
        elif score >= self.quality_thresholds["good"]:
            return "良好"
        elif score >= self.quality_thresholds["acceptable"]:
            return "可接受"
        else:
            return "需要改进"
    
    async def _generate_validation_report(
        self,
        query: str,
        documents: List[Document],
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """生成验证报告"""
        
        issues = []
        suggestions = []
        
        # 基于分数识别问题
        if scores["relevance"] < self.quality_thresholds["acceptable"]:
            issues.append("文档相关性不足")
            suggestions.append("重新优化查询词或扩大检索范围")
        
        if scores["sufficiency"] < self.quality_thresholds["acceptable"]:
            issues.append("信息充分性不足")
            suggestions.append("增加检索文档数量或使用多种检索策略")
        
        if scores["consistency"] < self.quality_thresholds["acceptable"]:
            issues.append("文档间存在不一致")
            suggestions.append("检查文档来源，筛选权威资料")
        
        if scores["accuracy"] < self.quality_thresholds["acceptable"]:
            issues.append("信息准确性存疑")
            suggestions.append("验证信息来源，参考官方文档")
        
        # 生成具体建议
        if scores["overall"] < self.quality_thresholds["good"]:
            suggestions.extend([
                "考虑使用多个知识源进行交叉验证",
                "优化检索策略以获得更好的结果",
                "人工审核关键信息的准确性"
            ])
        
        return {
            "query": query,
            "document_count": len(documents),
            "scores": scores,
            "quality_assessment": {
                "level": self._determine_quality_level(scores["overall"]),
                "strengths": self._identify_strengths(scores),
                "weaknesses": self._identify_weaknesses(scores)
            },
            "issues": issues,
            "suggestions": suggestions,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """识别优势"""
        
        strengths = []
        threshold = self.quality_thresholds["good"]
        
        if scores["relevance"] >= threshold:
            strengths.append("文档相关性良好")
        
        if scores["sufficiency"] >= threshold:
            strengths.append("信息充分性良好")
        
        if scores["consistency"] >= threshold:
            strengths.append("文档一致性良好")
        
        if scores["accuracy"] >= threshold:
            strengths.append("信息准确性良好")
        
        return strengths if strengths else ["整体质量需要提升"]
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """识别弱点"""
        
        weaknesses = []
        threshold = self.quality_thresholds["acceptable"]
        
        if scores["relevance"] < threshold:
            weaknesses.append("文档相关性不足")
        
        if scores["sufficiency"] < threshold:
            weaknesses.append("信息不够充分")
        
        if scores["consistency"] < threshold:
            weaknesses.append("存在信息不一致")
        
        if scores["accuracy"] < threshold:
            weaknesses.append("准确性有待验证")
        
        return weaknesses
    
    def _format_documents_for_validation(self, documents: List[Document]) -> str:
        """格式化文档用于验证"""
        
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            doc_text = f"文档 {i+1}:\n"
            doc_text += f"内容: {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}\n"
            
            if doc.metadata:
                doc_text += f"元数据: {doc.metadata}\n"
            
            formatted_docs.append(doc_text)
        
        return "\n\n".join(formatted_docs)
    
    def _extract_score_from_response(self, response_content: str) -> float:
        """从响应中提取分数"""
        
        import re
        
        # 查找数字分数（0.0-1.0格式）
        score_patterns = [
            r'(\d+\.\d+)',  # 小数格式
            r'(\d+)%',      # 百分比格式
            r'(\d+)/10',    # 分数格式
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response_content)
            if matches:
                score = float(matches[0])
                
                # 归一化到0-1范围
                if score > 1:
                    if score <= 10:
                        score = score / 10
                    elif score <= 100:
                        score = score / 100
                
                return min(1.0, max(0.0, score))
        
        # 如果没有找到数字分数，基于关键词估算
        content_lower = response_content.lower()
        
        if any(word in content_lower for word in ["优秀", "excellent", "很好", "perfect"]):
            return 0.9
        elif any(word in content_lower for word in ["良好", "good", "不错", "satisfactory"]):
            return 0.7
        elif any(word in content_lower for word in ["一般", "average", "可以", "acceptable"]):
            return 0.5
        elif any(word in content_lower for word in ["较差", "poor", "不好", "inadequate"]):
            return 0.3
        else:
            return 0.5  # 默认中等分数
    
    async def validate_analysis_result(
        self,
        analysis_type: str,
        analysis_result: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> ValidationResult:
        """验证分析结果"""
        
        logger.info(f"🔍 开始验证分析结果，类型: {analysis_type}")
        
        try:
            # 1. 结果完整性检查
            completeness_score = self._check_result_completeness(analysis_type, analysis_result)
            
            # 2. 结果合理性检查
            rationality_score = await self._check_result_rationality(
                analysis_type, analysis_result, input_data
            )
            
            # 3. 结果一致性检查
            consistency_score = self._check_result_consistency(analysis_result)
            
            # 4. 计算综合质量
            overall_quality = (completeness_score + rationality_score + consistency_score) / 3
            
            # 5. 生成验证结果
            is_valid = overall_quality >= self.quality_thresholds["acceptable"]
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence_score=overall_quality,
                validation_details={
                    "analysis_type": analysis_type,
                    "completeness_score": completeness_score,
                    "rationality_score": rationality_score,
                    "consistency_score": consistency_score,
                    "overall_quality": overall_quality
                },
                issues=self._identify_analysis_issues(analysis_type, analysis_result, {
                    "completeness": completeness_score,
                    "rationality": rationality_score,
                    "consistency": consistency_score
                }),
                suggestions=self._generate_analysis_suggestions(analysis_type, overall_quality)
            )
            
            logger.info(f"✅ 分析结果验证完成，质量分数: {overall_quality:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 分析结果验证失败: {str(e)}")
            raise
    
    def _check_result_completeness(
        self,
        analysis_type: str,
        analysis_result: Dict[str, Any]
    ) -> float:
        """检查结果完整性"""
        
        required_fields_map = {
            "NESMA_classification": ["function_type", "confidence_score", "justification"],
            "COSMIC_analysis": ["data_movements", "functional_processes", "cfp_total"],
            "process_identification": ["processes", "dependencies", "data_groups"],
            "comparison_analysis": ["nesma_total", "cosmic_total", "difference_analysis"]
        }
        
        required_fields = required_fields_map.get(analysis_type, [])
        if not required_fields:
            return 1.0  # 未知类型默认完整
        
        present_fields = sum(1 for field in required_fields if field in analysis_result)
        completeness = present_fields / len(required_fields)
        
        return completeness
    
    async def _check_result_rationality(
        self,
        analysis_type: str,
        analysis_result: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> float:
        """检查结果合理性"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是{analysis_type}分析专家，需要评估分析结果的合理性。

合理性评估标准：
1. 结果与输入数据是否匹配
2. 分析逻辑是否合理
3. 数值是否在合理范围内
4. 结论是否符合专业标准

评分范围：0.0-1.0（1.0表示完全合理）"""),
            ("human", """分析类型：{analysis_type}

输入数据：
{input_data}

分析结果：
{analysis_result}

请评估这个分析结果的合理性，返回0.0-1.0的分数。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                analysis_type=analysis_type,
                input_data=str(input_data),
                analysis_result=str(analysis_result)
            )
        )
        
        return self._extract_score_from_response(response.content)
    
    def _check_result_consistency(self, analysis_result: Dict[str, Any]) -> float:
        """检查结果一致性"""
        
        consistency_score = 1.0
        
        # 检查数值一致性
        if "total" in analysis_result and "details" in analysis_result:
            details = analysis_result["details"]
            if isinstance(details, list):
                calculated_total = sum(item.get("value", 0) for item in details if isinstance(item, dict))
                reported_total = analysis_result["total"]
                
                if reported_total > 0:
                    difference_ratio = abs(calculated_total - reported_total) / reported_total
                    if difference_ratio > 0.1:  # 10%以上差异认为不一致
                        consistency_score -= 0.3
        
        # 检查置信度一致性
        confidence_scores = []
        if isinstance(analysis_result, dict):
            for key, value in analysis_result.items():
                if "confidence" in key and isinstance(value, (int, float)):
                    confidence_scores.append(value)
        
        if len(confidence_scores) > 1:
            score_variance = max(confidence_scores) - min(confidence_scores)
            if score_variance > 0.5:  # 置信度差异过大
                consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _identify_analysis_issues(
        self,
        analysis_type: str,
        analysis_result: Dict[str, Any],
        scores: Dict[str, float]
    ) -> List[str]:
        """识别分析问题"""
        
        issues = []
        threshold = self.quality_thresholds["acceptable"]
        
        if scores["completeness"] < threshold:
            issues.append(f"{analysis_type}分析结果不完整")
        
        if scores["rationality"] < threshold:
            issues.append(f"{analysis_type}分析结果不合理")
        
        if scores["consistency"] < threshold:
            issues.append(f"{analysis_type}分析结果存在内部不一致")
        
        return issues
    
    def _generate_analysis_suggestions(
        self,
        analysis_type: str,
        overall_quality: float
    ) -> List[str]:
        """生成分析建议"""
        
        suggestions = []
        
        if overall_quality < self.quality_thresholds["good"]:
            suggestions.extend([
                f"重新检查{analysis_type}的分析逻辑",
                "验证输入数据的准确性",
                "参考相关标准和最佳实践",
                "考虑使用多种方法进行交叉验证"
            ])
        
        return suggestions


if __name__ == "__main__":
    # 测试质量验证智能体
    async def test_validator():
        agent = ValidatorAgent()
        
        # 测试知识验证
        test_documents = [
            Document(
                page_content="ILF是内部逻辑文件，由应用程序内部维护的数据组成。",
                metadata={"source": "NESMA官方文档"}
            ),
            Document(
                page_content="内部逻辑文件包含用户可识别的数据，并通过应用程序的功能过程维护。",
                metadata={"source": "NESMA指南"}
            )
        ]
        
        result = await agent.validate_retrieved_knowledge(
            query="什么是NESMA中的ILF",
            retrieved_documents=test_documents,
            knowledge_source="NESMA"
        )
        
        print(f"知识验证结果：")
        print(f"- 是否有效：{result.is_valid}")
        print(f"- 置信度：{result.confidence_score:.3f}")
        print(f"- 质量等级：{result.validation_details.get('quality_level')}")
        if result.issues:
            print(f"- 问题：{result.issues}")
        if result.suggestions:
            print(f"- 建议：{result.suggestions}")
    
    asyncio.run(test_validator()) 