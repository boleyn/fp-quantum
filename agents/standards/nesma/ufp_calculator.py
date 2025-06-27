"""
量子智能化功能点估算系统 - NESMA未调整功能点计算器智能体

基于NESMA v2.3+标准计算未调整功能点(UFP)
"""

import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import SpecializedAgent
from agents.knowledge.rule_retriever import RuleRetrieverAgent
from models.nesma_models import (
    NESMAFunctionType, NESMAFunctionClassification,
    NESMAComplexityLevel, NESMAComplexityCalculation
)
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class NESMAUFPCalculatorAgent(SpecializedAgent):
    """NESMA未调整功能点计算器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="nesma_ufp_calculator",
            specialty="nesma_ufp_calculation",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # NESMA功能点权重表
        self.function_point_weights = self._load_function_point_weights()
        self.calculation_history: List[Dict[str, Any]] = []
        
    def _load_function_point_weights(self) -> Dict[str, Dict[str, int]]:
        """加载NESMA功能点权重表"""
        return {
            "ILF": {
                "Low": 7,
                "Average": 10,
                "High": 15
            },
            "EIF": {
                "Low": 5,
                "Average": 7,
                "High": 10
            },
            "EI": {
                "Low": 3,
                "Average": 4,
                "High": 6
            },
            "EO": {
                "Low": 4,
                "Average": 5,
                "High": 7
            },
            "EQ": {
                "Low": 3,
                "Average": 4,
                "High": 6
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "UFP计算",
            "功能点权重应用",
            "分类汇总统计",
            "质量验证",
            "报告生成"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行UFP计算任务"""
        if task_name == "calculate_ufp":
            if "complexity_results" not in inputs:
                raise KeyError(f"缺少必需的输入参数 'complexity_results'。可用的键: {list(inputs.keys())}")
            return await self.calculate_ufp(inputs["complexity_results"])
        elif task_name == "calculate_function_points":
            return await self.calculate_function_points(
                inputs["classification"],
                inputs["complexity_result"]
            )
        elif task_name == "generate_ufp_report":
            return await self.generate_ufp_report(inputs["ufp_result"])
        elif task_name == "validate_ufp_calculation":
            return await self.validate_ufp_calculation(
                inputs["ufp_result"],
                inputs["complexity_results"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def calculate_ufp(
        self, 
        complexity_results: List[NESMAComplexityCalculation]
    ) -> Dict[str, Any]:
        """计算项目的总UFP"""
        
        if not complexity_results:
            raise ValueError("复杂度结果列表不能为空")
        
        # 1. 计算各功能的功能点
        function_point_details = []
        total_ufp = 0
        
        for complexity_result in complexity_results:
            fp_detail = await self._calculate_single_function_points(complexity_result)
            function_point_details.append(fp_detail)
            total_ufp += fp_detail["function_points"]
        
        # 2. 按功能类型分组统计
        type_summary = self._calculate_type_summary(function_point_details)
        
        # 3. 按复杂度分组统计
        complexity_summary = self._calculate_complexity_summary(function_point_details)
        
        # 4. 生成质量指标
        quality_metrics = await self._calculate_quality_metrics(
            function_point_details, 
            complexity_results
        )
        
        # 5. 构建UFP计算结果
        ufp_result = {
            "total_ufp": total_ufp,
            "function_count": len(complexity_results),
            "function_point_details": function_point_details,
            "type_summary": type_summary,
            "complexity_summary": complexity_summary,
            "quality_metrics": quality_metrics,
            "calculation_metadata": {
                "calculation_time": datetime.now(),
                "nesma_version": "v2.3+",
                "weight_table_version": "standard"
            }
        }
        
        # 6. 记录计算历史
        self.calculation_history.append(ufp_result)
        
        return ufp_result
    
    async def calculate_function_points(
        self,
        classification: NESMAFunctionClassification,
        complexity_result: NESMAComplexityCalculation
    ) -> Dict[str, Any]:
        """计算单个功能的功能点"""
        
        return await self._calculate_single_function_points(complexity_result)
    
    async def generate_ufp_report(self, ufp_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成UFP详细报告"""
        
        # 1. 创建执行摘要
        executive_summary = self._create_executive_summary(ufp_result)
        
        # 2. 生成详细分析
        detailed_analysis = await self._create_detailed_analysis(ufp_result)
        
        # 3. 创建可视化数据
        visualization_data = self._create_visualization_data(ufp_result)
        
        # 4. 生成建议和洞察
        insights_and_recommendations = await self._generate_insights(ufp_result)
        
        report = {
            "report_type": "NESMA_UFP_Report",
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "visualization_data": visualization_data,
            "insights_and_recommendations": insights_and_recommendations,
            "appendix": {
                "weight_table": self.function_point_weights,
                "calculation_details": ufp_result["function_point_details"],
                "quality_metrics": ufp_result["quality_metrics"]
            },
            "generation_time": datetime.now()
        }
        
        return report
    
    async def validate_ufp_calculation(
        self,
        ufp_result: Dict[str, Any],
        complexity_results: List[NESMAComplexityCalculation]
    ) -> Dict[str, Any]:
        """验证UFP计算结果"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "validation_issues": [],
            "suggestions": []
        }
        
        # 验证总数一致性
        expected_count = len(complexity_results)
        actual_count = ufp_result.get("function_count", 0)
        
        if expected_count != actual_count:
            validation_result["validation_issues"].append({
                "type": "count_mismatch",
                "message": f"功能数量不匹配：期望 {expected_count}，实际 {actual_count}"
            })
        
        # 验证功能点计算
        recalculated_total = 0
        for detail in ufp_result.get("function_point_details", []):
            function_type = detail["function_type"]
            complexity = detail["complexity"]
            expected_fp = self.function_point_weights[function_type][complexity]
            actual_fp = detail["function_points"]
            
            if expected_fp != actual_fp:
                validation_result["validation_issues"].append({
                    "type": "weight_calculation_error",
                    "message": f"功能 {detail['function_id']} 权重计算错误：期望 {expected_fp}，实际 {actual_fp}"
                })
            
            recalculated_total += expected_fp
        
        # 验证总UFP
        reported_total = ufp_result.get("total_ufp", 0)
        if recalculated_total != reported_total:
            validation_result["validation_issues"].append({
                "type": "total_ufp_mismatch",
                "message": f"总UFP不匹配：重算 {recalculated_total}，报告 {reported_total}"
            })
        
        # 验证汇总统计
        type_summary_issues = self._validate_type_summary(ufp_result)
        validation_result["validation_issues"].extend(type_summary_issues)
        
        # 计算验证分数
        if validation_result["validation_issues"]:
            validation_result["is_valid"] = False
            validation_result["confidence_score"] = max(0.1,
                1.0 - len(validation_result["validation_issues"]) * 0.15
            )
        
        # 生成改进建议
        if not validation_result["is_valid"]:
            validation_result["suggestions"] = self._generate_ufp_suggestions(
                validation_result["validation_issues"]
            )
        
        return validation_result
    
    async def _calculate_single_function_points(
        self, 
        complexity_result: NESMAComplexityCalculation
    ) -> Dict[str, Any]:
        """计算单个功能的功能点"""
        
        function_type = complexity_result.function_type
        complexity = complexity_result.complexity.value
        
        # 获取权重
        weight = self.function_point_weights[function_type][complexity]
        
        return {
            "function_id": complexity_result.function_id,
            "function_type": function_type,
            "complexity": complexity,
            "weight": weight,
            "function_points": weight,  # UFP中功能点 = 权重
            "det_count": complexity_result.det_count,
            "ret_count": complexity_result.ret_count,
            "calculation_details": complexity_result.calculation_details
        }
    
    def _calculate_type_summary(
        self, 
        function_point_details: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """按功能类型计算汇总"""
        
        type_summary = {}
        
        for detail in function_point_details:
            function_type = detail["function_type"]
            
            if function_type not in type_summary:
                type_summary[function_type] = {
                    "count": 0,
                    "total_fp": 0,
                    "complexity_breakdown": {"Low": 0, "Average": 0, "High": 0}
                }
            
            type_summary[function_type]["count"] += 1
            type_summary[function_type]["total_fp"] += detail["function_points"]
            type_summary[function_type]["complexity_breakdown"][detail["complexity"]] += 1
        
        # 计算每种类型的平均功能点
        for type_data in type_summary.values():
            if type_data["count"] > 0:
                type_data["average_fp"] = type_data["total_fp"] / type_data["count"]
        
        return type_summary
    
    def _calculate_complexity_summary(
        self, 
        function_point_details: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """按复杂度等级计算汇总"""
        
        complexity_summary = {
            "Low": {"count": 0, "total_fp": 0},
            "Average": {"count": 0, "total_fp": 0},
            "High": {"count": 0, "total_fp": 0}
        }
        
        for detail in function_point_details:
            complexity = detail["complexity"]
            complexity_summary[complexity]["count"] += 1
            complexity_summary[complexity]["total_fp"] += detail["function_points"]
        
        # 计算平均功能点和百分比
        total_functions = len(function_point_details)
        total_fp = sum(detail["function_points"] for detail in function_point_details)
        
        for complexity, data in complexity_summary.items():
            if data["count"] > 0:
                data["average_fp"] = data["total_fp"] / data["count"]
                data["percentage"] = (data["count"] / total_functions) * 100
                data["fp_percentage"] = (data["total_fp"] / total_fp) * 100 if total_fp > 0 else 0
        
        return complexity_summary
    
    async def _calculate_quality_metrics(
        self,
        function_point_details: List[Dict[str, Any]],
        complexity_results: List[NESMAComplexityCalculation]
    ) -> Dict[str, Any]:
        """计算质量指标"""
        
        # 基础统计
        total_fp = sum(detail["function_points"] for detail in function_point_details)
        function_count = len(function_point_details)
        
        # 计算平均置信度
        confidence_scores = []
        for complexity_result in complexity_results:
            confidence = complexity_result.calculation_details.get("confidence_score", 0.7)
            confidence_scores.append(confidence)
        
        average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.7
        
        # 计算复杂度分布均衡性
        complexity_counts = {"Low": 0, "Average": 0, "High": 0}
        for detail in function_point_details:
            complexity_counts[detail["complexity"]] += 1
        
        # 计算方差（衡量分布均衡性）
        complexity_values = list(complexity_counts.values())
        mean_count = sum(complexity_values) / len(complexity_values)
        variance = sum((count - mean_count) ** 2 for count in complexity_values) / len(complexity_values)
        distribution_balance = 1.0 / (1.0 + variance)  # 归一化到0-1
        
        # 计算功能类型覆盖度
        type_coverage = len(set(detail["function_type"] for detail in function_point_details)) / 5.0
        
        return {
            "average_fp_per_function": total_fp / function_count if function_count > 0 else 0,
            "average_confidence_score": average_confidence,
            "complexity_distribution_balance": distribution_balance,
            "function_type_coverage": type_coverage,
            "total_functions": function_count,
            "total_ufp": total_fp,
            "quality_score": (average_confidence + distribution_balance + type_coverage) / 3.0
        }
    
    def _create_executive_summary(self, ufp_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行摘要"""
        
        total_ufp = ufp_result["total_ufp"]
        function_count = ufp_result["function_count"]
        quality_score = ufp_result["quality_metrics"]["quality_score"]
        
        # 找出主要功能类型
        type_summary = ufp_result["type_summary"]
        dominant_type = max(type_summary.keys(), key=lambda k: type_summary[k]["total_fp"])
        
        # 找出主要复杂度
        complexity_summary = ufp_result["complexity_summary"]
        dominant_complexity = max(complexity_summary.keys(), key=lambda k: complexity_summary[k]["count"])
        
        return {
            "total_ufp": total_ufp,
            "function_count": function_count,
            "average_fp_per_function": total_ufp / function_count if function_count > 0 else 0,
            "quality_score": quality_score,
            "dominant_function_type": dominant_type,
            "dominant_complexity": dominant_complexity,
            "summary_text": f"""
项目总计 {total_ufp} 个未调整功能点（UFP），包含 {function_count} 个功能。
主要功能类型为 {dominant_type}，主要复杂度等级为 {dominant_complexity}。
整体质量评分：{quality_score:.2f}。
            """.strip()
        }
    
    async def _create_detailed_analysis(self, ufp_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建详细分析"""
        
        return {
            "type_analysis": {
                "summary": ufp_result["type_summary"],
                "insights": self._analyze_type_distribution(ufp_result["type_summary"])
            },
            "complexity_analysis": {
                "summary": ufp_result["complexity_summary"],
                "insights": self._analyze_complexity_distribution(ufp_result["complexity_summary"])
            },
            "quality_analysis": {
                "metrics": ufp_result["quality_metrics"],
                "assessment": self._assess_calculation_quality(ufp_result["quality_metrics"])
            }
        }
    
    def _create_visualization_data(self, ufp_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建可视化数据"""
        
        # 功能类型分布饼图数据
        type_pie_data = []
        for func_type, data in ufp_result["type_summary"].items():
            type_pie_data.append({
                "label": func_type,
                "value": data["total_fp"],
                "count": data["count"]
            })
        
        # 复杂度分布柱状图数据
        complexity_bar_data = []
        for complexity, data in ufp_result["complexity_summary"].items():
            complexity_bar_data.append({
                "label": complexity,
                "count": data["count"],
                "total_fp": data["total_fp"]
            })
        
        # 功能点详细表格数据
        function_table_data = ufp_result["function_point_details"]
        
        return {
            "type_distribution_pie": type_pie_data,
            "complexity_distribution_bar": complexity_bar_data,
            "function_details_table": function_table_data,
            "chart_config": {
                "type_pie": {
                    "title": "功能类型分布（按功能点）",
                    "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
                },
                "complexity_bar": {
                    "title": "复杂度分布",
                    "x_axis": "复杂度等级",
                    "y_axis": "功能数量"
                }
            }
        }
    
    async def _generate_insights(self, ufp_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成洞察和建议"""
        
        insights = []
        recommendations = []
        
        # 分析功能类型分布
        type_insights = self._analyze_type_distribution(ufp_result["type_summary"])
        insights.extend(type_insights)
        
        # 分析复杂度分布
        complexity_insights = self._analyze_complexity_distribution(ufp_result["complexity_summary"])
        insights.extend(complexity_insights)
        
        # 质量评估
        quality_metrics = ufp_result["quality_metrics"]
        if quality_metrics["quality_score"] < 0.7:
            recommendations.append("建议重新检查复杂度计算，提高估算质量")
        
        if quality_metrics["complexity_distribution_balance"] < 0.5:
            recommendations.append("复杂度分布不均衡，建议重新审查功能分类")
        
        if quality_metrics["function_type_coverage"] < 0.6:
            recommendations.append("功能类型覆盖度不足，可能遗漏某些功能类型")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "risk_assessment": self._assess_estimation_risks(ufp_result),
            "improvement_suggestions": self._suggest_improvements(ufp_result)
        }
    
    def _analyze_type_distribution(self, type_summary: Dict[str, Dict[str, Any]]) -> List[str]:
        """分析功能类型分布"""
        insights = []
        
        # 计算各类型占比
        total_fp = sum(data["total_fp"] for data in type_summary.values())
        
        for func_type, data in type_summary.items():
            percentage = (data["total_fp"] / total_fp) * 100 if total_fp > 0 else 0
            if percentage > 40:
                insights.append(f"{func_type}功能占主导地位（{percentage:.1f}%），可能影响整体估算精度")
            elif percentage < 5:
                insights.append(f"{func_type}功能较少（{percentage:.1f}%），可能存在识别遗漏")
        
        return insights
    
    def _analyze_complexity_distribution(self, complexity_summary: Dict[str, Dict[str, Any]]) -> List[str]:
        """分析复杂度分布"""
        insights = []
        
        total_count = sum(data["count"] for data in complexity_summary.values())
        
        for complexity, data in complexity_summary.items():
            percentage = (data["count"] / total_count) * 100 if total_count > 0 else 0
            if complexity == "High" and percentage > 30:
                insights.append(f"高复杂度功能较多（{percentage:.1f}%），项目可能存在高风险")
            elif complexity == "Low" and percentage > 60:
                insights.append(f"低复杂度功能占主体（{percentage:.1f}%），项目相对简单")
        
        return insights
    
    def _assess_calculation_quality(self, quality_metrics: Dict[str, Any]) -> str:
        """评估计算质量"""
        quality_score = quality_metrics["quality_score"]
        
        if quality_score >= 0.9:
            return "计算质量优秀，估算结果可信度高"
        elif quality_score >= 0.7:
            return "计算质量良好，估算结果基本可信"
        elif quality_score >= 0.5:
            return "计算质量一般，建议进一步验证"
        else:
            return "计算质量较低，需要重新评估"
    
    def _assess_estimation_risks(self, ufp_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估估算风险"""
        
        risks = []
        risk_level = "Low"
        
        quality_score = ufp_result["quality_metrics"]["quality_score"]
        if quality_score < 0.6:
            risks.append("计算质量偏低，可能影响估算准确性")
            risk_level = "High"
        
        # 检查复杂度分布
        complexity_summary = ufp_result["complexity_summary"]
        high_complexity_ratio = complexity_summary["High"]["percentage"]
        if high_complexity_ratio > 40:
            risks.append("高复杂度功能占比过高，实施风险较大")
            risk_level = "High" if risk_level != "High" else "High"
        
        # 检查功能类型分布
        type_summary = ufp_result["type_summary"]
        if len(type_summary) < 3:
            risks.append("功能类型覆盖不足，可能存在功能遗漏")
            risk_level = "Medium" if risk_level == "Low" else risk_level
        
        return {
            "risk_level": risk_level,
            "identified_risks": risks,
            "mitigation_suggestions": self._suggest_risk_mitigation(risks)
        }
    
    def _suggest_risk_mitigation(self, risks: List[str]) -> List[str]:
        """建议风险缓解措施"""
        suggestions = []
        
        for risk in risks:
            if "质量" in risk:
                suggestions.append("建议重新审查复杂度计算，增加专家评审环节")
            elif "复杂度" in risk:
                suggestions.append("建议细化高复杂度功能分解，降低实施风险")
            elif "类型" in risk:
                suggestions.append("建议重新梳理需求，确保功能类型识别完整")
        
        return suggestions
    
    def _suggest_improvements(self, ufp_result: Dict[str, Any]) -> List[str]:
        """建议改进措施"""
        suggestions = []
        
        quality_metrics = ufp_result["quality_metrics"]
        
        if quality_metrics["average_confidence_score"] < 0.8:
            suggestions.append("提高功能分析详细程度，增加计算置信度")
        
        if quality_metrics["complexity_distribution_balance"] < 0.6:
            suggestions.append("重新审查复杂度评估标准，确保分类一致性")
        
        if quality_metrics["function_type_coverage"] < 0.8:
            suggestions.append("完善需求分析，确保所有功能类型被正确识别")
        
        return suggestions
    
    def _validate_type_summary(self, ufp_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """验证类型汇总统计"""
        issues = []
        
        # 检查汇总数据一致性
        type_summary = ufp_result["type_summary"]
        function_details = ufp_result["function_point_details"]
        
        # 重新计算汇总数据进行验证
        recalc_summary = {}
        for detail in function_details:
            func_type = detail["function_type"]
            if func_type not in recalc_summary:
                recalc_summary[func_type] = {"count": 0, "total_fp": 0}
            recalc_summary[func_type]["count"] += 1
            recalc_summary[func_type]["total_fp"] += detail["function_points"]
        
        # 对比验证
        for func_type, data in type_summary.items():
            if func_type in recalc_summary:
                recalc_data = recalc_summary[func_type]
                if data["count"] != recalc_data["count"]:
                    issues.append({
                        "type": "type_count_mismatch",
                        "message": f"{func_type}类型计数不匹配"
                    })
                if data["total_fp"] != recalc_data["total_fp"]:
                    issues.append({
                        "type": "type_fp_mismatch", 
                        "message": f"{func_type}类型功能点总数不匹配"
                    })
        
        return issues
    
    def _generate_ufp_suggestions(self, validation_issues: List[Dict[str, str]]) -> List[str]:
        """生成UFP计算改进建议"""
        suggestions = []
        
        issue_types = [issue["type"] for issue in validation_issues]
        
        if "weight_calculation_error" in issue_types:
            suggestions.append("检查功能点权重表应用，确保计算正确")
        
        if "total_ufp_mismatch" in issue_types:
            suggestions.append("重新计算总UFP，检查汇总逻辑")
        
        if any("type_" in issue_type for issue_type in issue_types):
            suggestions.append("验证功能类型汇总统计，确保数据一致性")
        
        if "count_mismatch" in issue_types:
            suggestions.append("检查输入数据完整性，确保所有功能被包含")
        
        return suggestions or ["UFP计算验证通过，无需调整"]
    
    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """获取计算历史"""
        return self.calculation_history.copy()
    
    def get_weight_table(self) -> Dict[str, Dict[str, int]]:
        """获取功能点权重表"""
        return self.function_point_weights.copy()


async def create_nesma_ufp_calculator(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> NESMAUFPCalculatorAgent:
    """创建NESMA UFP计算器智能体"""
    return NESMAUFPCalculatorAgent(rule_retriever=rule_retriever, llm=llm)


if __name__ == "__main__":
    async def main():
        # 测试NESMA UFP计算器
        calculator = await create_nesma_ufp_calculator()
        
        # 测试复杂度结果
        test_complexity_results = [
            NESMAComplexityCalculation(
                function_id="func_001",
                function_type=NESMAFunctionType.ILF,
                det_count=8,
                ret_count=1,
                complexity=NESMAComplexityLevel.AVERAGE,
                complexity_matrix_used="ILF_test_matrix",
                calculation_steps=["test step 1", "test step 2"],
                calculation_details={"confidence_score": 0.9}
            ),
            NESMAComplexityCalculation(
                function_id="func_002",
                function_type=NESMAFunctionType.EI,
                det_count=5,
                ret_count=1,
                complexity=NESMAComplexityLevel.LOW,
                complexity_matrix_used="EI_test_matrix",
                calculation_steps=["test step 1", "test step 2"],
                calculation_details={"confidence_score": 0.8}
            )
        ]
        
        # 计算UFP
        ufp_result = await calculator.calculate_ufp(test_complexity_results)
        print(f"UFP计算结果: {ufp_result['total_ufp']} UFP")
        
        # 生成报告
        report = await calculator.generate_ufp_report(ufp_result)
        print(f"报告生成完成: {report['executive_summary']}")
        
    asyncio.run(main()) 