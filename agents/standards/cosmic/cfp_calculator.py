"""
量子智能化功能点估算系统 - COSMIC CFP计算器智能体

基于COSMIC v4.0+标准计算COSMIC功能点(CFP)
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
from models.cosmic_models import (
    COSMICDataMovement, COSMICFunctionalUser, COSMICBoundaryAnalysis
)
from models.project_models import ProjectInfo
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class COSMICCFPCalculatorAgent(SpecializedAgent):
    """COSMIC CFP计算器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="cosmic_cfp_calculator", 
            specialty="cosmic_cfp_calculation",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # COSMIC CFP计算规则
        self.cfp_rules = self._load_cfp_rules()
        self.calculation_history: List[Dict[str, Any]] = []
        
    def _load_cfp_rules(self) -> Dict[str, Any]:
        """加载COSMIC CFP计算规则"""
        return {
            "CFP计算原则": {
                "基本原则": "1个数据移动 = 1 CFP",
                "计算方法": "CFP = Entry数量 + Exit数量 + Read数量 + Write数量",
                "质量要求": [
                    "所有数据移动必须准确分类",
                    "避免重复计算",
                    "确保边界一致性",
                    "验证功能完整性"
                ]
            },
            "CFP统计": {
                "按类型统计": ["Entry CFP", "Exit CFP", "Read CFP", "Write CFP"],
                "按过程统计": "每个功能过程的CFP分布",
                "总计": "项目总CFP",
                "质量指标": ["平均CFP/过程", "分布均衡性", "计算置信度"]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "CFP计算",
            "数据移动统计",
            "质量验证",
            "CFP报告生成",
            "计算优化"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行CFP计算任务"""
        if task_name == "calculate_cfp":
            return await self.calculate_cfp(
                inputs["data_movements"],
                inputs["project_info"],
                inputs["boundary_analysis"]
            )
        elif task_name == "validate_cfp_calculation":
            return await self.validate_cfp_calculation(
                inputs["cfp_result"],
                inputs["data_movements"]
            )
        elif task_name == "generate_cfp_report":
            return await self.generate_cfp_report(inputs["cfp_result"])
        elif task_name == "optimize_cfp_calculation":
            return await self.optimize_cfp_calculation(
                inputs["cfp_result"],
                inputs["feedback"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def calculate_cfp(
        self,
        data_movements: List[COSMICDataMovement],
        project_info: ProjectInfo,
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> Dict[str, Any]:
        """计算项目总CFP"""
        
        # 1. 按类型统计数据移动
        type_statistics = self._calculate_type_statistics(data_movements)
        
        # 2. 按过程统计CFP
        process_statistics = self._calculate_process_statistics(data_movements)
        
        # 3. 计算总CFP
        total_cfp = len(data_movements)
        
        # 4. 生成质量指标
        quality_metrics = await self._calculate_quality_metrics(
            data_movements, 
            project_info,
            boundary_analysis
        )
        
        # 5. 构建CFP结果
        cfp_result = {
            "total_cfp": total_cfp,
            "data_movements_count": len(data_movements),
            "type_statistics": type_statistics,
            "process_statistics": process_statistics,
            "quality_metrics": quality_metrics,
            "calculation_metadata": {
                "calculation_time": datetime.now(),
                "cosmic_version": "v4.0+",
                "project_name": project_info.name,
                "calculation_principle": "1 数据移动 = 1 CFP"
            }
        }
        
        # 6. 记录计算历史
        self.calculation_history.append(cfp_result)
        
        return cfp_result
    
    async def validate_cfp_calculation(
        self,
        cfp_result: Dict[str, Any],
        data_movements: List[COSMICDataMovement]
    ) -> Dict[str, Any]:
        """验证CFP计算结果"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "validation_issues": [],
            "suggestions": []
        }
        
        # 验证总数一致性
        expected_cfp = len(data_movements)
        actual_cfp = cfp_result.get("total_cfp", 0)
        
        if expected_cfp != actual_cfp:
            validation_result["validation_issues"].append({
                "type": "cfp_count_mismatch",
                "message": f"CFP总数不匹配：期望 {expected_cfp}，实际 {actual_cfp}"
            })
        
        # 验证类型统计
        type_stats = cfp_result.get("type_statistics", {})
        type_sum = sum(type_stats.values())
        
        if type_sum != expected_cfp:
            validation_result["validation_issues"].append({
                "type": "type_statistics_mismatch",
                "message": f"类型统计总和不匹配：{type_sum} != {expected_cfp}"
            })
        
        # 验证数据移动完整性
        movement_issues = self._validate_data_movements_completeness(data_movements)
        validation_result["validation_issues"].extend(movement_issues)
        
        # 计算验证分数
        if validation_result["validation_issues"]:
            validation_result["is_valid"] = False
            validation_result["confidence_score"] = max(0.1,
                1.0 - len(validation_result["validation_issues"]) * 0.15
            )
        
        # 生成建议
        if not validation_result["is_valid"]:
            validation_result["suggestions"] = self._generate_cfp_suggestions(
                validation_result["validation_issues"]
            )
        
        return validation_result
    
    async def generate_cfp_report(self, cfp_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成CFP详细报告"""
        
        # 1. 执行摘要
        executive_summary = self._create_executive_summary(cfp_result)
        
        # 2. 详细分析
        detailed_analysis = await self._create_detailed_analysis(cfp_result)
        
        # 3. 可视化数据
        visualization_data = self._create_visualization_data(cfp_result)
        
        # 4. 洞察和建议
        insights = await self._generate_insights(cfp_result)
        
        report = {
            "report_type": "COSMIC_CFP_Report",
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "visualization_data": visualization_data,
            "insights_and_recommendations": insights,
            "appendix": {
                "cfp_calculation_details": cfp_result,
                "cosmic_principles": self.cfp_rules
            },
            "generation_time": datetime.now()
        }
        
        return report
    
    async def optimize_cfp_calculation(
        self,
        cfp_result: Dict[str, Any],
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于反馈优化CFP计算"""
        
        optimized_result = cfp_result.copy()
        
        # 应用反馈调整
        if "data_movement_corrections" in feedback:
            corrections = feedback["data_movement_corrections"]
            optimized_result = await self._apply_movement_corrections(
                optimized_result,
                corrections
            )
        
        if "calculation_adjustments" in feedback:
            adjustments = feedback["calculation_adjustments"]
            optimized_result = self._apply_calculation_adjustments(
                optimized_result,
                adjustments
            )
        
        # 重新计算质量指标
        optimized_result["quality_metrics"]["optimization_applied"] = True
        optimized_result["quality_metrics"]["optimization_time"] = datetime.now()
        
        return optimized_result
    
    def _calculate_type_statistics(
        self, 
        data_movements: List[COSMICDataMovement]
    ) -> Dict[str, int]:
        """按类型计算统计"""
        
        type_counts = {
            "Entry": 0,
            "Exit": 0, 
            "Read": 0,
            "Write": 0
        }
        
        for movement in data_movements:
            movement_type = movement.type.value
            type_counts[movement_type] += 1
        
        return type_counts
    
    def _calculate_process_statistics(
        self, 
        data_movements: List[COSMICDataMovement]
    ) -> Dict[str, Dict[str, Any]]:
        """按过程计算统计"""
        
        process_stats = {}
        
        for movement in data_movements:
            # 从movement ID中提取process ID
            process_id = movement.id.split("_")[0] if "_" in movement.id else "unknown"
            
            if process_id not in process_stats:
                process_stats[process_id] = {
                    "total_cfp": 0,
                    "movement_types": {"Entry": 0, "Exit": 0, "Read": 0, "Write": 0},
                    "data_groups": set()
                }
            
            process_stats[process_id]["total_cfp"] += 1
            process_stats[process_id]["movement_types"][movement.type.value] += 1
            process_stats[process_id]["data_groups"].add(movement.data_group)
        
        # 转换set为list便于序列化
        for stats in process_stats.values():
            stats["data_groups"] = list(stats["data_groups"])
            stats["data_groups_count"] = len(stats["data_groups"])
        
        return process_stats
    
    async def _calculate_quality_metrics(
        self,
        data_movements: List[COSMICDataMovement],
        project_info: ProjectInfo,
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> Dict[str, Any]:
        """计算质量指标"""
        
        total_cfp = len(data_movements)
        
        # 计算类型分布均衡性
        type_counts = self._calculate_type_statistics(data_movements)
        type_values = list(type_counts.values())
        mean_count = sum(type_values) / len(type_values) if type_values else 0
        variance = sum((count - mean_count) ** 2 for count in type_values) / len(type_values) if type_values else 0
        distribution_balance = 1.0 / (1.0 + variance) if variance > 0 else 1.0
        
        # 计算过程CFP分布
        process_stats = self._calculate_process_statistics(data_movements)
        process_cfps = [stats["total_cfp"] for stats in process_stats.values()]
        avg_cfp_per_process = sum(process_cfps) / len(process_cfps) if process_cfps else 0
        
        # 计算数据组多样性
        all_data_groups = set(movement.data_group for movement in data_movements)
        data_group_diversity = len(all_data_groups) / total_cfp if total_cfp > 0 else 0
        
        # 计算边界一致性分数
        boundary_consistency = await self._assess_boundary_consistency(
            data_movements,
            boundary_analysis
        )
        
        return {
            "total_cfp": total_cfp,
            "average_cfp_per_process": avg_cfp_per_process,
            "type_distribution_balance": distribution_balance,
            "data_group_diversity": data_group_diversity,
            "boundary_consistency_score": boundary_consistency,
            "functional_users_count": len(boundary_analysis.functional_users),
            "processes_count": len(process_stats),
            "quality_score": (distribution_balance + data_group_diversity + boundary_consistency) / 3.0
        }
    
    def _validate_data_movements_completeness(
        self, 
        data_movements: List[COSMICDataMovement]
    ) -> List[Dict[str, str]]:
        """验证数据移动完整性"""
        issues = []
        
        # 检查是否有数据移动
        if not data_movements:
            issues.append({
                "type": "no_data_movements",
                "message": "没有识别到任何数据移动"
            })
            return issues
        
        # 检查数据移动类型覆盖
        type_counts = self._calculate_type_statistics(data_movements)
        
        if type_counts["Entry"] == 0:
            issues.append({
                "type": "missing_entry_movements",
                "message": "缺少Entry类型数据移动"
            })
        
        if type_counts["Read"] == 0:
            issues.append({
                "type": "missing_read_movements", 
                "message": "缺少Read类型数据移动"
            })
        
        # 检查数据移动ID唯一性
        movement_ids = [movement.id for movement in data_movements]
        if len(movement_ids) != len(set(movement_ids)):
            issues.append({
                "type": "duplicate_movement_ids",
                "message": "存在重复的数据移动ID"
            })
        
        return issues
    
    def _generate_cfp_suggestions(
        self, 
        validation_issues: List[Dict[str, str]]
    ) -> List[str]:
        """生成CFP计算改进建议"""
        suggestions = []
        
        issue_types = [issue["type"] for issue in validation_issues]
        
        if "cfp_count_mismatch" in issue_types:
            suggestions.append("检查CFP计算逻辑，确保总数正确")
        
        if "type_statistics_mismatch" in issue_types:
            suggestions.append("验证类型统计计算，确保数据一致性")
        
        if "no_data_movements" in issue_types:
            suggestions.append("重新进行数据移动识别，确保功能完整性")
        
        if "missing_entry_movements" in issue_types:
            suggestions.append("补充Entry类型数据移动识别")
        
        if "missing_read_movements" in issue_types:
            suggestions.append("补充Read类型数据移动识别")
        
        if "duplicate_movement_ids" in issue_types:
            suggestions.append("检查并修正重复的数据移动ID")
        
        return suggestions or ["CFP计算验证通过，无需调整"]
    
    def _create_executive_summary(self, cfp_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行摘要"""
        
        total_cfp = cfp_result["total_cfp"]
        type_stats = cfp_result["type_statistics"]
        quality_score = cfp_result["quality_metrics"]["quality_score"]
        
        # 找出主导的数据移动类型
        dominant_type = max(type_stats.keys(), key=lambda k: type_stats[k])
        
        return {
            "total_cfp": total_cfp,
            "dominant_movement_type": dominant_type,
            "quality_score": quality_score,
            "processes_analyzed": cfp_result["quality_metrics"]["processes_count"],
            "summary_text": f"""
项目总计 {total_cfp} 个COSMIC功能点（CFP）。
主要数据移动类型为 {dominant_type}（{type_stats[dominant_type]} 个）。
质量评分：{quality_score:.2f}，共分析 {cfp_result["quality_metrics"]["processes_count"]} 个功能过程。
            """.strip()
        }
    
    async def _create_detailed_analysis(self, cfp_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建详细分析"""
        
        return {
            "type_breakdown": {
                "statistics": cfp_result["type_statistics"],
                "analysis": self._analyze_type_distribution(cfp_result["type_statistics"])
            },
            "process_breakdown": {
                "statistics": cfp_result["process_statistics"],
                "analysis": self._analyze_process_distribution(cfp_result["process_statistics"])
            },
            "quality_assessment": {
                "metrics": cfp_result["quality_metrics"],
                "assessment": self._assess_calculation_quality(cfp_result["quality_metrics"])
            }
        }
    
    def _create_visualization_data(self, cfp_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建可视化数据"""
        
        # 类型分布饼图
        type_pie_data = []
        for movement_type, count in cfp_result["type_statistics"].items():
            type_pie_data.append({
                "label": movement_type,
                "value": count,
                "percentage": (count / cfp_result["total_cfp"]) * 100 if cfp_result["total_cfp"] > 0 else 0
            })
        
        # 过程CFP柱状图
        process_bar_data = []
        for process_id, stats in cfp_result["process_statistics"].items():
            process_bar_data.append({
                "process": process_id,
                "cfp": stats["total_cfp"],
                "data_groups": stats["data_groups_count"]
            })
        
        return {
            "type_distribution_pie": type_pie_data,
            "process_distribution_bar": process_bar_data,
            "chart_config": {
                "type_pie": {
                    "title": "数据移动类型分布",
                    "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
                },
                "process_bar": {
                    "title": "功能过程CFP分布",
                    "x_axis": "功能过程",
                    "y_axis": "CFP数量"
                }
            }
        }
    
    async def _generate_insights(self, cfp_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成洞察和建议"""
        
        insights = []
        recommendations = []
        
        # 分析CFP规模
        total_cfp = cfp_result["total_cfp"]
        if total_cfp < 50:
            insights.append(f"项目规模较小（{total_cfp} CFP），适合小团队快速开发")
        elif total_cfp < 200:
            insights.append(f"项目规模中等（{total_cfp} CFP），需要合理的项目管理")
        else:
            insights.append(f"项目规模较大（{total_cfp} CFP），需要严格的项目管控")
        
        # 分析类型分布
        type_insights = self._analyze_type_distribution(cfp_result["type_statistics"])
        insights.extend(type_insights)
        
        # 质量评估
        quality_score = cfp_result["quality_metrics"]["quality_score"]
        if quality_score < 0.6:
            recommendations.append("建议重新审查数据移动识别，提高估算质量")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "risk_assessment": self._assess_project_risks(cfp_result),
            "improvement_suggestions": self._suggest_improvements(cfp_result)
        }
    
    def _analyze_type_distribution(self, type_statistics: Dict[str, int]) -> List[str]:
        """分析类型分布"""
        insights = []
        total = sum(type_statistics.values())
        
        for movement_type, count in type_statistics.items():
            percentage = (count / total) * 100 if total > 0 else 0
            if percentage > 40:
                insights.append(f"{movement_type}移动占主导（{percentage:.1f}%）")
            elif percentage < 10:
                insights.append(f"{movement_type}移动较少（{percentage:.1f}%）")
        
        return insights
    
    def _analyze_process_distribution(self, process_statistics: Dict[str, Dict[str, Any]]) -> List[str]:
        """分析过程分布"""
        insights = []
        
        cfp_values = [stats["total_cfp"] for stats in process_statistics.values()]
        if cfp_values:
            max_cfp = max(cfp_values)
            min_cfp = min(cfp_values)
            
            if max_cfp > min_cfp * 3:
                insights.append("功能过程复杂度差异较大，建议平衡功能分解")
        
        return insights
    
    def _assess_calculation_quality(self, quality_metrics: Dict[str, Any]) -> str:
        """评估计算质量"""
        quality_score = quality_metrics["quality_score"]
        
        if quality_score >= 0.9:
            return "计算质量优秀"
        elif quality_score >= 0.7:
            return "计算质量良好"
        elif quality_score >= 0.5:
            return "计算质量一般"
        else:
            return "计算质量需要改进"
    
    async def _assess_boundary_consistency(
        self,
        data_movements: List[COSMICDataMovement],
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> float:
        """评估边界一致性"""
        
        consistency_score = 1.0
        
        # 检查Entry/Exit移动是否涉及功能用户
        user_names = [user.name.lower() for user in boundary_analysis.functional_users]
        
        for movement in data_movements:
            if movement.type.value in ["Entry", "Exit"]:
                source_lower = movement.source.lower()
                target_lower = movement.target.lower()
                
                if movement.type.value == "Entry":
                    # Entry应该从功能用户到软件
                    if not any(user_name in source_lower for user_name in user_names):
                        consistency_score -= 0.1
                
                elif movement.type.value == "Exit":
                    # Exit应该从软件到功能用户
                    if not any(user_name in target_lower for user_name in user_names):
                        consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _assess_project_risks(self, cfp_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估项目风险"""
        
        risks = []
        risk_level = "Low"
        
        total_cfp = cfp_result["total_cfp"]
        quality_score = cfp_result["quality_metrics"]["quality_score"]
        
        # CFP规模风险
        if total_cfp > 500:
            risks.append("项目规模较大，存在管理复杂度风险")
            risk_level = "High"
        
        # 质量风险
        if quality_score < 0.6:
            risks.append("估算质量偏低，可能影响项目规划准确性")
            risk_level = "High" if risk_level != "High" else "High"
        
        # 分布风险
        type_stats = cfp_result["type_statistics"]
        total = sum(type_stats.values())
        if any((count/total) > 0.6 for count in type_stats.values() if total > 0):
            risks.append("数据移动类型分布不均衡，可能存在架构风险")
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
            if "规模" in risk:
                suggestions.append("建议采用敏捷开发，分阶段交付")
            elif "质量" in risk:
                suggestions.append("建议增加专家评审，提高估算准确性")
            elif "分布" in risk:
                suggestions.append("建议重新审查架构设计，平衡功能分布")
        
        return suggestions
    
    def _suggest_improvements(self, cfp_result: Dict[str, Any]) -> List[str]:
        """建议改进措施"""
        suggestions = []
        
        quality_metrics = cfp_result["quality_metrics"]
        
        if quality_metrics["type_distribution_balance"] < 0.6:
            suggestions.append("改进数据移动识别，确保类型分布合理")
        
        if quality_metrics["data_group_diversity"] < 0.3:
            suggestions.append("增加数据组分析精度，提高识别细粒度")
        
        if quality_metrics["boundary_consistency_score"] < 0.8:
            suggestions.append("重新审查边界定义，确保数据移动分类准确")
        
        return suggestions or ["CFP计算质量良好，无需特别改进"]
    
    async def _apply_movement_corrections(
        self,
        cfp_result: Dict[str, Any],
        corrections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用数据移动修正"""
        
        # 重新计算统计信息
        if "updated_movements" in corrections:
            updated_movements = corrections["updated_movements"]
            cfp_result["total_cfp"] = len(updated_movements)
            cfp_result["type_statistics"] = self._calculate_type_statistics(updated_movements)
            cfp_result["process_statistics"] = self._calculate_process_statistics(updated_movements)
        
        return cfp_result
    
    def _apply_calculation_adjustments(
        self,
        cfp_result: Dict[str, Any],
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用计算调整"""
        
        if "cfp_adjustment" in adjustments:
            adjustment = adjustments["cfp_adjustment"]
            cfp_result["total_cfp"] += adjustment
        
        return cfp_result
    
    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """获取计算历史"""
        return self.calculation_history.copy()


async def create_cosmic_cfp_calculator(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> COSMICCFPCalculatorAgent:
    """创建COSMIC CFP计算器智能体"""
    return COSMICCFPCalculatorAgent(rule_retriever=rule_retriever, llm=llm)


if __name__ == "__main__":
    async def main():
        # 测试COSMIC CFP计算器
        calculator = await create_cosmic_cfp_calculator()
        
        # 测试数据移动
        from models.cosmic_models import COSMICDataMovementType
        test_movements = [
            COSMICDataMovement(
                id="process1_entry",
                type=COSMICDataMovementType.ENTRY,
                source="用户",
                target="系统",
                data_group="用户输入",
                justification="用户输入数据"
            ),
            COSMICDataMovement(
                id="process1_read",
                type=COSMICDataMovementType.READ,
                source="数据库",
                target="系统",
                data_group="用户数据",
                justification="读取用户信息"
            )
        ]
        
        test_project = ProjectInfo(
            name="测试项目",
            description="COSMIC CFP计算测试",
            technology_stack=["Python"],
            business_domain="测试"
        )
        
        test_boundary = COSMICBoundaryAnalysis(
            software_boundary="测试边界",
            persistent_storage_boundary="测试存储",
            functional_users=[],
            boundary_reasoning="测试推理"
        )
        
        # 计算CFP
        cfp_result = await calculator.calculate_cfp(
            test_movements,
            test_project,
            test_boundary
        )
        
        print(f"总CFP: {cfp_result['total_cfp']}")
        print(f"类型统计: {cfp_result['type_statistics']}")
        
    asyncio.run(main()) 