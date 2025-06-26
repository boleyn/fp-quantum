"""
量子智能化功能点估算系统 - 报告生成器智能体

聚合估算结果，生成详细的分析报告和可视化图表
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
import json
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import SpecializedAgent
from models.project_models import ProjectInfo, EstimationStrategy
from models.nesma_models import NESMAFunctionClassification, NESMAComplexityLevel
from models.cosmic_models import COSMICDataMovement, COSMICFunctionalUser
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ReportGeneratorAgent(SpecializedAgent):
    """报告生成器智能体"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(
            agent_id="report_generator",
            specialty="report_generation",
            llm=llm
        )
        
        self.settings = get_settings()
        
        # 报告模板和历史
        self.report_templates = self._load_report_templates()
        self.generated_reports: List[Dict[str, Any]] = []
        
    def _load_report_templates(self) -> Dict[str, str]:
        """加载报告模板"""
        return {
            "executive_summary": """
## 📊 执行摘要

### 项目概览
- **项目名称**: {project_name}
- **业务领域**: {business_domain}
- **估算日期**: {estimation_date}
- **估算策略**: {estimation_strategy}

### 核心结果
- **总功能点**: {total_fp} FP
- **置信度**: {confidence_level}
- **估算方法**: {methods_used}

### 关键发现
{key_findings}
""",
            
            "detailed_analysis": """
## 🔍 详细分析

### 功能分解
{function_breakdown}

### 复杂度分析
{complexity_analysis}

### 质量指标
{quality_metrics}

### 风险评估
{risk_assessment}
""",
            
            "comparison_report": """
## ⚖️ 标准对比分析

### NESMA vs COSMIC 结果对比
{comparison_table}

### 差异分析
{difference_analysis}

### 推荐建议
{recommendations}
""",
            
            "appendix": """
## 📎 附录

### 详细数据
{detailed_data}

### 计算过程
{calculation_details}

### 参考标准
{reference_standards}
"""
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "结果聚合分析",
            "多格式报告生成",
            "可视化图表创建",
            "标准对比报告",
            "质量评估报告"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行报告生成任务"""
        if task_name == "generate_estimation_report":
            return await self.generate_estimation_report(
                inputs["project_info"],
                inputs["estimation_results"],
                inputs.get("format", "markdown")
            )
        elif task_name == "generate_comparison_report":
            return await self.generate_comparison_report(
                inputs["nesma_results"],
                inputs["cosmic_results"],
                inputs["project_info"]
            )
        elif task_name == "create_executive_summary":
            return await self.create_executive_summary(
                inputs["estimation_results"],
                inputs["project_info"]
            )
        elif task_name == "export_detailed_data":
            return await self.export_detailed_data(
                inputs["estimation_results"],
                inputs.get("format", "json")
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def generate_estimation_report(
        self,
        project_info: ProjectInfo,
        estimation_results: Dict[str, Any],
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """生成完整的估算报告"""
        
        # 1. 创建执行摘要
        executive_summary = await self.create_executive_summary(
            estimation_results, project_info
        )
        
        # 2. 生成详细分析
        detailed_analysis = await self._create_detailed_analysis(
            estimation_results, project_info
        )
        
        # 3. 创建可视化数据
        visualization_data = await self._create_visualization_data(
            estimation_results
        )
        
        # 4. 生成质量评估
        quality_assessment = await self._create_quality_assessment(
            estimation_results
        )
        
        # 5. 组装完整报告
        full_report = await self._assemble_full_report(
            executive_summary,
            detailed_analysis,
            visualization_data,
            quality_assessment,
            project_info,
            format
        )
        
        # 6. 记录生成历史
        self.generated_reports.append({
            "project_name": project_info.name,
            "generation_time": datetime.now(),
            "format": format,
            "report_size": len(str(full_report))
        })
        
        return full_report
    
    async def generate_comparison_report(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """生成标准对比报告"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是功能点估算专家，需要对NESMA和COSMIC两种标准的估算结果进行深度对比分析。

请从以下角度进行分析：
1. 数值差异及其原因
2. 方法论差异的影响
3. 适用性评估
4. 可信度对比
5. 实际应用建议

提供客观、专业的分析。"""),
            ("human", """项目信息：
项目名称：{project_name}
业务领域：{business_domain}

NESMA估算结果：
{nesma_results}

COSMIC估算结果：
{cosmic_results}

请生成详细的对比分析报告。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_name=project_info.name,
                business_domain=project_info.business_domain,
                nesma_results=json.dumps(nesma_results, ensure_ascii=False, indent=2),
                cosmic_results=json.dumps(cosmic_results, ensure_ascii=False, indent=2)
            )
        )
        
        # 生成对比图表数据
        comparison_charts = self._create_comparison_charts(nesma_results, cosmic_results)
        
        return {
            "comparison_analysis": response.content,
            "charts_data": comparison_charts,
            "summary_table": self._create_summary_table(nesma_results, cosmic_results),
            "recommendations": await self._generate_comparison_recommendations(
                nesma_results, cosmic_results, project_info
            )
        }
    
    async def create_executive_summary(
        self,
        estimation_results: Dict[str, Any],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """创建执行摘要"""
        
        # 提取关键指标
        total_fp = self._extract_total_fp(estimation_results)
        confidence_level = self._calculate_overall_confidence(estimation_results)
        methods_used = self._identify_methods_used(estimation_results)
        
        # 生成关键发现
        key_findings = await self._generate_key_findings(estimation_results, project_info)
        
        # 创建摘要内容
        summary_content = self.report_templates["executive_summary"].format(
            project_name=project_info.name,
            business_domain=project_info.business_domain,
            estimation_date=datetime.now().strftime("%Y-%m-%d"),
            estimation_strategy=estimation_results.get("strategy", "未知"),
            total_fp=total_fp,
            confidence_level=confidence_level,
            methods_used=", ".join(methods_used),
            key_findings=key_findings
        )
        
        return {
            "content": summary_content,
            "key_metrics": {
                "total_fp": total_fp,
                "confidence_level": confidence_level,
                "methods_used": methods_used
            }
        }
    
    async def export_detailed_data(
        self,
        estimation_results: Dict[str, Any],
        format: str = "json"
    ) -> Dict[str, Any]:
        """导出详细数据"""
        
        detailed_data = {
            "export_timestamp": datetime.now().isoformat(),
            "estimation_results": estimation_results,
            "metadata": {
                "format": format,
                "agent_id": self.agent_id,
                "version": "1.0"
            }
        }
        
        if format.lower() == "json":
            return {
                "format": "json",
                "data": detailed_data,
                "content": json.dumps(detailed_data, ensure_ascii=False, indent=2)
            }
        elif format.lower() == "csv":
            csv_content = await self._convert_to_csv(estimation_results)
            return {
                "format": "csv", 
                "data": detailed_data,
                "content": csv_content
            }
        else:
            return {
                "format": "raw",
                "data": detailed_data,
                "content": str(detailed_data)
            }
    
    async def _create_detailed_analysis(
        self,
        estimation_results: Dict[str, Any],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """创建详细分析"""
        
        # 功能分解分析
        function_breakdown = await self._analyze_function_breakdown(estimation_results)
        
        # 复杂度分析
        complexity_analysis = await self._analyze_complexity_distribution(estimation_results)
        
        # 质量指标分析
        quality_metrics = await self._analyze_quality_metrics(estimation_results)
        
        # 风险评估
        risk_assessment = await self._assess_estimation_risks(estimation_results, project_info)
        
        return {
            "function_breakdown": function_breakdown,
            "complexity_analysis": complexity_analysis,
            "quality_metrics": quality_metrics,
            "risk_assessment": risk_assessment
        }
    
    async def _create_visualization_data(
        self,
        estimation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建可视化数据"""
        
        visualization_data = {}
        
        # 功能类型分布饼图
        if "nesma_results" in estimation_results:
            nesma_distribution = self._create_nesma_distribution_chart(
                estimation_results["nesma_results"]
            )
            visualization_data["nesma_distribution"] = nesma_distribution
        
        if "cosmic_results" in estimation_results:
            cosmic_distribution = self._create_cosmic_distribution_chart(
                estimation_results["cosmic_results"]
            )
            visualization_data["cosmic_distribution"] = cosmic_distribution
        
        # 复杂度分布柱状图
        complexity_chart = self._create_complexity_chart(estimation_results)
        visualization_data["complexity_distribution"] = complexity_chart
        
        # 置信度趋势图
        confidence_chart = self._create_confidence_chart(estimation_results)
        visualization_data["confidence_trends"] = confidence_chart
        
        return visualization_data
    
    async def _create_quality_assessment(
        self,
        estimation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建质量评估"""
        
        quality_scores = {}
        
        # 数据完整性评分
        quality_scores["completeness"] = self._assess_data_completeness(estimation_results)
        
        # 一致性评分
        quality_scores["consistency"] = self._assess_data_consistency(estimation_results)
        
        # 可信度评分
        quality_scores["reliability"] = self._assess_reliability(estimation_results)
        
        # 总体质量评分
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        quality_scores["overall"] = overall_score
        
        # 质量建议
        quality_recommendations = self._generate_quality_recommendations(quality_scores)
        
        return {
            "scores": quality_scores,
            "recommendations": quality_recommendations,
            "assessment_timestamp": datetime.now()
        }
    
    async def _assemble_full_report(
        self,
        executive_summary: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        visualization_data: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        project_info: ProjectInfo,
        format: str
    ) -> Dict[str, Any]:
        """组装完整报告"""
        
        if format.lower() == "markdown":
            return await self._create_markdown_report(
                executive_summary, detailed_analysis, 
                visualization_data, quality_assessment, project_info
            )
        elif format.lower() == "html":
            return await self._create_html_report(
                executive_summary, detailed_analysis,
                visualization_data, quality_assessment, project_info
            )
        else:
            return {
                "format": format,
                "executive_summary": executive_summary,
                "detailed_analysis": detailed_analysis,
                "visualization_data": visualization_data,
                "quality_assessment": quality_assessment,
                "project_info": project_info.dict()
            }
    
    def _extract_total_fp(self, estimation_results: Dict[str, Any]) -> Union[int, str]:
        """提取总功能点数"""
        if "nesma_results" in estimation_results and "cosmic_results" in estimation_results:
            nesma_fp = estimation_results["nesma_results"].get("total_ufp", 0)
            cosmic_fp = estimation_results["cosmic_results"].get("total_cfp", 0)
            return f"NESMA: {nesma_fp}, COSMIC: {cosmic_fp}"
        elif "nesma_results" in estimation_results:
            return estimation_results["nesma_results"].get("total_ufp", 0)
        elif "cosmic_results" in estimation_results:
            return estimation_results["cosmic_results"].get("total_cfp", 0)
        else:
            return "未知"
    
    def _calculate_overall_confidence(self, estimation_results: Dict[str, Any]) -> str:
        """计算总体置信度"""
        confidence_scores = []
        
        if "nesma_results" in estimation_results:
            nesma_confidence = estimation_results["nesma_results"].get("average_confidence", 0)
            confidence_scores.append(nesma_confidence)
        
        if "cosmic_results" in estimation_results:
            cosmic_confidence = estimation_results["cosmic_results"].get("average_confidence", 0)
            confidence_scores.append(cosmic_confidence)
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence >= 0.8:
                return "高"
            elif avg_confidence >= 0.6:
                return "中"
            else:
                return "低"
        
        return "未知"
    
    def _identify_methods_used(self, estimation_results: Dict[str, Any]) -> List[str]:
        """识别使用的方法"""
        methods = []
        
        if "nesma_results" in estimation_results:
            methods.append("NESMA v2.3+")
        
        if "cosmic_results" in estimation_results:
            methods.append("COSMIC v4.0+")
        
        return methods or ["未知方法"]
    
    async def _generate_key_findings(
        self,
        estimation_results: Dict[str, Any],
        project_info: ProjectInfo
    ) -> str:
        """生成关键发现"""
        
        findings = []
        
        # 分析功能分布
        if "nesma_results" in estimation_results:
            nesma_classifications = estimation_results["nesma_results"].get("classifications", [])
            if nesma_classifications:
                most_common_type = max(
                    set(c.get("function_type") for c in nesma_classifications),
                    key=lambda x: sum(1 for c in nesma_classifications if c.get("function_type") == x)
                )
                findings.append(f"- 最常见的NESMA功能类型是{most_common_type}")
        
        # 分析复杂度
        if "complexity_analysis" in estimation_results:
            high_complexity_count = estimation_results["complexity_analysis"].get("high_complexity_count", 0)
            if high_complexity_count > 0:
                findings.append(f"- 识别出{high_complexity_count}个高复杂度功能")
        
        # 分析风险
        if "risks" in estimation_results:
            risk_count = len(estimation_results["risks"])
            findings.append(f"- 识别出{risk_count}个潜在风险")
        
        return "\n".join(findings) if findings else "- 标准的功能点估算结果"
    
    def _create_nesma_distribution_chart(self, nesma_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建NESMA分布图表数据"""
        
        classifications = nesma_results.get("classifications", [])
        if not classifications:
            return {"type": "pie", "data": [], "labels": []}
        
        # 统计各类型数量
        type_counts = {}
        for classification in classifications:
            func_type = classification.get("function_type", "Unknown")
            type_counts[func_type] = type_counts.get(func_type, 0) + 1
        
        return {
            "type": "pie",
            "title": "NESMA功能类型分布",
            "labels": list(type_counts.keys()),
            "data": list(type_counts.values())
        }
    
    def _create_cosmic_distribution_chart(self, cosmic_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建COSMIC分布图表数据"""
        
        data_movements = cosmic_results.get("data_movements", [])
        if not data_movements:
            return {"type": "pie", "data": [], "labels": []}
        
        # 统计各类型数量
        type_counts = {}
        for movement in data_movements:
            movement_type = movement.get("type", "Unknown")
            type_counts[movement_type] = type_counts.get(movement_type, 0) + 1
        
        return {
            "type": "pie",
            "title": "COSMIC数据移动类型分布",
            "labels": list(type_counts.keys()),
            "data": list(type_counts.values())
        }
    
    def _create_comparison_charts(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建对比图表"""
        
        return {
            "total_comparison": {
                "type": "bar",
                "title": "NESMA vs COSMIC 总功能点对比",
                "categories": ["NESMA", "COSMIC"],
                "data": [
                    nesma_results.get("total_ufp", 0),
                    cosmic_results.get("total_cfp", 0)
                ]
            },
            "confidence_comparison": {
                "type": "bar", 
                "title": "置信度对比",
                "categories": ["NESMA", "COSMIC"],
                "data": [
                    nesma_results.get("average_confidence", 0),
                    cosmic_results.get("average_confidence", 0)
                ]
            }
        }
    
    def _create_summary_table(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建摘要对比表"""
        
        return {
            "headers": ["指标", "NESMA", "COSMIC", "差异"],
            "rows": [
                [
                    "总功能点",
                    nesma_results.get("total_ufp", "N/A"),
                    cosmic_results.get("total_cfp", "N/A"),
                    self._calculate_difference(
                        nesma_results.get("total_ufp", 0),
                        cosmic_results.get("total_cfp", 0)
                    )
                ],
                [
                    "平均置信度",
                    f"{nesma_results.get('average_confidence', 0):.2f}",
                    f"{cosmic_results.get('average_confidence', 0):.2f}",
                    f"{abs(nesma_results.get('average_confidence', 0) - cosmic_results.get('average_confidence', 0)):.2f}"
                ]
            ]
        }
    
    def _calculate_difference(self, value1: Union[int, float], value2: Union[int, float]) -> str:
        """计算差异"""
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            diff = value1 - value2
            if value2 != 0:
                percentage = (diff / value2) * 100
                return f"{diff} ({percentage:+.1f}%)"
            else:
                return str(diff)
        return "N/A"
    
    async def _create_markdown_report(
        self,
        executive_summary: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        visualization_data: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """创建Markdown格式报告"""
        
        markdown_content = f"""# 📊 功能点估算报告

{executive_summary['content']}

{self.report_templates['detailed_analysis'].format(
    function_breakdown=detailed_analysis.get('function_breakdown', ''),
    complexity_analysis=detailed_analysis.get('complexity_analysis', ''),
    quality_metrics=detailed_analysis.get('quality_metrics', ''),
    risk_assessment=detailed_analysis.get('risk_assessment', '')
)}

## 📈 数据可视化

### 图表说明
本报告包含以下可视化数据：
{self._format_visualization_summary(visualization_data)}

## ✅ 质量评估

### 评估结果
- **数据完整性**: {quality_assessment['scores'].get('completeness', 0):.2f}
- **数据一致性**: {quality_assessment['scores'].get('consistency', 0):.2f}
- **结果可信度**: {quality_assessment['scores'].get('reliability', 0):.2f}
- **总体质量**: {quality_assessment['scores'].get('overall', 0):.2f}

### 质量建议
{chr(10).join(f"- {rec}" for rec in quality_assessment.get('recommendations', []))}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*生成工具: 量子智能化功能点估算系统*
"""
        
        return {
            "format": "markdown",
            "content": markdown_content,
            "metadata": {
                "project_name": project_info.name,
                "generation_time": datetime.now(),
                "total_lines": len(markdown_content.split('\n'))
            }
        }
    
    def _format_visualization_summary(self, visualization_data: Dict[str, Any]) -> str:
        """格式化可视化数据摘要"""
        summaries = []
        
        for chart_name, chart_data in visualization_data.items():
            if isinstance(chart_data, dict):
                chart_type = chart_data.get("type", "unknown")
                chart_title = chart_data.get("title", chart_name)
                summaries.append(f"- **{chart_title}** ({chart_type}图)")
        
        return "\n".join(summaries) if summaries else "- 无可视化数据"
    
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """获取报告生成历史"""
        return self.generated_reports.copy()
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """获取生成统计"""
        if not self.generated_reports:
            return {"total": 0}
        
        formats = [report["format"] for report in self.generated_reports]
        format_counts = {fmt: formats.count(fmt) for fmt in set(formats)}
        
        return {
            "total_reports": len(self.generated_reports),
            "format_distribution": format_counts,
            "recent_reports": self.generated_reports[-5:]
        }


# 工厂函数
async def create_report_generator(llm: Optional[BaseLanguageModel] = None) -> ReportGeneratorAgent:
    """创建报告生成器智能体"""
    generator = ReportGeneratorAgent(llm=llm)
    await generator.initialize()
    return generator


if __name__ == "__main__":
    async def main():
        # 测试报告生成器
        generator = await create_report_generator()
        
        # 模拟估算结果
        test_results = {
            "strategy": "DUAL_PARALLEL",
            "nesma_results": {
                "total_ufp": 245,
                "average_confidence": 0.85,
                "classifications": [
                    {"function_type": "EI", "confidence": 0.9},
                    {"function_type": "EO", "confidence": 0.8},
                    {"function_type": "EQ", "confidence": 0.9}
                ]
            },
            "cosmic_results": {
                "total_cfp": 198,
                "average_confidence": 0.82,
                "data_movements": [
                    {"type": "Entry", "confidence": 0.9},
                    {"type": "Exit", "confidence": 0.8},
                    {"type": "Read", "confidence": 0.8}
                ]
            }
        }
        
        # 测试项目信息
        test_project = ProjectInfo(
            name="电商平台用户系统",
            description="用户管理和订单处理系统",
            technology_stack=["Java", "Spring Boot", "MySQL"],
            business_domain="电商"
        )
        
        print("📄 报告生成测试:")
        
        # 生成执行摘要
        summary = await generator.execute(
            "create_executive_summary",
            {
                "estimation_results": test_results,
                "project_info": test_project
            }
        )
        print(f"\n执行摘要生成完成，总功能点: {summary['key_metrics']['total_fp']}")
        
        # 生成对比报告
        comparison = await generator.execute(
            "generate_comparison_report",
            {
                "nesma_results": test_results["nesma_results"],
                "cosmic_results": test_results["cosmic_results"],
                "project_info": test_project
            }
        )
        print(f"\n对比报告生成完成")
        
        # 生成完整报告
        full_report = await generator.execute(
            "generate_estimation_report",
            {
                "project_info": test_project,
                "estimation_results": test_results,
                "format": "markdown"
            }
        )
        print(f"\n完整报告生成完成，格式: {full_report['format']}")
        print(f"报告行数: {full_report['metadata']['total_lines']}")
        
        # 显示统计信息
        stats = generator.get_generation_statistics()
        print(f"\n📊 生成统计: {stats}")
    
    asyncio.run(main()) 