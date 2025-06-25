"""
量子智能化功能点估算系统 - 对比分析智能体

提供NESMA和COSMIC估算结果的对比分析和差异解释
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import SpecializedAgent
from models.nesma_models import NESMAFunctionClassification, NESMAComplexityResult
from models.cosmic_models import COSMICDataMovement, COSMICFunctionalProcess
from models.common_models import ComparisonResult
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ComparisonAnalyzerAgent(SpecializedAgent):
    """跨标准对比分析智能体"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(
            agent_id="comparison_analyzer",
            specialty="cross_standard_analysis",
            llm=llm
        )
        
        self.settings = get_settings()
        
        # 对比分析规则和知识
        self.comparison_frameworks = self._load_comparison_frameworks()
        self.difference_patterns = self._load_difference_patterns()
        
    def _load_comparison_frameworks(self) -> Dict[str, Any]:
        """加载对比分析框架"""
        return {
            "理论基础对比": {
                "NESMA": {
                    "核心理念": "功能规模度量",
                    "测量单位": "功能点(FP)",
                    "计算方法": "基于数据和事务功能",
                    "复杂度因子": "DET和RET"
                },
                "COSMIC": {
                    "核心理念": "软件规模度量",
                    "测量单位": "COSMIC功能点(CFP)",
                    "计算方法": "基于数据移动",
                    "复杂度因子": "数据移动数量"
                }
            },
            "适用场景对比": {
                "NESMA": [
                    "传统业务应用系统",
                    "管理信息系统",
                    "数据处理系统",
                    "报表系统"
                ],
                "COSMIC": [
                    "实时系统",
                    "嵌入式系统",
                    "控制系统",
                    "现代软件架构"
                ]
            },
            "估算精度对比": {
                "NESMA": {
                    "精度范围": "±15-25%",
                    "影响因素": ["功能分类准确性", "复杂度判断", "需求完整性"],
                    "优势": "成熟的标准，丰富的历史数据"
                },
                "COSMIC": {
                    "精度范围": "±10-20%",
                    "影响因素": ["数据移动识别", "边界定义", "功能过程分解"],
                    "优势": "更细粒度的度量，适应现代软件"
                }
            }
        }
    
    def _load_difference_patterns(self) -> Dict[str, List[str]]:
        """加载差异模式"""
        return {
            "常见差异原因": [
                "标准理论基础不同",
                "功能分类方法不同",
                "复杂度计算方式不同",
                "边界定义差异",
                "数据粒度差异"
            ],
            "差异类型": [
                "系统性差异 - 一致的高估或低估",
                "功能性差异 - 特定功能类型的差异",
                "复杂度差异 - 复杂功能的估算差异",
                "边界差异 - 系统边界理解不同"
            ],
            "可接受差异范围": [
                "小于20% - 正常差异范围",
                "20%-40% - 需要详细分析原因",
                "大于40% - 可能存在分析错误"
            ]
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "NESMA与COSMIC结果对比",
            "差异原因分析",
            "标准选择建议",
            "结果可信度评估",
            "综合估算建议"
        ]
    
    async def analyze_cross_standard_comparison(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> ComparisonResult:
        """执行跨标准对比分析"""
        
        logger.info("🔍 开始执行跨标准对比分析...")
        
        start_time = time.time()
        
        try:
            # 1. 基础数据对比
            basic_comparison = await self._perform_basic_comparison(nesma_results, cosmic_results)
            
            # 2. 详细差异分析
            detailed_analysis = await self._perform_detailed_analysis(
                nesma_results, cosmic_results, project_info
            )
            
            # 3. 差异原因解释
            difference_explanation = await self._explain_differences(
                basic_comparison, detailed_analysis, project_info
            )
            
            # 4. 标准选择建议
            standard_recommendation = await self._generate_standard_recommendation(
                basic_comparison, detailed_analysis, project_info
            )
            
            # 5. 综合分析报告
            comprehensive_report = await self._generate_comprehensive_report(
                basic_comparison, detailed_analysis, difference_explanation, standard_recommendation
            )
            
            processing_time = time.time() - start_time
            
            comparison_result = ComparisonResult(
                nesma_total=nesma_results.get("total_ufp", 0),
                cosmic_total=cosmic_results.get("total_cfp", 0),
                difference_percentage=basic_comparison["difference_percentage"],
                difference_analysis=detailed_analysis,
                explanation=difference_explanation,
                recommendation=standard_recommendation,
                comprehensive_report=comprehensive_report,
                processing_time=processing_time
            )
            
            logger.info(f"✅ 跨标准对比分析完成，耗时 {processing_time:.2f} 秒")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ 跨标准对比分析失败: {str(e)}")
            raise
    
    async def _perform_basic_comparison(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行基础数据对比"""
        
        nesma_total = nesma_results.get("total_ufp", 0)
        cosmic_total = cosmic_results.get("total_cfp", 0)
        
        if nesma_total == 0 or cosmic_total == 0:
            difference_percentage = 0
        else:
            difference_percentage = abs(nesma_total - cosmic_total) / max(nesma_total, cosmic_total) * 100
        
        # 基础统计信息
        nesma_function_count = len(nesma_results.get("classifications", []))
        cosmic_process_count = len(cosmic_results.get("functional_processes", []))
        cosmic_movement_count = len(cosmic_results.get("data_movements", []))
        
        return {
            "nesma_total": nesma_total,
            "cosmic_total": cosmic_total,
            "difference_absolute": abs(nesma_total - cosmic_total),
            "difference_percentage": difference_percentage,
            "higher_estimate": "NESMA" if nesma_total > cosmic_total else "COSMIC",
            "statistics": {
                "nesma_function_count": nesma_function_count,
                "cosmic_process_count": cosmic_process_count,
                "cosmic_movement_count": cosmic_movement_count,
                "avg_cfp_per_process": cosmic_total / max(cosmic_process_count, 1),
                "avg_ufp_per_function": nesma_total / max(nesma_function_count, 1)
            }
        }
    
    async def _perform_detailed_analysis(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行详细差异分析"""
        
        # 功能类型分布分析
        function_distribution_analysis = await self._analyze_function_distribution(
            nesma_results, cosmic_results
        )
        
        # 复杂度分布分析
        complexity_distribution_analysis = await self._analyze_complexity_distribution(
            nesma_results, cosmic_results
        )
        
        # 数据处理模式分析
        data_processing_analysis = await self._analyze_data_processing_patterns(
            nesma_results, cosmic_results
        )
        
        return {
            "function_distribution": function_distribution_analysis,
            "complexity_distribution": complexity_distribution_analysis,
            "data_processing": data_processing_analysis,
            "correlation_analysis": await self._perform_correlation_analysis(nesma_results, cosmic_results)
        }
    
    async def _analyze_function_distribution(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析功能类型分布"""
        
        # NESMA功能类型统计
        nesma_classifications = nesma_results.get("classifications", [])
        nesma_distribution = {}
        for classification in nesma_classifications:
            func_type = classification.get("function_type", "Unknown")
            nesma_distribution[func_type] = nesma_distribution.get(func_type, 0) + 1
        
        # COSMIC数据移动类型统计
        cosmic_movements = cosmic_results.get("data_movements", [])
        cosmic_distribution = {}
        for movement in cosmic_movements:
            movement_type = movement.get("type", "Unknown")
            cosmic_distribution[movement_type] = cosmic_distribution.get(movement_type, 0) + 1
        
        return {
            "nesma_function_types": nesma_distribution,
            "cosmic_movement_types": cosmic_distribution,
            "distribution_insights": self._generate_distribution_insights(nesma_distribution, cosmic_distribution)
        }
    
    async def _analyze_complexity_distribution(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析复杂度分布"""
        
        # NESMA复杂度分布
        nesma_complexity = {}
        for classification in nesma_results.get("classifications", []):
            complexity = classification.get("complexity", "Unknown")
            nesma_complexity[complexity] = nesma_complexity.get(complexity, 0) + 1
        
        # COSMIC过程复杂度分析（基于数据移动数量）
        cosmic_complexity = {"Low": 0, "Average": 0, "High": 0}
        for process in cosmic_results.get("functional_processes", []):
            movement_count = len(process.get("data_movements", []))
            if movement_count <= 4:
                cosmic_complexity["Low"] += 1
            elif movement_count <= 7:
                cosmic_complexity["Average"] += 1
            else:
                cosmic_complexity["High"] += 1
        
        return {
            "nesma_complexity": nesma_complexity,
            "cosmic_complexity": cosmic_complexity,
            "complexity_insights": self._generate_complexity_insights(nesma_complexity, cosmic_complexity)
        }
    
    async def _analyze_data_processing_patterns(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析数据处理模式"""
        
        # NESMA数据功能vs事务功能
        nesma_data_functions = 0
        nesma_transaction_functions = 0
        
        for classification in nesma_results.get("classifications", []):
            func_type = classification.get("function_type", "")
            if func_type in ["ILF", "EIF"]:
                nesma_data_functions += 1
            elif func_type in ["EI", "EO", "EQ"]:
                nesma_transaction_functions += 1
        
        # COSMIC数据移动模式
        cosmic_data_in = 0
        cosmic_data_out = 0
        cosmic_data_storage = 0
        
        for movement in cosmic_results.get("data_movements", []):
            movement_type = movement.get("type", "")
            if movement_type == "Entry":
                cosmic_data_in += 1
            elif movement_type == "Exit":
                cosmic_data_out += 1
            elif movement_type in ["Read", "Write"]:
                cosmic_data_storage += 1
        
        return {
            "nesma_data_vs_transaction": {
                "data_functions": nesma_data_functions,
                "transaction_functions": nesma_transaction_functions,
                "data_ratio": nesma_data_functions / max(nesma_data_functions + nesma_transaction_functions, 1)
            },
            "cosmic_movement_patterns": {
                "data_in": cosmic_data_in,
                "data_out": cosmic_data_out,
                "data_storage": cosmic_data_storage,
                "input_output_ratio": cosmic_data_in / max(cosmic_data_out, 1)
            }
        }
    
    async def _perform_correlation_analysis(
        self,
        nesma_results: Dict[str, Any],
        cosmic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行相关性分析"""
        
        # 计算功能数量相关性
        nesma_function_count = len(nesma_results.get("classifications", []))
        cosmic_process_count = len(cosmic_results.get("functional_processes", []))
        
        function_count_ratio = nesma_function_count / max(cosmic_process_count, 1)
        
        # 计算估算值相关性
        nesma_total = nesma_results.get("total_ufp", 0)
        cosmic_total = cosmic_results.get("total_cfp", 0)
        
        size_ratio = nesma_total / max(cosmic_total, 1)
        
        return {
            "function_count_correlation": function_count_ratio,
            "size_correlation": size_ratio,
            "correlation_insights": self._generate_correlation_insights(function_count_ratio, size_ratio)
        }
    
    def _generate_distribution_insights(
        self,
        nesma_distribution: Dict[str, int],
        cosmic_distribution: Dict[str, int]
    ) -> List[str]:
        """生成分布洞察"""
        
        insights = []
        
        # NESMA分布分析
        total_nesma = sum(nesma_distribution.values())
        if total_nesma > 0:
            data_function_ratio = (nesma_distribution.get("ILF", 0) + nesma_distribution.get("EIF", 0)) / total_nesma
            if data_function_ratio > 0.4:
                insights.append("NESMA分析显示数据功能占比较高，系统偏向数据密集型")
            elif data_function_ratio < 0.2:
                insights.append("NESMA分析显示事务功能占比较高，系统偏向处理密集型")
        
        # COSMIC分布分析
        total_cosmic = sum(cosmic_distribution.values())
        if total_cosmic > 0:
            storage_ratio = (cosmic_distribution.get("Read", 0) + cosmic_distribution.get("Write", 0)) / total_cosmic
            if storage_ratio > 0.5:
                insights.append("COSMIC分析显示存储操作较多，系统需要大量数据持久化")
        
        return insights
    
    def _generate_complexity_insights(
        self,
        nesma_complexity: Dict[str, int],
        cosmic_complexity: Dict[str, int]
    ) -> List[str]:
        """生成复杂度洞察"""
        
        insights = []
        
        # NESMA复杂度分析
        total_nesma = sum(nesma_complexity.values())
        if total_nesma > 0:
            high_complexity_ratio = nesma_complexity.get("High", 0) / total_nesma
            if high_complexity_ratio > 0.3:
                insights.append("NESMA分析显示高复杂度功能较多，系统设计复杂")
        
        # COSMIC复杂度分析
        total_cosmic = sum(cosmic_complexity.values())
        if total_cosmic > 0:
            simple_process_ratio = cosmic_complexity.get("Low", 0) / total_cosmic
            if simple_process_ratio > 0.6:
                insights.append("COSMIC分析显示简单功能过程较多，系统功能相对独立")
        
        return insights
    
    def _generate_correlation_insights(
        self,
        function_count_ratio: float,
        size_ratio: float
    ) -> List[str]:
        """生成相关性洞察"""
        
        insights = []
        
        if 0.8 <= function_count_ratio <= 1.2:
            insights.append("NESMA功能数量与COSMIC过程数量相近，分解粒度一致")
        elif function_count_ratio > 1.5:
            insights.append("NESMA功能分解更细，可能包含更多细节功能")
        elif function_count_ratio < 0.7:
            insights.append("COSMIC过程分解更细，功能过程识别更全面")
        
        if 0.9 <= size_ratio <= 1.1:
            insights.append("两种标准的估算结果非常接近，结果可信度高")
        elif size_ratio > 1.3:
            insights.append("NESMA估算值显著高于COSMIC，可能高估了功能复杂度")
        elif size_ratio < 0.7:
            insights.append("COSMIC估算值显著高于NESMA，可能识别了更多数据移动")
        
        return insights
    
    async def _explain_differences(
        self,
        basic_comparison: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解释差异原因"""
        
        difference_percentage = basic_comparison["difference_percentage"]
        
        # 根据差异程度分类
        if difference_percentage < 20:
            difference_category = "正常差异"
            explanation_focus = "标准差异"
        elif difference_percentage < 40:
            difference_category = "显著差异"
            explanation_focus = "方法论差异"
        else:
            difference_category = "异常差异"
            explanation_focus = "可能存在分析错误"
        
        # 生成详细解释
        detailed_explanation = await self._generate_detailed_explanation(
            basic_comparison, detailed_analysis, project_info, difference_category
        )
        
        return {
            "difference_category": difference_category,
            "explanation_focus": explanation_focus,
            "main_reasons": self._identify_main_reasons(detailed_analysis, difference_percentage),
            "detailed_explanation": detailed_explanation,
            "improvement_suggestions": self._generate_improvement_suggestions(difference_category, detailed_analysis)
        }
    
    async def _generate_detailed_explanation(
        self,
        basic_comparison: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        project_info: Dict[str, Any],
        difference_category: str
    ) -> str:
        """生成详细解释"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是功能点估算专家，需要解释NESMA和COSMIC估算结果的差异。

分析要点：
1. 标准理论基础差异
2. 功能分类方法差异
3. 复杂度计算差异
4. 项目特征影响
5. 可能的改进方向

请提供专业、详细的差异解释。"""),
            ("human", """项目信息：{project_info}

基础对比结果：{basic_comparison}

详细分析结果：{detailed_analysis}

差异类别：{difference_category}

请解释造成差异的主要原因，并提供专业建议。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_info=str(project_info),
                basic_comparison=str(basic_comparison),
                detailed_analysis=str(detailed_analysis),
                difference_category=difference_category
            )
        )
        
        return response.content
    
    def _identify_main_reasons(
        self,
        detailed_analysis: Dict[str, Any],
        difference_percentage: float
    ) -> List[str]:
        """识别主要差异原因"""
        
        reasons = []
        
        # 基于分布分析识别原因
        function_dist = detailed_analysis.get("function_distribution", {})
        nesma_types = function_dist.get("nesma_function_types", {})
        cosmic_types = function_dist.get("cosmic_movement_types", {})
        
        # 检查数据功能占比
        total_nesma = sum(nesma_types.values())
        if total_nesma > 0:
            data_ratio = (nesma_types.get("ILF", 0) + nesma_types.get("EIF", 0)) / total_nesma
            if data_ratio > 0.4:
                reasons.append("NESMA数据功能占比较高，可能影响总体估算")
        
        # 检查复杂度分布
        complexity_dist = detailed_analysis.get("complexity_distribution", {})
        nesma_complexity = complexity_dist.get("nesma_complexity", {})
        total_nesma_complexity = sum(nesma_complexity.values())
        if total_nesma_complexity > 0:
            high_ratio = nesma_complexity.get("High", 0) / total_nesma_complexity
            if high_ratio > 0.3:
                reasons.append("NESMA高复杂度功能较多，增加了估算值")
        
        # 检查数据移动模式
        total_cosmic = sum(cosmic_types.values())
        if total_cosmic > 0:
            storage_ratio = (cosmic_types.get("Read", 0) + cosmic_types.get("Write", 0)) / total_cosmic
            if storage_ratio > 0.5:
                reasons.append("COSMIC存储操作较多，可能增加了CFP计数")
        
        if not reasons:
            reasons.append("差异在正常范围内，主要由标准理论差异造成")
        
        return reasons
    
    def _generate_improvement_suggestions(
        self,
        difference_category: str,
        detailed_analysis: Dict[str, Any]
    ) -> List[str]:
        """生成改进建议"""
        
        suggestions = []
        
        if difference_category == "异常差异":
            suggestions.extend([
                "重新检查功能分类的准确性",
                "验证数据移动识别的完整性",
                "确认系统边界定义的一致性",
                "检查是否遗漏或重复计算功能"
            ])
        elif difference_category == "显著差异":
            suggestions.extend([
                "分析项目特征对标准选择的影响",
                "考虑使用混合方法进行验证",
                "细化功能分解和边界定义"
            ])
        else:
            suggestions.extend([
                "差异在可接受范围内",
                "可以基于项目特征选择更适合的标准",
                "建议保留两种估算结果作为参考"
            ])
        
        return suggestions
    
    async def _generate_standard_recommendation(
        self,
        basic_comparison: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成标准选择建议"""
        
        # 分析项目特征
        project_characteristics = self._analyze_project_characteristics(project_info)
        
        # 评估标准适用性
        nesma_suitability = self._evaluate_nesma_suitability(project_characteristics, detailed_analysis)
        cosmic_suitability = self._evaluate_cosmic_suitability(project_characteristics, detailed_analysis)
        
        # 生成推荐
        if nesma_suitability > cosmic_suitability + 0.2:
            recommended_standard = "NESMA"
            confidence = nesma_suitability
        elif cosmic_suitability > nesma_suitability + 0.2:
            recommended_standard = "COSMIC"
            confidence = cosmic_suitability
        else:
            recommended_standard = "BOTH"
            confidence = (nesma_suitability + cosmic_suitability) / 2
        
        return {
            "recommended_standard": recommended_standard,
            "confidence": confidence,
            "nesma_suitability": nesma_suitability,
            "cosmic_suitability": cosmic_suitability,
            "reasoning": self._generate_recommendation_reasoning(
                recommended_standard, nesma_suitability, cosmic_suitability, project_characteristics
            ),
            "use_case_guidance": self._generate_use_case_guidance(recommended_standard, project_characteristics)
        }
    
    def _analyze_project_characteristics(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析项目特征"""
        
        description = project_info.get("description", "").lower()
        technology_stack = [tech.lower() for tech in project_info.get("technology_stack", [])]
        business_domain = project_info.get("business_domain", "").lower()
        
        characteristics = {
            "is_traditional_business": any(term in description for term in ["管理", "业务", "信息系统", "数据处理"]),
            "is_realtime_system": any(term in description for term in ["实时", "控制", "监控", "嵌入式"]),
            "is_data_intensive": any(term in description for term in ["数据", "分析", "报表", "统计"]),
            "is_modern_architecture": any(tech in technology_stack for tech in ["微服务", "云原生", "api", "rest"]),
            "business_domain": business_domain,
            "technology_complexity": len(technology_stack)
        }
        
        return characteristics
    
    def _evaluate_nesma_suitability(
        self,
        characteristics: Dict[str, Any],
        detailed_analysis: Dict[str, Any]
    ) -> float:
        """评估NESMA适用性"""
        
        suitability_score = 0.5  # 基础分数
        
        # 传统业务系统加分
        if characteristics["is_traditional_business"]:
            suitability_score += 0.3
        
        # 数据密集型系统加分
        if characteristics["is_data_intensive"]:
            suitability_score += 0.2
        
        # 实时系统减分
        if characteristics["is_realtime_system"]:
            suitability_score -= 0.2
        
        # 现代架构减分
        if characteristics["is_modern_architecture"]:
            suitability_score -= 0.1
        
        # 基于分析结果调整
        function_dist = detailed_analysis.get("function_distribution", {})
        nesma_types = function_dist.get("nesma_function_types", {})
        total_functions = sum(nesma_types.values())
        
        if total_functions > 0:
            # 数据功能占比高时加分
            data_ratio = (nesma_types.get("ILF", 0) + nesma_types.get("EIF", 0)) / total_functions
            if data_ratio > 0.4:
                suitability_score += 0.1
        
        return max(0.0, min(1.0, suitability_score))
    
    def _evaluate_cosmic_suitability(
        self,
        characteristics: Dict[str, Any],
        detailed_analysis: Dict[str, Any]
    ) -> float:
        """评估COSMIC适用性"""
        
        suitability_score = 0.5  # 基础分数
        
        # 实时系统加分
        if characteristics["is_realtime_system"]:
            suitability_score += 0.3
        
        # 现代架构加分
        if characteristics["is_modern_architecture"]:
            suitability_score += 0.2
        
        # 传统业务系统减分
        if characteristics["is_traditional_business"]:
            suitability_score -= 0.1
        
        # 技术复杂度加分
        if characteristics["technology_complexity"] > 5:
            suitability_score += 0.1
        
        # 基于分析结果调整
        data_processing = detailed_analysis.get("data_processing", {})
        cosmic_patterns = data_processing.get("cosmic_movement_patterns", {})
        
        # 数据移动模式丰富时加分
        movement_types = sum(1 for v in cosmic_patterns.values() if v > 0)
        if movement_types >= 3:
            suitability_score += 0.1
        
        return max(0.0, min(1.0, suitability_score))
    
    def _generate_recommendation_reasoning(
        self,
        recommended_standard: str,
        nesma_suitability: float,
        cosmic_suitability: float,
        characteristics: Dict[str, Any]
    ) -> str:
        """生成推荐理由"""
        
        if recommended_standard == "NESMA":
            return f"NESMA更适合此项目（适用性：{nesma_suitability:.2f}），因为项目特征偏向传统业务系统"
        elif recommended_standard == "COSMIC":
            return f"COSMIC更适合此项目（适用性：{cosmic_suitability:.2f}），因为项目特征偏向现代软件架构"
        else:
            return f"两种标准都适用（NESMA：{nesma_suitability:.2f}，COSMIC：{cosmic_suitability:.2f}），建议同时使用"
    
    def _generate_use_case_guidance(
        self,
        recommended_standard: str,
        characteristics: Dict[str, Any]
    ) -> List[str]:
        """生成使用指导"""
        
        guidance = []
        
        if recommended_standard == "NESMA":
            guidance.extend([
                "重点关注数据功能和事务功能的准确分类",
                "仔细计算DET和RET以确定复杂度",
                "利用NESMA丰富的历史数据进行校准"
            ])
        elif recommended_standard == "COSMIC":
            guidance.extend([
                "清晰定义软件边界和功能用户",
                "全面识别所有数据移动",
                "确保功能过程分解的完整性"
            ])
        else:
            guidance.extend([
                "使用两种标准进行交叉验证",
                "关注估算结果的一致性",
                "基于项目阶段选择合适的标准"
            ])
        
        return guidance
    
    async def _generate_comprehensive_report(
        self,
        basic_comparison: Dict[str, Any],
        detailed_analysis: Dict[str, Any],
        difference_explanation: Dict[str, Any],
        standard_recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成综合分析报告"""
        
        return {
            "executive_summary": {
                "nesma_estimate": basic_comparison["nesma_total"],
                "cosmic_estimate": basic_comparison["cosmic_total"],
                "difference_percentage": basic_comparison["difference_percentage"],
                "recommended_standard": standard_recommendation["recommended_standard"],
                "confidence_level": standard_recommendation["confidence"]
            },
            "detailed_findings": {
                "function_analysis": detailed_analysis["function_distribution"],
                "complexity_analysis": detailed_analysis["complexity_distribution"],
                "correlation_analysis": detailed_analysis["correlation_analysis"]
            },
            "difference_analysis": {
                "category": difference_explanation["difference_category"],
                "main_reasons": difference_explanation["main_reasons"],
                "improvement_suggestions": difference_explanation["improvement_suggestions"]
            },
            "recommendations": {
                "standard_choice": standard_recommendation,
                "implementation_guidance": standard_recommendation["use_case_guidance"]
            },
            "quality_metrics": {
                "analysis_completeness": self._calculate_analysis_completeness(detailed_analysis),
                "result_reliability": self._calculate_result_reliability(basic_comparison, detailed_analysis)
            }
        }
    
    def _calculate_analysis_completeness(self, detailed_analysis: Dict[str, Any]) -> float:
        """计算分析完整性"""
        
        required_analyses = ["function_distribution", "complexity_distribution", "data_processing", "correlation_analysis"]
        completed_analyses = sum(1 for analysis in required_analyses if analysis in detailed_analysis)
        
        return completed_analyses / len(required_analyses)
    
    def _calculate_result_reliability(
        self,
        basic_comparison: Dict[str, Any],
        detailed_analysis: Dict[str, Any]
    ) -> float:
        """计算结果可靠性"""
        
        reliability_score = 1.0
        
        # 基于差异程度调整
        difference_percentage = basic_comparison["difference_percentage"]
        if difference_percentage > 40:
            reliability_score -= 0.3
        elif difference_percentage > 20:
            reliability_score -= 0.1
        
        # 基于统计数据调整
        statistics = basic_comparison.get("statistics", {})
        nesma_count = statistics.get("nesma_function_count", 0)
        cosmic_count = statistics.get("cosmic_process_count", 0)
        
        if nesma_count < 5 or cosmic_count < 5:
            reliability_score -= 0.2
        
        return max(0.0, reliability_score)


if __name__ == "__main__":
    # 测试对比分析智能体
    async def test_comparison_analyzer():
        agent = ComparisonAnalyzerAgent()
        
        # 测试数据
        nesma_results = {
            "total_ufp": 120,
            "classifications": [
                {"function_type": "ILF", "complexity": "Average"},
                {"function_type": "EI", "complexity": "High"},
                {"function_type": "EO", "complexity": "Low"}
            ]
        }
        
        cosmic_results = {
            "total_cfp": 95,
            "functional_processes": [
                {"data_movements": ["Entry", "Read", "Write", "Exit"]},
                {"data_movements": ["Entry", "Read"]}
            ],
            "data_movements": [
                {"type": "Entry"}, {"type": "Read"}, {"type": "Write"},
                {"type": "Exit"}, {"type": "Entry"}, {"type": "Read"}
            ]
        }
        
        project_info = {
            "description": "企业管理信息系统",
            "technology_stack": ["Java", "Spring", "MySQL"],
            "business_domain": "企业管理"
        }
        
        result = await agent.analyze_cross_standard_comparison(
            nesma_results, cosmic_results, project_info
        )
        
        print(f"对比分析结果：")
        print(f"- NESMA估算：{result.nesma_total} UFP")
        print(f"- COSMIC估算：{result.cosmic_total} CFP")
        print(f"- 差异百分比：{result.difference_percentage:.1f}%")
        print(f"- 推荐标准：{result.recommendation['recommended_standard']}")
    
    asyncio.run(test_comparison_analyzer()) 