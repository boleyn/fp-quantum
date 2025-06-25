"""
量子智能化功能点估算系统 - NESMA复杂度计算器智能体

基于NESMA v2.3+标准计算功能复杂度等级
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
    ComplexityLevel, NESMAComplexityResult
)
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class NESMAComplexityCalculatorAgent(SpecializedAgent):
    """NESMA复杂度计算器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="nesma_complexity_calculator",
            specialty="nesma_complexity_calculation",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # NESMA复杂度规则数据库
        self.complexity_rules = self._load_complexity_rules()
        self.calculation_history: List[NESMAComplexityResult] = []
        
    def _load_complexity_rules(self) -> Dict[str, Any]:
        """加载NESMA复杂度计算规则"""
        return {
            "ILF": {
                "complexity_matrix": {
                    "DET_1-19": {
                        "RET_1": "Low",
                        "RET_2-5": "Low", 
                        "RET_6+": "Average"
                    },
                    "DET_20-50": {
                        "RET_1": "Low",
                        "RET_2-5": "Average",
                        "RET_6+": "High"
                    },
                    "DET_51+": {
                        "RET_1": "Average",
                        "RET_2-5": "High",
                        "RET_6+": "High"
                    }
                },
                "DET_识别规则": [
                    "用户可识别的非重复字段",
                    "维护的数据元素",
                    "外键计为1个DET",
                    "系统生成字段一般不计入"
                ],
                "RET_识别规则": [
                    "用户可识别的子组",
                    "可选或重复的数据组",
                    "主记录类型计为1个RET",
                    "每个子类型计为独立RET"
                ]
            },
            "EIF": {
                "complexity_matrix": {
                    "DET_1-19": {
                        "RET_1": "Low",
                        "RET_2-5": "Low",
                        "RET_6+": "Average"
                    },
                    "DET_20-50": {
                        "RET_1": "Low", 
                        "RET_2-5": "Average",
                        "RET_6+": "High"
                    },
                    "DET_51+": {
                        "RET_1": "Average",
                        "RET_2-5": "High",
                        "RET_6+": "High"
                    }
                },
                "DET_识别规则": [
                    "应用引用的数据元素",
                    "从ILF中引用的字段",
                    "计数规则与ILF相同"
                ],
                "RET_识别规则": [
                    "引用的子组数量",
                    "与维护该文件的应用定义一致"
                ]
            },
            "EI": {
                "complexity_matrix": {
                    "DET_1-14": {
                        "FTR_0-1": "Low",
                        "FTR_2": "Low",
                        "FTR_3+": "Average"
                    },
                    "DET_15-25": {
                        "FTR_0-1": "Low",
                        "FTR_2": "Average", 
                        "FTR_3+": "High"
                    },
                    "DET_26+": {
                        "FTR_0-1": "Average",
                        "FTR_2": "High",
                        "FTR_3+": "High"
                    }
                },
                "DET_识别规则": [
                    "用户可识别的输入字段",
                    "应用接收的数据元素",
                    "控制信息也计入DET",
                    "重复字段只计算一次"
                ],
                "FTR_识别规则": [
                    "处理过程中引用的文件",
                    "更新的ILF计为FTR",
                    "读取的EIF计为FTR",
                    "控制文件也计入FTR"
                ]
            },
            "EO": {
                "complexity_matrix": {
                    "DET_1-19": {
                        "FTR_0-1": "Low",
                        "FTR_2-3": "Low",
                        "FTR_4+": "Average"
                    },
                    "DET_20-25": {
                        "FTR_0-1": "Low",
                        "FTR_2-3": "Average",
                        "FTR_4+": "High"
                    },
                    "DET_26+": {
                        "FTR_0-1": "Average", 
                        "FTR_2-3": "High",
                        "FTR_4+": "High"
                    }
                },
                "DET_识别规则": [
                    "用户可识别的输出字段",
                    "派生或计算的数据元素",
                    "报表中的汇总数据",
                    "用户可见的控制信息"
                ],
                "FTR_识别规则": [
                    "读取或处理的文件",
                    "用于计算的ILF和EIF",
                    "控制信息来源文件"
                ]
            },
            "EQ": {
                "complexity_matrix": {
                    "DET_1-19": {
                        "FTR_0-1": "Low",
                        "FTR_2-3": "Low", 
                        "FTR_4+": "Average"
                    },
                    "DET_20-25": {
                        "FTR_0-1": "Low",
                        "FTR_2-3": "Average",
                        "FTR_4+": "High"
                    },
                    "DET_26+": {
                        "FTR_0-1": "Average",
                        "FTR_2-3": "High", 
                        "FTR_4+": "High"
                    }
                },
                "DET_识别规则": [
                    "输入和输出的数据元素总和",
                    "查询条件字段",
                    "显示的数据字段",
                    "不重复计算相同字段"
                ],
                "FTR_识别规则": [
                    "查询引用的文件",
                    "检索的ILF和EIF",
                    "用于查询逻辑的文件"
                ]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "DET计数计算",
            "RET/FTR计数计算", 
            "复杂度等级确定",
            "计算过程解释",
            "批量复杂度计算"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行复杂度计算任务"""
        if task_name == "calculate_complexity":
            return await self.calculate_complexity(
                inputs["classification"],
                inputs["function_description"],
                inputs.get("detailed_data", {})
            )
        elif task_name == "calculate_batch_complexity":
            return await self.calculate_batch_complexity(inputs["classifications"])
        elif task_name == "explain_complexity_calculation":
            return await self.explain_complexity_calculation(
                inputs["function_type"],
                inputs["det_count"],
                inputs["ret_ftr_count"]
            )
        elif task_name == "validate_complexity_result":
            return await self.validate_complexity_result(
                inputs["complexity_result"],
                inputs["function_description"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def calculate_complexity(
        self,
        classification: NESMAFunctionClassification,
        function_description: str,
        detailed_data: Optional[Dict[str, Any]] = None
    ) -> NESMAComplexityResult:
        """计算功能复杂度"""
        
        # 1. 获取相关NESMA复杂度规则
        complexity_rules = await self._retrieve_complexity_rules(
            classification.function_type, 
            function_description
        )
        
        # 2. 计算DET数量
        det_count = await self._calculate_det_count(
            classification.function_type,
            function_description,
            detailed_data
        )
        
        # 3. 计算RET/FTR数量
        ret_ftr_count = await self._calculate_ret_ftr_count(
            classification.function_type,
            function_description,
            detailed_data
        )
        
        # 4. 确定复杂度等级
        complexity_level = self._determine_complexity_level(
            classification.function_type,
            det_count,
            ret_ftr_count
        )
        
        # 5. 生成计算详情
        calculation_details = {
            "function_type": classification.function_type,
            "det_count": det_count,
            "ret_ftr_count": ret_ftr_count,
            "applied_rules": complexity_rules,
            "calculation_matrix": self.complexity_rules[classification.function_type]["complexity_matrix"],
            "confidence_score": self._calculate_complexity_confidence(
                det_count, ret_ftr_count, detailed_data
            )
        }
        
        # 6. 构建复杂度结果
        complexity_result = NESMAComplexityResult(
            function_id=classification.function_id,
            complexity=complexity_level,
            det_count=det_count,
            ret_count=ret_ftr_count if classification.function_type in ["ILF", "EIF"] else 0,
            calculation_details=calculation_details
        )
        
        # 7. 记录计算历史
        self.calculation_history.append(complexity_result)
        
        return complexity_result
    
    async def calculate_batch_complexity(
        self, 
        classifications: List[NESMAFunctionClassification]
    ) -> List[NESMAComplexityResult]:
        """批量计算功能复杂度"""
        
        complexity_results = []
        
        for classification in classifications:
            try:
                complexity_result = await self.calculate_complexity(
                    classification,
                    f"批量计算功能: {classification.function_id}",
                    {}
                )
                complexity_results.append(complexity_result)
                
            except Exception as e:
                logger.error(f"计算功能 {classification.function_id} 复杂度时出错: {e}")
                # 使用保守估算
                fallback_result = self._create_fallback_complexity(classification)
                complexity_results.append(fallback_result)
        
        return complexity_results
    
    async def explain_complexity_calculation(
        self,
        function_type: NESMAFunctionType,
        det_count: int,
        ret_ftr_count: int
    ) -> Dict[str, Any]:
        """解释复杂度计算过程"""
        
        # 获取应用的复杂度矩阵
        matrix = self.complexity_rules[function_type]["complexity_matrix"]
        
        # 确定DET范围
        det_range = self._get_det_range(det_count)
        
        # 确定RET/FTR范围
        ret_ftr_range = self._get_ret_ftr_range(function_type, ret_ftr_count)
        
        # 获取复杂度结果
        complexity = matrix[det_range][ret_ftr_range]
        
        explanation = {
            "function_type": function_type,
            "input_counts": {
                "det_count": det_count,
                "ret_ftr_count": ret_ftr_count
            },
            "applied_ranges": {
                "det_range": det_range,
                "ret_ftr_range": ret_ftr_range
            },
            "complexity_result": complexity,
            "calculation_matrix": matrix,
            "explanation_text": f"""
根据NESMA标准，{function_type}功能的复杂度计算如下：

1. DET计数: {det_count} ({det_range})
2. {'RET' if function_type in ['ILF', 'EIF'] else 'FTR'}计数: {ret_ftr_count} ({ret_ftr_range})
3. 查询复杂度矩阵: {det_range} × {ret_ftr_range} = {complexity}

因此，该功能的复杂度等级为: {complexity}
            """.strip()
        }
        
        return explanation
    
    async def validate_complexity_result(
        self,
        complexity_result: NESMAComplexityResult,
        function_description: str
    ) -> Dict[str, Any]:
        """验证复杂度计算结果"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "validation_issues": [],
            "suggestions": []
        }
        
        # 验证DET计数合理性
        det_issues = self._validate_det_count(
            complexity_result.function_type,
            complexity_result.det_count,
            function_description
        )
        validation_result["validation_issues"].extend(det_issues)
        
        # 验证RET/FTR计数合理性
        ret_ftr_issues = self._validate_ret_ftr_count(
            complexity_result.function_type,
            complexity_result.ret_count,
            function_description
        )
        validation_result["validation_issues"].extend(ret_ftr_issues)
        
        # 验证复杂度等级一致性
        expected_complexity = self._determine_complexity_level(
            complexity_result.function_type,
            complexity_result.det_count,
            complexity_result.ret_count
        )
        
        if expected_complexity != complexity_result.complexity:
            validation_result["validation_issues"].append({
                "type": "complexity_mismatch",
                "message": f"复杂度等级不匹配，期望 {expected_complexity}，实际 {complexity_result.complexity}"
            })
        
        # 计算整体验证分数
        if validation_result["validation_issues"]:
            validation_result["is_valid"] = False
            validation_result["confidence_score"] = max(0.1, 
                1.0 - len(validation_result["validation_issues"]) * 0.2
            )
        
        # 生成改进建议
        if not validation_result["is_valid"]:
            validation_result["suggestions"] = self._generate_complexity_suggestions(
                complexity_result, validation_result["validation_issues"]
            )
        
        return validation_result
    
    async def _retrieve_complexity_rules(
        self, 
        function_type: NESMAFunctionType, 
        function_description: str
    ) -> List[str]:
        """检索相关的复杂度计算规则"""
        
        if not self.rule_retriever:
            return []
        
        # 构建查询
        query = f"NESMA {function_type} complexity calculation DET RET FTR rules {function_description}"
        
        try:
            # 检索相关规则
            retrieved_rules = await self.rule_retriever.execute_task(
                "retrieve_rules",
                {
                    "query": query,
                    "source_type": "NESMA",
                    "max_results": 5
                }
            )
            
            return retrieved_rules.get("rules", [])
            
        except Exception as e:
            logger.warning(f"检索复杂度规则失败: {e}")
            return []
    
    async def _calculate_det_count(
        self,
        function_type: NESMAFunctionType,
        function_description: str,
        detailed_data: Optional[Dict[str, Any]]
    ) -> int:
        """计算DET数量"""
        
        # 如果提供了详细数据，直接使用
        if detailed_data and "det_count" in detailed_data:
            return detailed_data["det_count"]
        
        # 使用LLM分析功能描述
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是NESMA复杂度计算专家，需要计算{function_type}功能的DET数量。

DET计算规则（{function_type}）：
{chr(10).join(self.complexity_rules[function_type]["DET_识别规则"])}

请仔细分析功能描述，识别所有符合条件的数据元素。

返回JSON格式：
{{
  "det_count": 数量,
  "identified_dets": ["DET1", "DET2", ...],
  "reasoning": "计算理由"
}}"""),
            ("human", f"功能描述：{function_description}\n\n请计算DET数量。")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # 解析DET计算结果
        det_result = await self._parse_det_calculation(response.content)
        
        return det_result.get("det_count", 5)  # 默认保守估计
    
    async def _calculate_ret_ftr_count(
        self,
        function_type: NESMAFunctionType,
        function_description: str,
        detailed_data: Optional[Dict[str, Any]]
    ) -> int:
        """计算RET/FTR数量"""
        
        # 如果提供了详细数据，直接使用
        count_key = "ret_count" if function_type in ["ILF", "EIF"] else "ftr_count"
        if detailed_data and count_key in detailed_data:
            return detailed_data[count_key]
        
        # 使用LLM分析功能描述
        count_type = "RET" if function_type in ["ILF", "EIF"] else "FTR"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是NESMA复杂度计算专家，需要计算{function_type}功能的{count_type}数量。

{count_type}计算规则（{function_type}）：
{chr(10).join(self.complexity_rules[function_type][f"{count_type}_识别规则"])}

请仔细分析功能描述，识别所有符合条件的{count_type}。

返回JSON格式：
{{
  "{count_type.lower()}_count": 数量,
  "identified_{count_type.lower()}s": ["{count_type}1", "{count_type}2", ...],
  "reasoning": "计算理由"
}}"""),
            ("human", f"功能描述：{function_description}\n\n请计算{count_type}数量。")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # 解析RET/FTR计算结果
        ret_ftr_result = await self._parse_ret_ftr_calculation(response.content, count_type)
        
        return ret_ftr_result.get(f"{count_type.lower()}_count", 2)  # 默认保守估计
    
    def _determine_complexity_level(
        self,
        function_type: NESMAFunctionType,
        det_count: int,
        ret_ftr_count: int
    ) -> ComplexityLevel:
        """根据DET和RET/FTR数量确定复杂度等级"""
        
        matrix = self.complexity_rules[function_type]["complexity_matrix"]
        
        # 确定DET范围
        det_range = self._get_det_range(det_count)
        
        # 确定RET/FTR范围
        ret_ftr_range = self._get_ret_ftr_range(function_type, ret_ftr_count)
        
        # 查询复杂度矩阵
        complexity_str = matrix[det_range][ret_ftr_range]
        
        # 转换为枚举
        if complexity_str == "Low":
            return ComplexityLevel.LOW
        elif complexity_str == "Average":
            return ComplexityLevel.AVERAGE
        else:
            return ComplexityLevel.HIGH
    
    def _get_det_range(self, det_count: int) -> str:
        """获取DET计数范围"""
        if det_count <= 14:
            return "DET_1-14"
        elif det_count <= 19:
            return "DET_1-19"
        elif det_count <= 25:
            return "DET_15-25" if det_count >= 15 else "DET_20-25"
        elif det_count <= 50:
            return "DET_20-50" if det_count >= 20 else "DET_26+"
        else:
            return "DET_51+"
    
    def _get_ret_ftr_range(self, function_type: NESMAFunctionType, count: int) -> str:
        """获取RET/FTR计数范围"""
        if function_type in ["ILF", "EIF"]:
            # RET范围
            if count == 1:
                return "RET_1"
            elif count <= 5:
                return "RET_2-5"
            else:
                return "RET_6+"
        else:
            # FTR范围
            if count <= 1:
                return "FTR_0-1"
            elif count <= 2:
                return "FTR_2"
            elif count <= 3:
                return "FTR_2-3"
            elif count <= 4:
                return "FTR_4+"
            else:
                return "FTR_4+"
    
    def _calculate_complexity_confidence(
        self,
        det_count: int,
        ret_ftr_count: int,
        detailed_data: Optional[Dict[str, Any]]
    ) -> float:
        """计算复杂度计算的置信度"""
        
        confidence = 0.7  # 基础置信度
        
        # 如果有详细数据，提高置信度
        if detailed_data:
            if "det_count" in detailed_data:
                confidence += 0.15
            if any(key in detailed_data for key in ["ret_count", "ftr_count"]):
                confidence += 0.15
        
        # 根据计数的合理性调整置信度
        if det_count > 0 and ret_ftr_count > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_fallback_complexity(
        self, 
        classification: NESMAFunctionClassification
    ) -> NESMAComplexityResult:
        """创建保守的复杂度估算"""
        
        # 使用保守的中等复杂度
        return NESMAComplexityResult(
            function_id=classification.function_id,
            complexity=ComplexityLevel.AVERAGE,
            det_count=10,  # 保守估计
            ret_count=2,   # 保守估计
            calculation_details={
                "fallback_calculation": True,
                "reason": "自动计算失败，使用保守估计",
                "confidence_score": 0.5
            }
        )
    
    async def _parse_det_calculation(self, response_content: str) -> Dict[str, Any]:
        """解析DET计算结果"""
        try:
            import json
            # 尝试提取JSON
            if "{" in response_content and "}" in response_content:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_str = response_content[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析DET计算结果失败: {e}")
        
        # 回退方案：简单数字提取
        import re
        numbers = re.findall(r'\d+', response_content)
        if numbers:
            return {"det_count": int(numbers[0])}
        
        return {"det_count": 5}
    
    async def _parse_ret_ftr_calculation(
        self, 
        response_content: str, 
        count_type: str
    ) -> Dict[str, Any]:
        """解析RET/FTR计算结果"""
        try:
            import json
            if "{" in response_content and "}" in response_content:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_str = response_content[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析{count_type}计算结果失败: {e}")
        
        # 回退方案：简单数字提取
        import re
        numbers = re.findall(r'\d+', response_content)
        if numbers:
            return {f"{count_type.lower()}_count": int(numbers[0])}
        
        return {f"{count_type.lower()}_count": 2}
    
    def _validate_det_count(
        self,
        function_type: NESMAFunctionType,
        det_count: int,
        function_description: str
    ) -> List[Dict[str, str]]:
        """验证DET计数的合理性"""
        issues = []
        
        # 检查DET计数范围
        if det_count <= 0:
            issues.append({
                "type": "invalid_det_count",
                "message": "DET计数不能为0或负数"
            })
        elif det_count > 100:
            issues.append({
                "type": "high_det_count", 
                "message": f"DET计数过高 ({det_count})，请确认是否正确"
            })
        
        return issues
    
    def _validate_ret_ftr_count(
        self,
        function_type: NESMAFunctionType,
        ret_ftr_count: int,
        function_description: str
    ) -> List[Dict[str, str]]:
        """验证RET/FTR计数的合理性"""
        issues = []
        
        count_type = "RET" if function_type in ["ILF", "EIF"] else "FTR"
        
        # 检查计数范围
        if ret_ftr_count <= 0:
            issues.append({
                "type": f"invalid_{count_type.lower()}_count",
                "message": f"{count_type}计数不能为0或负数"
            })
        elif ret_ftr_count > 20:
            issues.append({
                "type": f"high_{count_type.lower()}_count",
                "message": f"{count_type}计数过高 ({ret_ftr_count})，请确认是否正确"
            })
        
        return issues
    
    def _generate_complexity_suggestions(
        self,
        complexity_result: NESMAComplexityResult,
        validation_issues: List[Dict[str, str]]
    ) -> List[str]:
        """生成复杂度计算改进建议"""
        suggestions = []
        
        for issue in validation_issues:
            if "det_count" in issue["type"]:
                suggestions.append("建议重新检查DET识别，确保符合NESMA标准")
            elif "ret_count" in issue["type"] or "ftr_count" in issue["type"]:
                suggestions.append("建议重新检查RET/FTR识别，确认文件引用关系")
            elif "complexity_mismatch" in issue["type"]:
                suggestions.append("建议重新计算复杂度矩阵查找")
        
        if not suggestions:
            suggestions.append("复杂度计算验证通过，无需调整")
        
        return suggestions
    
    def get_calculation_history(self) -> List[NESMAComplexityResult]:
        """获取计算历史"""
        return self.calculation_history.copy()
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        if not self.calculation_history:
            return {"total_calculations": 0}
        
        complexity_counts = {}
        total_det = 0
        total_ret = 0
        
        for result in self.calculation_history:
            complexity = result.complexity.value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            total_det += result.det_count
            total_ret += result.ret_count
        
        return {
            "total_calculations": len(self.calculation_history),
            "complexity_distribution": complexity_counts,
            "average_det_count": total_det / len(self.calculation_history),
            "average_ret_count": total_ret / len(self.calculation_history),
            "last_calculation_time": self.calculation_history[-1].calculation_details.get("timestamp")
        }


async def create_nesma_complexity_calculator(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> NESMAComplexityCalculatorAgent:
    """创建NESMA复杂度计算器智能体"""
    return NESMAComplexityCalculatorAgent(rule_retriever=rule_retriever, llm=llm)


if __name__ == "__main__":
    async def main():
        # 测试NESMA复杂度计算器
        calculator = await create_nesma_complexity_calculator()
        
        # 测试分类
        test_classification = NESMAFunctionClassification(
            function_id="test_func_001",
            function_type=NESMAFunctionType.ILF,
            confidence_score=0.9,
            justification="测试内部逻辑文件",
            rules_applied=["ILF识别规则"]
        )
        
        # 计算复杂度
        complexity_result = await calculator.calculate_complexity(
            test_classification,
            "用户管理表，包含用户ID、姓名、邮箱、电话、部门、创建时间等字段",
            {"det_count": 8, "ret_count": 1}
        )
        
        print(f"复杂度计算结果: {complexity_result}")
        
    asyncio.run(main()) 