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
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.base.base_agent import SpecializedAgent
from agents.knowledge.rule_retriever import RuleRetrieverAgent
from models.nesma_models import (
    NESMAFunctionType, NESMAFunctionClassification,
    NESMAComplexityLevel, NESMAComplexityCalculation
)
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Pydantic输出模型
class DETCalculationResult(BaseModel):
    """DET计算结果模型"""
    det_count: int = Field(description="数据元素类型数量", ge=0)
    identified_dets: List[str] = Field(description="识别到的数据元素列表")
    reasoning: str = Field(description="计算推理过程")

class RETFTRCalculationResult(BaseModel):
    """RET/FTR计算结果模型"""
    count: int = Field(description="记录元素类型或文件类型引用数量", ge=0)
    identified_items: List[str] = Field(description="识别到的项目列表")
    reasoning: str = Field(description="计算推理过程")

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
        self.calculation_history: List[NESMAComplexityCalculation] = []
        
    def _load_complexity_rules(self) -> Dict[str, Any]:
        """加载NESMA复杂度计算规则"""
        return {
            "ILF": {
                "complexity_matrix": {
                    "DET_1-14": {
                        "RET_1": "Low",
                        "RET_2-5": "Low", 
                        "RET_6+": "Average"
                    },
                    "DET_15-25": {
                        "RET_1": "Low",
                        "RET_2-5": "Average",
                        "RET_6+": "High"
                    },
                    "DET_26+": {
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
                    "DET_1-14": {
                        "RET_1": "Low",
                        "RET_2-5": "Low",
                        "RET_6+": "Average"
                    },
                    "DET_15-25": {
                        "RET_1": "Low", 
                        "RET_2-5": "Average",
                        "RET_6+": "High"
                    },
                    "DET_26+": {
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
            result = await self.calculate_complexity(
                inputs["classification"],
                inputs["function_description"],
                inputs.get("detailed_data", {})
            )
            # 将 NESMAComplexityCalculation 对象转换为字典
            return {
                "function_id": result.function_id,
                "function_type": result.function_type.value,
                "det_count": result.det_count,
                "ret_count": result.ret_count,
                "complexity": result.complexity.value,
                "complexity_matrix_used": result.complexity_matrix_used,
                "calculation_steps": result.calculation_steps,
                "calculation_details": result.calculation_details
            }
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
    ) -> NESMAComplexityCalculation:
        """计算功能复杂度"""
        
        logger.info(f"🔍 开始计算复杂度:")
        logger.info(f"  - 功能ID: {classification.function_id}")
        logger.info(f"  - 功能类型: {classification.function_type}")
        logger.info(f"  - 功能名称: {getattr(classification, 'function_name', '未知')}")
        logger.info(f"  - 功能描述: {function_description}")
        logger.info(f"  - 详细数据: {detailed_data}")
        
        try:
            # 1. 获取相关NESMA复杂度规则
            logger.info("📋 获取NESMA复杂度规则...")
            complexity_rules = await self._retrieve_complexity_rules(
                classification.function_type, 
                function_description
            )
            logger.info(f"✅ 获取到 {len(complexity_rules)} 条规则")
            
            # 2. 计算DET数量
            logger.info("🔢 开始计算DET数量...")
            det_count = await self._calculate_det_count(
                classification.function_type,
                function_description,
                detailed_data
            )
            logger.info(f"✅ DET计算完成: {det_count}")
            
            # 3. 计算RET/FTR数量
            count_type = "RET" if classification.function_type in ["ILF", "EIF"] else "FTR"
            logger.info(f"🔢 开始计算{count_type}数量...")
            ret_ftr_count = await self._calculate_ret_ftr_count(
                classification.function_type,
                function_description,
                detailed_data
            )
            logger.info(f"✅ {count_type}计算完成: {ret_ftr_count}")
            
            # 4. 确定复杂度等级
            logger.info("📊 确定复杂度等级...")
            complexity_level = self._determine_complexity_level(
                classification.function_type,
                det_count,
                ret_ftr_count
            )
            logger.info(f"✅ 复杂度确定: {complexity_level}")
            
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
            complexity_result = NESMAComplexityCalculation(
                function_id=classification.function_id,
                function_type=classification.function_type,
                det_count=det_count,
                ret_count=ret_ftr_count if classification.function_type in ["ILF", "EIF"] else 0,
                complexity=complexity_level,
                complexity_matrix_used=f"{classification.function_type}_complexity_matrix",
                calculation_steps=[
                    f"1. 分析功能类型: {classification.function_type}",
                    f"2. 计算DET数量: {det_count}",
                    f"3. 计算{'RET' if classification.function_type in ['ILF', 'EIF'] else 'FTR'}数量: {ret_ftr_count}",
                    f"4. 查询复杂度矩阵确定等级: {complexity_level}"
                ],
                calculation_details=calculation_details
            )
            
            # 7. 记录计算历史
            self.calculation_history.append(complexity_result)
            
            logger.info(f"🎉 复杂度计算完成: {complexity_result.function_id} -> {complexity_level} (DET:{det_count}, {count_type}:{ret_ftr_count})")
            
            return complexity_result
            
        except Exception as e:
            logger.error(f"❌ 复杂度计算出现异常: {e}")
            logger.error(f"异常类型: {type(e).__name__}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            
            # 返回保守估算
            logger.warning("⚠️ 使用保守估算替代")
            return self._create_fallback_complexity(classification)
    
    async def calculate_batch_complexity(
        self, 
        classifications: List[NESMAFunctionClassification]
    ) -> List[NESMAComplexityCalculation]:
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
        complexity_result: NESMAComplexityCalculation,
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
            logger.warning("⚠️ rule_retriever 未初始化，无法检索规则")
            return []
        
        # 构建查询
        query = f"NESMA {function_type} complexity calculation DET RET FTR rules {function_description}"
        
        # 🔥 记录查询详情
        logger.info(f"📋 获取NESMA复杂度规则...")
        logger.info(f"🔍 规则检索参数:")
        logger.info(f"  - 功能类型: {function_type}")
        logger.info(f"  - 功能描述: {function_description}")
        logger.info(f"  - 构建的查询: {query}")
        
        try:
            # 检索相关规则
            from models.common_models import EstimationStandard
            knowledge_result = await self.rule_retriever.retrieve_rules(
                query=query,
                standard=EstimationStandard.NESMA,
                min_chunks=3
            )
            
            # 🔥 记录检索结果详情
            logger.info(f"📊 检索结果分析:")
            logger.info(f"  - 查询状态: {'成功' if knowledge_result else '失败'}")
            if knowledge_result:
                logger.info(f"  - 返回块数: {knowledge_result.total_chunks}")
                logger.info(f"  - 检索耗时: {knowledge_result.processing_time_ms}ms")
                logger.info(f"  - 数据类型: {type(knowledge_result.retrieved_chunks)}")
                
                if knowledge_result.retrieved_chunks:
                    logger.info(f"  - 块列表长度: {len(knowledge_result.retrieved_chunks)}")
                    for i, chunk in enumerate(knowledge_result.retrieved_chunks):
                        logger.info(f"    块{i+1}: ID={chunk.get('chunk_id', 'unknown')}, 长度={len(chunk.get('content', ''))}")
                        logger.info(f"          相关性={chunk.get('relevance_score', 0):.3f}")
                        logger.info(f"          内容预览: {chunk.get('content', '')[:100]}...")
            
            # 提取规则文本
            rules = []
            if knowledge_result and knowledge_result.retrieved_chunks:
                for chunk in knowledge_result.retrieved_chunks:
                    content = chunk.get('content', '')
                    if content:
                        rules.append(content)
                        logger.info(f"✅ 添加规则: 长度={len(content)}")
            
            logger.info(f"✅ 获取到 {len(rules)} 条规则")
            
            # 🔥 如果没有规则，记录详细信息用于调试
            if not rules:
                logger.warning("⚠️ 没有获取到任何规则，调试信息:")
                logger.warning(f"  - rule_retriever存在: {self.rule_retriever is not None}")
                logger.warning(f"  - knowledge_result存在: {knowledge_result is not None}")
                if knowledge_result:
                    logger.warning(f"  - retrieved_chunks存在: {knowledge_result.retrieved_chunks is not None}")
                    logger.warning(f"  - retrieved_chunks类型: {type(knowledge_result.retrieved_chunks)}")
                    logger.warning(f"  - retrieved_chunks内容: {knowledge_result.retrieved_chunks}")
            
            return rules
            
        except Exception as e:
            logger.error(f"❌ 检索复杂度规则失败: {str(e)}")
            import traceback
            logger.error(f"❌ 异常堆栈: {traceback.format_exc()}")
            return []
    
    async def _calculate_det_count(
        self,
        function_type: NESMAFunctionType,
        function_description: str,
        detailed_data: Optional[Dict[str, Any]]
    ) -> int:
        """计算DET数量"""
        
        # 定义DET计算工具
        @tool
        def calculate_det_count(
            det_count: int, 
            identified_dets: List[str], 
            reasoning: str
        ) -> dict:
            """计算NESMA功能的DET数量
            
            Args:
                det_count: 数据元素类型数量，必须大于等于0
                identified_dets: 识别到的数据元素列表
                reasoning: 计算推理过程
            """
            return {
                "det_count": det_count,
                "identified_dets": identified_dets,
                "reasoning": reasoning
            }
        
        # 创建带工具的LLM
        llm_with_tools = self.llm.bind_tools([calculate_det_count])
        
        # 获取相关规则
        rules = await self._retrieve_complexity_rules(function_type, function_description)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是NESMA复杂度计算专家，专门计算{function_type.value}类型功能的DET数量。

DET (Data Element Type) 识别规则：
{chr(10).join(rules[:3])}

请仔细分析功能描述，识别所有数据元素，并使用calculate_det_count工具返回结果。"""),
            ("human", """功能描述：{function_description}

详细数据：{detailed_data}

请计算此功能的DET数量。""")
        ])
        
        messages = prompt.format_messages(
            function_description=function_description,
            detailed_data=str(detailed_data) if detailed_data else "无详细数据"
        )
        
        try:
            response = await llm_with_tools.ainvoke(messages)
            
            # 解析工具调用结果
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                result_data = tool_call["args"]
                det_count = result_data.get("det_count", 5)
                
                logger.info(f"✅ DET计算成功: {det_count}")
                logger.info(f"   识别的DET: {result_data.get('identified_dets', [])}")
                
                return max(1, det_count)  # 确保至少为1
            else:
                logger.warning("LLM未使用工具调用，使用默认值")
                return 5
                
        except Exception as e:
            logger.error(f"❌ DET计算失败: {str(e)}")
            return 5  # 默认值
    
    async def _calculate_ret_ftr_count(
        self,
        function_type: NESMAFunctionType,
        function_description: str,
        detailed_data: Optional[Dict[str, Any]]
    ) -> int:
        """计算RET/FTR数量"""
        
        count_type = "RET" if function_type in ["ILF", "EIF"] else "FTR"
        
        # 定义RET/FTR计算工具
        @tool
        def calculate_ret_ftr_count(
            count: int, 
            identified_items: List[str], 
            reasoning: str
        ) -> dict:
            """计算NESMA功能的RET/FTR数量
            
            Args:
                count: RET/FTR数量，必须大于等于0
                identified_items: 识别到的项目列表
                reasoning: 计算推理过程
            """
            return {
                "count": count,
                "identified_items": identified_items,
                "reasoning": reasoning
            }
        
        # 创建带工具的LLM
        llm_with_tools = self.llm.bind_tools([calculate_ret_ftr_count])
        
        # 获取相关规则
        rules = await self._retrieve_complexity_rules(function_type, function_description)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是NESMA复杂度计算专家，专门计算{function_type.value}类型功能的{count_type}数量。

{count_type} 识别规则：
{chr(10).join(rules[3:6] if len(rules) > 3 else rules)}

请仔细分析功能描述，识别所有{count_type}，并使用calculate_ret_ftr_count工具返回结果。"""),
            ("human", """功能描述：{function_description}

详细数据：{detailed_data}

请计算此功能的{count_type}数量。""")
        ])
        
        messages = prompt.format_messages(
            function_description=function_description,
            detailed_data=str(detailed_data) if detailed_data else "无详细数据",
            count_type=count_type
        )
        
        try:
            response = await llm_with_tools.ainvoke(messages)
            
            # 解析工具调用结果
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                result_data = tool_call["args"]
                count = result_data.get("count", 2)
                
                logger.info(f"✅ {count_type}计算成功: {count}")
                logger.info(f"   识别的{count_type}: {result_data.get('identified_items', [])}")
                
                return max(1, count)  # 确保至少为1
            else:
                logger.warning("LLM未使用工具调用，使用默认值")
                return 2
                
        except Exception as e:
            logger.error(f"❌ {count_type}计算失败: {str(e)}")
            return 2  # 默认值
    
    def _determine_complexity_level(
        self,
        function_type: NESMAFunctionType,
        det_count: int,
        ret_ftr_count: int
    ) -> NESMAComplexityLevel:
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
            return NESMAComplexityLevel.LOW
        elif complexity_str == "Average":
            return NESMAComplexityLevel.AVERAGE
        else:
            return NESMAComplexityLevel.HIGH
    
    def _get_det_range(self, det_count: int) -> str:
        """获取DET计数范围"""
        if det_count <= 14:
            return "DET_1-14"
        elif det_count <= 25:
            return "DET_15-25"
        else:
            return "DET_26+"
    
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
            # FTR范围 - 根据NESMA复杂度矩阵定义
            if count <= 1:
                return "FTR_0-1"
            elif count == 2:
                return "FTR_2"
            elif count >= 3:
                return "FTR_3+"
            else:
                return "FTR_0-1"
    
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
    ) -> NESMAComplexityCalculation:
        """创建保守的复杂度估算"""
        
        # 使用保守的中等复杂度
        return NESMAComplexityCalculation(
            function_id=classification.function_id,
            function_type=classification.function_type,
            det_count=10,  # 保守估计
            ret_count=2,   # 保守估计
            complexity=NESMAComplexityLevel.AVERAGE,
            complexity_matrix_used=f"{classification.function_type}_fallback_matrix",
            calculation_steps=[
                "1. 自动计算失败",
                "2. 使用保守估计",
                "3. 复杂度设为Average"
            ],
            calculation_details={
                "fallback_calculation": True,
                "reason": "自动计算失败，使用保守估计",
                "confidence_score": 0.5
            }
        )
    
    async def _parse_det_calculation(self, response_content: str) -> Dict[str, Any]:
        """解析DET计算结果 - 已废弃，保留用于兼容性"""
        logger.warning("使用已废弃的JSON解析方法，建议使用工具调用")
        return {"det_count": 5, "reasoning": "使用默认值"}
    
    async def _parse_ret_ftr_calculation(
        self, 
        response_content: str, 
        count_type: str
    ) -> Dict[str, Any]:
        """解析RET/FTR计算结果 - 已废弃，保留用于兼容性"""
        logger.warning("使用已废弃的JSON解析方法，建议使用工具调用")
        return {f"{count_type.lower()}_count": 2, "reasoning": "使用默认值"}
    
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
        complexity_result: NESMAComplexityCalculation,
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
    
    def get_calculation_history(self) -> List[NESMAComplexityCalculation]:
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