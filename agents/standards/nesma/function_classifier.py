"""
量子智能化功能点估算系统 - NESMA功能分类器智能体

基于NESMA v2.3+标准进行功能类型智能分类
"""

import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from agents.base.base_agent import SpecializedAgent
from agents.knowledge.rule_retriever import RuleRetrieverAgent
from models.nesma_models import (
    NESMAFunctionType, NESMAFunctionClassification
)
from models.project_models import ProcessDetails
from models.common_models import ConfidenceLevel, ValidationResult
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

# 结构化输出模型
class FunctionClassificationResult(BaseModel):
    """功能分类结果的结构化输出模型"""
    function_type: str = Field(description="功能类型：ILF、EIF、EI、EO、EQ之一")
    confidence_score: float = Field(description="置信度分数，0.0-1.0之间", ge=0.0, le=1.0)
    justification: str = Field(description="详细的分类理由")
    key_indicators: List[str] = Field(description="关键指标列表")
    rules_applied: List[str] = Field(description="应用的规则列表")


class NESMAFunctionClassifierAgent(SpecializedAgent):
    """NESMA功能分类器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="nesma_function_classifier",
            specialty="nesma_function_classification",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # NESMA分类规则数据库
        self.classification_rules = self._load_classification_rules()
        self.classification_history: List[NESMAFunctionClassification] = []
        
    def _load_classification_rules(self) -> Dict[str, Any]:
        """加载NESMA分类规则"""
        return {
            "ILF": {
                "定义": "内部逻辑文件：应用内维护的一组逻辑相关的数据",
                "关键特征": [
                    "由应用内部维护",
                    "通过外部输入进行更新",
                    "包含多个数据元素",
                    "有主键标识"
                ],
                "识别词汇": [
                    "存储", "维护", "管理", "表", "数据库", "记录",
                    "主数据", "基础数据", "配置数据", "用户信息"
                ],
                "排除条件": [
                    "仅供查询不能更新",
                    "来自外部系统", 
                    "临时文件",
                    "日志文件"
                ]
            },
            "EIF": {
                "定义": "外部接口文件：被另一个应用维护的一组逻辑相关的数据",
                "关键特征": [
                    "由外部应用维护",
                    "仅供本应用引用",
                    "不能由本应用更新",
                    "跨系统数据"
                ],
                "识别词汇": [
                    "外部", "接口", "引用", "查询", "第三方",
                    "其他系统", "共享数据", "同步数据"
                ],
                "排除条件": [
                    "可以更新",
                    "应用内部数据",
                    "临时接口"
                ]
            },
            "EI": {
                "定义": "外部输入：从应用边界外处理数据或控制信息的功能",
                "关键特征": [
                    "数据从外部进入",
                    "更新ILF",
                    "维护数据完整性",
                    "包含业务逻辑"
                ],
                "识别词汇": [
                    "输入", "新增", "添加", "创建", "录入", "导入",
                    "更新", "修改", "编辑", "维护", "保存"
                ],
                "排除条件": [
                    "仅查询功能",
                    "仅输出功能",
                    "重复数据处理"
                ]
            },
            "EO": {
                "定义": "外部输出：向应用边界外发送数据或控制信息的功能",
                "关键特征": [
                    "包含派生数据",
                    "有计算逻辑",
                    "创建数据供外部使用",
                    "含有处理逻辑"
                ],
                "识别词汇": [
                    "报表", "统计", "分析", "计算", "汇总", "生成",
                    "导出", "打印", "发送", "传输"
                ],
                "排除条件": [
                    "仅检索功能",
                    "无计算逻辑",
                    "重复查询"
                ]
            },
            "EQ": {
                "定义": "外部查询：从应用边界外检索数据的功能",
                "关键特征": [
                    "检索数据展示",
                    "不更新ILF",
                    "输入输出结合",
                    "简单检索逻辑"
                ],
                "识别词汇": [
                    "查询", "检索", "搜索", "查找", "浏览", "显示",
                    "列表", "详情", "信息", "状态"
                ],
                "排除条件": [
                    "更新功能",
                    "复杂计算",
                    "派生数据"
                ]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "NESMA功能类型识别",
            "分类规则匹配",
            "置信度评估",
            "分类理由生成",
            "批量功能分类"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行NESMA分类任务"""
        if task_name == "classify_function":
            return await self.classify_function(
                inputs["function_description"],
                inputs.get("process_details", None)
            )
        elif task_name == "classify_batch_functions":
            return await self.classify_batch_functions(inputs["functions"])
        elif task_name == "validate_classification":
            return await self.validate_classification(
                inputs["classification"],
                inputs["function_description"]
            )
        elif task_name == "explain_classification":
            return await self.explain_classification(
                inputs["function_type"],
                inputs["function_description"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def classify_function(
        self,
        function_description: str,
        process_details: Optional[ProcessDetails] = None
    ) -> NESMAFunctionClassification:
        """对单个功能进行NESMA分类"""
        
        # 1. 获取相关NESMA规则
        nesma_rules = await self._retrieve_nesma_rules(function_description)
        
        # 2. 使用LLM进行分类
        classification = await self._llm_classify_function(
            function_description, 
            process_details, 
            nesma_rules
        )
        
        # 3. 验证分类结果
        validated_classification = await self._validate_classification_result(
            classification, 
            function_description
        )
        
        # 4. 记录分类历史
        self.classification_history.append(validated_classification)
        
        return validated_classification
    
    async def classify_batch_functions(
        self, 
        functions: List[Dict[str, Any]]
    ) -> List[NESMAFunctionClassification]:
        """批量功能分类"""
        
        classifications = []
        
        for func_info in functions:
            try:
                classification = await self.classify_function(
                    func_info["description"],
                    func_info.get("process_details", None)
                )
                classifications.append(classification)
            except Exception as e:
                logger.error(f"功能分类失败: {func_info.get('id', 'unknown')}, 错误: {str(e)}")
                # 创建失败的分类记录
                failed_classification = NESMAFunctionClassification(
                    function_id=func_info.get("id", "unknown"),
                    function_name=func_info.get("name", "未知功能"),  # 添加function_name
                    function_description=func_info.get("description", ""),  # 添加function_description
                    function_type=NESMAFunctionType.EI,  # 默认类型
                    confidence_score=0.0,
                    justification=f"分类失败: {str(e)}",
                    rules_applied=[]
                )
                classifications.append(failed_classification)
        
        return classifications
    
    async def validate_classification(
        self,
        classification: NESMAFunctionClassification,
        function_description: str
    ) -> Dict[str, Any]:
        """验证分类结果的合理性"""
        
        # 获取规则匹配度
        rule_match = self._check_rule_match(
            classification.function_type, 
            function_description
        )
        
        # 检查常见错误
        potential_errors = self._check_common_errors(
            classification.function_type,
            function_description
        )
        
        # 计算总体验证分数
        validation_score = self._calculate_validation_score(
            classification.confidence_score,
            rule_match["score"],
            len(potential_errors)
        )
        
        return {
            "is_valid": validation_score > 0.7,
            "validation_score": validation_score,
            "rule_match": rule_match,
            "potential_errors": potential_errors,
            "suggestions": self._generate_validation_suggestions(
                classification, 
                potential_errors
            )
        }
    
    async def explain_classification(
        self,
        function_type: NESMAFunctionType,
        function_description: str
    ) -> Dict[str, Any]:
        """解释分类原因和依据"""
        
        rules = self.classification_rules[function_type.value]
        
        # 分析功能描述中的关键词匹配
        keyword_matches = self._analyze_keyword_matches(
            function_description, 
            rules["识别词汇"]
        )
        
        # 检查排除条件
        exclusion_checks = self._check_exclusion_conditions(
            function_description,
            rules["排除条件"]
        )
        
        return {
            "function_type": function_type.value,
            "definition": rules["定义"],
            "key_characteristics": rules["关键特征"],
            "keyword_matches": keyword_matches,
            "exclusion_checks": exclusion_checks,
            "classification_confidence": self._calculate_explanation_confidence(
                keyword_matches, 
                exclusion_checks
            )
        }
    
    async def _retrieve_nesma_rules(self, function_description: str) -> List[str]:
        """检索相关的NESMA分类规则"""
        if not self.rule_retriever:
            return []
        
        try:
            # 构建查询
            query = f"NESMA 功能分类 {function_description[:100]}"
            
            result = await self.rule_retriever.retrieve_rules(
                query=query,
                standard="NESMA",
                min_chunks=3
            )
            
            if result and result.retrieved_chunks:
                # 处理不同类型的chunk格式（字典或对象）
                def get_chunk_content(chunk):
                    if isinstance(chunk, dict):
                        return chunk.get('content', '')
                    else:
                        return getattr(chunk, 'content', '')
                
                return [get_chunk_content(chunk) for chunk in result.retrieved_chunks]
            
        except Exception as e:
            logger.warning(f"NESMA规则检索失败: {str(e)}")
        
        return []
    
    async def _llm_classify_function(
        self,
        function_description: str,
        process_details: Optional[ProcessDetails],
        nesma_rules: List[str]
    ) -> NESMAFunctionClassification:
        """使用LLM进行功能分类"""
        
        # 定义分类工具
        @tool
        def classify_nesma_function(
            function_type: str,
            confidence_score: float,
            justification: str,
            key_indicators: List[str],
            rules_applied: List[str]
        ) -> dict:
            """对功能进行NESMA分类
            
            Args:
                function_type: 功能类型，必须是ILF、EIF、EI、EO、EQ之一
                confidence_score: 置信度分数，0.0-1.0之间
                justification: 详细的分类理由
                key_indicators: 关键指标列表
                rules_applied: 应用的规则列表
            """
            return {
                "function_type": function_type,
                "confidence_score": confidence_score,
                "justification": justification,
                "key_indicators": key_indicators,
                "rules_applied": rules_applied
            }
        
        # 创建带工具的LLM
        llm_with_tools = self.llm.bind_tools([classify_nesma_function])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是NESMA功能点分类专家，需要将功能分类为以下五种类型之一：

ILF (Internal Logical File): 内部逻辑文件 - 应用内维护的数据
EIF (External Interface File): 外部接口文件 - 外部应用维护的数据  
EI (External Input): 外部输入 - 处理输入数据并更新内部文件
EO (External Output): 外部输出 - 向外部发送处理后的数据
EQ (External Query): 外部查询 - 检索数据进行展示

分类原则：
1. 关注数据的来源和去向
2. 判断是否包含业务逻辑处理
3. 区分数据维护和数据检索
4. 考虑跨应用边界的特征

请使用classify_nesma_function工具返回分类结果。"""),
            ("human", """功能描述：{function_description}

流程上下文：{process_context}

NESMA规则参考：
{nesma_rules}

请对此功能进行NESMA分类。""")
        ])
        
        process_context = ""
        if process_details:
            process_context = f"流程名称: {process_details.name}\n流程描述: {process_details.description}\n数据组: {process_details.data_groups}"
        
        response = await llm_with_tools.ainvoke(
            prompt.format_messages(
                function_description=function_description,
                process_context=process_context,
                nesma_rules="\n".join(nesma_rules[:5])  # 限制规则数量
            )
        )
        
        # 解析工具调用结果
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            result_data = tool_call["args"]
            
            # 验证功能类型
            function_type_str = result_data.get("function_type", "EI")
            try:
                function_type = NESMAFunctionType(function_type_str)
            except ValueError:
                logger.warning(f"无效的功能类型: {function_type_str}，使用默认值EI")
                function_type = NESMAFunctionType.EI
            
            return NESMAFunctionClassification(
                function_id="auto_generated",
                function_name=f"{function_type.value}功能",
                function_description=function_description,
                function_type=function_type,
                confidence_score=result_data.get("confidence_score", 0.7),
                justification=result_data.get("justification", "LLM工具调用分类"),
                rules_applied=result_data.get("rules_applied", ["NESMA基础规则"]),
                classification_confidence=result_data.get("confidence_score", 0.7)
            )
        else:
            # 如果没有工具调用，使用保守分类
            logger.warning("LLM未使用工具调用，使用保守分类")
            return await self._conservative_classify(function_description)
    
    async def _validate_classification_result(
        self,
        classification: NESMAFunctionClassification,
        function_description: str
    ) -> NESMAFunctionClassification:
        """验证并可能调整分类结果"""
        
        # 检查分类合理性
        validation = await self.validate_classification(
            classification, 
            function_description
        )
        
        # 如果验证失败且置信度较低，尝试重新分类
        if not validation["is_valid"] and classification.confidence_score < 0.7:
            logger.warning(f"分类验证失败，尝试重新分类: {function_description[:50]}...")
            
            # 使用更保守的分类策略
            adjusted_classification = await self._conservative_classify(
                function_description
            )
            return adjusted_classification
        
        return classification
    
    async def _conservative_classify(
        self, 
        function_description: str
    ) -> NESMAFunctionClassification:
        """保守的分类策略（当主要分类失败时使用）"""
        
        description_lower = function_description.lower()
        
        # 简单的关键词匹配
        if any(keyword in description_lower for keyword in ["新增", "添加", "创建", "录入", "输入"]):
            function_type = NESMAFunctionType.EI
            justification = "基于关键词匹配，识别为外部输入功能"
        elif any(keyword in description_lower for keyword in ["查询", "检索", "搜索", "显示", "列表"]):
            function_type = NESMAFunctionType.EQ
            justification = "基于关键词匹配，识别为外部查询功能"
        elif any(keyword in description_lower for keyword in ["报表", "统计", "导出", "计算"]):
            function_type = NESMAFunctionType.EO
            justification = "基于关键词匹配，识别为外部输出功能"
        elif any(keyword in description_lower for keyword in ["存储", "维护", "管理", "数据"]):
            function_type = NESMAFunctionType.ILF
            justification = "基于关键词匹配，识别为内部逻辑文件"
        else:
            function_type = NESMAFunctionType.EI  # 默认分类
            justification = "无法明确识别，默认分类为外部输入"
        
        return NESMAFunctionClassification(
            function_id="auto_generated",
            function_name=f"{function_type.value}功能",
            function_description=function_description,
            function_type=function_type,
            confidence_score=0.6,  # 保守的置信度
            justification=justification,
            rules_applied=["保守分类策略"]
        )
    
    def _check_rule_match(
        self, 
        function_type: NESMAFunctionType, 
        function_description: str
    ) -> Dict[str, Any]:
        """检查规则匹配情况"""
        
        rules = self.classification_rules[function_type.value]
        description_lower = function_description.lower()
        
        # 检查识别词汇匹配
        matched_keywords = [
            keyword for keyword in rules["识别词汇"] 
            if keyword in description_lower
        ]
        
        # 检查排除条件
        exclusion_violations = [
            condition for condition in rules["排除条件"]
            if any(word in description_lower for word in condition.split())
        ]
        
        # 计算匹配分数
        keyword_score = len(matched_keywords) / len(rules["识别词汇"])
        exclusion_penalty = len(exclusion_violations) * 0.2
        rule_score = max(0.0, keyword_score - exclusion_penalty)
        
        return {
            "score": rule_score,
            "matched_keywords": matched_keywords,
            "exclusion_violations": exclusion_violations,
            "keyword_match_rate": keyword_score
        }
    
    def _check_common_errors(
        self, 
        function_type: NESMAFunctionType, 
        function_description: str
    ) -> List[str]:
        """检查常见分类错误"""
        
        errors = []
        description_lower = function_description.lower()
        
        # ILF常见错误
        if function_type == NESMAFunctionType.ILF:
            if any(word in description_lower for word in ["查询", "显示", "列表"]):
                errors.append("ILF不应包含查询展示功能")
            if "外部" in description_lower:
                errors.append("ILF应为内部数据，不应涉及外部")
        
        # EIF常见错误
        elif function_type == NESMAFunctionType.EIF:
            if any(word in description_lower for word in ["更新", "修改", "维护"]):
                errors.append("EIF不能被本应用更新")
            if "内部" in description_lower:
                errors.append("EIF应为外部数据")
        
        # EI常见错误  
        elif function_type == NESMAFunctionType.EI:
            if "查询" in description_lower and "输入" not in description_lower:
                errors.append("纯查询功能应分类为EQ")
            if "报表" in description_lower or "统计" in description_lower:
                errors.append("报表统计功能应分类为EO")
        
        # EO常见错误
        elif function_type == NESMAFunctionType.EO:
            if "查询" in description_lower and "计算" not in description_lower:
                errors.append("简单查询应分类为EQ")
            if "输入" in description_lower:
                errors.append("输入功能应分类为EI")
        
        # EQ常见错误
        elif function_type == NESMAFunctionType.EQ:
            if any(word in description_lower for word in ["更新", "新增", "修改"]):
                errors.append("更新功能应分类为EI")
            if any(word in description_lower for word in ["计算", "统计", "报表"]):
                errors.append("计算功能应分类为EO")
        
        return errors
    
    def _calculate_validation_score(
        self, 
        confidence_score: float, 
        rule_match_score: float, 
        error_count: int
    ) -> float:
        """计算验证分数"""
        
        base_score = (confidence_score + rule_match_score) / 2
        error_penalty = error_count * 0.15
        
        return max(0.0, min(1.0, base_score - error_penalty))
    
    def _generate_validation_suggestions(
        self,
        classification: NESMAFunctionClassification,
        potential_errors: List[str]
    ) -> List[str]:
        """生成验证建议"""
        
        suggestions = []
        
        if classification.confidence_score < 0.7:
            suggestions.append("置信度较低，建议人工复核")
        
        if potential_errors:
            suggestions.append("存在潜在分类错误，建议检查：" + "; ".join(potential_errors))
        
        if not classification.rules_applied:
            suggestions.append("未明确应用分类规则，建议补充规则依据")
        
        if not suggestions:
            suggestions.append("分类结果看起来合理")
        
        return suggestions
    
    def _analyze_keyword_matches(
        self, 
        function_description: str, 
        keywords: List[str]
    ) -> Dict[str, Any]:
        """分析关键词匹配情况"""
        
        description_lower = function_description.lower()
        matched = []
        
        for keyword in keywords:
            if keyword in description_lower:
                matched.append(keyword)
        
        return {
            "matched_keywords": matched,
            "match_count": len(matched),
            "match_rate": len(matched) / len(keywords) if keywords else 0
        }
    
    def _check_exclusion_conditions(
        self, 
        function_description: str, 
        exclusions: List[str]
    ) -> Dict[str, Any]:
        """检查排除条件"""
        
        description_lower = function_description.lower()
        violated = []
        
        for exclusion in exclusions:
            if any(word in description_lower for word in exclusion.split()):
                violated.append(exclusion)
        
        return {
            "violated_conditions": violated,
            "violation_count": len(violated),
            "is_valid": len(violated) == 0
        }
    
    def _calculate_explanation_confidence(
        self, 
        keyword_matches: Dict[str, Any], 
        exclusion_checks: Dict[str, Any]
    ) -> float:
        """计算解释置信度"""
        
        match_score = keyword_matches["match_rate"]
        exclusion_penalty = exclusion_checks["violation_count"] * 0.3
        
        return max(0.0, min(1.0, match_score - exclusion_penalty))
    
    async def _parse_classification_response(
        self, 
        response_content: str, 
        function_description: str
    ) -> NESMAFunctionClassification:
        """解析分类响应 - 已废弃，保留用于兼容性"""
        logger.warning("使用已废弃的JSON解析方法，建议使用工具调用")
        return await self._conservative_classify(function_description)
    
    def get_classification_history(self) -> List[NESMAFunctionClassification]:
        """获取分类历史"""
        return self.classification_history.copy()
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """获取分类统计"""
        if not self.classification_history:
            return {"total": 0}
        
        type_counts = {}
        confidence_scores = []
        
        for classification in self.classification_history:
            func_type = classification.function_type.value
            type_counts[func_type] = type_counts.get(func_type, 0) + 1
            confidence_scores.append(classification.confidence_score)
        
        return {
            "total": len(self.classification_history),
            "type_distribution": type_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores)
        }


# 工厂函数
async def create_nesma_function_classifier(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> NESMAFunctionClassifierAgent:
    """创建NESMA功能分类器智能体"""
    classifier = NESMAFunctionClassifierAgent(rule_retriever=rule_retriever, llm=llm)
    await classifier.initialize()
    return classifier


if __name__ == "__main__":
    async def main():
        # 测试NESMA功能分类器
        classifier = await create_nesma_function_classifier()
        
        # 测试功能描述
        test_functions = [
            "用户可以录入个人基本信息，包括姓名、身份证号、联系方式等",
            "系统显示用户信息列表，支持按姓名和部门查询",
            "生成月度销售统计报表，包含各产品线的销售额和增长率",
            "维护产品基础信息，包括产品编码、名称、价格、库存等",
            "查询外部供应商的产品价格信息"
        ]
        
        print("🔍 NESMA功能分类测试:")
        for func_desc in test_functions:
            classification = await classifier.execute(
                "classify_function",
                {"function_description": func_desc}
            )
            print(f"\n功能: {func_desc[:50]}...")
            print(f"分类: {classification.function_type.value}")
            print(f"置信度: {classification.confidence_score:.2f}")
            print(f"理由: {classification.justification}")
        
        # 显示统计信息
        stats = classifier.get_classification_statistics()
        print(f"\n📊 分类统计: {stats}")
    
    asyncio.run(main()) 