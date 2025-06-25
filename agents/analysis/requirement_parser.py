"""
量子智能化功能点估算系统 - 需求解析智能体

从自然语言需求描述中提取结构化的功能信息
"""

import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import re

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import SpecializedAgent
from models.project_models import ProjectInfo, ProcessDetails
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RequirementParserAgent(SpecializedAgent):
    """需求解析智能体"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(
            agent_id="requirement_parser",
            specialty="natural_language_processing",
            llm=llm
        )
        
        self.settings = get_settings()
        
        # 解析历史和模式识别
        self.parsing_history: List[Dict[str, Any]] = []
        self.common_patterns = self._load_common_patterns()
        
    def _load_common_patterns(self) -> Dict[str, List[str]]:
        """加载常见的需求表达模式"""
        return {
            "功能动词": [
                "可以", "能够", "实现", "提供", "支持", "允许", "帮助",
                "管理", "维护", "处理", "生成", "创建", "查询", "更新"
            ],
            "实体标识": [
                "用户", "管理员", "系统", "数据", "信息", "报表", "文件",
                "订单", "产品", "客户", "员工", "部门", "项目"
            ],
            "操作类型": [
                "增加", "删除", "修改", "查询", "导入", "导出", "打印",
                "发送", "接收", "审核", "审批", "统计", "分析"
            ],
            "连接词": [
                "通过", "基于", "根据", "按照", "包括", "包含", "涉及",
                "关于", "针对", "对于", "以及", "和", "或者"
            ]
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "自然语言需求解析",
            "功能边界识别",
            "实体关系提取",
            "业务流程梳理",
            "需求结构化转换"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行需求解析任务"""
        if task_name == "parse_requirements":
            return await self.parse_requirements(inputs["requirement_text"])
        elif task_name == "extract_functional_entities":
            return await self.extract_functional_entities(inputs["requirement_text"])
        elif task_name == "identify_business_processes":
            return await self.identify_business_processes(inputs["requirement_text"])
        elif task_name == "validate_parsed_requirements":
            return await self.validate_parsed_requirements(
                inputs["parsed_requirements"],
                inputs["original_text"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def parse_requirements(self, requirement_text: str) -> Dict[str, Any]:
        """解析需求文本，提取结构化信息"""
        
        # 1. 预处理需求文本
        processed_text = self._preprocess_text(requirement_text)
        
        # 2. 识别功能模块
        functional_modules = await self._identify_functional_modules(processed_text)
        
        # 3. 提取业务实体
        business_entities = await self.extract_functional_entities(processed_text)
        
        # 4. 识别业务流程
        business_processes = await self.identify_business_processes(processed_text)
        
        # 5. 构建解析结果
        parsing_result = {
            "original_text": requirement_text,
            "processed_text": processed_text,
            "functional_modules": functional_modules,
            "business_entities": business_entities,
            "business_processes": business_processes,
            "parsing_confidence": self._calculate_parsing_confidence(
                functional_modules, business_entities, business_processes
            ),
            "parsing_timestamp": datetime.now()
        }
        
        # 6. 记录解析历史
        self.parsing_history.append(parsing_result)
        
        return parsing_result
    
    async def extract_functional_entities(self, requirement_text: str) -> Dict[str, List[str]]:
        """提取功能相关的实体"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是需求分析专家，需要从需求描述中提取关键的功能实体。

请识别以下类型的实体：
1. 用户角色：谁会使用这些功能
2. 业务对象：系统处理的主要数据或信息
3. 功能操作：用户可以执行的操作
4. 输入输出：数据的来源和去向
5. 业务规则：约束条件和业务逻辑

请返回JSON格式的结果。"""),
            ("human", """需求描述：
{requirement_text}

请提取功能相关的实体，并按类型分组。

返回格式：
{{
  "用户角色": ["角色1", "角色2"],
  "业务对象": ["对象1", "对象2"], 
  "功能操作": ["操作1", "操作2"],
  "输入输出": ["输入1", "输出1"],
  "业务规则": ["规则1", "规则2"]
}}""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(requirement_text=requirement_text)
        )
        
        # 解析实体提取结果
        return await self._parse_entity_extraction(response.content)
    
    async def identify_business_processes(self, requirement_text: str) -> List[ProcessDetails]:
        """识别业务流程"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是业务流程分析专家，需要从需求描述中识别独立的业务流程。

一个业务流程应该具备：
1. 明确的开始和结束条件
2. 清晰的处理步骤
3. 涉及的数据或实体
4. 产生的输出或结果

请识别每个独立的业务流程，并描述其特征。"""),
            ("human", """需求描述：
{requirement_text}

请识别其中的业务流程，每个流程包含：
- 流程名称
- 流程描述
- 涉及的数据组
- 主要步骤
- 输入输出

返回JSON格式的流程列表。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(requirement_text=requirement_text)
        )
        
        # 解析业务流程
        return await self._parse_business_processes(response.content)
    
    async def validate_parsed_requirements(
        self, 
        parsed_requirements: Dict[str, Any], 
        original_text: str
    ) -> Dict[str, Any]:
        """验证解析结果的完整性和准确性"""
        
        validation_result = {
            "completeness_score": 0.0,
            "accuracy_score": 0.0,
            "consistency_score": 0.0,
            "overall_score": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # 1. 完整性检查
        completeness_issues = self._check_completeness(parsed_requirements, original_text)
        validation_result["completeness_score"] = 1.0 - len(completeness_issues) * 0.2
        validation_result["issues"].extend(completeness_issues)
        
        # 2. 准确性检查
        accuracy_issues = self._check_accuracy(parsed_requirements, original_text)
        validation_result["accuracy_score"] = 1.0 - len(accuracy_issues) * 0.15
        validation_result["issues"].extend(accuracy_issues)
        
        # 3. 一致性检查
        consistency_issues = self._check_consistency(parsed_requirements)
        validation_result["consistency_score"] = 1.0 - len(consistency_issues) * 0.1
        validation_result["issues"].extend(consistency_issues)
        
        # 4. 计算总体评分
        validation_result["overall_score"] = (
            validation_result["completeness_score"] * 0.4 +
            validation_result["accuracy_score"] * 0.4 +
            validation_result["consistency_score"] * 0.2
        )
        
        # 5. 生成改进建议
        validation_result["suggestions"] = self._generate_improvement_suggestions(
            validation_result
        )
        
        return validation_result
    
    def _preprocess_text(self, text: str) -> str:
        """预处理需求文本"""
        
        # 去除多余空白
        processed = re.sub(r'\s+', ' ', text.strip())
        
        # 标准化标点符号
        processed = re.sub(r'[，。；：！？]', '。', processed)
        
        # 分句处理
        sentences = processed.split('。')
        cleaned_sentences = [s.strip() for s in sentences if s.strip()]
        
        return '。'.join(cleaned_sentences)
    
    async def _identify_functional_modules(self, text: str) -> List[Dict[str, Any]]:
        """识别功能模块"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是系统架构师，需要从需求描述中识别功能模块。

功能模块是相对独立的功能集合，通常包含：
- 相关的业务功能
- 共同的数据处理
- 特定的用户群体
- 明确的业务边界

请识别主要的功能模块。"""),
            ("human", """需求描述：
{text}

请识别功能模块，每个模块包含：
- 模块名称
- 模块描述
- 主要功能
- 涉及用户
- 相关数据

返回JSON格式的模块列表。""")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages(text=text))
        
        try:
            import json
            if "```json" in response.content:
                json_start = response.content.find("```json") + 7
                json_end = response.content.find("```", json_start)
                json_content = response.content[json_start:json_end].strip()
            else:
                json_content = response.content
            
            modules = json.loads(json_content)
            return modules if isinstance(modules, list) else []
            
        except Exception as e:
            logger.warning(f"解析功能模块失败: {str(e)}")
            return []
    
    async def _parse_entity_extraction(self, response_content: str) -> Dict[str, List[str]]:
        """解析实体提取结果"""
        
        try:
            import json
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content
            
            entities = json.loads(json_content)
            
            # 确保返回标准格式
            standard_entities = {
                "用户角色": entities.get("用户角色", []),
                "业务对象": entities.get("业务对象", []),
                "功能操作": entities.get("功能操作", []),
                "输入输出": entities.get("输入输出", []),
                "业务规则": entities.get("业务规则", [])
            }
            
            return standard_entities
            
        except Exception as e:
            logger.warning(f"解析实体提取结果失败: {str(e)}")
            return {
                "用户角色": [],
                "业务对象": [],
                "功能操作": [],
                "输入输出": [],
                "业务规则": []
            }
    
    async def _parse_business_processes(self, response_content: str) -> List[ProcessDetails]:
        """解析业务流程"""
        
        try:
            import json
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content
            
            processes_data = json.loads(json_content)
            
            processes = []
            for i, process_data in enumerate(processes_data):
                if isinstance(process_data, dict):
                    process = ProcessDetails(
                        id=f"process_{i+1}",
                        name=process_data.get("流程名称", f"流程{i+1}"),
                        description=process_data.get("流程描述", ""),
                        data_groups=process_data.get("涉及的数据组", []),
                        dependencies=process_data.get("依赖关系", [])
                    )
                    processes.append(process)
            
            return processes
            
        except Exception as e:
            logger.warning(f"解析业务流程失败: {str(e)}")
            return []
    
    def _calculate_parsing_confidence(
        self, 
        functional_modules: List[Dict[str, Any]], 
        business_entities: Dict[str, List[str]], 
        business_processes: List[ProcessDetails]
    ) -> float:
        """计算解析置信度"""
        
        confidence_factors = []
        
        # 功能模块数量合理性
        module_count = len(functional_modules)
        if 1 <= module_count <= 10:
            confidence_factors.append(0.9)
        elif module_count == 0:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.7)
        
        # 实体提取完整性
        entity_count = sum(len(entities) for entities in business_entities.values())
        if entity_count >= 5:
            confidence_factors.append(0.9)
        elif entity_count >= 2:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # 业务流程识别
        process_count = len(business_processes)
        if process_count >= 1:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _check_completeness(
        self, 
        parsed_requirements: Dict[str, Any], 
        original_text: str
    ) -> List[str]:
        """检查解析完整性"""
        
        issues = []
        
        # 检查关键元素是否存在
        if not parsed_requirements.get("functional_modules"):
            issues.append("未识别到功能模块")
        
        if not parsed_requirements.get("business_entities", {}).get("用户角色"):
            issues.append("未识别到用户角色")
        
        if not parsed_requirements.get("business_processes"):
            issues.append("未识别到业务流程")
        
        # 检查文本覆盖度（简化实现）
        text_length = len(original_text)
        if text_length > 500:
            # 对于长文本，检查是否有足够的结构化信息
            total_extracted = (
                len(parsed_requirements.get("functional_modules", [])) +
                sum(len(entities) for entities in parsed_requirements.get("business_entities", {}).values()) +
                len(parsed_requirements.get("business_processes", []))
            )
            if total_extracted < text_length // 100:  # 启发式规则
                issues.append("解析信息相对于文本长度偏少")
        
        return issues
    
    def _check_accuracy(
        self, 
        parsed_requirements: Dict[str, Any], 
        original_text: str
    ) -> List[str]:
        """检查解析准确性"""
        
        issues = []
        original_lower = original_text.lower()
        
        # 检查提取的实体是否在原文中存在
        business_entities = parsed_requirements.get("business_entities", {})
        
        for category, entities in business_entities.items():
            for entity in entities:
                if len(entity) > 2 and entity.lower() not in original_lower:
                    issues.append(f"实体'{entity}'在原文中未找到")
        
        # 检查功能模块名称合理性
        functional_modules = parsed_requirements.get("functional_modules", [])
        for module in functional_modules:
            module_name = module.get("模块名称", "")
            if module_name and len(module_name) < 2:
                issues.append(f"功能模块名称过短: {module_name}")
        
        return issues
    
    def _check_consistency(self, parsed_requirements: Dict[str, Any]) -> List[str]:
        """检查解析一致性"""
        
        issues = []
        
        # 检查业务流程与功能模块的一致性
        functional_modules = parsed_requirements.get("functional_modules", [])
        business_processes = parsed_requirements.get("business_processes", [])
        
        module_names = {module.get("模块名称", "") for module in functional_modules}
        process_names = {process.name for process in business_processes}
        
        # 简化的一致性检查
        if len(module_names) > 0 and len(process_names) > 0:
            # 检查是否有完全重复的名称
            duplicates = module_names.intersection(process_names)
            if duplicates:
                issues.append(f"模块和流程名称重复: {list(duplicates)}")
        
        # 检查业务实体的合理性
        business_entities = parsed_requirements.get("business_entities", {})
        user_roles = business_entities.get("用户角色", [])
        business_objects = business_entities.get("业务对象", [])
        
        if len(user_roles) == 0 and len(business_objects) > 0:
            issues.append("识别了业务对象但缺少用户角色")
        
        return issues
    
    def _generate_improvement_suggestions(self, validation_result: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        
        suggestions = []
        
        if validation_result["completeness_score"] < 0.7:
            suggestions.append("建议补充缺失的功能模块、用户角色或业务流程信息")
        
        if validation_result["accuracy_score"] < 0.7:
            suggestions.append("建议检查提取的实体是否与原文一致")
        
        if validation_result["consistency_score"] < 0.7:
            suggestions.append("建议检查各组件之间的逻辑一致性")
        
        if validation_result["overall_score"] > 0.8:
            suggestions.append("解析质量良好，可以进行下一步处理")
        
        return suggestions
    
    def get_parsing_history(self) -> List[Dict[str, Any]]:
        """获取解析历史"""
        return self.parsing_history.copy()
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """获取解析统计"""
        if not self.parsing_history:
            return {"total": 0}
        
        confidence_scores = [result["parsing_confidence"] for result in self.parsing_history]
        
        return {
            "total_parsed": len(self.parsing_history),
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores),
            "recent_parsing": self.parsing_history[-5:]  # 最近5次解析
        }


# 工厂函数
async def create_requirement_parser(llm: Optional[BaseLanguageModel] = None) -> RequirementParserAgent:
    """创建需求解析智能体"""
    parser = RequirementParserAgent(llm=llm)
    await parser.initialize()
    return parser


if __name__ == "__main__":
    async def main():
        # 测试需求解析器
        parser = await create_requirement_parser()
        
        # 测试需求文本
        test_requirement = """
        电商平台用户管理系统需要支持以下功能：
        1. 用户注册：新用户可以通过手机号或邮箱注册账号，需要验证身份
        2. 用户登录：支持多种登录方式，包括密码、短信验证码、第三方登录
        3. 个人信息管理：用户可以查看和修改个人基本信息，包括头像、昵称、联系方式
        4. 订单管理：用户可以查看历史订单、订单详情，支持订单搜索和筛选
        5. 地址管理：用户可以添加、编辑、删除收货地址，设置默认地址
        6. 系统管理员可以查看用户统计报表，包括注册量、活跃度等指标
        """
        
        print("📋 需求解析测试:")
        parsing_result = await parser.execute(
            "parse_requirements",
            {"requirement_text": test_requirement}
        )
        
        print(f"\n功能模块数量: {len(parsing_result['functional_modules'])}")
        print(f"业务流程数量: {len(parsing_result['business_processes'])}")
        print(f"解析置信度: {parsing_result['parsing_confidence']:.2f}")
        
        print(f"\n识别的用户角色: {parsing_result['business_entities']['用户角色']}")
        print(f"识别的业务对象: {parsing_result['business_entities']['业务对象']}")
        
        # 验证解析结果
        validation = await parser.execute(
            "validate_parsed_requirements",
            {
                "parsed_requirements": parsing_result,
                "original_text": test_requirement
            }
        )
        
        print(f"\n验证结果:")
        print(f"总体评分: {validation['overall_score']:.2f}")
        print(f"问题: {validation['issues']}")
        print(f"建议: {validation['suggestions']}")
    
    asyncio.run(main()) 