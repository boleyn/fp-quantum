"""
量子智能化功能点估算系统 - COSMIC功能用户识别智能体

基于COSMIC v4.0+标准识别功能用户和定义软件边界
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
from models.cosmic_models import COSMICFunctionalUser, COSMICBoundaryAnalysis
from models.project_models import ProjectInfo, ProcessDetails
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class COSMICFunctionalUserAgent(SpecializedAgent):
    """COSMIC功能用户识别智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="cosmic_functional_user_agent",
            specialty="cosmic_functional_user_identification",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # COSMIC功能用户识别规则
        self.functional_user_rules = self._load_functional_user_rules()
        self.identification_history: List[COSMICFunctionalUser] = []
        
    def _load_functional_user_rules(self) -> Dict[str, Any]:
        """加载COSMIC功能用户识别规则"""
        return {
            "功能用户定义": {
                "核心概念": "功能用户是向软件发送数据移动或从软件接收数据移动的用户",
                "识别原则": [
                    "功能用户必须与软件有数据交换",
                    "可以是人类用户、其他软件或硬件设备",
                    "必须触发或接收功能过程",
                    "不能是软件内部的组件"
                ]
            },
            "常见功能用户类型": {
                "人类用户": [
                    "最终用户：直接使用软件完成业务任务",
                    "管理员用户：配置和管理软件系统",
                    "操作员：监控和操作软件运行"
                ],
                "外部系统": [
                    "其他软件系统：通过API或接口交换数据",
                    "数据库系统：存储和检索持久化数据",
                    "第三方服务：提供外部功能或数据"
                ],
                "硬件设备": [
                    "传感器：向软件发送测量数据",
                    "执行器：接收软件控制指令",
                    "打印机：接收软件输出内容"
                ]
            },
            "识别方法": {
                "数据流分析": "分析谁向软件发送数据，谁从软件接收数据",
                "用例分析": "从用例中识别参与者和他们的交互",
                "接口分析": "分析软件的所有外部接口",
                "业务流程分析": "从业务流程中识别外部参与者"
            },
            "边界定义": {
                "软件边界": "被测量软件与其环境之间的概念边界",
                "边界原则": [
                    "边界内是被测量的软件",
                    "边界外是功能用户和其他外部实体",
                    "数据移动穿越这个边界",
                    "边界必须清晰且一致"
                ]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "功能用户识别",
            "软件边界定义", 
            "用户类型分类",
            "边界验证",
            "用户需求分析"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行功能用户识别任务"""
        if task_name == "identify_functional_users":
            return await self.identify_functional_users(
                inputs["project_info"],
                inputs.get("process_details", [])
            )
        elif task_name == "define_software_boundary":
            return await self.define_software_boundary(
                inputs["project_info"],
                inputs["functional_users"]
            )
        elif task_name == "validate_functional_users":
            return await self.validate_functional_users(
                inputs["functional_users"],
                inputs["project_description"]
            )
        elif task_name == "analyze_user_interactions":
            return await self.analyze_user_interactions(
                inputs["functional_users"],
                inputs["process_details"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def identify_functional_users(
        self,
        project_info: ProjectInfo,
        process_details: Optional[List[ProcessDetails]] = None
    ) -> List[COSMICFunctionalUser]:
        """识别项目的功能用户"""
        
        # 1. 获取相关COSMIC规则
        cosmic_rules = await self._retrieve_cosmic_rules(project_info.description)
        
        # 2. 使用LLM分析项目描述
        identified_users = await self._llm_identify_functional_users(
            project_info,
            process_details or [],
            cosmic_rules
        )
        
        # 3. 验证和完善识别结果
        validated_users = await self._validate_and_refine_users(
            identified_users,
            project_info
        )
        
        # 4. 记录识别历史
        self.identification_history.extend(validated_users)
        
        return validated_users
    
    async def define_software_boundary(
        self,
        project_info: ProjectInfo,
        functional_users: List[COSMICFunctionalUser]
    ) -> COSMICBoundaryAnalysis:
        """定义软件边界"""
        
        # 1. 分析软件组件和边界
        boundary_definition = await self._analyze_software_boundary(
            project_info,
            functional_users
        )
        
        # 2. 定义持久存储边界
        storage_boundary = await self._define_storage_boundary(
            project_info,
            functional_users
        )
        
        # 3. 生成边界推理说明
        boundary_reasoning = await self._generate_boundary_reasoning(
            project_info,
            functional_users,
            boundary_definition,
            storage_boundary
        )
        
        return COSMICBoundaryAnalysis(
            software_boundary=boundary_definition,
            persistent_storage_boundary=storage_boundary,
            functional_users=functional_users,
            boundary_reasoning=boundary_reasoning
        )
    
    async def validate_functional_users(
        self,
        functional_users: List[COSMICFunctionalUser],
        project_description: str
    ) -> Dict[str, Any]:
        """验证功能用户识别结果"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "validation_issues": [],
            "suggestions": []
        }
        
        # 验证用户完整性
        completeness_issues = await self._check_user_completeness(
            functional_users,
            project_description
        )
        validation_result["validation_issues"].extend(completeness_issues)
        
        # 验证用户定义准确性
        accuracy_issues = await self._check_user_accuracy(functional_users)
        validation_result["validation_issues"].extend(accuracy_issues)
        
        # 验证边界一致性
        consistency_issues = self._check_boundary_consistency(functional_users)
        validation_result["validation_issues"].extend(consistency_issues)
        
        # 计算整体验证分数
        if validation_result["validation_issues"]:
            validation_result["is_valid"] = False
            validation_result["confidence_score"] = max(0.1,
                1.0 - len(validation_result["validation_issues"]) * 0.15
            )
        
        # 生成改进建议
        if not validation_result["is_valid"]:
            validation_result["suggestions"] = self._generate_user_suggestions(
                validation_result["validation_issues"]
            )
        
        return validation_result
    
    async def analyze_user_interactions(
        self,
        functional_users: List[COSMICFunctionalUser],
        process_details: List[ProcessDetails]
    ) -> Dict[str, Any]:
        """分析功能用户交互模式"""
        
        interaction_analysis = {
            "user_interaction_matrix": {},
            "interaction_patterns": [],
            "data_flow_summary": {},
            "complexity_assessment": {}
        }
        
        # 构建用户交互矩阵
        for user in functional_users:
            interaction_analysis["user_interaction_matrix"][user.id] = {
                "name": user.name,
                "description": user.description,
                "interaction_processes": [],
                "data_sent": [],
                "data_received": []
            }
        
        # 分析每个业务过程中的用户交互
        for process in process_details:
            process_interactions = await self._analyze_process_interactions(
                process,
                functional_users
            )
            
            # 更新交互矩阵
            for interaction in process_interactions:
                user_id = interaction["user_id"]
                if user_id in interaction_analysis["user_interaction_matrix"]:
                    user_data = interaction_analysis["user_interaction_matrix"][user_id]
                    user_data["interaction_processes"].append(process.name)
                    user_data["data_sent"].extend(interaction.get("data_sent", []))
                    user_data["data_received"].extend(interaction.get("data_received", []))
        
        # 识别交互模式
        interaction_analysis["interaction_patterns"] = self._identify_interaction_patterns(
            interaction_analysis["user_interaction_matrix"]
        )
        
        # 生成数据流汇总
        interaction_analysis["data_flow_summary"] = self._summarize_data_flows(
            interaction_analysis["user_interaction_matrix"]
        )
        
        # 评估交互复杂度
        interaction_analysis["complexity_assessment"] = self._assess_interaction_complexity(
            functional_users,
            interaction_analysis["user_interaction_matrix"]
        )
        
        return interaction_analysis
    
    async def _retrieve_cosmic_rules(self, project_description: str) -> List[str]:
        """检索相关的COSMIC规则"""
        
        if not self.rule_retriever:
            return []
        
        query = f"COSMIC functional user identification boundary definition {project_description}"
        
        try:
            retrieved_rules = await self.rule_retriever.execute_task(
                "retrieve_rules",
                {
                    "query": query,
                    "source_type": "COSMIC",
                    "max_results": 5
                }
            )
            
            return retrieved_rules.get("rules", [])
            
        except Exception as e:
            logger.warning(f"检索COSMIC规则失败: {e}")
            return []
    
    async def _llm_identify_functional_users(
        self,
        project_info: ProjectInfo,
        process_details: List[ProcessDetails],
        cosmic_rules: List[str]
    ) -> List[COSMICFunctionalUser]:
        """使用LLM识别功能用户"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是COSMIC功能点估算专家，需要识别项目的功能用户。

功能用户识别原则：
1. 功能用户是向软件发送数据或从软件接收数据的外部实体
2. 可以是人类用户、其他软件系统或硬件设备
3. 必须与软件有明确的数据交换
4. 不能是软件内部的组件

相关COSMIC规则：
{cosmic_rules}

请仔细分析项目信息，识别所有功能用户。

返回JSON格式：
{{
  "functional_users": [
    {{
      "id": "user_id",
      "name": "用户名称",
      "description": "用户描述",
      "user_type": "human|software|hardware",
      "boundary_definition": "边界定义说明",
      "data_interactions": ["发送的数据", "接收的数据"]
    }}
  ]
}}"""),
            ("human", """项目信息：
项目名称：{project_name}
项目描述：{project_description}
技术栈：{technology_stack}
业务领域：{business_domain}

业务过程：
{process_details}

请识别所有功能用户。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                cosmic_rules="\n".join(cosmic_rules),
                project_name=project_info.name,
                project_description=project_info.description,
                technology_stack=", ".join(project_info.technology_stack),
                business_domain=project_info.business_domain,
                process_details="\n".join([f"- {p.name}: {p.description}" for p in process_details])
            )
        )
        
        # 解析LLM响应
        return await self._parse_functional_users_response(response.content)
    
    async def _validate_and_refine_users(
        self,
        identified_users: List[COSMICFunctionalUser],
        project_info: ProjectInfo
    ) -> List[COSMICFunctionalUser]:
        """验证和完善用户识别结果"""
        
        validated_users = []
        
        for user in identified_users:
            # 验证用户定义的合理性
            if self._is_valid_functional_user(user, project_info):
                validated_users.append(user)
            else:
                # 尝试修正用户定义
                corrected_user = await self._correct_functional_user(user, project_info)
                if corrected_user:
                    validated_users.append(corrected_user)
        
        # 检查是否遗漏重要用户
        additional_users = await self._identify_missing_users(validated_users, project_info)
        validated_users.extend(additional_users)
        
        return validated_users
    
    async def _analyze_software_boundary(
        self,
        project_info: ProjectInfo,
        functional_users: List[COSMICFunctionalUser]
    ) -> str:
        """分析软件边界"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是COSMIC边界分析专家，需要定义软件边界。

软件边界定义原则：
1. 边界内是被测量的软件组件
2. 边界外是功能用户和外部实体
3. 边界必须清晰、一致且完整
4. 数据移动穿越这个边界

请根据项目信息和功能用户，定义清晰的软件边界。"""),
            ("human", """项目信息：
{project_description}

已识别的功能用户：
{functional_users}

请定义软件边界，说明哪些组件在边界内，哪些在边界外。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_description=project_info.description,
                functional_users="\n".join([f"- {u.name}: {u.description}" for u in functional_users])
            )
        )
        
        return response.content
    
    async def _define_storage_boundary(
        self,
        project_info: ProjectInfo,
        functional_users: List[COSMICFunctionalUser]
    ) -> str:
        """定义持久存储边界"""
        
        # 分析项目中的数据存储组件
        storage_components = []
        
        # 从技术栈推断存储组件
        for tech in project_info.technology_stack:
            if any(db in tech.lower() for db in ["mysql", "postgresql", "mongodb", "redis", "database"]):
                storage_components.append(tech)
        
        # 使用LLM分析存储边界
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是COSMIC存储边界分析专家。

持久存储边界定义原则：
1. 标识软件用于存储和检索数据的持久存储
2. 通常包括数据库、文件系统等
3. 存储边界与软件边界可能不同
4. 用于识别Read和Write数据移动

请定义持久存储边界。"""),
            ("human", """项目信息：
{project_description}

技术栈：{technology_stack}

已识别存储组件：{storage_components}

请定义持久存储边界。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_description=project_info.description,
                technology_stack=", ".join(project_info.technology_stack),
                storage_components=", ".join(storage_components) if storage_components else "未明确"
            )
        )
        
        return response.content
    
    async def _generate_boundary_reasoning(
        self,
        project_info: ProjectInfo,
        functional_users: List[COSMICFunctionalUser],
        boundary_definition: str,
        storage_boundary: str
    ) -> str:
        """生成边界定义推理说明"""
        
        reasoning = f"""
## COSMIC边界定义推理

### 软件边界分析
{boundary_definition}

### 持久存储边界分析  
{storage_boundary}

### 功能用户验证
基于边界定义，验证以下功能用户的合理性：
"""
        
        for user in functional_users:
            reasoning += f"\n- **{user.name}**: {user.description}"
            reasoning += f"\n  位置：边界外，通过数据移动与软件交互"
        
        reasoning += f"""

### 边界一致性检查
1. 所有功能用户都位于软件边界外
2. 数据移动清晰穿越边界
3. 持久存储边界明确定义
4. 边界定义支持后续数据移动分类

### 项目特征
- 业务领域：{project_info.business_domain}
- 技术架构：{', '.join(project_info.technology_stack)}
- 功能用户数量：{len(functional_users)}
        """
        
        return reasoning.strip()
    
    async def _check_user_completeness(
        self,
        functional_users: List[COSMICFunctionalUser],
        project_description: str
    ) -> List[Dict[str, str]]:
        """检查用户识别完整性"""
        issues = []
        
        # 检查用户数量合理性
        if len(functional_users) == 0:
            issues.append({
                "type": "no_users_identified",
                "message": "未识别到任何功能用户"
            })
        elif len(functional_users) == 1:
            issues.append({
                "type": "insufficient_users",
                "message": "只识别到一个功能用户，可能存在遗漏"
            })
        
        # 检查用户类型多样性
        user_types = set(user.description.split()[0] for user in functional_users)
        if len(user_types) < 2 and len(functional_users) > 2:
            issues.append({
                "type": "limited_user_diversity",
                "message": "功能用户类型单一，建议检查是否遗漏其他类型用户"
            })
        
        return issues
    
    async def _check_user_accuracy(
        self, 
        functional_users: List[COSMICFunctionalUser]
    ) -> List[Dict[str, str]]:
        """检查用户定义准确性"""
        issues = []
        
        for user in functional_users:
            # 检查用户描述完整性
            if len(user.description.split()) < 5:
                issues.append({
                    "type": "insufficient_user_description",
                    "message": f"用户 {user.name} 描述过于简单"
                })
            
            # 检查边界定义
            if not user.boundary_definition or len(user.boundary_definition.split()) < 10:
                issues.append({
                    "type": "insufficient_boundary_definition",
                    "message": f"用户 {user.name} 边界定义不够详细"
                })
        
        return issues
    
    def _check_boundary_consistency(
        self, 
        functional_users: List[COSMICFunctionalUser]
    ) -> List[Dict[str, str]]:
        """检查边界一致性"""
        issues = []
        
        # 检查用户名称唯一性
        user_names = [user.name for user in functional_users]
        if len(user_names) != len(set(user_names)):
            issues.append({
                "type": "duplicate_user_names",
                "message": "存在重复的用户名称"
            })
        
        # 检查用户ID唯一性
        user_ids = [user.id for user in functional_users]
        if len(user_ids) != len(set(user_ids)):
            issues.append({
                "type": "duplicate_user_ids", 
                "message": "存在重复的用户ID"
            })
        
        return issues
    
    def _generate_user_suggestions(
        self, 
        validation_issues: List[Dict[str, str]]
    ) -> List[str]:
        """生成用户识别改进建议"""
        suggestions = []
        
        issue_types = [issue["type"] for issue in validation_issues]
        
        if "no_users_identified" in issue_types:
            suggestions.append("重新分析项目需求，识别所有与软件交互的外部实体")
        
        if "insufficient_users" in issue_types:
            suggestions.append("检查是否遗漏管理员用户、外部系统或其他类型用户")
        
        if "limited_user_diversity" in issue_types:
            suggestions.append("考虑不同类型的功能用户：人类用户、外部系统、硬件设备")
        
        if any("description" in issue_type for issue_type in issue_types):
            suggestions.append("完善用户描述，包含用户角色、职责和交互方式")
        
        if any("boundary" in issue_type for issue_type in issue_types):
            suggestions.append("明确定义用户与软件的边界关系和数据交换方式")
        
        return suggestions or ["功能用户识别验证通过，无需调整"]
    
    def _is_valid_functional_user(
        self, 
        user: COSMICFunctionalUser, 
        project_info: ProjectInfo
    ) -> bool:
        """验证功能用户定义是否有效"""
        
        # 基本验证
        if not user.name or not user.description:
            return False
        
        # 检查是否符合COSMIC功能用户定义
        valid_indicators = [
            "数据" in user.description,
            "交互" in user.description,
            any(keyword in user.description for keyword in ["用户", "系统", "设备", "服务"]),
            len(user.description.split()) >= 5
        ]
        
        return sum(valid_indicators) >= 2
    
    async def _correct_functional_user(
        self, 
        user: COSMICFunctionalUser, 
        project_info: ProjectInfo
    ) -> Optional[COSMICFunctionalUser]:
        """修正功能用户定义"""
        
        # 简单修正策略：增强描述
        if len(user.description.split()) < 5:
            enhanced_description = f"{user.description}，与{project_info.name}系统进行数据交互"
            
            return COSMICFunctionalUser(
                id=user.id,
                name=user.name,
                description=enhanced_description,
                boundary_definition=user.boundary_definition or f"{user.name}位于软件边界外，通过接口与系统交互"
            )
        
        return user
    
    async def _identify_missing_users(
        self, 
        identified_users: List[COSMICFunctionalUser], 
        project_info: ProjectInfo
    ) -> List[COSMICFunctionalUser]:
        """识别可能遗漏的用户"""
        
        additional_users = []
        identified_names = {user.name.lower() for user in identified_users}
        
        # 基于项目特征推断可能的用户
        common_users = {
            "管理员": "系统管理员，负责系统配置和用户管理",
            "数据库": "持久存储系统，存储和检索业务数据",
            "外部API": "第三方服务接口，提供外部功能或数据"
        }
        
        for user_name, description in common_users.items():
            if user_name.lower() not in identified_names:
                # 检查项目是否可能需要这类用户
                if self._project_likely_needs_user(project_info, user_name, description):
                    additional_users.append(COSMICFunctionalUser(
                        id=f"auto_generated_{user_name.lower()}",
                        name=user_name,
                        description=description,
                        boundary_definition=f"{user_name}位于软件边界外，与系统进行必要的数据交换"
                    ))
        
        return additional_users
    
    def _project_likely_needs_user(
        self, 
        project_info: ProjectInfo, 
        user_name: str, 
        description: str
    ) -> bool:
        """判断项目是否可能需要某类用户"""
        
        project_text = f"{project_info.description} {' '.join(project_info.technology_stack)}"
        
        user_keywords = {
            "管理员": ["管理", "配置", "权限", "admin"],
            "数据库": ["数据", "存储", "database", "mysql", "postgresql"],
            "外部API": ["api", "接口", "第三方", "外部", "集成"]
        }
        
        keywords = user_keywords.get(user_name, [])
        return any(keyword.lower() in project_text.lower() for keyword in keywords)
    
    async def _parse_functional_users_response(
        self, 
        response_content: str
    ) -> List[COSMICFunctionalUser]:
        """解析LLM响应的功能用户"""
        
        try:
            import json
            # 尝试提取JSON
            if "{" in response_content and "}" in response_content:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_str = response_content[start:end]
                data = json.loads(json_str)
                
                users = []
                for user_data in data.get("functional_users", []):
                    user = COSMICFunctionalUser(
                        id=user_data.get("id", f"user_{len(users)}"),
                        name=user_data.get("name", "未命名用户"),
                        description=user_data.get("description", ""),
                        boundary_definition=user_data.get("boundary_definition", "")
                    )
                    users.append(user)
                
                return users
                
        except Exception as e:
            logger.warning(f"解析功能用户响应失败: {e}")
        
        # 回退方案：基于文本解析
        return self._parse_users_from_text(response_content)
    
    def _parse_users_from_text(self, text: str) -> List[COSMICFunctionalUser]:
        """从文本中解析功能用户"""
        
        users = []
        lines = text.split('\n')
        
        current_user = None
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                # 新用户开始
                if current_user:
                    users.append(current_user)
                
                user_name = line.lstrip('- *').split(':')[0].strip()
                user_desc = line.split(':', 1)[1].strip() if ':' in line else ""
                
                current_user = COSMICFunctionalUser(
                    id=f"user_{len(users)}",
                    name=user_name,
                    description=user_desc,
                    boundary_definition=f"{user_name}位于软件边界外"
                )
        
        if current_user:
            users.append(current_user)
        
        return users
    
    async def _analyze_process_interactions(
        self,
        process: ProcessDetails,
        functional_users: List[COSMICFunctionalUser]
    ) -> List[Dict[str, Any]]:
        """分析业务过程中的用户交互"""
        
        interactions = []
        
        # 基于过程描述推断用户交互
        for user in functional_users:
            interaction = {
                "user_id": user.id,
                "process_name": process.name,
                "data_sent": [],
                "data_received": []
            }
            
            # 简单的关键词匹配推断交互
            if any(keyword in process.description.lower() for keyword in ["输入", "提交", "创建"]):
                interaction["data_sent"].append(f"来自{user.name}的输入数据")
            
            if any(keyword in process.description.lower() for keyword in ["显示", "输出", "报告"]):
                interaction["data_received"].append(f"向{user.name}的输出数据")
            
            # 如果有交互，添加到列表
            if interaction["data_sent"] or interaction["data_received"]:
                interactions.append(interaction)
        
        return interactions
    
    def _identify_interaction_patterns(
        self, 
        user_interaction_matrix: Dict[str, Any]
    ) -> List[str]:
        """识别交互模式"""
        
        patterns = []
        
        # 分析交互频率
        high_interaction_users = []
        low_interaction_users = []
        
        for user_id, user_data in user_interaction_matrix.items():
            interaction_count = len(user_data["interaction_processes"])
            if interaction_count > 3:
                high_interaction_users.append(user_data["name"])
            elif interaction_count == 1:
                low_interaction_users.append(user_data["name"])
        
        if high_interaction_users:
            patterns.append(f"高频交互用户：{', '.join(high_interaction_users)}")
        
        if low_interaction_users:
            patterns.append(f"低频交互用户：{', '.join(low_interaction_users)}")
        
        # 分析数据流方向
        bidirectional_users = []
        input_only_users = []
        output_only_users = []
        
        for user_id, user_data in user_interaction_matrix.items():
            has_input = bool(user_data["data_sent"])
            has_output = bool(user_data["data_received"])
            
            if has_input and has_output:
                bidirectional_users.append(user_data["name"])
            elif has_input:
                input_only_users.append(user_data["name"])
            elif has_output:
                output_only_users.append(user_data["name"])
        
        if bidirectional_users:
            patterns.append(f"双向交互用户：{', '.join(bidirectional_users)}")
        
        if input_only_users:
            patterns.append(f"仅输入用户：{', '.join(input_only_users)}")
        
        if output_only_users:
            patterns.append(f"仅输出用户：{', '.join(output_only_users)}")
        
        return patterns
    
    def _summarize_data_flows(
        self, 
        user_interaction_matrix: Dict[str, Any]
    ) -> Dict[str, Any]:
        """汇总数据流"""
        
        summary = {
            "total_users": len(user_interaction_matrix),
            "input_flows": 0,
            "output_flows": 0,
            "bidirectional_users": 0
        }
        
        for user_data in user_interaction_matrix.values():
            if user_data["data_sent"]:
                summary["input_flows"] += len(user_data["data_sent"])
            if user_data["data_received"]:
                summary["output_flows"] += len(user_data["data_received"])
            if user_data["data_sent"] and user_data["data_received"]:
                summary["bidirectional_users"] += 1
        
        return summary
    
    def _assess_interaction_complexity(
        self,
        functional_users: List[COSMICFunctionalUser],
        user_interaction_matrix: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估交互复杂度"""
        
        total_interactions = sum(
            len(user_data["interaction_processes"]) 
            for user_data in user_interaction_matrix.values()
        )
        
        average_interactions = total_interactions / len(functional_users) if functional_users else 0
        
        complexity_level = "Low"
        if average_interactions > 5:
            complexity_level = "High"
        elif average_interactions > 2:
            complexity_level = "Medium"
        
        return {
            "total_interactions": total_interactions,
            "average_interactions_per_user": average_interactions,
            "complexity_level": complexity_level,
            "assessment": f"用户交互复杂度：{complexity_level}，平均每用户参与 {average_interactions:.1f} 个交互过程"
        }
    
    def get_identification_history(self) -> List[COSMICFunctionalUser]:
        """获取识别历史"""
        return self.identification_history.copy()


async def create_cosmic_functional_user_agent(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> COSMICFunctionalUserAgent:
    """创建COSMIC功能用户识别智能体"""
    return COSMICFunctionalUserAgent(rule_retriever=rule_retriever, llm=llm)


if __name__ == "__main__":
    async def main():
        # 测试COSMIC功能用户识别智能体
        agent = await create_cosmic_functional_user_agent()
        
        # 测试项目信息
        test_project = ProjectInfo(
            name="电商管理系统",
            description="包含用户管理、商品管理、订单处理、支付集成等功能的电商平台",
            technology_stack=["Python", "Django", "MySQL", "Redis"],
            business_domain="电商"
        )
        
        # 识别功能用户
        functional_users = await agent.identify_functional_users(test_project)
        print(f"识别到 {len(functional_users)} 个功能用户:")
        for user in functional_users:
            print(f"- {user.name}: {user.description}")
        
        # 定义软件边界
        boundary_analysis = await agent.define_software_boundary(test_project, functional_users)
        print(f"\n边界分析:\n{boundary_analysis.boundary_reasoning}")
        
    asyncio.run(main()) 