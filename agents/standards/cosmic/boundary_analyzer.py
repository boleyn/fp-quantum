"""
量子智能化功能点估算系统 - COSMIC边界分析器智能体

专门负责COSMIC软件边界和持久存储边界的详细分析
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
from models.project_models import ProjectInfo
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class COSMICBoundaryAnalyzerAgent(SpecializedAgent):
    """COSMIC边界分析器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="cosmic_boundary_analyzer",
            specialty="cosmic_boundary_analysis",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # COSMIC边界分析规则
        self.boundary_rules = self._load_boundary_rules()
        self.analysis_history: List[COSMICBoundaryAnalysis] = []
        
    def _load_boundary_rules(self) -> Dict[str, Any]:
        """加载COSMIC边界分析规则"""
        return {
            "软件边界": {
                "定义": "被测量软件与其环境之间的概念边界",
                "核心原则": [
                    "边界内包含所有被测量的软件组件",
                    "边界外包含所有功能用户和外部实体",
                    "边界必须清晰、一致且完整",
                    "数据移动穿越这个边界进行交互"
                ],
                "识别方法": [
                    "确定软件的核心功能组件",
                    "识别软件的外部接口",
                    "分析软件与环境的交互点",
                    "明确软件的职责范围"
                ],
                "常见错误": [
                    "将功能用户包含在软件边界内",
                    "边界定义过于宽泛或狭窄",
                    "忽略重要的软件组件",
                    "边界定义不一致"
                ]
            },
            "持久存储边界": {
                "定义": "软件用于存储和检索数据的持久存储系统边界",
                "核心原则": [
                    "标识所有持久存储组件",
                    "区分应用数据和配置数据",
                    "明确存储访问模式",
                    "支持Read和Write数据移动识别"
                ],
                "存储类型": [
                    "关系数据库：MySQL、PostgreSQL、Oracle等",
                    "NoSQL数据库：MongoDB、Redis、Elasticsearch等",
                    "文件系统：本地文件、网络存储等",
                    "云存储：对象存储、块存储等"
                ],
                "分析要点": [
                    "存储层次和结构",
                    "数据访问模式",
                    "存储边界与软件边界的关系",
                    "数据一致性要求"
                ]
            },
            "边界验证": {
                "一致性检查": [
                    "所有功能用户都在软件边界外",
                    "所有软件组件都在软件边界内",
                    "数据移动方向符合边界定义",
                    "边界定义支持完整的功能分析"
                ],
                "完整性检查": [
                    "所有重要接口都已识别",
                    "所有存储组件都已包含",
                    "边界覆盖所有系统功能",
                    "没有遗漏关键组件"
                ]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "软件边界定义",
            "持久存储边界分析",
            "边界一致性验证",
            "架构组件识别",
            "边界优化建议"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行边界分析任务"""
        if task_name == "analyze_software_boundary":
            return await self.analyze_software_boundary(
                inputs["project_info"],
                inputs["functional_users"]
            )
        elif task_name == "analyze_storage_boundary":
            return await self.analyze_storage_boundary(
                inputs["project_info"],
                inputs.get("architecture_info", {})
            )
        elif task_name == "validate_boundary_definition":
            return await self.validate_boundary_definition(
                inputs["boundary_analysis"],
                inputs["project_info"]
            )
        elif task_name == "optimize_boundary_definition":
            return await self.optimize_boundary_definition(
                inputs["boundary_analysis"],
                inputs["feedback"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def analyze_software_boundary(
        self,
        project_info: ProjectInfo,
        functional_users: List[COSMICFunctionalUser]
    ) -> Dict[str, Any]:
        """分析软件边界"""
        
        # 1. 获取相关COSMIC边界规则
        boundary_rules = await self._retrieve_boundary_rules(project_info.description)
        
        # 2. 分析软件架构和组件
        software_components = await self._identify_software_components(project_info)
        
        # 3. 定义软件边界
        boundary_definition = await self._define_software_boundary(
            project_info,
            functional_users,
            software_components,
            boundary_rules
        )
        
        # 4. 验证边界一致性
        boundary_validation = await self._validate_software_boundary(
            boundary_definition,
            functional_users,
            software_components
        )
        
        return {
            "boundary_definition": boundary_definition,
            "software_components": software_components,
            "validation_result": boundary_validation,
            "analysis_metadata": {
                "analysis_time": datetime.now(),
                "functional_users_count": len(functional_users),
                "components_identified": len(software_components)
            }
        }
    
    async def analyze_storage_boundary(
        self,
        project_info: ProjectInfo,
        architecture_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """分析持久存储边界"""
        
        # 1. 识别存储组件
        storage_components = await self._identify_storage_components(
            project_info,
            architecture_info
        )
        
        # 2. 分析存储架构
        storage_architecture = await self._analyze_storage_architecture(
            storage_components,
            project_info
        )
        
        # 3. 定义存储边界
        storage_boundary_definition = await self._define_storage_boundary(
            storage_components,
            storage_architecture,
            project_info
        )
        
        # 4. 分析数据访问模式
        access_patterns = await self._analyze_data_access_patterns(
            storage_components,
            project_info
        )
        
        return {
            "storage_boundary_definition": storage_boundary_definition,
            "storage_components": storage_components,
            "storage_architecture": storage_architecture,
            "access_patterns": access_patterns,
            "analysis_metadata": {
                "analysis_time": datetime.now(),
                "storage_components_count": len(storage_components),
                "architecture_complexity": self._assess_storage_complexity(storage_components)
            }
        }
    
    async def validate_boundary_definition(
        self,
        boundary_analysis: COSMICBoundaryAnalysis,
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """验证边界定义"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "validation_issues": [],
            "suggestions": []
        }
        
        # 验证软件边界
        software_boundary_issues = await self._validate_software_boundary_definition(
            boundary_analysis.software_boundary,
            boundary_analysis.functional_users,
            project_info
        )
        validation_result["validation_issues"].extend(software_boundary_issues)
        
        # 验证存储边界
        storage_boundary_issues = await self._validate_storage_boundary_definition(
            boundary_analysis.persistent_storage_boundary,
            project_info
        )
        validation_result["validation_issues"].extend(storage_boundary_issues)
        
        # 验证边界一致性
        consistency_issues = self._validate_boundary_consistency(boundary_analysis)
        validation_result["validation_issues"].extend(consistency_issues)
        
        # 计算整体验证分数
        if validation_result["validation_issues"]:
            validation_result["is_valid"] = False
            validation_result["confidence_score"] = max(0.1,
                1.0 - len(validation_result["validation_issues"]) * 0.12
            )
        
        # 生成改进建议
        if not validation_result["is_valid"]:
            validation_result["suggestions"] = self._generate_boundary_suggestions(
                validation_result["validation_issues"]
            )
        
        return validation_result
    
    async def optimize_boundary_definition(
        self,
        boundary_analysis: COSMICBoundaryAnalysis,
        feedback: Dict[str, Any]
    ) -> COSMICBoundaryAnalysis:
        """基于反馈优化边界定义"""
        
        optimized_software_boundary = boundary_analysis.software_boundary
        optimized_storage_boundary = boundary_analysis.persistent_storage_boundary
        optimized_reasoning = boundary_analysis.boundary_reasoning
        
        # 应用软件边界反馈
        if "software_boundary_feedback" in feedback:
            software_feedback = feedback["software_boundary_feedback"]
            optimized_software_boundary = await self._apply_software_boundary_feedback(
                boundary_analysis.software_boundary,
                software_feedback
            )
        
        # 应用存储边界反馈
        if "storage_boundary_feedback" in feedback:
            storage_feedback = feedback["storage_boundary_feedback"]
            optimized_storage_boundary = await self._apply_storage_boundary_feedback(
                boundary_analysis.persistent_storage_boundary,
                storage_feedback
            )
        
        # 更新推理说明
        if optimized_software_boundary != boundary_analysis.software_boundary or \
           optimized_storage_boundary != boundary_analysis.persistent_storage_boundary:
            optimized_reasoning = await self._update_boundary_reasoning(
                optimized_software_boundary,
                optimized_storage_boundary,
                boundary_analysis.functional_users,
                feedback
            )
        
        return COSMICBoundaryAnalysis(
            software_boundary=optimized_software_boundary,
            persistent_storage_boundary=optimized_storage_boundary,
            functional_users=boundary_analysis.functional_users,
            boundary_reasoning=optimized_reasoning
        )
    
    async def _retrieve_boundary_rules(self, project_description: str) -> List[str]:
        """检索相关的COSMIC边界规则"""
        
        if not self.rule_retriever:
            return []
        
        query = f"COSMIC software boundary persistent storage boundary definition {project_description}"
        
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
            logger.warning(f"检索COSMIC边界规则失败: {e}")
            return []
    
    async def _identify_software_components(
        self,
        project_info: ProjectInfo
    ) -> List[Dict[str, Any]]:
        """识别软件组件"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是软件架构分析专家，需要识别项目的软件组件。

软件组件识别原则：
1. 识别应用的核心功能模块
2. 识别技术架构组件
3. 识别业务逻辑组件
4. 区分内部组件和外部依赖

请基于项目信息识别所有软件组件。

返回JSON格式：
{{
  "software_components": [
    {{
      "name": "组件名称",
      "type": "business_logic|data_access|presentation|infrastructure",
      "description": "组件描述",
      "internal": true|false,
      "interfaces": ["接口1", "接口2"]
    }}
  ]
}}"""),
            ("human", """项目信息：
项目名称：{project_name}
项目描述：{project_description}
技术栈：{technology_stack}
业务领域：{business_domain}

请识别软件组件。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_name=project_info.name,
                project_description=project_info.description,
                technology_stack=", ".join(project_info.technology_stack),
                business_domain=project_info.business_domain
            )
        )
        
        # 解析组件
        return await self._parse_software_components(response.content)
    
    async def _define_software_boundary(
        self,
        project_info: ProjectInfo,
        functional_users: List[COSMICFunctionalUser],
        software_components: List[Dict[str, Any]],
        boundary_rules: List[str]
    ) -> str:
        """定义软件边界"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是COSMIC边界定义专家，需要明确定义软件边界。

边界定义原则：
1. 边界内：所有被测量的软件组件
2. 边界外：所有功能用户和外部系统
3. 边界必须清晰、一致、完整
4. 支持数据移动的准确分类

相关规则：
{boundary_rules}

请基于组件分析和功能用户，明确定义软件边界。"""),
            ("human", """项目信息：
{project_description}

软件组件：
{software_components}

功能用户：
{functional_users}

请明确定义软件边界，说明哪些在边界内，哪些在边界外。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                boundary_rules="\n".join(boundary_rules),
                project_description=project_info.description,
                software_components=self._format_components(software_components),
                functional_users="\n".join([f"- {u.name}: {u.description}" for u in functional_users])
            )
        )
        
        return response.content
    
    async def _identify_storage_components(
        self,
        project_info: ProjectInfo,
        architecture_info: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """识别存储组件"""
        
        storage_keywords = {
            "database": ["mysql", "postgresql", "oracle", "sql server"],
            "nosql": ["mongodb", "redis", "elasticsearch", "cassandra"],
            "cache": ["redis", "memcached", "hazelcast"],
            "file": ["filesystem", "s3", "hdfs", "nfs"],
            "queue": ["rabbitmq", "kafka", "activemq"]
        }
        
        identified_storage = []
        
        # 从技术栈推断存储组件
        for tech in project_info.technology_stack:
            tech_lower = tech.lower()
            for storage_type, keywords in storage_keywords.items():
                if any(keyword in tech_lower for keyword in keywords):
                    identified_storage.append({
                        "name": tech,
                        "type": storage_type,
                        "description": f"{storage_type}存储组件",
                        "technology": tech,
                        "persistence_level": "persistent" if storage_type != "cache" else "temporary"
                    })
        
        # 如果架构信息中有存储信息
        if architecture_info and "storage" in architecture_info:
            for storage_info in architecture_info["storage"]:
                identified_storage.append(storage_info)
        
        # 如果没有识别到存储组件，添加默认存储
        if not identified_storage:
            identified_storage.append({
                "name": "默认数据库",
                "type": "database",
                "description": "项目的主要数据存储",
                "technology": "未指定",
                "persistence_level": "persistent"
            })
        
        return identified_storage
    
    async def _analyze_storage_architecture(
        self,
        storage_components: List[Dict[str, Any]],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """分析存储架构"""
        
        architecture = {
            "storage_layers": [],
            "data_flow": [],
            "access_patterns": [],
            "consistency_requirements": []
        }
        
        # 分析存储层次
        for component in storage_components:
            layer = {
                "component": component["name"],
                "type": component["type"],
                "role": self._determine_storage_role(component),
                "access_level": self._determine_access_level(component)
            }
            architecture["storage_layers"].append(layer)
        
        # 分析数据流
        if len(storage_components) > 1:
            architecture["data_flow"] = self._analyze_inter_storage_flow(storage_components)
        
        return architecture
    
    async def _define_storage_boundary(
        self,
        storage_components: List[Dict[str, Any]],
        storage_architecture: Dict[str, Any],
        project_info: ProjectInfo
    ) -> str:
        """定义存储边界"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是COSMIC存储边界定义专家。

存储边界定义原则：
1. 包含所有持久存储组件
2. 区分应用数据和配置数据
3. 明确存储访问模式
4. 支持Read和Write数据移动识别

请定义清晰的持久存储边界。"""),
            ("human", """项目信息：
{project_description}

存储组件：
{storage_components}

存储架构：
{storage_architecture}

请定义持久存储边界。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_description=project_info.description,
                storage_components=self._format_storage_components(storage_components),
                storage_architecture=str(storage_architecture)
            )
        )
        
        return response.content
    
    async def _analyze_data_access_patterns(
        self,
        storage_components: List[Dict[str, Any]],
        project_info: ProjectInfo
    ) -> List[Dict[str, Any]]:
        """分析数据访问模式"""
        
        patterns = []
        
        for component in storage_components:
            pattern = {
                "storage_component": component["name"],
                "read_patterns": self._identify_read_patterns(component, project_info),
                "write_patterns": self._identify_write_patterns(component, project_info),
                "access_frequency": self._estimate_access_frequency(component),
                "data_types": self._identify_data_types(component, project_info)
            }
            patterns.append(pattern)
        
        return patterns
    
    async def _validate_software_boundary_definition(
        self,
        software_boundary: str,
        functional_users: List[COSMICFunctionalUser],
        project_info: ProjectInfo
    ) -> List[Dict[str, str]]:
        """验证软件边界定义"""
        issues = []
        
        # 检查边界描述完整性
        if len(software_boundary.split()) < 20:
            issues.append({
                "type": "insufficient_boundary_description",
                "message": "软件边界描述过于简单，需要更详细的说明"
            })
        
        # 检查功能用户是否被错误包含在边界内
        for user in functional_users:
            if user.name.lower() in software_boundary.lower() and "边界内" in software_boundary:
                issues.append({
                    "type": "user_in_boundary",
                    "message": f"功能用户 {user.name} 不应该在软件边界内"
                })
        
        # 检查是否明确说明了边界内外的组件
        if "边界内" not in software_boundary and "边界外" not in software_boundary:
            issues.append({
                "type": "unclear_boundary_definition",
                "message": "边界定义不够明确，应该明确说明边界内外的组件"
            })
        
        return issues
    
    async def _validate_storage_boundary_definition(
        self,
        storage_boundary: str,
        project_info: ProjectInfo
    ) -> List[Dict[str, str]]:
        """验证存储边界定义"""
        issues = []
        
        # 检查是否包含存储组件说明
        storage_keywords = ["数据库", "存储", "database", "storage", "文件系统"]
        if not any(keyword in storage_boundary.lower() for keyword in storage_keywords):
            issues.append({
                "type": "missing_storage_components",
                "message": "存储边界定义中缺少存储组件说明"
            })
        
        # 检查是否说明了持久性
        persistence_keywords = ["持久", "永久", "persistent", "permanent"]
        if not any(keyword in storage_boundary.lower() for keyword in persistence_keywords):
            issues.append({
                "type": "missing_persistence_definition",
                "message": "存储边界定义中缺少持久性说明"
            })
        
        return issues
    
    def _validate_boundary_consistency(
        self,
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> List[Dict[str, str]]:
        """验证边界一致性"""
        issues = []
        
        # 检查软件边界和存储边界的一致性
        software_boundary = boundary_analysis.software_boundary.lower()
        storage_boundary = boundary_analysis.persistent_storage_boundary.lower()
        
        # 检查是否存在冲突的表述
        if "数据库在边界内" in software_boundary and "数据库在边界外" in storage_boundary:
            issues.append({
                "type": "boundary_conflict",
                "message": "软件边界和存储边界对数据库位置的描述存在冲突"
            })
        
        # 检查推理说明是否与边界定义一致
        reasoning = boundary_analysis.boundary_reasoning.lower()
        if "边界内" in reasoning:
            # 检查推理中的组件是否与边界定义一致
            pass  # 可以进一步实现详细的一致性检查
        
        return issues
    
    def _generate_boundary_suggestions(
        self,
        validation_issues: List[Dict[str, str]]
    ) -> List[str]:
        """生成边界定义改进建议"""
        suggestions = []
        
        issue_types = [issue["type"] for issue in validation_issues]
        
        if "insufficient_boundary_description" in issue_types:
            suggestions.append("详细描述软件边界，包含所有内部组件和外部接口")
        
        if "user_in_boundary" in issue_types:
            suggestions.append("确保所有功能用户都位于软件边界外")
        
        if "unclear_boundary_definition" in issue_types:
            suggestions.append("明确列出边界内和边界外的组件，使边界定义更加清晰")
        
        if "missing_storage_components" in issue_types:
            suggestions.append("明确识别和描述所有存储组件在存储边界中的位置")
        
        if "missing_persistence_definition" in issue_types:
            suggestions.append("说明存储的持久性特征和数据访问模式")
        
        if "boundary_conflict" in issue_types:
            suggestions.append("检查并解决软件边界和存储边界定义中的冲突")
        
        return suggestions or ["边界定义验证通过，无需调整"]
    
    def _parse_software_components(self, response_content: str) -> List[Dict[str, Any]]:
        """解析软件组件"""
        try:
            import json
            if "{" in response_content and "}" in response_content:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_str = response_content[start:end]
                data = json.loads(json_str)
                return data.get("software_components", [])
        except Exception as e:
            logger.warning(f"解析软件组件失败: {e}")
        
        # 回退方案：基于技术栈生成默认组件
        return [
            {
                "name": "业务逻辑层",
                "type": "business_logic",
                "description": "核心业务逻辑处理",
                "internal": True,
                "interfaces": ["API接口"]
            },
            {
                "name": "数据访问层",
                "type": "data_access",
                "description": "数据库访问和数据持久化",
                "internal": True,
                "interfaces": ["数据库接口"]
            }
        ]
    
    def _format_components(self, components: List[Dict[str, Any]]) -> str:
        """格式化组件信息"""
        formatted = ""
        for comp in components:
            formatted += f"- {comp['name']} ({comp['type']}): {comp['description']}\n"
        return formatted
    
    def _format_storage_components(self, storage_components: List[Dict[str, Any]]) -> str:
        """格式化存储组件信息"""
        formatted = ""
        for comp in storage_components:
            formatted += f"- {comp['name']} ({comp['type']}): {comp['description']}\n"
        return formatted
    
    def _determine_storage_role(self, component: Dict[str, Any]) -> str:
        """确定存储组件角色"""
        if component["type"] == "database":
            return "主要数据存储"
        elif component["type"] == "cache":
            return "缓存存储"
        elif component["type"] == "queue":
            return "消息存储"
        elif component["type"] == "file":
            return "文件存储"
        else:
            return "辅助存储"
    
    def _determine_access_level(self, component: Dict[str, Any]) -> str:
        """确定访问级别"""
        if component.get("persistence_level") == "persistent":
            return "持久访问"
        else:
            return "临时访问"
    
    def _analyze_inter_storage_flow(self, storage_components: List[Dict[str, Any]]) -> List[str]:
        """分析存储间数据流"""
        flows = []
        
        # 简单的数据流推断
        databases = [c for c in storage_components if c["type"] == "database"]
        caches = [c for c in storage_components if c["type"] == "cache"]
        
        if databases and caches:
            flows.append(f"从 {databases[0]['name']} 到 {caches[0]['name']} 的缓存数据流")
        
        return flows
    
    def _identify_read_patterns(self, component: Dict[str, Any], project_info: ProjectInfo) -> List[str]:
        """识别读取模式"""
        if component["type"] == "database":
            return ["查询业务数据", "验证用户信息", "加载配置信息"]
        elif component["type"] == "cache":
            return ["快速数据检索", "会话信息读取"]
        else:
            return ["数据访问"]
    
    def _identify_write_patterns(self, component: Dict[str, Any], project_info: ProjectInfo) -> List[str]:
        """识别写入模式"""
        if component["type"] == "database":
            return ["保存业务数据", "更新状态信息", "记录操作日志"]
        elif component["type"] == "cache":
            return ["缓存热点数据", "存储会话信息"]
        else:
            return ["数据写入"]
    
    def _estimate_access_frequency(self, component: Dict[str, Any]) -> str:
        """估算访问频率"""
        if component["type"] == "cache":
            return "高频"
        elif component["type"] == "database":
            return "中频"
        else:
            return "低频"
    
    def _identify_data_types(self, component: Dict[str, Any], project_info: ProjectInfo) -> List[str]:
        """识别数据类型"""
        if "电商" in project_info.business_domain:
            return ["用户数据", "商品数据", "订单数据", "支付数据"]
        elif "金融" in project_info.business_domain:
            return ["账户数据", "交易数据", "风险数据", "合规数据"]
        else:
            return ["业务数据", "用户数据", "系统数据"]
    
    def _assess_storage_complexity(self, storage_components: List[Dict[str, Any]]) -> str:
        """评估存储复杂度"""
        if len(storage_components) == 1:
            return "简单"
        elif len(storage_components) <= 3:
            return "中等"
        else:
            return "复杂"
    
    async def _apply_software_boundary_feedback(
        self,
        current_boundary: str,
        feedback: Dict[str, Any]
    ) -> str:
        """应用软件边界反馈"""
        
        if feedback.get("add_components"):
            current_boundary += f"\n补充组件：{', '.join(feedback['add_components'])}"
        
        if feedback.get("remove_components"):
            # 简单处理：添加说明
            current_boundary += f"\n排除组件：{', '.join(feedback['remove_components'])}"
        
        if feedback.get("clarification"):
            current_boundary += f"\n补充说明：{feedback['clarification']}"
        
        return current_boundary
    
    async def _apply_storage_boundary_feedback(
        self,
        current_boundary: str,
        feedback: Dict[str, Any]
    ) -> str:
        """应用存储边界反馈"""
        
        if feedback.get("add_storage"):
            current_boundary += f"\n补充存储：{', '.join(feedback['add_storage'])}"
        
        if feedback.get("modify_persistence"):
            current_boundary += f"\n持久性调整：{feedback['modify_persistence']}"
        
        return current_boundary
    
    async def _update_boundary_reasoning(
        self,
        software_boundary: str,
        storage_boundary: str,
        functional_users: List[COSMICFunctionalUser],
        feedback: Dict[str, Any]
    ) -> str:
        """更新边界推理说明"""
        
        reasoning = f"""
## 优化后的COSMIC边界定义推理

### 软件边界
{software_boundary}

### 持久存储边界
{storage_boundary}

### 优化说明
基于反馈进行了以下优化：
"""
        
        if feedback.get("software_boundary_feedback"):
            reasoning += f"\n- 软件边界优化：{feedback['software_boundary_feedback'].get('reason', '用户反馈')}"
        
        if feedback.get("storage_boundary_feedback"):
            reasoning += f"\n- 存储边界优化：{feedback['storage_boundary_feedback'].get('reason', '用户反馈')}"
        
        reasoning += f"\n\n### 功能用户验证\n"
        for user in functional_users:
            reasoning += f"- {user.name}: 位于软件边界外，符合COSMIC定义\n"
        
        return reasoning.strip()
    
    def get_analysis_history(self) -> List[COSMICBoundaryAnalysis]:
        """获取分析历史"""
        return self.analysis_history.copy()


async def create_cosmic_boundary_analyzer(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> COSMICBoundaryAnalyzerAgent:
    """创建COSMIC边界分析器智能体"""
    return COSMICBoundaryAnalyzerAgent(rule_retriever=rule_retriever, llm=llm)


if __name__ == "__main__":
    async def main():
        # 测试COSMIC边界分析器
        analyzer = await create_cosmic_boundary_analyzer()
        
        # 测试项目信息
        test_project = ProjectInfo(
            name="用户管理系统",
            description="包含用户注册、登录、权限管理等功能的Web应用",
            technology_stack=["Python", "Django", "PostgreSQL", "Redis"],
            business_domain="用户管理"
        )
        
        test_users = [
            COSMICFunctionalUser(
                id="user_001",
                name="系统用户",
                description="使用系统的最终用户",
                boundary_definition="用户位于软件边界外"
            ),
            COSMICFunctionalUser(
                id="admin_001",
                name="系统管理员",
                description="管理系统配置的管理员",
                boundary_definition="管理员位于软件边界外"
            )
        ]
        
        # 分析软件边界
        software_boundary_result = await analyzer.analyze_software_boundary(
            test_project,
            test_users
        )
        
        print("软件边界分析:")
        print(software_boundary_result["boundary_definition"])
        
        # 分析存储边界
        storage_boundary_result = await analyzer.analyze_storage_boundary(test_project)
        
        print("\n存储边界分析:")
        print(storage_boundary_result["storage_boundary_definition"])
        
    asyncio.run(main()) 