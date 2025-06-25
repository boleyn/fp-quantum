"""
量子智能化功能点估算系统 - 标准推荐器智能体

基于项目特征智能推荐NESMA/COSMIC标准的最优组合
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
from models.project_models import (
    ProjectInfo, EstimationStandard, EstimationStrategy, 
    StandardRecommendation
)
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class StandardRecommenderAgent(SpecializedAgent):
    """标准推荐器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="standard_recommender",
            specialty="functional_point_standards",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # 标准特征数据库
        self.standard_characteristics = self._load_standard_characteristics()
        self.recommendation_history: List[StandardRecommendation] = []
        
    def _load_standard_characteristics(self) -> Dict[str, Any]:
        """加载标准特征数据库"""
        return {
            "NESMA": {
                "适用技术栈": [
                    "传统企业应用", "数据处理系统", "ERP系统", 
                    "CRM系统", "财务管理系统", "报表系统"
                ],
                "适用架构": [
                    "单体应用", "传统三层架构", "CS架构", 
                    "主机系统", "数据仓库"
                ],
                "开发模式": ["瀑布模型", "V模型", "增量模型"],
                "业务特征": [
                    "数据密集型", "事务处理", "批处理", 
                    "报表生成", "数据维护"
                ],
                "优势": [
                    "成熟标准", "历史数据丰富", "行业认可度高",
                    "适合传统项目", "计算相对简单"
                ],
                "劣势": [
                    "对现代架构支持有限", "实时系统适应性差",
                    "微服务架构不友好"
                ]
            },
            "COSMIC": {
                "适用技术栈": [
                    "现代Web应用", "移动应用", "微服务架构",
                    "云原生应用", "API服务", "实时系统"
                ],
                "适用架构": [
                    "微服务", "SOA", "RESTful API", "事件驱动",
                    "容器化应用", "serverless"
                ],
                "开发模式": ["敏捷开发", "DevOps", "持续集成"],
                "业务特征": [
                    "用户交互丰富", "实时处理", "分布式处理",
                    "多渠道集成", "数据流处理"
                ],
                "优势": [
                    "现代架构适应性强", "精确度高", "用户导向",
                    "适合复杂交互", "国际标准"
                ],
                "劣势": [
                    "学习成本高", "计算复杂", "历史数据相对较少"
                ]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "项目特征分析",
            "标准适用性评估", 
            "标准组合推荐",
            "风险评估",
            "置信度计算"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行标准推荐任务"""
        if task_name == "recommend_standards":
            return await self.recommend_standards(inputs["project_info"])
        elif task_name == "analyze_project_characteristics":
            return await self.analyze_project_characteristics(inputs["project_info"])
        elif task_name == "compare_standards":
            return await self.compare_standards(
                inputs["project_info"], 
                inputs.get("specific_standards", None)
            )
        elif task_name == "evaluate_dual_standard_necessity":
            return await self.evaluate_dual_standard_necessity(inputs["project_info"])
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def recommend_standards(self, project_info: ProjectInfo) -> StandardRecommendation:
        """推荐最适合的功能点估算标准"""
        
        # 1. 分析项目特征
        project_analysis = await self.analyze_project_characteristics(project_info)
        
        # 2. 获取相关知识
        knowledge_context = await self._retrieve_standard_knowledge(project_info)
        
        # 3. 使用LLM进行深度分析
        recommendation = await self._llm_recommend_standards(
            project_info, 
            project_analysis, 
            knowledge_context
        )
        
        # 4. 记录推荐历史
        self.recommendation_history.append(recommendation)
        
        return recommendation
    
    async def analyze_project_characteristics(self, project_info: ProjectInfo) -> Dict[str, Any]:
        """分析项目特征"""
        
        characteristics = {
            "技术现代化程度": self._assess_technology_modernity(project_info.technology_stack),
            "架构复杂度": self._assess_architecture_complexity(project_info),
            "业务领域匹配": self._assess_domain_fit(project_info.business_domain),
            "规模预估": self._estimate_project_scale(project_info),
            "交互复杂度": self._assess_interaction_complexity(project_info.description),
            "数据处理特征": self._assess_data_processing_characteristics(project_info)
        }
        
        return characteristics
    
    async def compare_standards(
        self, 
        project_info: ProjectInfo,
        specific_standards: Optional[List[EstimationStandard]] = None
    ) -> Dict[str, Any]:
        """对比标准适用性"""
        
        standards_to_compare = specific_standards or [
            EstimationStandard.NESMA, 
            EstimationStandard.COSMIC
        ]
        
        comparison_result = {}
        
        for standard in standards_to_compare:
            fit_score = await self._calculate_standard_fit_score(project_info, standard)
            comparison_result[standard.value] = {
                "适用性评分": fit_score,
                "优势": self._get_standard_advantages(standard, project_info),
                "劣势": self._get_standard_disadvantages(standard, project_info),
                "推荐场景": self._get_recommended_scenarios(standard)
            }
        
        return comparison_result
    
    async def evaluate_dual_standard_necessity(self, project_info: ProjectInfo) -> Dict[str, Any]:
        """评估是否需要双标准估算"""
        
        # 分析项目特征
        characteristics = await self.analyze_project_characteristics(project_info)
        
        # 计算各标准的适用性
        nesma_fit = await self._calculate_standard_fit_score(project_info, EstimationStandard.NESMA)
        cosmic_fit = await self._calculate_standard_fit_score(project_info, EstimationStandard.COSMIC)
        
        # 判断是否需要双标准
        fit_difference = abs(nesma_fit - cosmic_fit)
        
        if fit_difference < 0.2:  # 两个标准都很适合
            dual_necessity = "高"
            reason = "两种标准都很适合，双标准可提供更全面的视角"
        elif min(nesma_fit, cosmic_fit) > 0.6:  # 两个标准都适合
            dual_necessity = "中"
            reason = "两种标准都适合，可考虑双标准进行对比验证"
        else:  # 一个标准明显更适合
            dual_necessity = "低"
            reason = "一种标准明显更适合，双标准的投入产出比较低"
        
        return {
            "dual_necessity": dual_necessity,
            "reason": reason,
            "nesma_fit_score": nesma_fit,
            "cosmic_fit_score": cosmic_fit,
            "recommended_strategy": self._recommend_estimation_strategy(
                nesma_fit, cosmic_fit, dual_necessity
            )
        }
    
    async def _retrieve_standard_knowledge(self, project_info: ProjectInfo) -> Dict[str, Any]:
        """检索标准相关知识"""
        if not self.rule_retriever:
            return {}
        
        # 构建查询
        queries = [
            f"{project_info.business_domain} 功能点估算标准选择",
            f"{' '.join(project_info.technology_stack)} NESMA COSMIC 适用性",
            "功能点估算标准对比 选择原则"
        ]
        
        knowledge_results = {}
        for query in queries:
            try:
                result = await self.rule_retriever.retrieve_rules(
                    query=query,
                    standard=None,  # 不限定标准
                    min_chunks=2
                )
                knowledge_results[query] = result
            except Exception as e:
                logger.warning(f"知识检索失败 {query}: {str(e)}")
        
        return knowledge_results
    
    async def _llm_recommend_standards(
        self,
        project_info: ProjectInfo,
        project_analysis: Dict[str, Any],
        knowledge_context: Dict[str, Any]
    ) -> StandardRecommendation:
        """使用LLM进行标准推荐"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是功能点估算标准专家，需要基于项目特征推荐最适合的估算标准。

请综合考虑：
1. 项目技术栈和架构特征
2. 业务领域和应用类型
3. 开发模式和团队特点
4. 精度要求和时间约束
5. 历史经验和对比需求

标准选择原则：
- NESMA：适合传统企业应用、数据处理系统、瀑布开发
- COSMIC：适合现代应用、微服务架构、敏捷开发
- 双标准：复杂项目、高精度要求、需要对比验证

请返回JSON格式的推荐结果。"""),
            ("human", """项目信息：
项目名称：{name}
项目描述：{description}
技术栈：{technology_stack}
业务领域：{business_domain}

项目特征分析：
{project_analysis}

相关知识上下文：
{knowledge_context}

请推荐最适合的功能点估算标准。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                name=project_info.name,
                description=project_info.description,
                technology_stack=", ".join(project_info.technology_stack),
                business_domain=project_info.business_domain,
                project_analysis=str(project_analysis),
                knowledge_context=str(knowledge_context)
            )
        )
        
        # 解析响应并创建推荐结果
        return await self._parse_llm_recommendation(response.content, project_info)
    
    def _assess_technology_modernity(self, technology_stack: List[str]) -> float:
        """评估技术现代化程度"""
        modern_techs = {
            "微服务", "docker", "kubernetes", "react", "vue", "angular",
            "nodejs", "python", "go", "rust", "typescript", "graphql",
            "mongodb", "redis", "elasticsearch", "kafka", "rabbitmq",
            "aws", "azure", "gcp", "serverless", "lambda"
        }
        
        legacy_techs = {
            "cobol", "fortran", "mainframe", "as400", "vb6", "asp",
            "oracle forms", "powerbuilder", "delphi", "foxpro"
        }
        
        stack_lower = [tech.lower() for tech in technology_stack]
        modern_count = sum(1 for tech in stack_lower if any(modern in tech for modern in modern_techs))
        legacy_count = sum(1 for tech in stack_lower if any(legacy in tech for legacy in legacy_techs))
        
        total_count = len(technology_stack)
        if total_count == 0:
            return 0.5
        
        modernity_score = (modern_count - legacy_count * 0.5) / total_count
        return max(0.0, min(1.0, modernity_score))
    
    def _assess_architecture_complexity(self, project_info: ProjectInfo) -> float:
        """评估架构复杂度"""
        complexity_indicators = [
            "微服务", "分布式", "集群", "负载均衡", "消息队列",
            "缓存", "搜索引擎", "大数据", "实时", "流处理"
        ]
        
        description_lower = project_info.description.lower()
        tech_stack_lower = " ".join(project_info.technology_stack).lower()
        full_text = f"{description_lower} {tech_stack_lower}"
        
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in full_text)
        return min(1.0, complexity_count / 5.0)  # 最多5个指标达到1.0
    
    def _assess_domain_fit(self, business_domain: str) -> Dict[str, float]:
        """评估业务领域匹配度"""
        domain_lower = business_domain.lower()
        
        nesma_domains = ["金融", "银行", "保险", "erp", "财务", "hr", "制造", "供应链"]
        cosmic_domains = ["电商", "社交", "游戏", "移动", "iot", "ai", "大数据", "实时"]
        
        nesma_fit = sum(1 for d in nesma_domains if d in domain_lower) / len(nesma_domains)
        cosmic_fit = sum(1 for d in cosmic_domains if d in domain_lower) / len(cosmic_domains)
        
        return {"nesma": nesma_fit, "cosmic": cosmic_fit}
    
    def _estimate_project_scale(self, project_info: ProjectInfo) -> str:
        """估算项目规模"""
        description = project_info.description.lower()
        tech_count = len(project_info.technology_stack)
        
        scale_indicators = {
            "大型": ["大型", "企业级", "平台", "系统", "集群"],
            "中型": ["中型", "应用", "服务", "模块"],
            "小型": ["小型", "工具", "插件", "组件"]
        }
        
        for scale, indicators in scale_indicators.items():
            if any(indicator in description for indicator in indicators):
                return scale
        
        # 基于技术栈数量判断
        if tech_count >= 8:
            return "大型"
        elif tech_count >= 4:
            return "中型"
        else:
            return "小型"
    
    def _assess_interaction_complexity(self, description: str) -> float:
        """评估交互复杂度"""
        interaction_keywords = [
            "用户界面", "交互", "多端", "移动", "web", "api", 
            "集成", "第三方", "实时", "推送", "聊天", "协作"
        ]
        
        description_lower = description.lower()
        interaction_count = sum(1 for keyword in interaction_keywords if keyword in description_lower)
        
        return min(1.0, interaction_count / 6.0)
    
    def _assess_data_processing_characteristics(self, project_info: ProjectInfo) -> Dict[str, bool]:
        """评估数据处理特征"""
        full_text = f"{project_info.description} {' '.join(project_info.technology_stack)}".lower()
        
        return {
            "批处理": any(keyword in full_text for keyword in ["批处理", "batch", "定时", "调度"]),
            "实时处理": any(keyword in full_text for keyword in ["实时", "stream", "kafka", "实时"]),
            "数据分析": any(keyword in full_text for keyword in ["分析", "报表", "统计", "bi", "数据仓库"]),
            "事务处理": any(keyword in full_text for keyword in ["事务", "transaction", "acid", "数据库"])
        }
    
    async def _calculate_standard_fit_score(
        self, 
        project_info: ProjectInfo, 
        standard: EstimationStandard
    ) -> float:
        """计算标准适用性评分"""
        
        characteristics = await self.analyze_project_characteristics(project_info)
        
        if standard == EstimationStandard.NESMA:
            # NESMA适用性评分
            score = 0.0
            score += (1.0 - characteristics["技术现代化程度"]) * 0.3  # 传统技术更适合
            score += (1.0 - characteristics["架构复杂度"]) * 0.2  # 简单架构更适合
            score += characteristics["业务领域匹配"]["nesma"] * 0.3
            score += 0.2 if characteristics["数据处理特征"]["批处理"] else 0.0
            return min(1.0, score)
            
        elif standard == EstimationStandard.COSMIC:
            # COSMIC适用性评分
            score = 0.0
            score += characteristics["技术现代化程度"] * 0.3  # 现代技术更适合
            score += characteristics["架构复杂度"] * 0.2  # 复杂架构更适合
            score += characteristics["业务领域匹配"]["cosmic"] * 0.3
            score += characteristics["交互复杂度"] * 0.2
            return min(1.0, score)
        
        return 0.5  # 默认中等适用性
    
    def _get_standard_advantages(
        self, 
        standard: EstimationStandard, 
        project_info: ProjectInfo
    ) -> List[str]:
        """获取标准优势"""
        return self.standard_characteristics[standard.value]["优势"]
    
    def _get_standard_disadvantages(
        self, 
        standard: EstimationStandard, 
        project_info: ProjectInfo
    ) -> List[str]:
        """获取标准劣势"""
        return self.standard_characteristics[standard.value]["劣势"]
    
    def _get_recommended_scenarios(self, standard: EstimationStandard) -> List[str]:
        """获取推荐场景"""
        return self.standard_characteristics[standard.value]["适用技术栈"]
    
    def _recommend_estimation_strategy(
        self, 
        nesma_fit: float, 
        cosmic_fit: float, 
        dual_necessity: str
    ) -> EstimationStrategy:
        """推荐估算策略"""
        
        if dual_necessity == "高":
            return EstimationStrategy.DUAL_COMPARISON
        elif dual_necessity == "中":
            return EstimationStrategy.DUAL_PARALLEL
        else:
            # 选择适用性更高的单一标准
            if nesma_fit > cosmic_fit:
                return EstimationStrategy.NESMA_ONLY
            else:
                return EstimationStrategy.COSMIC_ONLY
    
    async def _parse_llm_recommendation(
        self, 
        response_content: str, 
        project_info: ProjectInfo
    ) -> StandardRecommendation:
        """解析LLM推荐结果"""
        
        # 简化实现：实际应该解析LLM的JSON响应
        # 这里提供一个默认的推荐结构
        
        nesma_fit = await self._calculate_standard_fit_score(project_info, EstimationStandard.NESMA)
        cosmic_fit = await self._calculate_standard_fit_score(project_info, EstimationStandard.COSMIC)
        
        if nesma_fit > 0.7 and cosmic_fit > 0.7:
            # 两个都很适合
            recommended_standards = [EstimationStandard.NESMA, EstimationStandard.COSMIC]
            strategy = EstimationStrategy.DUAL_COMPARISON
            confidence_score = min(nesma_fit, cosmic_fit)
        elif nesma_fit > cosmic_fit and nesma_fit > 0.6:
            recommended_standards = [EstimationStandard.NESMA]
            strategy = EstimationStrategy.NESMA_ONLY
            confidence_score = nesma_fit
        elif cosmic_fit > nesma_fit and cosmic_fit > 0.6:
            recommended_standards = [EstimationStandard.COSMIC]
            strategy = EstimationStrategy.COSMIC_ONLY
            confidence_score = cosmic_fit
        else:
            # 都不太适合，推荐双标准
            recommended_standards = [EstimationStandard.NESMA, EstimationStandard.COSMIC]
            strategy = EstimationStrategy.DUAL_PARALLEL
            confidence_score = max(nesma_fit, cosmic_fit)
        
        return StandardRecommendation(
            recommended_standards=recommended_standards,
            strategy=strategy,
            confidence_score=confidence_score,
            reasoning=response_content,
            expected_differences=f"NESMA适用性: {nesma_fit:.2f}, COSMIC适用性: {cosmic_fit:.2f}"
        )
    
    def get_recommendation_history(self) -> List[StandardRecommendation]:
        """获取推荐历史"""
        return self.recommendation_history.copy()


# 工厂函数
async def create_standard_recommender(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> StandardRecommenderAgent:
    """创建标准推荐器智能体"""
    recommender = StandardRecommenderAgent(rule_retriever=rule_retriever, llm=llm)
    await recommender.initialize()
    return recommender


if __name__ == "__main__":
    async def main():
        # 测试标准推荐器
        recommender = await create_standard_recommender()
        
        # 测试项目
        test_project = ProjectInfo(
            name="电商平台用户系统",
            description="现代化电商平台的用户管理和订单处理系统，包括实时推荐、支付集成、移动端支持",
            technology_stack=["Java", "Spring Boot", "Redis", "MySQL", "React", "微服务"],
            business_domain="电商",
            complexity_level="高"
        )
        
        # 获取推荐
        recommendation = await recommender.execute(
            "recommend_standards",
            {"project_info": test_project}
        )
        print(f"标准推荐结果: {recommendation}")
        
        # 分析项目特征
        characteristics = await recommender.execute(
            "analyze_project_characteristics", 
            {"project_info": test_project}
        )
        print(f"项目特征分析: {characteristics}")
        
        # 评估双标准必要性
        dual_evaluation = await recommender.execute(
            "evaluate_dual_standard_necessity",
            {"project_info": test_project}
        )
        print(f"双标准评估: {dual_evaluation}")
    
    asyncio.run(main()) 