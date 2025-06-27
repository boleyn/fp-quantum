"""
量子智能化功能点估算系统 - 工作流编排器智能体

基于DeepSeek-R1的中枢思考智能体，负责复杂决策和工作流规划
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import BaseAgent
from models.project_models import (
    ProjectInfo, EstimationStrategy, ProcessDetails
)
from graph.state_definitions import WorkflowState, StandardRecommendation
from models.common_models import ProcessingStatus, ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class WorkflowOrchestratorAgent(BaseAgent):
    """工作流编排器智能体 - 使用DeepSeek-R1进行深度思考决策"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(agent_id="workflow_orchestrator")
        
        # 使用DeepSeek-R1作为思考模型
        self.reasoning_llm = llm or self._create_reasoning_llm()
        self.settings = get_settings()
        
        # 工作流状态管理
        self.current_state = WorkflowState.STARTING
        self.execution_plan: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        
    def _create_reasoning_llm(self) -> BaseLanguageModel:
        """创建推理LLM（DeepSeek-R1）"""
        return ChatOpenAI(
            model="deepseek-reasoner",  # DeepSeek-R1
            api_key=self.settings.llm.deepseek_api_key,
            base_url=self.settings.llm.deepseek_api_base,
            temperature=0.1,
            max_tokens=8000,
            timeout=60  # 给思考链足够时间
        )
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行编排任务"""
        if task_name == "plan_estimation_strategy":
            return await self.plan_estimation_strategy(inputs["project_info"])
        elif task_name == "analyze_standard_fit":
            return await self.analyze_standard_fit(
                inputs["project_info"], 
                inputs.get("additional_context", {})
            )
        elif task_name == "handle_execution_error":
            return await self.handle_execution_error(
                inputs["error_info"],
                inputs["current_state"],
                inputs.get("retry_count", 0)
            )
        elif task_name == "optimize_workflow":
            return await self.optimize_workflow(
                inputs["current_results"],
                inputs["performance_metrics"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def plan_estimation_strategy(self, project_info: ProjectInfo) -> Dict[str, Any]:
        """使用R1思考链制定估算策略"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个功能点估算专家，需要深度分析项目特征，制定最优的估算策略。
            
请使用<thinking>标签进行深度思考，考虑以下因素：
1. 项目规模和复杂度
2. 技术栈特性和业务领域
3. NESMA vs COSMIC标准的适用性
4. 预期精度要求和时间约束
5. 风险评估和应对方案

基于思考结果，制定详细的执行策略。"""),
            ("human", """项目信息：
项目名称：{name}
项目描述：{description}
技术栈：{technology_stack}
业务领域：{business_domain}
复杂度级别：{complexity_level}
用户偏好：{user_preferences}

请制定最优的功能点估算策略。""")
        ])
        
        response = await self.reasoning_llm.ainvoke(
            prompt.format_messages(
                name=project_info.name,
                description=project_info.description,
                technology_stack=", ".join(project_info.technology_stack),
                business_domain=project_info.business_domain,
                complexity_level=project_info.complexity_level or "未知",
                user_preferences=str(project_info.user_preferences)
            )
        )
        
        # 解析推理结果
        strategy_result = await self._parse_strategy_response(response.content, project_info)
        
        # 记录决策历史
        self._record_decision("plan_estimation_strategy", {
            "project_info": project_info.dict(),
            "strategy": strategy_result,
            "reasoning": response.content
        })
        
        return strategy_result
    
    async def analyze_standard_fit(
        self, 
        project_info: ProjectInfo,
        additional_context: Dict[str, Any] = None
    ) -> StandardRecommendation:
        """分析标准适用性"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是功能点估算标准专家，需要深度分析NESMA和COSMIC标准的适用性。

<thinking>
请考虑以下标准选择原则：

NESMA适用场景：
- 传统企业应用系统
- 数据处理为主的业务系统
- 需要与历史数据对比的项目
- 瀑布式开发模式

COSMIC适用场景：
- 现代软件应用
- 实时系统和嵌入式软件
- 面向服务的架构
- 敏捷/DevOps开发模式
- 复杂的用户交互系统

需要综合评估：
1. 项目技术特征与标准匹配度
2. 业务领域的标准偏好
3. 团队熟悉度和历史经验
4. 客户需求和交付标准
5. 维护和更新的便利性
</thinking>

基于深度分析，推荐最适合的标准组合。"""),
            ("human", """项目信息：
- 技术栈：{technology_stack}
- 业务领域：{business_domain}
- 项目描述：{description}
- 额外上下文：{additional_context}

请推荐最适合的功能点估算标准。""")
        ])
        
        response = await self.reasoning_llm.ainvoke(
            prompt.format_messages(
                technology_stack=", ".join(project_info.technology_stack),
                business_domain=project_info.business_domain,
                description=project_info.description,
                additional_context=str(additional_context or {})
            )
        )
        
        # 解析标准推荐
        recommendation = await self._parse_standard_recommendation(response.content, project_info)
        
        self._record_decision("analyze_standard_fit", {
            "project_info": project_info.dict(),
            "recommendation": recommendation.dict(),
            "reasoning": response.content
        })
        
        return recommendation
    
    async def handle_execution_error(
        self,
        error_info: Dict[str, Any],
        current_state: WorkflowState,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """智能错误处理和恢复策略"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能工作流错误处理专家，需要分析错误原因并制定恢复策略。

<thinking>
错误分析维度：
1. 错误类型和严重性
2. 错误发生的工作流阶段
3. 已尝试的重试次数
4. 系统状态和资源情况
5. 用户影响程度

恢复策略选项：
- 直接重试（简单错误）
- 调整参数重试（参数问题）
- 切换策略（策略不当）
- 降级处理（资源不足）
- 人工介入（复杂错误）
- 终止流程（致命错误）
</thinking>

制定具体的错误恢复方案。"""),
            ("human", """错误信息：
错误类型：{error_type}
错误消息：{error_message}
当前状态：{current_state}
重试次数：{retry_count}
错误时间：{occurred_at}

请制定恢复策略。""")
        ])
        
        response = await self.reasoning_llm.ainvoke(
            prompt.format_messages(
                error_type=error_info.get("error_type", "未知"),
                error_message=error_info.get("error_message", ""),
                current_state=current_state.value,
                retry_count=retry_count,
                occurred_at=error_info.get("occurred_at", datetime.now())
            )
        )
        
        # 解析恢复策略
        recovery_plan = await self._parse_recovery_strategy(response.content, error_info)
        
        self._record_decision("handle_execution_error", {
            "error_info": error_info,
            "current_state": current_state.value,
            "retry_count": retry_count,
            "recovery_plan": recovery_plan,
            "reasoning": response.content
        })
        
        return recovery_plan
    
    async def optimize_workflow(
        self,
        current_results: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """工作流优化建议"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是工作流优化专家，基于执行结果和性能指标提供优化建议。

<thinking>
优化分析维度：
1. 执行效率和时间消耗
2. 结果质量和准确性
3. 资源使用和成本效益
4. 用户体验和满意度
5. 系统稳定性和可靠性

优化策略：
- 并行化处理
- 缓存优化
- 参数调优
- 流程简化
- 资源调配
- 算法优化
</thinking>

提供具体的优化方案。"""),
            ("human", """当前结果：{current_results}
性能指标：{performance_metrics}

请提供工作流优化建议。""")
        ])
        
        response = await self.reasoning_llm.ainvoke(
            prompt.format_messages(
                current_results=str(current_results),
                performance_metrics=str(performance_metrics)
            )
        )
        
        optimization_plan = await self._parse_optimization_plan(response.content)
        
        self._record_decision("optimize_workflow", {
            "current_results": current_results,
            "performance_metrics": performance_metrics,
            "optimization_plan": optimization_plan,
            "reasoning": response.content
        })
        
        return optimization_plan
    
    async def _parse_strategy_response(
        self, 
        response_content: str, 
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """解析策略响应"""
        # 这里可以使用更精细的解析逻辑，或调用LLM进行结构化提取
        return {
            "recommended_strategy": EstimationStrategy.DUAL_PARALLEL,
            "confidence_score": 0.85,
            "reasoning": response_content,
            "execution_order": ["standard_identification", "process_analysis", "parallel_estimation"],
            "risk_factors": ["技术栈复杂性", "业务需求不明确"],
            "mitigation_plans": ["增加技术分析", "细化需求澄清"]
        }
    
    async def _parse_standard_recommendation(
        self, 
        response_content: str, 
        project_info: ProjectInfo
    ) -> StandardRecommendation:
        """解析标准推荐"""
        # 简化实现，实际可以使用更复杂的解析逻辑
        return StandardRecommendation(
            recommended_standard="NESMA+COSMIC",
            confidence_score=0.8,
            reasoning=response_content,
            alternative_standards=["NESMA", "COSMIC"]
        )
    
    async def _parse_recovery_strategy(
        self, 
        response_content: str, 
        error_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析恢复策略"""
        return {
            "action": "retry_with_adjustment",
            "adjustments": ["降低并发度", "增加超时时间"],
            "max_retries": 2,
            "fallback_strategy": "降级处理",
            "reasoning": response_content
        }
    
    async def _parse_optimization_plan(self, response_content: str) -> Dict[str, Any]:
        """解析优化计划"""
        return {
            "optimizations": [
                {"type": "cache", "description": "增加结果缓存"},
                {"type": "parallel", "description": "并行处理独立任务"}
            ],
            "expected_improvement": "30%性能提升",
            "implementation_priority": "high",
            "reasoning": response_content
        }
    
    def _record_decision(self, decision_type: str, details: Dict[str, Any]):
        """记录决策历史"""
        decision_record = {
            "timestamp": datetime.now(),
            "decision_type": decision_type,
            "details": details,
            "session_id": self.session_id
        }
        self.decision_history.append(decision_record)
        
        # 限制历史记录数量
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-50:]
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """获取决策历史"""
        return self.decision_history.copy()
    
    def get_current_execution_plan(self) -> List[Dict[str, Any]]:
        """获取当前执行计划"""
        return self.execution_plan.copy()


# 工厂函数
async def create_workflow_orchestrator(llm: Optional[BaseLanguageModel] = None) -> WorkflowOrchestratorAgent:
    """创建工作流编排器智能体"""
    orchestrator = WorkflowOrchestratorAgent(llm=llm)
    await orchestrator.initialize()
    return orchestrator


if __name__ == "__main__":
    async def main():
        # 测试工作流编排器
        orchestrator = await create_workflow_orchestrator()
        
        # 测试项目
        test_project = ProjectInfo(
            name="电商平台功能点估算",
            description="大型电商平台的用户管理和订单处理系统",
            technology_stack=["Java", "Spring Boot", "MySQL", "Redis"],
            business_domain="电商",
            complexity_level="高"
        )
        
        # 制定策略
        strategy = await orchestrator.execute(
            "plan_estimation_strategy",
            {"project_info": test_project}
        )
        print(f"估算策略: {strategy}")
        
        # 分析标准适用性
        recommendation = await orchestrator.execute(
            "analyze_standard_fit",
            {"project_info": test_project}
        )
        print(f"标准推荐: {recommendation}")
    
    asyncio.run(main()) 