"""
量子智能化功能点估算系统 - 主工作流图

基于LangGraph的状态机驱动工作流，实现NESMA和COSMIC双标准智能化估算
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Literal, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .state_definitions import WorkflowGraphState, WorkflowState, ProcessingStage
from .node_functions import (
    # 启动和路由节点
    start_workflow_node,
    recommend_standard_node,
    
    # 需求解析节点
    parse_requirements_node,
    identify_processes_node,
    
    # NESMA工作流节点
    nesma_classify_functions_node,
    nesma_calculate_complexity_node,
    nesma_calculate_ufp_node,
    
    # COSMIC工作流节点
    cosmic_identify_users_node,
    cosmic_analyze_boundary_node,
    cosmic_classify_movements_node,
    cosmic_calculate_cfp_node,
    
    # 验证节点
    validate_results_node,
    
    # 对比和输出节点
    compare_standards_node,
    generate_report_node,
    
    # 错误处理节点
    handle_error_node,
    complete_workflow_node
)

logger = logging.getLogger(__name__)


def create_workflow_graph() -> StateGraph:
    """创建主工作流图"""
    
    # 创建状态图
    workflow = StateGraph(WorkflowGraphState)
    
    # ==================== 添加节点 ====================
    
    # 启动和路由节点
    workflow.add_node("start_workflow", start_workflow_node)
    workflow.add_node("recommend_standard", recommend_standard_node)
    
    # 需求解析节点
    workflow.add_node("parse_requirements", parse_requirements_node)
    workflow.add_node("identify_processes", identify_processes_node)
    
    # NESMA工作流节点
    workflow.add_node("nesma_classify_functions", nesma_classify_functions_node)
    workflow.add_node("nesma_calculate_complexity", nesma_calculate_complexity_node)
    workflow.add_node("nesma_calculate_ufp", nesma_calculate_ufp_node)
    
    # COSMIC工作流节点
    workflow.add_node("cosmic_identify_users", cosmic_identify_users_node)
    workflow.add_node("cosmic_analyze_boundary", cosmic_analyze_boundary_node)
    workflow.add_node("cosmic_classify_movements", cosmic_classify_movements_node)
    workflow.add_node("cosmic_calculate_cfp", cosmic_calculate_cfp_node)
    
    # 验证节点
    workflow.add_node("validate_results", validate_results_node)
    
    # 对比和输出节点
    workflow.add_node("compare_standards", compare_standards_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # 错误处理节点
    workflow.add_node("handle_error", handle_error_node)
    
    # ==================== 设置入口点 ====================
    workflow.set_entry_point("start_workflow")
    
    # ==================== 添加边和条件路由 ====================
    
    # 启动流程
    workflow.add_edge("start_workflow", "recommend_standard")
    workflow.add_edge("recommend_standard", "parse_requirements")
    
    # 需求解析流程
    workflow.add_edge("parse_requirements", "identify_processes")
    workflow.add_conditional_edges(
        "identify_processes",
        after_process_identification,
        {
            "nesma_only": "nesma_classify_functions",
            "cosmic_only": "cosmic_identify_users", 
            "dual_standard": "nesma_classify_functions",  # 双标准时先走NESMA，然后会路由到COSMIC
            "handle_error": "handle_error"
        }
    )
    
    # NESMA工作流
    workflow.add_conditional_edges(
        "nesma_classify_functions",
        after_nesma_classification,
        {
            "nesma_calculate_complexity": "nesma_calculate_complexity",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "nesma_calculate_complexity",
        after_nesma_complexity,
        {
            "nesma_calculate_ufp": "nesma_calculate_ufp",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "nesma_calculate_ufp",
        after_nesma_ufp,
        {
            "cosmic_identify_users": "cosmic_identify_users",
            "compare_standards": "compare_standards",
            "generate_report": "generate_report",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    # COSMIC工作流
    workflow.add_conditional_edges(
        "cosmic_identify_users",
        after_cosmic_functional_users,
        {
            "cosmic_analyze_boundary": "cosmic_analyze_boundary",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "cosmic_analyze_boundary",
        after_cosmic_boundaries,
        {
            "cosmic_classify_movements": "cosmic_classify_movements",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "cosmic_classify_movements",
        after_cosmic_data_movements,
        {
            "cosmic_calculate_cfp": "cosmic_calculate_cfp",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "cosmic_calculate_cfp",
        after_cosmic_cfp,
        {
            "nesma_classify_functions": "nesma_classify_functions",
            "compare_standards": "compare_standards",
            "generate_report": "generate_report",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    # 验证结果
    workflow.add_conditional_edges(
        "validate_results",
        after_validation,
        {
            "compare_standards": "compare_standards",
            "generate_report": "generate_report",
            "handle_error": "handle_error"
        }
    )
    
    # 对比和输出
    workflow.add_edge("compare_standards", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # 错误处理
    workflow.add_conditional_edges(
        "handle_error",
        after_error_handling,
        {
            "generate_report": "generate_report",
            END: END
        }
    )
    
    return workflow


# ==================== 条件路由函数 ====================

def route_strategy_decision(state: WorkflowGraphState) -> Literal["parse_requirements", "handle_error"]:
    """标准路由决策"""
    try:
        if not state.standard_recommendation:
            logger.warning("缺少标准推荐结果")
            return "handle_error"
        
        if not state.selected_strategy:
            logger.warning("缺少选择的策略")
            return "handle_error"
        
        return "parse_requirements"
        
    except Exception as e:
        logger.error(f"标准路由决策失败: {e}")
        return "handle_error"


def after_process_identification(state: WorkflowGraphState) -> Literal[
    "nesma_only", "cosmic_only", "dual_standard", "handle_error"
]:
    """流程识别后的路由决策"""
    try:
        strategy = state.selected_strategy
        
        if not strategy:
            logger.error("缺少估算策略信息")
            return "handle_error"
        
        from models.common_models import EstimationStrategy
        
        if strategy == EstimationStrategy.NESMA_ONLY:
            logger.info("🎯 选择NESMA单一标准估算")
            return "nesma_only"
        elif strategy == EstimationStrategy.COSMIC_ONLY:
            logger.info("🎯 选择COSMIC单一标准估算")
            return "cosmic_only"
        elif strategy == EstimationStrategy.DUAL_PARALLEL:
            logger.info("🎯 选择双标准并行估算")
            return "dual_standard"
        else:
            logger.error(f"未知的估算策略: {strategy}")
            return "handle_error"
            
    except Exception as e:
        logger.error(f"流程识别后路由失败: {str(e)}")
        return "handle_error"


# 删除不使用的after_knowledge_retrieval函数


def after_nesma_classification(state: WorkflowGraphState) -> Literal[
    "nesma_calculate_complexity", "validate_results", "handle_error"
]:
    """NESMA分类后的路由决策"""
    try:
        if not state.nesma_results:
            return "validate_results"
        
        classifications = state.nesma_classifications or []
        
        if not classifications:
            return "validate_results"
        
        return "nesma_calculate_complexity"
        
    except Exception as e:
        logger.error(f"NESMA分类后路由失败: {str(e)}")
        return "handle_error"


def after_nesma_complexity(state: WorkflowGraphState) -> Literal[
    "nesma_calculate_ufp", "validate_results", "handle_error"
]:
    """NESMA复杂度计算后的路由决策"""
    try:
        if not state.nesma_results:
            return "validate_results"
        
        complexity_results = state.nesma_complexity_results or []
        
        if not complexity_results:
            return "validate_results"
        
        return "nesma_calculate_ufp"
        
    except Exception as e:
        logger.error(f"NESMA复杂度计算后路由失败: {str(e)}")
        return "handle_error"


def after_nesma_ufp(state: WorkflowGraphState) -> Literal[
    "cosmic_identify_users", "compare_standards", 
    "generate_report", "validate_results", "handle_error"
]:
    """NESMA UFP计算后的路由决策"""
    try:
        strategy = state.selected_strategy
        
        if not state.nesma_results or not state.nesma_results.total_ufp:
            return "validate_results"
        
        from models.common_models import EstimationStrategy
        
        if strategy == EstimationStrategy.NESMA_ONLY:
            return "generate_report"
        elif strategy == EstimationStrategy.DUAL_PARALLEL:
            if state.cosmic_results and state.cosmic_results.total_cfp:
                return "compare_standards"
            else:
                return "cosmic_identify_users"
        else:
            return "validate_results"
            
    except Exception as e:
        logger.error(f"NESMA UFP计算后路由失败: {str(e)}")
        return "handle_error"


def after_cosmic_functional_users(state: WorkflowGraphState) -> Literal[
    "cosmic_analyze_boundary", "validate_results", "handle_error"
]:
    """COSMIC功能用户识别后的路由决策"""
    try:
        if not state.cosmic_results:
            return "validate_results"
        
        functional_users = state.cosmic_functional_users or []
        
        if not functional_users:
            return "validate_results"
        
        return "cosmic_analyze_boundary"
        
    except Exception as e:
        logger.error(f"COSMIC功能用户识别后路由失败: {str(e)}")
        return "handle_error"


def after_cosmic_boundaries(state: WorkflowGraphState) -> Literal[
    "cosmic_classify_movements", "validate_results", "handle_error"
]:
    """COSMIC边界分析后的路由决策"""
    try:
        if not state.cosmic_results or not state.cosmic_boundary_analysis:
            return "validate_results"
        
        return "cosmic_classify_movements"
        
    except Exception as e:
        logger.error(f"COSMIC边界分析后路由失败: {str(e)}")
        return "handle_error"


def after_cosmic_data_movements(state: WorkflowGraphState) -> Literal[
    "cosmic_calculate_cfp", "validate_results", "handle_error"
]:
    """COSMIC数据移动分类后的路由决策"""
    try:
        if not state.cosmic_results:
            return "validate_results"
        
        data_movements = state.cosmic_data_movements or []
        
        if not data_movements:
            return "validate_results"
        
        return "cosmic_calculate_cfp"
        
    except Exception as e:
        logger.error(f"COSMIC数据移动分类后路由失败: {str(e)}")
        return "handle_error"


def after_cosmic_cfp(state: WorkflowGraphState) -> Literal[
    "nesma_classify_functions", "compare_standards", 
    "generate_report", "validate_results", "handle_error"
]:
    """COSMIC CFP计算后的路由决策"""
    try:
        strategy = state.selected_strategy
        
        if not state.cosmic_results or not state.cosmic_results.total_cfp:
            return "validate_results"
        
        from models.common_models import EstimationStrategy
        
        if strategy == EstimationStrategy.COSMIC_ONLY:
            return "generate_report"
        elif strategy == EstimationStrategy.DUAL_PARALLEL:
            if state.nesma_results and state.nesma_results.total_ufp:
                return "compare_standards"
            else:
                return "nesma_classify_functions"
        else:
            return "validate_results"
            
    except Exception as e:
        logger.error(f"COSMIC CFP计算后路由失败: {str(e)}")
        return "handle_error"


def after_validation(state: WorkflowGraphState) -> Literal[
    "compare_standards", "generate_report", "handle_error"
]:
    """验证后的路由决策"""
    try:
        validation_results = state.validation_results
        logger.info(f"🔍 验证后路由 - 验证结果: {validation_results}")
        
        is_valid = True
        
        if validation_results:
            if hasattr(validation_results, 'validation_status'):
                is_valid = validation_results.validation_status != "failed"
            elif isinstance(validation_results, dict):
                overall_validation = validation_results.get("overall_validation")
                if overall_validation and hasattr(overall_validation, 'is_valid'):
                    is_valid = overall_validation.is_valid
                elif isinstance(overall_validation, dict):
                    is_valid = overall_validation.get("is_valid", True)
            elif isinstance(validation_results, list) and validation_results:
                first_result = validation_results[0]
                if hasattr(first_result, 'is_valid'):
                    is_valid = first_result.is_valid
                elif isinstance(first_result, dict):
                    is_valid = first_result.get("is_valid", True)
        
        logger.info(f"🔍 验证状态: is_valid={is_valid}")
        
        if not is_valid:
            logger.warning("验证失败，跳转到错误处理")
            return "handle_error"
        
        strategy = state.selected_strategy
        
        has_nesma = bool(state.nesma_results and state.nesma_results.total_ufp)
        has_cosmic = bool(state.cosmic_results and state.cosmic_results.total_cfp)
        
        logger.info(f"🔍 结果检查: strategy={strategy}, NESMA={has_nesma}, COSMIC={has_cosmic}")
        
        from models.common_models import EstimationStrategy
        
        if (strategy == EstimationStrategy.DUAL_PARALLEL and 
            has_nesma and has_cosmic):
            return "compare_standards"
        else:
            return "generate_report"
            
    except Exception as e:
        logger.error(f"验证后路由失败: {str(e)}")
        import traceback
        logger.error(f"验证后路由异常堆栈: {traceback.format_exc()}")
        return "handle_error"


def after_error_handling(state: WorkflowGraphState) -> Literal["generate_report", END]:
    """错误处理后的路由决策"""
    try:
        if not state:
            logger.error("状态为空，终止工作流")
            return END
        
        current_state = state.current_state
        
        if current_state == WorkflowState.REPORT_GENERATION_PENDING:
            return "generate_report"
        
        try:
            has_nesma_results = bool(state.nesma_results and state.nesma_results.total_ufp)
            has_cosmic_results = bool(state.cosmic_results and state.cosmic_results.total_cfp)
            
            logger.info(f"🔍 错误处理路由检查: NESMA={has_nesma_results}, COSMIC={has_cosmic_results}")
            
            if has_nesma_results or has_cosmic_results:
                return "generate_report"
            else:
                return END
                
        except Exception as e:
            logger.error(f"❌ 检查结果状态时出错: {e}")
            return END
            
    except Exception as e:
        logger.error(f"错误处理后路由失败: {e}")
        return END


def after_retry(state: WorkflowGraphState) -> Literal[
    "recommend_standard", "parse_requirements", "identify_processes",
    "nesma_classify_functions", "cosmic_identify_users",
    "validate_results", "handle_error", END
]:
    """重试后的路由决策"""
    try:
        retry_info = state.retry_info or {}
        failed_step = retry_info.get("failed_step")
        retry_count = state.retry_count or 0
        max_retries = state.max_retries or 3
        
        if retry_count >= max_retries:
            return "handle_error"
        
        step_mapping = {
            "recommend_standard": "recommend_standard",
            "parse_requirements": "parse_requirements",
            "identify_processes": "identify_processes",
            "nesma_classify_functions": "nesma_classify_functions",
            "cosmic_identify_users": "cosmic_identify_users",
            "validate_results": "validate_results"
        }
        
        return step_mapping.get(failed_step, "handle_error")
        
    except Exception as e:
        logger.error(f"重试后路由失败: {e}")
        return "handle_error"


# ==================== 工作流编译和创建函数 ====================

def create_compiled_workflow() -> StateGraph:
    """创建编译后的工作流"""
    try:
        workflow = create_workflow_graph()
        compiled_workflow = workflow.compile()
        logger.info("✅ 工作流编译完成")
        return compiled_workflow
    except Exception as e:
        logger.error(f"❌ 工作流编译失败: {str(e)}")
        raise


def get_workflow_config() -> Dict[str, Any]:
    """获取工作流配置"""
    return {
        "version": "1.0.0",
        "max_retries": 3,
        "timeout_seconds": 300,
        "checkpointing_enabled": True
    }


def visualize_workflow(workflow: StateGraph, output_path: str = "workflow_graph.png"):
    """可视化工作流图"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        import networkx as nx
        
        G = nx.DiGraph()
        
        nodes = [
            "start_workflow", "recommend_standard", "route_by_standard",
            "parse_requirements", "identify_processes", "retrieve_knowledge",
            "nesma_classify_functions", "nesma_calculate_complexity", "nesma_calculate_ufp",
            "cosmic_identify_functional_users", "cosmic_analyze_boundaries", 
            "cosmic_classify_data_movements", "cosmic_calculate_cfp",
            "validate_results", "compare_standards", "generate_report",
            "handle_error", "retry_failed_step"
        ]
        
        G.add_nodes_from(nodes)
        
        edges = [
            ("start_workflow", "recommend_standard"),
            ("recommend_standard", "route_by_standard"),
            ("route_by_standard", "parse_requirements"),
            ("parse_requirements", "identify_processes"),
            ("identify_processes", "nesma_classify_functions"),
            ("identify_processes", "cosmic_identify_functional_users"),
            ("nesma_classify_functions", "nesma_calculate_complexity"),
            ("nesma_calculate_complexity", "nesma_calculate_ufp"),
            ("cosmic_identify_functional_users", "cosmic_analyze_boundaries"),
            ("cosmic_analyze_boundaries", "cosmic_classify_data_movements"),
            ("cosmic_classify_data_movements", "cosmic_calculate_cfp"),
            ("nesma_calculate_ufp", "compare_standards"),
            ("cosmic_calculate_cfp", "compare_standards"),
            ("compare_standards", "generate_report")
        ]
        
        G.add_edges_from(edges)
        
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.7)
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title("量子智能化功能点估算系统 - 工作流图", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"工作流图已保存到: {output_path}")
        
    except ImportError:
        logger.warning("matplotlib 和 networkx 未安装，无法生成可视化图形")
    except Exception as e:
        logger.error(f"生成工作流可视化失败: {e}")


# ==================== 工作流包装类 ====================

class FPEstimationWorkflow:
    """功能点估算工作流包装类"""
    
    def __init__(self):
        """初始化工作流"""
        self.compiled_workflow = None
        self.current_state = None
        self.session_id = None
        
    async def initialize(self, project_info, strategy, requirements, user_preferences=None):
        """初始化工作流状态"""
        import uuid
        from datetime import datetime
        from models.common_models import EstimationStrategy
        
        self.session_id = str(uuid.uuid4())
        
        self.current_state = {
            "session_id": self.session_id,
            "project_info": project_info,
            "current_state": WorkflowState.STARTING,
            "execution_log": [],
            
            "user_requirements": requirements,
            "user_preferences": user_preferences or {},
            
            "standard_recommendation": None,
            "selected_strategy": strategy,
            
            "requirement_analysis": None,
            "identified_processes": None,
            "current_process_context": None,
            
            "nesma_results": None,
            "nesma_classifications": [],
            "nesma_complexity_results": [],
            "nesma_estimation_result": None,
            
            "cosmic_results": None,
            "cosmic_functional_users": [],
            "cosmic_data_movements": [],
            "cosmic_boundary_analysis": None,
            "cosmic_estimation_result": None,
            
            "validation_results": None,
            
            "comparison_analysis": None,
            "final_report": None,
            
            "error_message": None,
            "retry_count": 0,
            "max_retries": 3,
            
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "processing_stats": {}
        }
        
        self.compiled_workflow = create_compiled_workflow()
        
        return self.session_id
    
    async def execute(self):
        """执行工作流"""
        if not self.compiled_workflow or not self.current_state:
            raise ValueError("工作流未正确初始化")
        
        try:
            final_state = await self.compiled_workflow.ainvoke(
                self.current_state,
                config={"configurable": {"thread_id": self.session_id}}
            )
            
            self.current_state = final_state
            return final_state
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            if self.current_state:
                self.current_state["error_message"] = str(e)
                self.current_state["current_state"] = WorkflowState.ERROR_ENCOUNTERED
            raise
    
    async def get_current_state(self):
        """获取当前状态"""
        return self.current_state
    
    def get_session_id(self):
        """获取会话ID"""
        return self.session_id


# ==================== 向后兼容函数 ====================

async def create_function_point_workflow():
    """创建功能点估算工作流 - 向后兼容的函数"""
    return create_compiled_workflow()


def create_workflow():
    """创建工作流 - 另一个向后兼容的函数"""
    return create_compiled_workflow()


# ==================== 导出接口 ====================

__all__ = [
    "create_workflow_graph",
    "create_compiled_workflow", 
    "get_workflow_config",
    "visualize_workflow",
    "FPEstimationWorkflow"
]


if __name__ == "__main__":
    # 测试工作流创建
    try:
        workflow = create_compiled_workflow()
        print("✅ 工作流图创建成功")
        
        # 生成可视化图
        try:
            visualize_workflow(workflow)
            print("✅ 工作流可视化图生成成功")
        except Exception as e:
            print(f"⚠️ 工作流可视化失败: {e}")
            
    except Exception as e:
        print(f"❌ 工作流图创建失败: {e}") 