"""
量子智能化功能点估算系统 - 主工作流图

基于LangGraph的状态机驱动工作流，实现NESMA和COSMIC双标准智能化估算
"""

import logging
from typing import Dict, Any, Literal, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .state_definitions import WorkflowGraphState, WorkflowState
from .node_functions import (
    # 启动和路由节点
    start_workflow_node,
    recommend_standard_node,
    route_by_standard_node,
    
    # 需求解析节点
    parse_requirements_node,
    identify_processes_node,
    
    # NESMA工作流节点
    nesma_classify_functions_node,
    nesma_calculate_complexity_node,
    nesma_calculate_ufp_node,
    
    # COSMIC工作流节点
    cosmic_identify_functional_users_node,
    cosmic_analyze_boundaries_node,
    cosmic_classify_data_movements_node,
    cosmic_calculate_cfp_node,
    
    # 知识和验证节点
    retrieve_knowledge_node,
    validate_results_node,
    
    # 对比和输出节点
    compare_standards_node,
    generate_report_node,
    
    # 错误处理节点
    handle_error_node,
    retry_failed_step_node
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
    workflow.add_node("route_by_standard", route_by_standard_node)
    
    # 需求解析节点
    workflow.add_node("parse_requirements", parse_requirements_node)
    workflow.add_node("identify_processes", identify_processes_node)
    
    # NESMA工作流节点
    workflow.add_node("nesma_classify_functions", nesma_classify_functions_node)
    workflow.add_node("nesma_calculate_complexity", nesma_calculate_complexity_node)
    workflow.add_node("nesma_calculate_ufp", nesma_calculate_ufp_node)
    
    # COSMIC工作流节点
    workflow.add_node("cosmic_identify_functional_users", cosmic_identify_functional_users_node)
    workflow.add_node("cosmic_analyze_boundaries", cosmic_analyze_boundaries_node)
    workflow.add_node("cosmic_classify_data_movements", cosmic_classify_data_movements_node)
    workflow.add_node("cosmic_calculate_cfp", cosmic_calculate_cfp_node)
    
    # 知识和验证节点
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("validate_results", validate_results_node)
    
    # 对比和输出节点
    workflow.add_node("compare_standards", compare_standards_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # 错误处理节点
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("retry_failed_step", retry_failed_step_node)
    
    # ==================== 设置入口点 ====================
    workflow.set_entry_point("start_workflow")
    
    # ==================== 添加边和条件路由 ====================
    
    # 启动流程
    workflow.add_edge("start_workflow", "recommend_standard")
    workflow.add_edge("recommend_standard", "route_by_standard")
    
    # 标准路由后的条件分支
    workflow.add_conditional_edges(
        "route_by_standard",
        route_strategy_decision,
        {
            "parse_requirements": "parse_requirements",
            "handle_error": "handle_error"
        }
    )
    
    # 需求解析流程
    workflow.add_edge("parse_requirements", "identify_processes")
    workflow.add_conditional_edges(
        "identify_processes",
        after_process_identification,
        {
            "nesma_only": "nesma_classify_functions",
            "cosmic_only": "cosmic_identify_functional_users", 
            "dual_standard": "retrieve_knowledge",
            "handle_error": "handle_error"
        }
    )
    
    # 知识检索（双标准情况）
    workflow.add_conditional_edges(
        "retrieve_knowledge",
        after_knowledge_retrieval,
        {
            "nesma_classify_functions": "nesma_classify_functions",
            "cosmic_identify_functional_users": "cosmic_identify_functional_users",
            "validate_results": "validate_results",
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
            "cosmic_identify_functional_users": "cosmic_identify_functional_users",
            "compare_standards": "compare_standards",
            "generate_report": "generate_report",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    # COSMIC工作流
    workflow.add_conditional_edges(
        "cosmic_identify_functional_users",
        after_cosmic_functional_users,
        {
            "cosmic_analyze_boundaries": "cosmic_analyze_boundaries",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "cosmic_analyze_boundaries",
        after_cosmic_boundaries,
        {
            "cosmic_classify_data_movements": "cosmic_classify_data_movements",
            "validate_results": "validate_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "cosmic_classify_data_movements",
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
    
    # 验证流程
    workflow.add_conditional_edges(
        "validate_results",
        after_validation,
        {
            "compare_standards": "compare_standards",
            "generate_report": "generate_report",
            "retry_failed_step": "retry_failed_step",
            "handle_error": "handle_error"
        }
    )
    
    # 对比和报告生成
    workflow.add_edge("compare_standards", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # 错误处理流程
    workflow.add_conditional_edges(
        "handle_error",
        after_error_handling,
        {
            "retry_failed_step": "retry_failed_step",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "retry_failed_step",
        after_retry,
        {
            "recommend_standard": "recommend_standard",
            "parse_requirements": "parse_requirements",
            "identify_processes": "identify_processes",
            "nesma_classify_functions": "nesma_classify_functions",
            "cosmic_identify_functional_users": "cosmic_identify_functional_users",
            "validate_results": "validate_results",
            "handle_error": "handle_error",
            END: END
        }
    )
    
    return workflow


# ==================== 条件路由函数 ====================

def route_strategy_decision(state: WorkflowGraphState) -> Literal["parse_requirements", "handle_error"]:
    """标准路由决策"""
    try:
        if not state.get("standard_recommendation"):
            logger.warning("缺少标准推荐结果")
            return "handle_error"
        
        if not state.get("selected_strategy"):
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
        strategy = state.get("selected_strategy")
        if not strategy:
            return "handle_error"
        
        if strategy == "NESMA_ONLY":
            return "nesma_only"
        elif strategy == "COSMIC_ONLY":
            return "cosmic_only"
        elif strategy in ["DUAL_PARALLEL", "DUAL_COMPARISON"]:
            return "dual_standard"
        else:
            logger.warning(f"未知策略: {strategy}")
            return "handle_error"
            
    except Exception as e:
        logger.error(f"流程识别后路由失败: {e}")
        return "handle_error"


def after_knowledge_retrieval(state: WorkflowGraphState) -> Literal[
    "nesma_classify_functions", "cosmic_identify_functional_users", 
    "validate_results", "handle_error"
]:
    """知识检索后的路由决策"""
    try:
        strategy = state.get("selected_strategy")
        if not strategy:
            return "handle_error"
        
        # 检查知识检索是否成功
        if not state.get("knowledge_base_results"):
            return "validate_results"  # 即使没有知识也继续，但需要验证
        
        # 根据策略决定下一步
        if strategy == "DUAL_PARALLEL":
            # 并行处理时，可以同时启动两个标准，这里选择先启动NESMA
            return "nesma_classify_functions"
        elif strategy == "DUAL_COMPARISON":
            # 对比模式时，也先启动NESMA
            return "nesma_classify_functions"
        else:
            return "handle_error"
            
    except Exception as e:
        logger.error(f"知识检索后路由失败: {e}")
        return "handle_error"


def after_nesma_classification(state: WorkflowGraphState) -> Literal[
    "nesma_calculate_complexity", "validate_results", "handle_error"
]:
    """NESMA分类后的路由决策"""
    try:
        classifications = state.get("nesma_classifications", [])
        if not classifications:
            return "validate_results"  # 没有分类结果，需要验证
        
        return "nesma_calculate_complexity"
        
    except Exception as e:
        logger.error(f"NESMA分类后路由失败: {e}")
        return "handle_error"


def after_nesma_complexity(state: WorkflowGraphState) -> Literal[
    "nesma_calculate_ufp", "validate_results", "handle_error"
]:
    """NESMA复杂度计算后的路由决策"""
    try:
        complexity_results = state.get("nesma_complexity_results", [])
        if not complexity_results:
            return "validate_results"
        
        return "nesma_calculate_ufp"
        
    except Exception as e:
        logger.error(f"NESMA复杂度计算后路由失败: {e}")
        return "handle_error"


def after_nesma_ufp(state: WorkflowGraphState) -> Literal[
    "cosmic_identify_functional_users", "compare_standards", 
    "generate_report", "validate_results", "handle_error"
]:
    """NESMA UFP计算后的路由决策"""
    try:
        strategy = state.get("selected_strategy")
        ufp_total = state.get("nesma_ufp_total")
        
        if ufp_total is None:
            return "validate_results"
        
        if strategy == "NESMA_ONLY":
            return "generate_report"
        elif strategy in ["DUAL_PARALLEL", "DUAL_COMPARISON"]:
            # 检查是否已有COSMIC结果
            cosmic_cfp = state.get("cosmic_cfp_total")
            if cosmic_cfp is not None:
                return "compare_standards"
            else:
                return "cosmic_identify_functional_users"
        else:
            return "handle_error"
            
    except Exception as e:
        logger.error(f"NESMA UFP计算后路由失败: {e}")
        return "handle_error"


def after_cosmic_functional_users(state: WorkflowGraphState) -> Literal[
    "cosmic_analyze_boundaries", "validate_results", "handle_error"
]:
    """COSMIC功能用户识别后的路由决策"""
    try:
        functional_users = state.get("cosmic_functional_users", [])
        if not functional_users:
            return "validate_results"
        
        return "cosmic_analyze_boundaries"
        
    except Exception as e:
        logger.error(f"COSMIC功能用户识别后路由失败: {e}")
        return "handle_error"


def after_cosmic_boundaries(state: WorkflowGraphState) -> Literal[
    "cosmic_classify_data_movements", "validate_results", "handle_error"
]:
    """COSMIC边界分析后的路由决策"""
    try:
        boundary_analysis = state.get("cosmic_boundary_analysis")
        if not boundary_analysis:
            return "validate_results"
        
        return "cosmic_classify_data_movements"
        
    except Exception as e:
        logger.error(f"COSMIC边界分析后路由失败: {e}")
        return "handle_error"


def after_cosmic_data_movements(state: WorkflowGraphState) -> Literal[
    "cosmic_calculate_cfp", "validate_results", "handle_error"
]:
    """COSMIC数据移动分类后的路由决策"""
    try:
        data_movements = state.get("cosmic_data_movements", [])
        if not data_movements:
            return "validate_results"
        
        return "cosmic_calculate_cfp"
        
    except Exception as e:
        logger.error(f"COSMIC数据移动分类后路由失败: {e}")
        return "handle_error"


def after_cosmic_cfp(state: WorkflowGraphState) -> Literal[
    "nesma_classify_functions", "compare_standards", 
    "generate_report", "validate_results", "handle_error"
]:
    """COSMIC CFP计算后的路由决策"""
    try:
        strategy = state.get("selected_strategy")
        cfp_total = state.get("cosmic_cfp_total")
        
        if cfp_total is None:
            return "validate_results"
        
        if strategy == "COSMIC_ONLY":
            return "generate_report"
        elif strategy in ["DUAL_PARALLEL", "DUAL_COMPARISON"]:
            # 检查是否已有NESMA结果
            nesma_ufp = state.get("nesma_ufp_total")
            if nesma_ufp is not None:
                return "compare_standards"
            else:
                return "nesma_classify_functions"
        else:
            return "handle_error"
            
    except Exception as e:
        logger.error(f"COSMIC CFP计算后路由失败: {e}")
        return "handle_error"


def after_validation(state: WorkflowGraphState) -> Literal[
    "compare_standards", "generate_report", "retry_failed_step", "handle_error"
]:
    """验证后的路由决策"""
    try:
        validation_reports = state.get("validation_reports", [])
        
        # 检查是否有严重验证问题
        has_critical_issues = any(
            report.get("severity") == "critical" 
            for report in validation_reports
        )
        
        if has_critical_issues:
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 3)
            
            if retry_count < max_retries:
                return "retry_failed_step"
            else:
                return "handle_error"
        
        # 检查是否需要对比分析
        strategy = state.get("selected_strategy")
        nesma_ufp = state.get("nesma_ufp_total")
        cosmic_cfp = state.get("cosmic_cfp_total")
        
        if strategy in ["DUAL_PARALLEL", "DUAL_COMPARISON"] and nesma_ufp is not None and cosmic_cfp is not None:
            return "compare_standards"
        else:
            return "generate_report"
            
    except Exception as e:
        logger.error(f"验证后路由失败: {e}")
        return "handle_error"


def after_error_handling(state: WorkflowGraphState) -> Literal["retry_failed_step", END]:
    """错误处理后的路由决策"""
    try:
        error_resolution = state.get("error_resolution", {})
        action = error_resolution.get("action", "terminate")
        
        if action == "retry":
            return "retry_failed_step"
        else:
            return END
            
    except Exception as e:
        logger.error(f"错误处理后路由失败: {e}")
        return END


def after_retry(state: WorkflowGraphState) -> Literal[
    "recommend_standard", "parse_requirements", "identify_processes",
    "nesma_classify_functions", "cosmic_identify_functional_users",
    "validate_results", "handle_error", END
]:
    """重试后的路由决策"""
    try:
        retry_info = state.get("retry_info", {})
        failed_step = retry_info.get("failed_step")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if retry_count >= max_retries:
            return "handle_error"
        
        # 根据失败的步骤决定重试点
        step_mapping = {
            "recommend_standard": "recommend_standard",
            "parse_requirements": "parse_requirements",
            "identify_processes": "identify_processes",
            "nesma_classify_functions": "nesma_classify_functions",
            "cosmic_identify_functional_users": "cosmic_identify_functional_users",
            "validate_results": "validate_results"
        }
        
        return step_mapping.get(failed_step, "handle_error")
        
    except Exception as e:
        logger.error(f"重试后路由失败: {e}")
        return "handle_error"


# ==================== 工作流编译和创建函数 ====================

def create_compiled_workflow() -> StateGraph:
    """创建并编译工作流图"""
    try:
        # 创建工作流图
        workflow = create_workflow_graph()
        
        # 设置检查点保存器（用于状态持久化）
        memory = MemorySaver()
        
        # 编译工作流
        compiled_workflow = workflow.compile(checkpointer=memory)
        
        logger.info("工作流图编译成功")
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"工作流图编译失败: {e}")
        raise


def get_workflow_config() -> Dict[str, Any]:
    """获取工作流配置"""
    return {
        "recursion_limit": 50,  # 最大递归深度
        "max_retries": 3,       # 最大重试次数
        "timeout": 300,         # 超时时间（秒）
        "enable_debugging": False,  # 是否启用调试模式
        "checkpoint_namespace": "fp_quantum_workflow"  # 检查点命名空间
    }


def visualize_workflow(workflow: StateGraph, output_path: str = "workflow_graph.png"):
    """可视化工作流图"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        import networkx as nx
        
        # 创建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点（这里简化处理，实际应该从workflow对象中提取）
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
        
        # 添加主要边（简化版本）
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
        
        # 绘制图形
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6)
        
        # 绘制标签
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


# ==================== 导出接口 ====================

__all__ = [
    "create_workflow_graph",
    "create_compiled_workflow", 
    "get_workflow_config",
    "visualize_workflow"
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