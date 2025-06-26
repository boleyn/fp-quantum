"""
量子智能化功能点估算系统 - 工作流状态定义

定义LangGraph工作流中使用的所有状态结构
"""

from enum import Enum
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

from models.project_models import ProjectInfo, StandardRecommendation, EstimationStrategy
from models.nesma_models import (
    NESMAFunctionClassification, 
    NESMAComplexityCalculation, 
    NESMAEstimationResult
)
from models.cosmic_models import (
    COSMICFunctionalUser, 
    COSMICDataMovement, 
    COSMICBoundaryAnalysis,
    COSMICEstimationResult
)
from models.common_models import ProcessingStatus, ValidationResult


class ProcessingStage(str, Enum):
    """处理阶段枚举"""
    INITIALIZATION = "INITIALIZATION"
    REQUIREMENT_ANALYSIS = "REQUIREMENT_ANALYSIS"
    STANDARD_SELECTION = "STANDARD_SELECTION"
    NESMA_PROCESSING = "NESMA_PROCESSING"
    COSMIC_PROCESSING = "COSMIC_PROCESSING"
    VALIDATION = "VALIDATION"
    COMPARISON = "COMPARISON"
    REPORT_GENERATION = "REPORT_GENERATION"
    COMPLETION = "COMPLETION"
    ERROR_HANDLING = "ERROR_HANDLING"


class WorkflowState(str, Enum):
    """工作流状态枚举"""
    # 初始化状态
    STARTING = "STARTING"
    REQUIREMENT_INPUT_RECEIVED = "REQUIREMENT_INPUT_RECEIVED"
    
    # 标准选择阶段
    STANDARD_IDENTIFICATION_PENDING = "STANDARD_IDENTIFICATION_PENDING"
    STANDARD_RECOMMENDATION_READY = "STANDARD_RECOMMENDATION_READY"
    STANDARD_ROUTING_COMPLETED = "STANDARD_ROUTING_COMPLETED"
    
    # 需求解析阶段
    PROCESS_IDENTIFICATION_PENDING = "PROCESS_IDENTIFICATION_PENDING"
    PROCESSES_IDENTIFIED = "PROCESSES_IDENTIFIED"
    
    # NESMA处理阶段
    NESMA_PROCESSING_PENDING = "NESMA_PROCESSING_PENDING"
    NESMA_CLASSIFICATION_COMPLETED = "NESMA_CLASSIFICATION_COMPLETED"
    NESMA_CALCULATION_COMPLETED = "NESMA_CALCULATION_COMPLETED"
    
    # COSMIC处理阶段
    COSMIC_PROCESSING_PENDING = "COSMIC_PROCESSING_PENDING"
    COSMIC_ANALYSIS_COMPLETED = "COSMIC_ANALYSIS_COMPLETED"
    COSMIC_CALCULATION_COMPLETED = "COSMIC_CALCULATION_COMPLETED"
    
    # 最终阶段
    CROSS_STANDARD_COMPARISON_PENDING = "CROSS_STANDARD_COMPARISON_PENDING"
    REPORT_GENERATION_PENDING = "REPORT_GENERATION_PENDING"
    COMPLETED = "COMPLETED"
    
    # 错误处理
    ERROR_ENCOUNTERED = "ERROR_ENCOUNTERED"
    TERMINATED = "TERMINATED"


class ProcessDetails(TypedDict):
    """功能流程详情"""
    id: str
    name: str
    description: str
    data_groups: List[str]
    dependencies: List[str]
    metadata: Dict[str, Any]


class WorkflowGraphState(TypedDict):
    """LangGraph工作流状态"""
    # 基础信息
    session_id: str
    project_info: ProjectInfo
    current_state: WorkflowState
    execution_log: List[Dict[str, Any]]
    
    # 标准选择
    standard_recommendation: Optional[StandardRecommendation]
    selected_strategy: Optional[EstimationStrategy]
    
    # 需求解析结果
    identified_processes: Optional[List[ProcessDetails]]
    current_process_context: Optional[ProcessDetails]
    
    # NESMA结果
    nesma_classifications: List[NESMAFunctionClassification]
    nesma_complexity_results: List[NESMAComplexityCalculation]
    nesma_estimation_result: Optional[NESMAEstimationResult]
    
    # COSMIC结果
    cosmic_functional_users: List[COSMICFunctionalUser]
    cosmic_data_movements: List[COSMICDataMovement]
    cosmic_boundary_analysis: Optional[COSMICBoundaryAnalysis]
    cosmic_estimation_result: Optional[COSMICEstimationResult]
    
    # 质量验证
    validation_results: List[ValidationResult]
    
    # 最终输出
    comparison_analysis: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    
    # 错误处理
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    
    # 元数据
    created_at: datetime
    updated_at: datetime
    processing_stats: Dict[str, Any]


def create_initial_state(
    session_id: str,
    project_info: ProjectInfo
) -> WorkflowGraphState:
    """创建初始工作流状态"""
    return WorkflowGraphState(
        session_id=session_id,
        project_info=project_info,
        current_state=WorkflowState.STARTING,
        execution_log=[],
        
        standard_recommendation=None,
        selected_strategy=None,
        
        identified_processes=None,
        current_process_context=None,
        
        nesma_classifications=[],
        nesma_complexity_results=[],
        nesma_estimation_result=None,
        
        cosmic_functional_users=[],
        cosmic_data_movements=[],
        cosmic_boundary_analysis=None,
        cosmic_estimation_result=None,
        
        validation_results=[],
        
        comparison_analysis=None,
        final_report=None,
        
        error_message=None,
        retry_count=0,
        max_retries=3,
        
        created_at=datetime.now(),
        updated_at=datetime.now(),
        processing_stats={}
    )


def transition_state(
    state: WorkflowGraphState, 
    new_state: WorkflowState, 
    message: str = ""
) -> WorkflowGraphState:
    """状态转换"""
    old_state = state["current_state"]
    state["current_state"] = new_state
    state["updated_at"] = datetime.now()
    
    # 记录状态转换日志
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "state_transition",
        "agent_id": "workflow_manager",
        "from_state": old_state,
        "to_state": new_state,
        "message": message,
        "status": "success"
    }
    state["execution_log"].append(log_entry)
    
    return state


def update_execution_log(
    state: WorkflowGraphState,
    agent_id: str,
    action: str,
    status: str,
    duration_ms: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None
) -> WorkflowGraphState:
    """更新执行日志"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "action": action,
        "status": status,
        "duration_ms": duration_ms,
        "details": details or {}
    }
    
    state["execution_log"].append(log_entry)
    state["updated_at"] = datetime.now()
    
    return state


def is_retry_needed(state: WorkflowGraphState) -> bool:
    """检查是否需要重试"""
    return (
        state["current_state"] == WorkflowState.ERROR_ENCOUNTERED and 
        state["retry_count"] < state["max_retries"]
    )


def increment_retry_attempt(state: WorkflowGraphState) -> WorkflowGraphState:
    """增加重试次数"""
    state["retry_count"] += 1
    state["updated_at"] = datetime.now()
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "retry_attempt",
        "agent_id": "workflow_manager",
        "retry_count": state["retry_count"],
        "max_retries": state["max_retries"],
        "status": "info"
    }
    state["execution_log"].append(log_entry)
    
    return state