"""
量子智能化功能点估算系统 - 工作流状态定义

定义LangGraph工作流中使用的所有状态结构
"""

import uuid
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from models.project_models import ProjectInfo, EstimationStrategy, ProcessDetails
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
# from models.common_models import ProcessingStatus, ValidationResult


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


class ExecutionLogEntry(BaseModel):
    """执行日志条目 - Pydantic模型"""
    timestamp: str = Field(..., description="时间戳")
    agent_id: str = Field(..., description="智能体ID")
    action: str = Field(..., description="执行动作")
    status: str = Field(..., description="执行状态")
    duration_ms: Optional[int] = Field(None, description="执行时长(毫秒)")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")
    from_state: Optional[str] = Field(None, description="从状态")
    to_state: Optional[str] = Field(None, description="到状态")
    message: Optional[str] = Field(None, description="消息")
    retry_count: Optional[int] = Field(None, description="重试次数")
    max_retries: Optional[int] = Field(None, description="最大重试次数")


class UserPreferences(BaseModel):
    """用户偏好 - Pydantic模型"""
    preferred_standard: Optional[str] = Field(None, description="偏好标准")
    complexity_level: Optional[str] = Field(None, description="复杂度级别")
    report_format: Optional[str] = Field(None, description="报告格式")
    additional_requirements: List[str] = Field(default_factory=list, description="额外需求")


class StandardRecommendation(BaseModel):
    """标准推荐 - Pydantic模型"""
    recommended_standard: str = Field(..., description="推荐标准")
    confidence_score: float = Field(..., description="置信度")
    reasoning: str = Field(..., description="推荐理由")
    alternative_standards: List[str] = Field(default_factory=list, description="备选标准")


class RequirementAnalysis(BaseModel):
    """需求分析结果 - Pydantic模型"""
    functional_requirements: List[str] = Field(default_factory=list, description="功能需求")
    non_functional_requirements: List[str] = Field(default_factory=list, description="非功能需求")
    complexity_factors: List[str] = Field(default_factory=list, description="复杂度因素")
    risk_factors: List[str] = Field(default_factory=list, description="风险因素")
    estimated_effort: Optional[str] = Field(None, description="预估工作量")
    
    # 新增字段以匹配实际使用
    functional_modules: List[Dict[str, Any]] = Field(default_factory=list, description="功能模块")
    business_entities: Dict[str, Any] = Field(default_factory=dict, description="业务实体")
    business_processes: List[ProcessDetails] = Field(default_factory=list, description="业务流程")
    data_groups: List[str] = Field(default_factory=list, description="数据组")
    analysis_confidence: float = Field(default=0.0, description="分析置信度")
    parsing_issues: List[str] = Field(default_factory=list, description="解析问题")
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict, description="分析元数据")
    original_analysis: Optional[Dict[str, Any]] = Field(None, description="原始分析结果")


class NESMAResults(BaseModel):
    """NESMA结果 - Pydantic模型"""
    total_ufp: Optional[int] = Field(None, description="总UFP")
    function_count: Optional[int] = Field(None, description="功能点数量")
    complexity_distribution: Dict[str, int] = Field(default_factory=dict, description="复杂度分布")
    data_functions: List[Dict[str, Any]] = Field(default_factory=list, description="数据功能")
    transaction_functions: List[Dict[str, Any]] = Field(default_factory=list, description="事务功能")
    confidence_level: Optional[str] = Field(None, description="置信度")


class COSMICResults(BaseModel):
    """COSMIC结果 - Pydantic模型"""
    total_cfp: Optional[int] = Field(None, description="总CFP")
    data_movement_count: Optional[int] = Field(None, description="数据移动数量")
    functional_users: List[Dict[str, Any]] = Field(default_factory=list, description="功能用户")
    data_groups: List[Dict[str, Any]] = Field(default_factory=list, description="数据组")
    boundary_analysis: Optional[Dict[str, Any]] = Field(None, description="边界分析")
    confidence_level: Optional[str] = Field(None, description="置信度")
    data_movements: List[COSMICDataMovement] = Field(default_factory=list, description="数据移动")

class ValidationResults(BaseModel):
    """验证结果 - Pydantic模型"""
    nesma_validation: Optional[Dict[str, Any]] = Field(None, description="NESMA验证结果")
    cosmic_validation: Optional[Dict[str, Any]] = Field(None, description="COSMIC验证结果")
    overall_validation: Optional[Dict[str, Any]] = Field(None, description="整体验证结果")
    validation_status: str = Field(default="pending", description="验证状态")
    validation_score: Optional[float] = Field(None, description="验证分数")
    validation_issues: List[str] = Field(default_factory=list, description="验证问题")
    recommendations: List[str] = Field(default_factory=list, description="建议")


class ComparisonAnalysis(BaseModel):
    """对比分析 - Pydantic模型"""
    nesma_vs_cosmic: Dict[str, Any] = Field(default_factory=dict, description="NESMA vs COSMIC对比")
    accuracy_comparison: Dict[str, float] = Field(default_factory=dict, description="准确性对比")
    confidence_comparison: Dict[str, str] = Field(default_factory=dict, description="置信度对比")
    recommended_approach: Optional[str] = Field(None, description="推荐方法")
    reasoning: Optional[str] = Field(None, description="推荐理由")


class ReportContent(BaseModel):
    """报告内容 - Pydantic模型"""
    content: Optional[str] = Field(None, description="报告内容")
    file_path: Optional[str] = Field(None, description="文件路径")
    error: Optional[str] = Field(None, description="错误信息")


class FinalReport(BaseModel):
    """最终报告 - Pydantic模型"""
    markdown: Optional[ReportContent] = Field(None, description="Markdown格式")
    excel: Optional[ReportContent] = Field(None, description="Excel格式")
    word: Optional[ReportContent] = Field(None, description="Word格式")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="生成时间")


class ProcessingStats(BaseModel):
    """处理统计 - Pydantic模型"""
    total_duration_ms: int = Field(default=0, description="总耗时")
    agent_execution_times: Dict[str, int] = Field(default_factory=dict, description="智能体执行时间")
    memory_usage_mb: Optional[float] = Field(None, description="内存使用")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU使用率")


class QualityMetrics(BaseModel):
    """质量指标 - Pydantic模型"""
    accuracy_score: Optional[float] = Field(None, description="准确性分数")
    completeness_score: Optional[float] = Field(None, description="完整性分数")
    consistency_score: Optional[float] = Field(None, description="一致性分数")
    confidence_score: Optional[float] = Field(None, description="置信度分数")


class PerformanceMetrics(BaseModel):
    """性能指标 - Pydantic模型"""
    response_time_ms: int = Field(default=0, description="响应时间")
    throughput_requests_per_second: Optional[float] = Field(None, description="吞吐量")
    error_rate: Optional[float] = Field(None, description="错误率")
    success_rate: Optional[float] = Field(None, description="成功率")


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
    REQUIREMENT_PARSING_COMPLETED = "REQUIREMENT_PARSING_COMPLETED"
    PROCESS_IDENTIFICATION_PENDING = "PROCESS_IDENTIFICATION_PENDING"
    PROCESSES_IDENTIFIED = "PROCESSES_IDENTIFIED"
    
    # NESMA处理阶段
    NESMA_PROCESSING_PENDING = "NESMA_PROCESSING_PENDING"
    NESMA_CLASSIFICATION_COMPLETED = "NESMA_CLASSIFICATION_COMPLETED"
    NESMA_COMPLEXITY_COMPLETED = "NESMA_COMPLEXITY_COMPLETED"
    NESMA_CALCULATION_COMPLETED = "NESMA_CALCULATION_COMPLETED"
    
    # COSMIC处理阶段
    COSMIC_PROCESSING_PENDING = "COSMIC_PROCESSING_PENDING"
    COSMIC_ANALYSIS_COMPLETED = "COSMIC_ANALYSIS_COMPLETED"
    COSMIC_CALCULATION_COMPLETED = "COSMIC_CALCULATION_COMPLETED"
    
    # 验证阶段
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    
    # 对比分析阶段
    COMPARISON_ANALYSIS_COMPLETED = "COMPARISON_ANALYSIS_COMPLETED"
    
    # 最终阶段
    CROSS_STANDARD_COMPARISON_PENDING = "CROSS_STANDARD_COMPARISON_PENDING"
    REPORT_GENERATION_PENDING = "REPORT_GENERATION_PENDING"
    REPORT_COMPLETED = "REPORT_COMPLETED"
    COMPLETED = "COMPLETED"
    
    # 错误处理
    ERROR_ENCOUNTERED = "ERROR_ENCOUNTERED"
    TERMINATED = "TERMINATED"


class WorkflowGraphState(BaseModel):
    """LangGraph工作流状态 - 完整的Pydantic模型"""
    
    # 基础信息
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会话ID")
    project_info: ProjectInfo = Field(..., description="项目信息")
    current_state: WorkflowState = Field(default=WorkflowState.STARTING, description="当前状态")
    execution_log: List[ExecutionLogEntry] = Field(default_factory=list, description="执行日志")
    
    # 用户输入
    user_requirements: str = Field(default="", description="用户需求")
    user_preferences: UserPreferences = Field(default_factory=UserPreferences, description="用户偏好")
    
    # 标准选择
    standard_recommendation: Optional[StandardRecommendation] = Field(None, description="标准推荐")
    selected_strategy: Optional[EstimationStrategy] = Field(None, description="选择的策略")
    
    # 需求解析结果
    requirement_analysis: Optional[RequirementAnalysis] = Field(None, description="需求分析结果")
    identified_processes: Optional[List[ProcessDetails]] = Field(None, description="识别的流程")
    current_process_context: Optional[ProcessDetails] = Field(None, description="当前流程上下文")
    
    # NESMA结果
    nesma_results: Optional[NESMAResults] = Field(None, description="NESMA结果")
    nesma_classifications: List[NESMAFunctionClassification] = Field(default_factory=list, description="NESMA分类结果")
    nesma_complexity_results: List[NESMAComplexityCalculation] = Field(default_factory=list, description="NESMA复杂度结果")
    nesma_estimation_result: Optional[NESMAEstimationResult] = Field(None, description="NESMA估算结果")
    
    # COSMIC结果
    cosmic_results: Optional[COSMICResults] = Field(None, description="COSMIC结果")
    cosmic_functional_users: List[COSMICFunctionalUser] = Field(default_factory=list, description="COSMIC功能用户")
    cosmic_data_movements: List[COSMICDataMovement] = Field(default_factory=list, description="COSMIC数据移动")
    cosmic_boundary_analysis: Optional[COSMICBoundaryAnalysis] = Field(None, description="COSMIC边界分析")
    cosmic_estimation_result: Optional[COSMICEstimationResult] = Field(None, description="COSMIC估算结果")
    
    # 质量验证
    validation_results: Optional[ValidationResults] = Field(None, description="验证结果")
    
    # 最终输出
    comparison_analysis: Optional[ComparisonAnalysis] = Field(None, description="对比分析")
    final_report: Optional[FinalReport] = Field(None, description="最终报告")
    
    # 错误处理
    error_message: Optional[str] = Field(None, description="错误消息")
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats, description="处理统计")
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics, description="质量指标")
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics, description="性能指标")
    workflow_start_time: datetime = Field(default_factory=datetime.utcnow, description="工作流开始时间")

    retry_info: Dict[str, Any] = Field(default_factory=dict, description="重试信息")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


def create_initial_state(
    session_id: str,
    project_info: ProjectInfo
) -> WorkflowGraphState:
    """创建初始工作流状态"""
    return WorkflowGraphState(
        session_id=session_id,
        project_info=project_info
    )


def transition_state(
    state: WorkflowGraphState, 
    new_state: WorkflowState, 
    message: str = ""
) -> WorkflowGraphState:
    """状态转换"""
    old_state = state.current_state
    state.current_state = new_state
    state.updated_at = datetime.utcnow()
    
    # 记录状态转换日志
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": "state_transition",
        "agent_id": "workflow_manager",
        "from_state": old_state,
        "to_state": new_state,
        "message": message,
        "status": "success"
    }
    state.execution_log.append(log_entry)
    
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
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": agent_id,
        "action": action,
        "status": status,
        "duration_ms": duration_ms,
        "details": details or {}
    }
    
    state.execution_log.append(log_entry)
    state.updated_at = datetime.utcnow()
    
    return state


def is_retry_needed(state: WorkflowGraphState) -> bool:
    """检查是否需要重试"""
    return (
        state.current_state == WorkflowState.ERROR_ENCOUNTERED and 
        state.retry_count < state.max_retries
    )


def increment_retry_attempt(state: WorkflowGraphState) -> WorkflowGraphState:
    """增加重试次数"""
    state.retry_count += 1
    state.updated_at = datetime.utcnow()
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": "retry_attempt",
        "agent_id": "workflow_manager",
        "retry_count": state.retry_count,
        "max_retries": state.max_retries,
        "status": "info"
    }
    state.execution_log.append(log_entry)
    
    return state