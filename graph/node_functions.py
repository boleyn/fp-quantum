"""
量子智能化功能点估算系统 - LangGraph节点函数

实现工作流中的各个节点逻辑
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from graph.state_definitions import (
    WorkflowGraphState, WorkflowState, ProcessingStage,
    transition_state, update_execution_log, is_retry_needed,
    increment_retry_attempt
)
from models.project_models import EstimationStrategy, ProcessDetails
from agents.standards.standard_recommender import StandardRecommenderAgent
from agents.analysis.requirement_parser import RequirementParserAgent
from agents.analysis.process_identifier import ProcessIdentifierAgent
from agents.analysis.comparison_analyzer import ComparisonAnalyzerAgent
from agents.standards.nesma.function_classifier import NESMAFunctionClassifierAgent
from agents.standards.nesma.complexity_calculator import NESMAComplexityCalculatorAgent
from agents.standards.nesma.ufp_calculator import NESMAUFPCalculatorAgent
from agents.standards.cosmic.functional_user_agent import COSMICFunctionalUserAgent
from agents.standards.cosmic.boundary_analyzer import COSMICBoundaryAnalyzerAgent
from agents.standards.cosmic.data_movement_classifier import COSMICDataMovementClassifierAgent
from agents.standards.cosmic.cfp_calculator import COSMICCFPCalculatorAgent
from agents.knowledge.validator import ValidatorAgent
from agents.output.report_generator import ReportGeneratorAgent

# 添加必要的导入
from agents.knowledge.rule_retriever import create_rule_retriever_agent
from knowledge_base.vector_stores.pgvector_store import create_pgvector_store
from knowledge_base.embeddings.embedding_models import get_embedding_model
from config.settings import get_settings

logger = logging.getLogger(__name__)

# 全局智能体缓存
_cached_agents = {}

async def get_or_create_rule_retriever():
    """获取或创建rule_retriever智能体（全局缓存）"""
    if 'rule_retriever' not in _cached_agents:
        try:
            logger.info("🔧 初始化rule_retriever智能体...")
            
            # 获取配置
            settings = get_settings()
            
            # 创建embeddings
            embeddings = get_embedding_model()
            
            # 创建向量存储
            vector_store = await create_pgvector_store(embeddings)
            
            # 创建LLM
            from config.settings import get_llm
            llm = get_llm()
            
            # 创建rule_retriever
            rule_retriever = await create_rule_retriever_agent(
                llm=llm,
                embeddings=embeddings,
                vector_store=vector_store
            )
            
            _cached_agents['rule_retriever'] = rule_retriever
            logger.info("✅ rule_retriever智能体初始化完成")
            
        except Exception as e:
            logger.warning(f"⚠️ rule_retriever初始化失败: {e}")
            _cached_agents['rule_retriever'] = None
    
    return _cached_agents.get('rule_retriever')


# 初始化节点
async def start_workflow_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """工作流启动节点"""
    
    logger.info(f"🚀 启动工作流，会话ID: {state.session_id}")
    
    try:
        # 记录启动日志
        state = update_execution_log(
            state,
            agent_id="workflow_manager",
            action="start_workflow",
            status="success",
            details={
                "project_name": state.project_info.name,
                "user_requirements_length": len(state.user_requirements)
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.REQUIREMENT_INPUT_RECEIVED,
            "工作流启动完成"
        )
        
        logger.info("✅ 工作流启动成功")
        return state
        
    except Exception as e:
        logger.error(f"❌ 工作流启动失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 标准推荐节点
async def recommend_standard_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """标准推荐节点"""
    
    logger.info("🎯 开始标准推荐分析...")
    
    start_time = time.time()
    
    try:
        # 创建标准推荐智能体
        recommender = StandardRecommenderAgent()
        
        # 执行标准推荐
        recommendation = await recommender.recommend_standards(
            project_info=state.project_info
        )
        
        # 更新状态 - 使用正确的StandardRecommendation属性
        state.standard_recommendation = recommendation
        
        # 根据推荐的标准设置策略
        if recommendation.recommended_standard == "NESMA":
            state.selected_strategy = EstimationStrategy.NESMA_ONLY
        elif recommendation.recommended_standard == "COSMIC":
            state.selected_strategy = EstimationStrategy.COSMIC_ONLY
        elif recommendation.recommended_standard == "NESMA+COSMIC":
            state.selected_strategy = EstimationStrategy.DUAL_PARALLEL
        else:
            # 默认策略
            state.selected_strategy = EstimationStrategy.NESMA_ONLY
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="standard_recommender",
            action="recommend_standard",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "recommended_standard": recommendation.recommended_standard,
                "confidence_score": recommendation.confidence_score
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.STANDARD_RECOMMENDATION_READY,
            f"标准推荐完成: {recommendation.recommended_standard}"
        )
        
        logger.info(f"✅ 标准推荐完成: {recommendation.recommended_standard}")
        return state
        
    except Exception as e:
        error_msg = f"标准推荐失败: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # 详细的错误处理
        if "'StandardRecommenderAgent' object has no attribute 'recommend_estimation_standard'" in str(e):
            error_msg = "方法名不匹配：使用了错误的方法名 'recommend_estimation_standard'，应该是 'recommend_standards'"
        elif "recommend_standards" in str(e):
            error_msg = f"标准推荐智能体执行失败: {str(e)}"
        
        state.error_message = error_msg
        
        # 记录错误日志
        state = update_execution_log(
            state,
            agent_id="standard_recommender",
            action="recommend_standard",
            status="error",
            details={"error": error_msg, "exception_type": type(e).__name__}
        )
        
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, error_msg)


# 需求解析节点
async def parse_requirements_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """需求解析节点"""
    
    logger.info("📝 开始需求解析...")
    
    start_time = time.time()
    
    try:
        # 创建需求解析智能体
        parser = RequirementParserAgent()
        
        # 检查用户需求是否存在
        user_requirements = state.user_requirements
        if not user_requirements:
            raise ValueError("用户需求不存在，无法进行需求解析")
        
        # 执行需求解析
        analysis_result = await parser.parse_requirements(user_requirements)
        
        # 🔥 增强需求解析结果的检查和处理
        if not analysis_result:
            raise ValueError("需求解析返回空结果")
        
        # 更新状态 - 创建RequirementAnalysis实例
        from graph.state_definitions import RequirementAnalysis
        
        # 处理business_processes字段 - 确保是ProcessDetails对象列表
        business_processes = []
        raw_business_processes = analysis_result.get("business_processes", [])
        
        for i, process_data in enumerate(raw_business_processes):
            if isinstance(process_data, dict):
                # 如果是字典，转换为ProcessDetails对象
                process = ProcessDetails(
                    id=f"process_{i+1}",
                    name=process_data.get("流程名称", process_data.get("name", f"流程{i+1}")),
                    description=process_data.get("流程描述", process_data.get("description", "")),
                    data_groups=process_data.get("涉及的数据组", process_data.get("data_groups", [])),
                    dependencies=process_data.get("依赖关系", process_data.get("dependencies", [])),
                    inputs=process_data.get("inputs", []),
                    outputs=process_data.get("outputs", []),
                    business_rules=process_data.get("business_rules", []),
                    complexity_indicators=process_data.get("complexity_indicators", {}),
                    metadata=process_data
                )
                business_processes.append(process)
            elif isinstance(process_data, ProcessDetails):
                # 如果已经是ProcessDetails对象，直接使用
                business_processes.append(process_data)
            else:
                # 其他情况，创建默认对象
                process = ProcessDetails(
                    id=f"process_{i+1}",
                    name=f"流程{i+1}",
                    description=str(process_data),
                    data_groups=[],
                    dependencies=[],
                    inputs=[],
                    outputs=[],
                    business_rules=[],
                    complexity_indicators={},
                    metadata={"source": "fallback", "original_data": process_data}
                )
                business_processes.append(process)
        
        state.requirement_analysis = RequirementAnalysis(
            functional_modules=analysis_result.get("functional_modules", []),
            business_entities=analysis_result.get("business_entities", {}),
            business_processes=business_processes,
            data_groups=analysis_result.get("data_groups", []),
            analysis_confidence=analysis_result.get("parsing_confidence", 0.0),
            parsing_issues=analysis_result.get("parsing_issues", []),
            analysis_metadata=analysis_result.get("metadata", {}),
            original_analysis=analysis_result,
            # 设置基本字段
            functional_requirements=analysis_result.get("functional_requirements", []),
            non_functional_requirements=analysis_result.get("non_functional_requirements", []),
            complexity_factors=analysis_result.get("complexity_factors", []),
            risk_factors=analysis_result.get("risk_factors", []),
            estimated_effort=analysis_result.get("estimated_effort")
        )
        
        # 🔥 验证需求解析结果的结构
        logger.info(f"📋 需求解析结果检查:")
        logger.info(f"  - 结果类型: {type(state.requirement_analysis)}")
        logger.info(f"  - 结果键: {list(state.requirement_analysis.__dict__.keys()) if hasattr(state.requirement_analysis, '__dict__') else 'Not a model'}")
        
        if hasattr(state.requirement_analysis, 'functional_modules'):
            functional_modules = state.requirement_analysis.functional_modules
            business_processes = state.requirement_analysis.business_processes
            logger.info(f"  - 功能模块数量: {len(functional_modules)}")
            logger.info(f"  - 业务流程数量: {len(business_processes)}")
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="requirement_parser",
            action="parse_requirements",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "functional_modules_count": len(state.requirement_analysis.functional_modules),
                "analysis_confidence": state.requirement_analysis.analysis_confidence,
                "has_business_processes": len(state.requirement_analysis.business_processes) > 0
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.REQUIREMENT_PARSING_COMPLETED,
            "需求解析完成"
        )
        
        logger.info("✅ 需求解析完成")
        return state
        
    except Exception as e:
        error_msg = f"需求解析失败: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # 记录详细错误信息
        import traceback
        logger.error(f"❌ 需求解析异常堆栈: {traceback.format_exc()}")
        
        state.error_message = error_msg
        
        # 记录错误日志
        state = update_execution_log(
            state,
            agent_id="requirement_parser",
            action="parse_requirements",
            status="error",
            details={"error": error_msg, "exception_type": type(e).__name__}
        )
        
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, error_msg)


# 流程识别节点
async def identify_processes_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """流程识别节点"""
    
    logger.info("🔍 开始流程识别...")
    
    start_time = time.time()
    
    try:
        # 🔥 增强状态检查
        requirement_analysis = state.requirement_analysis
        if not requirement_analysis:
            error_msg = "需求解析结果不存在，无法进行流程识别"
            logger.error(f"❌ {error_msg}")
            
            # 尝试从状态中获取更多信息用于调试
            logger.error(f"🔍 状态检查详情:")
            logger.error(f"  - 当前状态: {state.current_state}")
            logger.error(f"  - 用户需求是否存在: {bool(state.user_requirements)}")
            logger.error(f"  - 项目信息是否存在: {bool(state.project_info)}")
            logger.error(f"  - 需求分析结果是否存在: {bool(state.requirement_analysis)}")
            
            raise ValueError(error_msg)
        
        # 🔥 验证需求解析结果的结构
        logger.info(f"📋 需求解析结果检查:")
        logger.info(f"  - 结果类型: {type(requirement_analysis)}")
        
        if hasattr(requirement_analysis, 'functional_modules'):
            functional_modules = requirement_analysis.functional_modules
            business_processes = requirement_analysis.business_processes
            logger.info(f"  - 功能模块数量: {len(functional_modules)}")
            logger.info(f"  - 业务流程数量: {len(business_processes)}")
        
        # 创建流程识别智能体
        identifier = ProcessIdentifierAgent()
        
        # 检查项目信息
        project_info = state.project_info
        if not project_info:
            raise ValueError("项目信息不存在，无法进行流程识别")
        
        # 执行流程识别
        processes = await identifier.identify_processes(
            requirement_analysis=requirement_analysis,
            project_info=project_info
        )
        
        # 🔥 验证流程识别结果
        if not processes:
            logger.warning("⚠️ 流程识别返回空结果，创建默认流程")
            # 创建一个默认流程以避免完全失败
            from models.project_models import ProcessDetails
            processes = [
                ProcessDetails(
                    id="default_process_001",
                    name="默认功能流程",
                    description="基于需求解析结果生成的默认功能流程",
                    data_groups=["默认数据组"],
                    dependencies=[],
                    inputs=[],
                    outputs=[],
                    business_rules=[],
                    complexity_indicators={},
                    metadata={"source": "fallback", "confidence": 0.3}
                )
            ]
        
        # 更新状态 - 直接使用返回的ProcessDetails对象
        state.identified_processes = processes
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="process_identifier",
            action="identify_processes",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "processes_count": len(processes),
                "process_names": [p.name for p in processes]
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.PROCESSES_IDENTIFIED,
            f"识别出 {len(processes)} 个功能流程"
        )
        
        logger.info(f"✅ 流程识别完成，识别出 {len(processes)} 个流程")
        return state
        
    except Exception as e:
        error_msg = f"流程识别失败: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # 记录详细错误信息
        import traceback
        logger.error(f"❌ 流程识别异常堆栈: {traceback.format_exc()}")
        
        # 详细的错误处理
        if "Can't instantiate abstract class ProcessIdentifierAgent" in str(e):
            error_msg = "ProcessIdentifierAgent类没有实现抽象方法 '_execute_task'"
        elif "_execute_task" in str(e):
            error_msg = f"流程识别智能体方法实现错误: {str(e)}"
        elif "需求解析结果不存在" in str(e):
            # 这种情况在上面已经处理过了，直接传递错误消息
            pass
        
        state.error_message = error_msg
        
        # 记录错误日志
        state = update_execution_log(
            state,
            agent_id="process_identifier",
            action="identify_processes", 
            status="error",
            details={"error": error_msg, "exception_type": type(e).__name__}
        )
        
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, error_msg)


# NESMA功能分类节点
async def nesma_classify_functions_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """NESMA功能分类节点"""
    
    logger.info("🎯 开始NESMA功能分类...")
    
    start_time = time.time()
    
    try:
        # 获取或创建rule_retriever
        rule_retriever = await get_or_create_rule_retriever()
        
        # 创建LLM
        from config.settings import get_llm
        llm = get_llm()
        
        # 创建NESMA功能分类智能体，传入rule_retriever
        classifier = NESMAFunctionClassifierAgent(
            rule_retriever=rule_retriever,
            llm=llm
        )
        
        logger.info(f"📋 功能分类器初始化完成，rule_retriever: {'✅' if rule_retriever else '❌'}")
        
        classifications = []
        processes = state.identified_processes or []
        
        for process in processes:
            # 为每个流程进行功能分类
            classification = await classifier.classify_function(
                function_description=process.description,
                process_details=process
            )
            classifications.append(classification)
        
        # 初始化NESMA结果
        if not state.nesma_results:
            state.nesma_results = {
                "function_classifications": [],
                "complexity_results": [],
                "total_ufp": 0,
                "ufp_breakdown": {},
                "estimation_confidence": 0.0,
                "classification_issues": [],
                "calculation_metadata": {}
            }
        
        state.nesma_results["function_classifications"] = classifications
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="nesma_classifier",
            action="classify_functions",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "classifications_count": len(classifications),
                "function_types": [c.function_type for c in classifications]
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.NESMA_CLASSIFICATION_COMPLETED,
            f"NESMA功能分类完成，分类 {len(classifications)} 个功能"
        )
        
        logger.info(f"✅ NESMA功能分类完成")
        return state
        
    except Exception as e:
        error_msg = f"NESMA功能分类失败: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # 详细的错误处理
        if "'nesma_results'" in str(e):
            error_msg = "状态字段 'nesma_results' 访问错误，可能未正确初始化"
        elif "classify_function" in str(e):
            error_msg = f"NESMA功能分类方法执行错误: {str(e)}"
        
        state.error_message = error_msg
        
        # 记录错误日志
        state = update_execution_log(
            state,
            agent_id="nesma_classifier",
            action="classify_functions",
            status="error",
            details={"error": error_msg, "exception_type": type(e).__name__}
        )
        
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, error_msg)


# NESMA复杂度计算节点
async def nesma_calculate_complexity_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """NESMA复杂度计算节点"""
    
    logger.info("⚙️ 开始NESMA复杂度计算...")
    
    start_time = time.time()
    
    try:
        # 获取或创建rule_retriever
        rule_retriever = await get_or_create_rule_retriever()
        
        # 创建LLM
        from config.settings import get_llm
        llm = get_llm()
        
        # 创建NESMA复杂度计算智能体，传入rule_retriever
        calculator = NESMAComplexityCalculatorAgent(
            rule_retriever=rule_retriever,
            llm=llm
        )
        
        logger.info(f"📋 复杂度计算器初始化完成，rule_retriever: {'✅' if rule_retriever else '❌'}")
        
        complexity_results = []
        classifications = state.nesma_results["function_classifications"]
        
        for classification in classifications:
            # 为每个分类计算复杂度
            complexity = await calculator.execute(
                task_name="calculate_complexity",
                inputs={
                    "classification": classification,
                    "function_description": classification.function_description,
                    "detailed_data": {}
                }
            )
            complexity_results.append(complexity)
        
        state.nesma_results["complexity_results"] = complexity_results
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="nesma_complexity_calculator",
            action="calculate_complexity",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "complexity_results_count": len(complexity_results)
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.NESMA_COMPLEXITY_COMPLETED,
            "NESMA复杂度计算完成"
        )
        
        logger.info("✅ NESMA复杂度计算完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ NESMA复杂度计算失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# NESMA UFP计算节点
async def nesma_calculate_ufp_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """NESMA UFP计算节点"""
    
    logger.info("🧮 开始NESMA UFP计算...")
    
    start_time = time.time()
    
    try:
        # 创建NESMA UFP计算智能体
        calculator = NESMAUFPCalculatorAgent()
        
        # 获取分类和复杂度结果，确保数据存在
        function_classifications = state.nesma_results.function_classifications or []
        complexity_results = state.nesma_results.complexity_results or []
        
        # 如果复杂度结果不足，用默认值补充
        if len(complexity_results) < len(function_classifications):
            complexity_results.extend([{"complexity": "AVERAGE"}] * (len(function_classifications) - len(complexity_results)))
        
        # 将字典转换为NESMAComplexityCalculation对象
        from models.nesma_models import NESMAComplexityCalculation, NESMAComplexityLevel, NESMAFunctionType
        
        # 复杂度映射字典
        complexity_mapping = {
            "Low": NESMAComplexityLevel.LOW,
            "Average": NESMAComplexityLevel.AVERAGE,
            "High": NESMAComplexityLevel.HIGH
        }
        
        complexity_objects = []
        for i, (c, cr) in enumerate(zip(function_classifications, complexity_results)):
            try:
                # 获取复杂度字符串并转换为枚举
                complexity_str = cr.get("complexity", "Low")
                complexity_enum = complexity_mapping.get(complexity_str, NESMAComplexityLevel.LOW)
                
                # 创建NESMAComplexityCalculation对象
                complexity_obj = NESMAComplexityCalculation(
                    function_id=cr.get("function_id", f"func_{i}"),
                    function_type=NESMAFunctionType(c.function_type.value if hasattr(c, 'function_type') else c.get("function_type", "EI")),
                    complexity=complexity_enum,
                    det_count=cr.get("det_count", 0),
                    ret_count=cr.get("ret_count", 0),
                    calculation_details=cr.get("calculation_details", {}),
                    complexity_matrix_used=cr.get("complexity_matrix_used", "default"),
                    calculation_steps=cr.get("calculation_steps", [])
                )
                complexity_objects.append(complexity_obj)
            except Exception as e:
                logger.warning(f"⚠️ 转换复杂度对象失败 {i}: {e}, 使用默认值")
                # 创建默认对象
                complexity_obj = NESMAComplexityCalculation(
                    function_id=f"func_{i}",
                    function_type=NESMAFunctionType.EI,
                    complexity=NESMAComplexityLevel.LOW,
                    det_count=1,
                    ret_count=1,
                    calculation_details={},
                    complexity_matrix_used="default",
                    calculation_steps=[]
                )
                complexity_objects.append(complexity_obj)
        
        # 执行UFP计算
        ufp_result = await calculator.execute(
            task_name="calculate_ufp",
            inputs={
                "complexity_results": complexity_objects,
                "project_info": state.project_info
            }
        )
        
        # 更新NESMA结果
        state.nesma_results["total_ufp"] = ufp_result.get("total_ufp", 0)
        state.nesma_results["ufp_breakdown"] = ufp_result.get("ufp_breakdown", {})
        state.nesma_results["estimation_confidence"] = ufp_result.get("confidence_score", 0.0)
        state.nesma_results["calculation_metadata"] = ufp_result.get("calculation_details", {})
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="nesma_ufp_calculator",
            action="calculate_ufp",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "total_ufp": ufp_result.get("total_ufp", 0),
                "confidence_score": ufp_result.get("confidence_score", 0.0)
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.NESMA_CALCULATION_COMPLETED,
            f"NESMA UFP计算完成: {ufp_result.get('total_ufp', 0)} UFP"
        )
        
        logger.info(f"✅ NESMA UFP计算完成: {ufp_result.get('total_ufp', 0)} UFP")
        return state
        
    except Exception as e:
        logger.error(f"❌ NESMA UFP计算失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# COSMIC功能用户识别节点
async def cosmic_identify_users_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """COSMIC功能用户识别节点"""
    
    logger.info("👥 开始COSMIC功能用户识别...")
    
    start_time = time.time()
    
    try:
        # 创建COSMIC功能用户智能体
        user_agent = COSMICFunctionalUserAgent()
        
        # 执行功能用户识别
        functional_users = await user_agent.execute(
            task_name="identify_functional_users",
            inputs={
                "project_info": state.project_info,
                "requirement_analysis": state.requirement_analysis,
                "identified_processes": state.identified_processes
            }
        )
        
        # 初始化COSMIC结果
        if not state.cosmic_results:
            state.cosmic_results = {
                "functional_users": [],
                "boundary_analysis": None,
                "data_movements": [],
                "functional_processes": [],
                "total_cfp": 0,
                "cfp_breakdown": {},
                "estimation_confidence": 0.0,
                "analysis_issues": [],
                "calculation_metadata": {}
            }
        
        state.cosmic_results.functional_users = functional_users if isinstance(functional_users, list) else functional_users.get("functional_users", [])
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="cosmic_functional_user",
            action="identify_users",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "functional_users_count": len(functional_users),
                "user_names": [user.name for user in functional_users]
            }
        )
        
        logger.info(f"✅ COSMIC功能用户识别完成，识别出 {len(functional_users)} 个功能用户")
        return state
        
    except Exception as e:
        logger.error(f"❌ COSMIC功能用户识别失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# COSMIC边界分析节点
async def cosmic_analyze_boundary_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """COSMIC边界分析节点"""
    
    logger.info("🏗️ 开始COSMIC边界分析...")
    
    start_time = time.time()
    
    try:
        # 创建COSMIC边界分析智能体
        boundary_analyzer = COSMICBoundaryAnalyzerAgent()
        
        # 执行边界分析
        boundary_analysis = await boundary_analyzer.analyze_system_boundary(
            project_info=state.project_info,
            functional_users=state.cosmic_results.functional_users,
            requirement_analysis=state.requirement_analysis
        )
        
        state.cosmic_results.boundary_analysis= boundary_analysis
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="cosmic_boundary_analyzer",
            action="analyze_boundary",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "software_boundary": boundary_analysis.software_boundary,
                "storage_boundary": boundary_analysis.persistent_storage_boundary
            }
        )
        
        logger.info("✅ COSMIC边界分析完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ COSMIC边界分析失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# COSMIC数据移动分类节点
async def cosmic_classify_movements_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """COSMIC数据移动分类节点"""
    
    logger.info("🔄 开始COSMIC数据移动分类...")
    
    start_time = time.time()
    
    try:
        # 创建COSMIC数据移动分类智能体
        movement_classifier = COSMICDataMovementClassifierAgent()
        
        # 执行数据移动分类
        data_movements = await movement_classifier.classify_data_movements(
            identified_processes=state.identified_processes,
            functional_users=state.cosmic_results.functional_users,
            boundary_analysis=state.cosmic_results.boundary_analysis
        )
        
        state.cosmic_results.data_movements= data_movements
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="cosmic_movement_classifier",
            action="classify_movements",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "data_movements_count": len(data_movements),
                "movement_types": [dm.type for dm in data_movements]
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.COSMIC_ANALYSIS_COMPLETED,
            f"COSMIC数据移动分类完成，识别出 {len(data_movements)} 个数据移动"
        )
        
        logger.info(f"✅ COSMIC数据移动分类完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ COSMIC数据移动分类失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# COSMIC CFP计算节点
async def cosmic_calculate_cfp_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """COSMIC CFP计算节点"""
    
    logger.info("🧮 开始COSMIC CFP计算...")
    
    start_time = time.time()
    
    try:
        # 创建COSMIC CFP计算智能体
        cfp_calculator = COSMICCFPCalculatorAgent()
        
        # 执行CFP计算
        cfp_result = await cfp_calculator.calculate_cosmic_function_points(
            data_movements=state.cosmic_results.data_movements,
            functional_processes=state.identified_processes,
            boundary_analysis=state.cosmic_results.boundary_analysis
        )
        
        # 更新COSMIC结果
        state.cosmic_results.total_cfp= cfp_result.total_cfp
        state.cosmic_results.cfp_breakdown = cfp_result.cfp_breakdown
        state.cosmic_results.estimation_confidence = cfp_result.confidence_score
        state.cosmic_results.functional_processes = cfp_result.functional_processes
        state.cosmic_results.calculation_metadata = cfp_result.calculation_details
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="cosmic_cfp_calculator",
            action="calculate_cfp",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "total_cfp": cfp_result.total_cfp,
                "confidence_score": cfp_result.confidence_score
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.COSMIC_CALCULATION_COMPLETED,
            f"COSMIC CFP计算完成: {cfp_result.total_cfp} CFP"
        )
        
        logger.info(f"✅ COSMIC CFP计算完成: {cfp_result.total_cfp} CFP")
        return state
        
    except Exception as e:
        logger.error(f"❌ COSMIC CFP计算失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 结果验证节点
async def validate_results_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """结果验证节点"""
    
    logger.info("✅ 开始结果验证...")
    
    start_time = time.time()
    
    try:
        # 创建验证智能体
        validator = ValidatorAgent()
        
        # 初始化验证结果
        nesma_validation = None
        cosmic_validation = None
        
        # 验证NESMA结果
        if state.nesma_results:
            nesma_validation = await validator.validate_analysis_result(
                analysis_type="NESMA_estimation",
                analysis_result=state.nesma_results,
                input_data=state.requirement_analysis
            )
        
        # 验证COSMIC结果
        if state.cosmic_results:
            cosmic_validation = await validator.validate_analysis_result(
                analysis_type="COSMIC_estimation",
                analysis_result=state.cosmic_results,
                input_data=state.requirement_analysis
            )
        
        # 计算整体验证分数
        overall_validation = _calculate_overall_validation({
            "nesma_validation": nesma_validation,
            "cosmic_validation": cosmic_validation
        })
        
        # 创建ValidationResults对象
        from graph.state_definitions import ValidationResults
        
        validation_results = ValidationResults(
            nesma_validation=nesma_validation,
            cosmic_validation=cosmic_validation,
            overall_validation=overall_validation.model_dump() if overall_validation else None,
            validation_status="completed",
            validation_score=overall_validation.confidence_score if overall_validation else 0.0,
            validation_issues=[],
            recommendations=[]
        )
        
        state.validation_results = validation_results
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="validator",
            action="validate_results",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "overall_validation_score": validation_results.validation_score
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.VALIDATION_COMPLETED,
            "结果验证完成"
        )
        
        logger.info("✅ 结果验证完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ 结果验证失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 对比分析节点
async def compare_standards_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """跨标准对比分析节点"""
    
    logger.info("📊 开始跨标准对比分析...")
    
    start_time = time.time()
    
    try:
        # 检查是否有两种标准的结果
        if not state.nesma_results or not state.cosmic_results:
            logger.warning("缺少完整的估算结果，跳过对比分析")
            return transition_state(
                state,
                WorkflowState.REPORT_GENERATION_PENDING,
                "跳过对比分析，缺少完整结果"
            )
        
        # 创建对比分析智能体
        analyzer = ComparisonAnalyzerAgent()
        
        # 执行对比分析
        comparison_result = await analyzer.analyze_cross_standard_comparison(
            nesma_results=state.nesma_results,
            cosmic_results=state.cosmic_results,
            project_info=state.project_info.__dict__
        )
        
        state.comparison_analysis = comparison_result
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="comparison_analyzer",
            action="compare_standards",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "difference_percentage": comparison_result.difference_percentage,
                "recommended_standard": comparison_result.recommendation.get("recommended_standard")
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.COMPARISON_ANALYSIS_COMPLETED,
            "跨标准对比分析完成"
        )
        
        logger.info("✅ 跨标准对比分析完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ 跨标准对比分析失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 报告生成节点
async def generate_report_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """报告生成节点"""
    
    logger.info("📄 开始生成最终报告...")
    
    start_time = time.time()
    
    try:
        # 创建报告生成智能体
        generator = ReportGeneratorAgent()
        
        # 🔥 支持多种报告格式 - 默认生成Excel和Word格式
        report_formats = ["excel", "word", "markdown"]
        generated_reports = {}
        
        for format_type in report_formats:
            try:
                # 生成报告
                report_data = await generator.generate_estimation_report(
                    project_info=state.project_info,
                    estimation_results={
                        "nesma_results": state.nesma_results.model_dump() if state.nesma_results else None,
                        "cosmic_results": state.cosmic_results.model_dump() if state.cosmic_results else None,
                        "comparison_result": state.comparison_analysis.model_dump() if state.comparison_analysis else None,
                        "validation_results": state.validation_results.model_dump() if state.validation_results else None,
                        "execution_log": [log.dict() for log in state.execution_log] if state.execution_log else []
                    },
                    format=format_type
                )
                
                # 将ReportContent转换为Pydantic模型
                from graph.state_definitions import ReportContent
                if format_type == "markdown":
                    generated_reports["markdown"] = ReportContent(
                        content=report_data.get("content"),
                        file_path=report_data.get("file_path"),
                        error=report_data.get("error")
                    )
                elif format_type == "excel":
                    generated_reports["excel"] = ReportContent(
                        content=report_data.get("content"),
                        file_path=report_data.get("file_path"),
                        error=report_data.get("error")
                    )
                elif format_type == "word":
                    generated_reports["word"] = ReportContent(
                        content=report_data.get("content"),
                        file_path=report_data.get("file_path"),
                        error=report_data.get("error")
                    )
                
                logger.info(f"✅ {format_type.upper()}格式报告生成完成")
                
            except Exception as e:
                logger.error(f"❌ {format_type.upper()}格式报告生成失败: {str(e)}")
                # 创建错误报告
                from graph.state_definitions import ReportContent
                generated_reports[format_type] = ReportContent(
                    content=None,
                    file_path=None,
                    error=str(e)
                )
        
        # 创建FinalReport Pydantic模型
        from graph.state_definitions import FinalReport
        final_report = FinalReport(
            markdown=generated_reports.get("markdown"),
            excel=generated_reports.get("excel"),
            word=generated_reports.get("word")
        )
        
        # 更新状态
        state.final_report = final_report
        
        # 记录执行时间
        execution_time = int((time.time() - start_time) * 1000)
        state = update_execution_log(
            state,
            "report_generator",
            "generate_final_report",
            "success",
            execution_time,
            {"formats_generated": list(generated_reports.keys())}
        )
        
        logger.info(f"📄 报告生成完成，耗时: {execution_time}ms")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ 报告生成失败: {str(e)}")
        
        # 创建错误报告
        from graph.state_definitions import FinalReport, ReportContent
        error_report = ReportContent(
            content=None,
            file_path=None,
            error=str(e)
        )
        
        state.final_report = FinalReport(
            markdown=error_report,
            excel=error_report,
            word=error_report
        )
        
        # 记录错误
        execution_time = int((time.time() - start_time) * 1000)
        state = update_execution_log(
            state,
            "report_generator",
            "generate_final_report",
            "error",
            execution_time,
            {"error": str(e)}
        )
        
        return state


# 完成节点
async def complete_workflow_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """工作流完成节点"""
    
    logger.info("🎉 工作流执行完成")
    
    try:
        # 计算最终质量指标
        state.quality_metrics.accuracy_score = _calculate_overall_confidence(state)
        
        # 记录完成日志
        state = update_execution_log(
            state,
            agent_id="workflow_manager",
            action="complete_workflow",
            status="success",
            details={
                "total_duration": state.processing_stats.total_duration_ms,
                "overall_confidence": state.quality_metrics.accuracy_score,
                "nesma_ufp": state.nesma_results.total_ufp if state.nesma_results else None,
                "cosmic_cfp": state.cosmic_results.total_cfp if state.cosmic_results else None
            }
        )
        
        # 转换到最终状态
        state = transition_state(
            state,
            WorkflowState.COMPLETED,
            "工作流执行成功完成"
        )
        
        logger.info("✅ 工作流执行成功完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ 工作流完成失败: {str(e)}")
        state.error_message = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 错误处理节点
async def handle_error_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """错误处理节点"""
    
    # 🔥 改进错误消息处理，避免None
    error_msg = state.error_message
    if not error_msg or error_msg == "None":
        # 尝试从状态中提取更多错误信息
        last_log = state.execution_log or []
        if last_log:
            last_entry = last_log[-1]
            if isinstance(last_entry, dict) and last_entry.get("status") == "error":
                error_msg = last_entry.get("details", {}).get("error", "工作流执行遇到未知错误")
            else:
                error_msg = "工作流执行遇到未知错误"
        else:
            error_msg = "工作流执行遇到未知错误"
    
    logger.error(f"❌ 处理工作流错误: {error_msg}")
    
    # 🔥 记录错误详情用于调试
    logger.error(f"🔍 错误状态详情:")
    logger.error(f"  - 当前状态: {state.current_state}")
    logger.error(f"  - 重试次数: {state.retry_count}")
    logger.error(f"  - 最大重试: {state.max_retries}")
    logger.error(f"  - 错误消息: {repr(error_msg)}")
    
    try:
        # 🔥 特殊处理：如果当前状态是VALIDATION_COMPLETED且错误信息涉及流程识别
        # 说明是状态流转出现问题，应该直接继续到下一步而不是重新执行流程识别
        current_state = state.current_state
        
        # 🔥 增加更多特殊处理情况
        if (current_state == WorkflowState.VALIDATION_COMPLETED and 
            ("流程识别失败" in error_msg and "需求解析结果不存在" in error_msg) or 
            error_msg == "工作流执行遇到未知错误"):
            
            logger.info("🔧 检测到状态流转错误，直接跳转到报告生成阶段")
            
            # 🔥 修复状态访问问题 - 正确处理Pydantic模型
            try:
                # 直接访问Pydantic模型属性，不使用字典访问
                has_nesma_results = bool(state.nesma_results and state.nesma_results.total_ufp)
                has_cosmic_results = bool(state.cosmic_results and state.cosmic_results.total_cfp)
                
                logger.info(f"🔍 结果检查: NESMA={has_nesma_results}, COSMIC={has_cosmic_results}")
                logger.info(f"🔍 NESMA结果: {state.nesma_results}")
                logger.info(f"🔍 COSMIC结果: {state.cosmic_results}")
                
            except Exception as e:
                logger.error(f"❌ 检查结果时出错: {e}")
                has_nesma_results = False
                has_cosmic_results = False
            
            if has_nesma_results or has_cosmic_results:
                # 有结果，直接生成报告
                return transition_state(
                    state,
                    WorkflowState.REPORT_GENERATION_PENDING,
                    "跳过错误状态，直接生成报告"
                )
            else:
                # 没有有效结果，终止工作流
                logger.error("❌ 没有有效的估算结果，无法生成报告")
                return transition_state(
                    state,
                    WorkflowState.TERMINATED,
                    f"工作流终止: 没有有效的估算结果"
                )
        
        # 检查是否需要重试
        if is_retry_needed(state):
            # 增加重试次数
            state = increment_retry_attempt(state)
            
            logger.info(f"🔄 准备重试，第 {state.retry_count} 次")
            
            # 转换到重试状态
            return transition_state(
                state,
                WorkflowState.STARTING,
                f"准备重试: {error_msg}"
            )
        else:
            # 重试次数已达上限，终止工作流
            logger.error("❌ 重试次数已达上限，终止工作流")
            
            return transition_state(
                state,
                WorkflowState.TERMINATED,
                f"工作流终止: {error_msg}"
            )
            
    except Exception as e:
        logger.error(f"❌ 错误处理失败: {str(e)}")
        import traceback
        logger.error(f"❌ 错误处理异常堆栈: {traceback.format_exc()}")
        return transition_state(
            state,
            WorkflowState.TERMINATED,
            f"错误处理失败: {str(e)}"
        )


# 辅助函数
def _calculate_overall_validation(validation_results: Dict[str, Any]) -> Optional[Any]:
    """计算整体验证结果"""
    
    scores = []
    
    if "nesma_validation" in validation_results:
        scores.append(validation_results["nesma_validation"].confidence_score)
    
    if "cosmic_validation" in validation_results:
        scores.append(validation_results["cosmic_validation"].confidence_score)
    
    if not scores:
        return None
    
    # 创建虚拟的整体验证结果
    from models.common_models import ValidationResult
    
    overall_score = sum(scores) / len(scores)
    
    return ValidationResult(
        is_valid=overall_score >= 0.5,
        confidence_score=overall_score,
        confidence_level=_score_to_confidence_level(overall_score),
        errors=[],
        warnings=[],
        suggestions=[],
        metadata={"component_scores": scores}
    )


def _score_to_confidence_level(score: float):
    """将分数转换为置信度等级"""
    from models.common_models import ConfidenceLevel
    if score >= 0.8:
        return ConfidenceLevel.VERY_HIGH
    elif score >= 0.6:
        return ConfidenceLevel.HIGH
    elif score >= 0.4:
        return ConfidenceLevel.MEDIUM
    elif score >= 0.2:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


def _calculate_overall_confidence(state: WorkflowGraphState) -> float:
    """计算整体置信度"""
    
    confidence_scores = []
    
    # NESMA置信度
    if state.nesma_results and hasattr(state.nesma_results, 'confidence_level'):
        # 将置信度等级转换为分数
        confidence_level = state.nesma_results.confidence_level
        if confidence_level == "HIGH":
            confidence_scores.append(0.8)
        elif confidence_level == "MEDIUM":
            confidence_scores.append(0.6)
        elif confidence_level == "LOW":
            confidence_scores.append(0.4)
        else:
            confidence_scores.append(0.5)  # 默认中等置信度
    
    # COSMIC置信度
    if state.cosmic_results and hasattr(state.cosmic_results, 'confidence_level'):
        # 将置信度等级转换为分数
        confidence_level = state.cosmic_results.confidence_level
        if confidence_level == "HIGH":
            confidence_scores.append(0.8)
        elif confidence_level == "MEDIUM":
            confidence_scores.append(0.6)
        elif confidence_level == "LOW":
            confidence_scores.append(0.4)
        else:
            confidence_scores.append(0.5)  # 默认中等置信度
    
    # 验证置信度
    if state.validation_results and hasattr(state.validation_results, 'validation_score'):
        validation_score = state.validation_results.validation_score
        if validation_score is not None:
            confidence_scores.append(validation_score)
    
    if not confidence_scores:
        return 0.0
    
    return sum(confidence_scores) / len(confidence_scores) 