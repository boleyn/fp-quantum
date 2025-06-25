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

logger = logging.getLogger(__name__)


# 初始化节点
async def start_workflow_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """工作流启动节点"""
    
    logger.info(f"🚀 启动工作流，会话ID: {state['session_id']}")
    
    try:
        # 记录启动日志
        state = update_execution_log(
            state,
            agent_id="workflow_manager",
            action="start_workflow",
            status="success",
            details={
                "project_name": state["project_info"].name,
                "user_requirements_length": len(state["user_requirements"])
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
        state["current_error"] = str(e)
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
        recommendation = await recommender.recommend_estimation_standard(
            project_info=state["project_info"],
            user_preferences=state["user_preferences"]
        )
        
        # 更新状态
        state["standard_recommendation"] = {
            "recommended_strategy": recommendation.strategy,
            "confidence_score": recommendation.confidence_score,
            "reasoning": recommendation.reasoning,
            "nesma_suitability": getattr(recommendation, "nesma_suitability", 0.5),
            "cosmic_suitability": getattr(recommendation, "cosmic_suitability", 0.5),
            "project_characteristics": getattr(recommendation, "project_characteristics", {}),
            "recommendation_details": getattr(recommendation, "recommendation_details", {})
        }
        
        state["selected_strategy"] = recommendation.strategy
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="standard_recommender",
            action="recommend_standard",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "recommended_strategy": recommendation.strategy,
                "confidence_score": recommendation.confidence_score
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.STANDARD_RECOMMENDATION_READY,
            f"标准推荐完成: {recommendation.strategy}"
        )
        
        logger.info(f"✅ 标准推荐完成: {recommendation.strategy}")
        return state
        
    except Exception as e:
        logger.error(f"❌ 标准推荐失败: {str(e)}")
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 需求解析节点
async def parse_requirements_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """需求解析节点"""
    
    logger.info("📝 开始需求解析...")
    
    start_time = time.time()
    
    try:
        # 创建需求解析智能体
        parser = RequirementParserAgent()
        
        # 执行需求解析
        analysis_result = await parser.parse_requirements(
            requirement_text=state["user_requirements"],
            project_info=state["project_info"]
        )
        
        # 更新状态
        state["requirement_analysis"] = {
            "functional_modules": analysis_result.get("functional_modules", []),
            "business_entities": analysis_result.get("business_entities", {}),
            "business_processes": analysis_result.get("business_processes", []),
            "data_groups": analysis_result.get("data_groups", []),
            "analysis_confidence": analysis_result.get("confidence_score", 0.0),
            "parsing_issues": analysis_result.get("parsing_issues", []),
            "analysis_metadata": analysis_result.get("metadata", {})
        }
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="requirement_parser",
            action="parse_requirements",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "functional_modules_count": len(state["requirement_analysis"]["functional_modules"]),
                "analysis_confidence": state["requirement_analysis"]["analysis_confidence"]
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
        logger.error(f"❌ 需求解析失败: {str(e)}")
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 流程识别节点
async def identify_processes_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """流程识别节点"""
    
    logger.info("🔍 开始流程识别...")
    
    start_time = time.time()
    
    try:
        # 创建流程识别智能体
        identifier = ProcessIdentifierAgent()
        
        # 执行流程识别
        processes = await identifier.identify_processes(
            requirement_analysis=state["requirement_analysis"],
            project_info=state["project_info"]
        )
        
        # 更新状态
        state["identified_processes"] = processes
        
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
        logger.error(f"❌ 流程识别失败: {str(e)}")
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# NESMA功能分类节点
async def nesma_classify_functions_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """NESMA功能分类节点"""
    
    logger.info("🎯 开始NESMA功能分类...")
    
    start_time = time.time()
    
    try:
        # 创建NESMA功能分类智能体
        classifier = NESMAFunctionClassifierAgent()
        
        classifications = []
        processes = state["identified_processes"] or []
        
        for process in processes:
            # 为每个流程进行功能分类
            classification = await classifier.classify_function_type(
                process_detail=process,
                project_context=state["project_info"]
            )
            classifications.append(classification)
        
        # 初始化NESMA结果
        if not state["nesma_results"]:
            state["nesma_results"] = {
                "function_classifications": [],
                "complexity_results": [],
                "total_ufp": 0,
                "ufp_breakdown": {},
                "estimation_confidence": 0.0,
                "classification_issues": [],
                "calculation_metadata": {}
            }
        
        state["nesma_results"]["function_classifications"] = classifications
        
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
        logger.error(f"❌ NESMA功能分类失败: {str(e)}")
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# NESMA复杂度计算节点
async def nesma_calculate_complexity_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """NESMA复杂度计算节点"""
    
    logger.info("⚙️ 开始NESMA复杂度计算...")
    
    start_time = time.time()
    
    try:
        # 创建NESMA复杂度计算智能体
        calculator = NESMAComplexityCalculatorAgent()
        
        complexity_results = []
        classifications = state["nesma_results"]["function_classifications"]
        
        for classification in classifications:
            # 为每个分类计算复杂度
            complexity = await calculator.calculate_complexity(
                classification=classification,
                detailed_requirements=state["requirement_analysis"]
            )
            complexity_results.append(complexity)
        
        state["nesma_results"]["complexity_results"] = complexity_results
        
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
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# NESMA UFP计算节点
async def nesma_calculate_ufp_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """NESMA UFP计算节点"""
    
    logger.info("🧮 开始NESMA UFP计算...")
    
    start_time = time.time()
    
    try:
        # 创建NESMA UFP计算智能体
        calculator = NESMAUFPCalculatorAgent()
        
        # 执行UFP计算
        ufp_result = await calculator.calculate_unadjusted_function_points(
            classifications=state["nesma_results"]["function_classifications"],
            complexity_results=state["nesma_results"]["complexity_results"]
        )
        
        # 更新NESMA结果
        state["nesma_results"]["total_ufp"] = ufp_result.total_ufp
        state["nesma_results"]["ufp_breakdown"] = ufp_result.ufp_breakdown
        state["nesma_results"]["estimation_confidence"] = ufp_result.confidence_score
        state["nesma_results"]["calculation_metadata"] = ufp_result.calculation_details
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="nesma_ufp_calculator",
            action="calculate_ufp",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "total_ufp": ufp_result.total_ufp,
                "confidence_score": ufp_result.confidence_score
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.NESMA_CALCULATION_COMPLETED,
            f"NESMA UFP计算完成: {ufp_result.total_ufp} UFP"
        )
        
        logger.info(f"✅ NESMA UFP计算完成: {ufp_result.total_ufp} UFP")
        return state
        
    except Exception as e:
        logger.error(f"❌ NESMA UFP计算失败: {str(e)}")
        state["current_error"] = str(e)
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
        functional_users = await user_agent.identify_functional_users(
            project_info=state["project_info"],
            requirement_analysis=state["requirement_analysis"],
            identified_processes=state["identified_processes"]
        )
        
        # 初始化COSMIC结果
        if not state["cosmic_results"]:
            state["cosmic_results"] = {
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
        
        state["cosmic_results"]["functional_users"] = functional_users
        
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
        state["current_error"] = str(e)
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
            project_info=state["project_info"],
            functional_users=state["cosmic_results"]["functional_users"],
            requirement_analysis=state["requirement_analysis"]
        )
        
        state["cosmic_results"]["boundary_analysis"] = boundary_analysis
        
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
        state["current_error"] = str(e)
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
            identified_processes=state["identified_processes"],
            functional_users=state["cosmic_results"]["functional_users"],
            boundary_analysis=state["cosmic_results"]["boundary_analysis"]
        )
        
        state["cosmic_results"]["data_movements"] = data_movements
        
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
        state["current_error"] = str(e)
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
            data_movements=state["cosmic_results"]["data_movements"],
            functional_processes=state["identified_processes"],
            boundary_analysis=state["cosmic_results"]["boundary_analysis"]
        )
        
        # 更新COSMIC结果
        state["cosmic_results"]["total_cfp"] = cfp_result.total_cfp
        state["cosmic_results"]["cfp_breakdown"] = cfp_result.cfp_breakdown
        state["cosmic_results"]["estimation_confidence"] = cfp_result.confidence_score
        state["cosmic_results"]["functional_processes"] = cfp_result.functional_processes
        state["cosmic_results"]["calculation_metadata"] = cfp_result.calculation_details
        
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
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 结果验证节点
async def validate_results_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """结果验证节点"""
    
    logger.info("✅ 开始结果验证...")
    
    start_time = time.time()
    
    try:
        # 创建验证智能体
        validator = ValidatorAgent()
        
        validation_results = {}
        
        # 验证NESMA结果
        if state["nesma_results"]:
            nesma_validation = await validator.validate_analysis_result(
                analysis_type="NESMA_estimation",
                analysis_result=state["nesma_results"],
                input_data=state["requirement_analysis"]
            )
            validation_results["nesma_validation"] = nesma_validation
        
        # 验证COSMIC结果
        if state["cosmic_results"]:
            cosmic_validation = await validator.validate_analysis_result(
                analysis_type="COSMIC_estimation",
                analysis_result=state["cosmic_results"],
                input_data=state["requirement_analysis"]
            )
            validation_results["cosmic_validation"] = cosmic_validation
        
        # 计算整体验证分数
        overall_validation = self._calculate_overall_validation(validation_results)
        validation_results["overall_validation"] = overall_validation
        
        state["validation_results"] = validation_results
        
        processing_time = time.time() - start_time
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="validator",
            action="validate_results",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "overall_validation_score": overall_validation.confidence_score if overall_validation else 0.0
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
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 对比分析节点
async def compare_standards_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """跨标准对比分析节点"""
    
    logger.info("📊 开始跨标准对比分析...")
    
    start_time = time.time()
    
    try:
        # 检查是否有两种标准的结果
        if not state["nesma_results"] or not state["cosmic_results"]:
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
            nesma_results=state["nesma_results"],
            cosmic_results=state["cosmic_results"],
            project_info=state["project_info"].__dict__
        )
        
        state["comparison_result"] = comparison_result
        
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
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 报告生成节点
async def generate_report_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """报告生成节点"""
    
    logger.info("📄 开始生成最终报告...")
    
    start_time = time.time()
    
    try:
        # 创建报告生成智能体
        generator = ReportGeneratorAgent()
        
        # 生成报告
        final_report = await generator.generate_comprehensive_report(
            project_info=state["project_info"],
            nesma_results=state["nesma_results"],
            cosmic_results=state["cosmic_results"],
            comparison_result=state["comparison_result"],
            validation_results=state["validation_results"],
            execution_log=state["execution_log"]
        )
        
        state["final_report"] = final_report
        
        processing_time = time.time() - start_time
        
        # 更新性能指标
        state["performance_metrics"]["total_processing_time"] = (
            datetime.now() - state["workflow_start_time"]
        ).total_seconds()
        
        # 记录执行日志
        state = update_execution_log(
            state,
            agent_id="report_generator",
            action="generate_report",
            status="success",
            duration_ms=int(processing_time * 1000),
            details={
                "report_sections": len(final_report.get("sections", [])),
                "total_pages": final_report.get("metadata", {}).get("total_pages", 0)
            }
        )
        
        # 转换状态
        state = transition_state(
            state,
            WorkflowState.REPORT_COMPLETED,
            "最终报告生成完成"
        )
        
        logger.info("✅ 最终报告生成完成")
        return state
        
    except Exception as e:
        logger.error(f"❌ 报告生成失败: {str(e)}")
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 完成节点
async def complete_workflow_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """工作流完成节点"""
    
    logger.info("🎉 工作流执行完成")
    
    try:
        # 计算最终质量指标
        state["quality_metrics"]["overall_confidence"] = self._calculate_overall_confidence(state)
        
        # 记录完成日志
        state = update_execution_log(
            state,
            agent_id="workflow_manager",
            action="complete_workflow",
            status="success",
            details={
                "total_duration": state["performance_metrics"]["total_processing_time"],
                "overall_confidence": state["quality_metrics"]["overall_confidence"],
                "nesma_ufp": state["nesma_results"]["total_ufp"] if state["nesma_results"] else None,
                "cosmic_cfp": state["cosmic_results"]["total_cfp"] if state["cosmic_results"] else None
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
        state["current_error"] = str(e)
        return transition_state(state, WorkflowState.ERROR_ENCOUNTERED, str(e))


# 错误处理节点
async def handle_error_node(state: WorkflowGraphState) -> WorkflowGraphState:
    """错误处理节点"""
    
    logger.error(f"❌ 处理工作流错误: {state['current_error']}")
    
    try:
        # 检查是否需要重试
        if is_retry_needed(state):
            # 增加重试次数
            state = increment_retry_attempt(
                state,
                retry_reason=state["current_error"],
                retry_strategy="simple"
            )
            
            logger.info(f"🔄 准备重试，第 {state['retry_config']['current_attempt']} 次")
            
            # 转换到重试状态
            return transition_state(
                state,
                WorkflowState.RETRY_PENDING,
                f"准备重试: {state['current_error']}"
            )
        else:
            # 重试次数已达上限，终止工作流
            logger.error("❌ 重试次数已达上限，终止工作流")
            
            return transition_state(
                state,
                WorkflowState.TERMINATED,
                f"工作流终止: {state['current_error']}"
            )
            
    except Exception as e:
        logger.error(f"❌ 错误处理失败: {str(e)}")
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
        validation_details={"component_scores": scores},
        issues=[],
        suggestions=[]
    )


def _calculate_overall_confidence(state: WorkflowGraphState) -> float:
    """计算整体置信度"""
    
    confidence_scores = []
    
    # NESMA置信度
    if state["nesma_results"]:
        confidence_scores.append(state["nesma_results"]["estimation_confidence"])
    
    # COSMIC置信度
    if state["cosmic_results"]:
        confidence_scores.append(state["cosmic_results"]["estimation_confidence"])
    
    # 验证置信度
    if state["validation_results"] and state["validation_results"]["overall_validation"]:
        confidence_scores.append(state["validation_results"]["overall_validation"].confidence_score)
    
    if not confidence_scores:
        return 0.0
    
    return sum(confidence_scores) / len(confidence_scores) 