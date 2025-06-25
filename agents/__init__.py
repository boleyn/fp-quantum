"""
量子智能化功能点估算系统 - 智能体模块

提供完整的功能点估算智能体生态系统
"""

# 基础智能体
from .base.base_agent import BaseAgent, SpecializedAgent

# 编排者智能体
from .orchestrator.workflow_orchestrator import WorkflowOrchestratorAgent

# 标准推荐智能体
from .standards.standard_recommender import StandardRecommenderAgent

# NESMA专精智能体
from .standards.nesma.function_classifier import NESMAFunctionClassifierAgent
from .standards.nesma.complexity_calculator import NESMAComplexityCalculatorAgent
from .standards.nesma.ufp_calculator import NESMAUFPCalculatorAgent

# COSMIC专精智能体  
from .standards.cosmic.functional_user_agent import COSMICFunctionalUserAgent
from .standards.cosmic.data_movement_classifier import COSMICDataMovementClassifierAgent
from .standards.cosmic.boundary_analyzer import COSMICBoundaryAnalyzerAgent
from .standards.cosmic.cfp_calculator import COSMICCFPCalculatorAgent

# 分析智能体
from .analysis.requirement_parser import RequirementParserAgent

# 知识管理智能体
from .knowledge.rule_retriever import RuleRetrieverAgent

# 输出处理智能体
from .output.report_generator import ReportGeneratorAgent

__all__ = [
    # 基础智能体
    "BaseAgent",
    "SpecializedAgent",
    
    # 编排者
    "WorkflowOrchestratorAgent",
    
    # 标准推荐
    "StandardRecommenderAgent",
    
    # NESMA智能体
    "NESMAFunctionClassifierAgent",
    "NESMAComplexityCalculatorAgent", 
    "NESMAUFPCalculatorAgent",
    
    # COSMIC智能体
    "COSMICFunctionalUserAgent",
    "COSMICDataMovementClassifierAgent",
    "COSMICBoundaryAnalyzerAgent",
    "COSMICCFPCalculatorAgent",
    
    # 分析智能体
    "RequirementParserAgent",
    
    # 知识管理
    "RuleRetrieverAgent",
    
    # 输出处理
    "ReportGeneratorAgent"
]

# 版本信息
__version__ = "0.1.0"
__author__ = "量子智能化功能点估算系统开发团队"
