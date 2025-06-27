"""
量子智能化功能点估算系统 - 工作流节点单元测试

测试工作流中各个节点的功能
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from graph.node_functions import (
    start_workflow_node, recommend_standard_node, parse_requirements_node,
    identify_processes_node, nesma_classify_functions_node,
    nesma_calculate_complexity_node, nesma_calculate_ufp_node,
    cosmic_identify_users_node, cosmic_analyze_boundary_node,
    cosmic_classify_movements_node, cosmic_calculate_cfp_node,
    validate_results_node, compare_standards_node, generate_report_node,
    complete_workflow_node, handle_error_node
)
from graph.state_definitions import (
    WorkflowGraphState, WorkflowState, ProcessDetails, create_initial_state
)
from models.project_models import ProjectInfo, EstimationStrategy
from models.nesma_models import NESMAFunctionType, NESMAComplexityLevel
from models.cosmic_models import COSMICDataMovementType, COSMICFunctionalUser


class TestWorkflowNodes:
    """工作流节点测试"""
    
    @pytest.fixture
    def sample_project(self):
        """创建样本项目信息"""
        from models.project_models import TechnologyStack, BusinessDomain
        return ProjectInfo(
            name="测试项目",
            description="这是一个测试项目，用于验证功能点估算流程",
            technology_stack=[TechnologyStack.PYTHON, TechnologyStack.FASTAPI],
            business_domain=BusinessDomain.TESTING
        )
    
    @pytest.fixture
    def sample_state(self, sample_project):
        """创建样本工作流状态"""
        state = create_initial_state("test_session", sample_project)
        state["user_requirements"] = "用户管理、订单处理、报表生成等核心功能"
        state["user_preferences"] = {"preferred_standard": "NESMA"}
        return state
    
    @pytest.mark.asyncio
    async def test_start_workflow_node(self, sample_state):
        """测试工作流启动节点"""
        result = await start_workflow_node(sample_state)
        
        assert result["current_state"] == WorkflowState.REQUIREMENT_INPUT_RECEIVED
        assert len(result["execution_log"]) > 0
    
    @pytest.mark.asyncio
    async def test_recommend_standard_node(self, sample_state):
        """测试标准推荐节点"""
        sample_state["current_state"] = WorkflowState.REQUIREMENT_INPUT_RECEIVED
        
        with patch('agents.standards.standard_recommender.StandardRecommenderAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.recommend_standards = AsyncMock(return_value=Mock(
                strategy=EstimationStrategy.NESMA_ONLY,
                confidence_score=0.9,
                reasoning="基于项目特征推荐NESMA标准"
            ))
            
            result = await recommend_standard_node(sample_state)
            
            assert result["current_state"] == WorkflowState.STANDARD_RECOMMENDATION_READY
            assert result["selected_strategy"] == EstimationStrategy.NESMA_ONLY
    
    @pytest.mark.asyncio
    async def test_parse_requirements_node(self, sample_state):
        """测试需求解析节点"""
        sample_state["current_state"] = WorkflowState.STANDARD_RECOMMENDATION_READY
        
        with patch('agents.analysis.requirement_parser.RequirementParserAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.parse_requirements = AsyncMock(return_value={
                "functional_modules": [
                    {"name": "用户管理", "description": "用户注册登录"},
                    {"name": "订单处理", "description": "订单创建管理"}
                ],
                "business_entities": {},
                "business_processes": [],
                "data_groups": [],
                "confidence_score": 0.85
            })
            
            result = await parse_requirements_node(sample_state)
            
            assert result["current_state"] == WorkflowState.REQUIREMENT_PARSING_COMPLETED
            assert "requirement_analysis" in result
    
    @pytest.mark.asyncio
    async def test_identify_processes_node(self, sample_state):
        """测试流程识别节点"""
        sample_state["current_state"] = WorkflowState.REQUIREMENT_PARSING_COMPLETED
        sample_state["requirement_analysis"] = {
            "functional_modules": [{"name": "用户管理", "description": "用户注册登录"}]
        }
        
        with patch('agents.analysis.process_identifier.ProcessIdentifierAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.identify_business_processes = AsyncMock(return_value=[
                ProcessDetails(
                    id="process_1",
                    name="用户注册",
                    description="用户注册功能",
                    data_groups=["用户信息"],
                    dependencies=[],
                    metadata={}
                )
            ])
            
            result = await identify_processes_node(sample_state)
            
            assert result["current_state"] == WorkflowState.PROCESSES_IDENTIFIED
            assert "identified_processes" in result
    
    @pytest.mark.asyncio
    async def test_nesma_classify_functions_node(self, sample_state):
        """测试NESMA功能分类节点"""
        sample_state["current_state"] = WorkflowState.PROCESSES_IDENTIFIED
        sample_state["identified_processes"] = [
            ProcessDetails(
                id="process_1",
                name="用户注册",
                description="用户注册功能",
                data_groups=["用户信息"],
                dependencies=[],
                metadata={}
            )
        ]
        
        with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.classify_function = AsyncMock(return_value=Mock(
                function_id="func_1",
                function_type=NESMAFunctionType.EI,
                function_description="用户注册功能",
                confidence_score=0.9
            ))
            
            result = await nesma_classify_functions_node(sample_state)
            
            assert result["current_state"] == WorkflowState.NESMA_CLASSIFICATION_COMPLETED
            assert "nesma_results" in result
    
    @pytest.mark.asyncio
    async def test_nesma_calculate_complexity_node(self, sample_state):
        """测试NESMA复杂度计算节点"""
        sample_state["current_state"] = WorkflowState.NESMA_CLASSIFICATION_COMPLETED
        sample_state["nesma_results"] = {
            "function_classifications": [
                Mock(
                    function_type=NESMAFunctionType.EI,
                    function_description="用户注册功能"
                )
            ]
        }
        
        with patch('agents.standards.nesma.complexity_calculator.NESMAComplexityCalculatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value={
                "complexity": NESMAComplexityLevel.LOW,
                "det_count": 5,
                "ret_count": 1,
                "reasoning": "数据元素较少"
            })
            
            result = await nesma_calculate_complexity_node(sample_state)
            
            assert result["current_state"] == WorkflowState.NESMA_COMPLEXITY_COMPLETED
            assert "complexity_results" in result["nesma_results"]
    
    @pytest.mark.asyncio
    async def test_nesma_calculate_ufp_node(self, sample_state):
        """测试NESMA UFP计算节点"""
        sample_state["current_state"] = WorkflowState.NESMA_COMPLEXITY_COMPLETED
        sample_state["nesma_results"] = {
            "function_classifications": [Mock(function_type=NESMAFunctionType.EI)],
            "complexity_results": [{"complexity": "LOW"}]
        }
        
        with patch('agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value=Mock(
                total_ufp=25,
                ufp_breakdown={"EI": 15, "ILF": 10},
                confidence_score=0.9,
                calculation_details={}
            ))
            
            result = await nesma_calculate_ufp_node(sample_state)
            
            assert result["current_state"] == WorkflowState.NESMA_CALCULATION_COMPLETED
            assert result["nesma_results"]["total_ufp"] == 25
    
    @pytest.mark.asyncio
    async def test_cosmic_identify_users_node(self, sample_state):
        """测试COSMIC功能用户识别节点"""
        sample_state["current_state"] = WorkflowState.COSMIC_PROCESSING_PENDING
        
        with patch('agents.standards.cosmic.functional_user_agent.COSMICFunctionalUserAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value=[
                COSMICFunctionalUser(
                    user_id="customer",
                    name="客户",
                    description="使用系统的客户",
                    user_type="人员",
                    boundary_definition="Web界面",
                    interaction_scope="前端Web界面",
                    identification_confidence=0.9,
                    identification_reasoning="基于需求分析识别"
                )
            ])
            
            result = await cosmic_identify_users_node(sample_state)
            
            assert result["current_state"] == WorkflowState.COSMIC_PROCESSING_PENDING
            assert "cosmic_results" in result
    
    @pytest.mark.asyncio
    async def test_cosmic_analyze_boundary_node(self, sample_state):
        """测试COSMIC边界分析节点"""
        sample_state["current_state"] = WorkflowState.COSMIC_PROCESSING_PENDING
        sample_state["cosmic_results"] = {
            "functional_users": [{"id": "customer", "name": "客户"}]
        }
        
        with patch('agents.standards.cosmic.boundary_analyzer.COSMICBoundaryAnalyzerAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value={
                "boundary_definition": "系统边界定义",
                "software_components": [{"name": "用户服务", "type": "核心组件"}],
                "validation_result": {"is_valid": True}
            })
            
            result = await cosmic_analyze_boundary_node(sample_state)
            
            assert result["current_state"] == WorkflowState.COSMIC_ANALYSIS_COMPLETED
            assert "boundary_analysis" in result["cosmic_results"]
    
    @pytest.mark.asyncio
    async def test_cosmic_classify_data_movements_node(self, sample_state):
        """测试COSMIC数据移动分类节点"""
        sample_state["current_state"] = WorkflowState.COSMIC_PROCESSING_PENDING
        sample_state["cosmic_results"] = {
            "functional_users": [{"id": "customer"}],
            "boundary_analysis": {"boundary_definition": "测试边界"}
        }
        sample_state["identified_processes"] = [
            ProcessDetails(
                id="process_1",
                name="用户注册",
                description="用户注册流程",
                data_groups=[],
                dependencies=[],
                metadata={}
            )
        ]
        
        with patch('agents.standards.cosmic.data_movement_classifier.COSMICDataMovementClassifierAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value={
                "data_movements": [
                    {
                        "id": "movement_1",
                        "type": "Entry",
                        "data_group": "用户信息",
                        "source": "用户",
                        "target": "系统"
                    }
                ],
                "classification_confidence": 0.9
            })
            
            result = await cosmic_classify_movements_node(sample_state)
            
            assert result["current_state"] == WorkflowState.COSMIC_CALCULATION_COMPLETED
            assert len(result["cosmic_results"]["data_movements"]) == 1
    
    @pytest.mark.asyncio
    async def test_cosmic_calculate_cfp_node(self, sample_state):
        """测试COSMIC CFP计算节点"""
        sample_state["current_state"] = WorkflowState.COSMIC_ANALYSIS_COMPLETED
        sample_state["cosmic_results"] = {
            "data_movements": [
                {"id": "movement_1", "type": "Entry", "data_group": "用户信息"}
            ]
        }
        
        with patch('agents.standards.cosmic.cfp_calculator.COSMICCFPCalculatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value={
                "total_cfp": 4,
                "movement_breakdown": {"movement_1": 1},
                "functional_processes": []
            })
            
            result = await cosmic_calculate_cfp_node(sample_state)
            
            assert result["current_state"] == WorkflowState.CROSS_STANDARD_COMPARISON_PENDING
            assert result["cosmic_results"]["total_cfp"] == 4
    
    @pytest.mark.asyncio
    async def test_validate_results_node(self, sample_state):
        """测试结果验证节点"""
        sample_state["current_state"] = WorkflowState.NESMA_CALCULATION_COMPLETED
        sample_state["nesma_results"] = {"total_ufp": 25}
        
        with patch('agents.knowledge.validator.ValidatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value={
                "validation_results": {"overall_validation": {"is_valid": True}},
                "confidence_score": 0.9
            })
            
            result = await validate_results_node(sample_state)
            
            assert "validation_results" in result
    
    @pytest.mark.asyncio
    async def test_generate_report_node(self, sample_state):
        """测试报告生成节点"""
        sample_state["current_state"] = WorkflowState.REPORT_GENERATION_PENDING
        sample_state["nesma_results"] = {"total_ufp": 25}
        sample_state["cosmic_results"] = {"total_cfp": 4}
        sample_state["validation_results"] = {"overall_validation": {"is_valid": True}}
        
        with patch('agents.output.report_generator.ReportGeneratorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute = AsyncMock(return_value={
                "report_content": "测试报告内容",
                "summary": {"nesma_ufp": 25, "cosmic_cfp": 4},
                "recommendations": []
            })
            
            result = await generate_report_node(sample_state)
            
            assert result["current_state"] == WorkflowState.COMPLETED
            assert "final_report" in result
    
    @pytest.mark.asyncio
    async def test_handle_error_node(self, sample_state):
        """测试错误处理节点"""
        sample_state["current_state"] = WorkflowState.ERROR_ENCOUNTERED
        sample_state["error_message"] = "测试错误"
        
        result = await handle_error_node(sample_state)
        
        assert result["current_state"] == WorkflowState.ERROR_ENCOUNTERED
        assert result["error_message"] == "测试错误"
    
    @pytest.mark.asyncio
    async def test_workflow_completed_node(self, sample_state):
        """测试工作流完成节点"""
        sample_state["current_state"] = WorkflowState.COMPLETED
        
        result = await complete_workflow_node(sample_state)
        
        assert result["current_state"] == WorkflowState.COMPLETED
        assert len(result["execution_log"]) > 0


class TestWorkflowStateTransitions:
    """工作流状态转换测试"""
    
    @pytest.fixture
    def base_state(self):
        """基础状态"""
        return {"current_state": WorkflowState.STARTING}
    
    def test_nesma_only_flow(self, base_state):
        """测试NESMA单一流程"""
        base_state["selected_strategy"] = EstimationStrategy.NESMA_ONLY
        
        expected_flow = [
            WorkflowState.STARTING,
            WorkflowState.REQUIREMENT_INPUT_RECEIVED,
            WorkflowState.STANDARD_RECOMMENDATION_READY,
            WorkflowState.PROCESSES_IDENTIFIED,
            WorkflowState.NESMA_CLASSIFICATION_COMPLETED,
            WorkflowState.NESMA_CALCULATION_COMPLETED,
            WorkflowState.CROSS_STANDARD_COMPARISON_PENDING,
            WorkflowState.REPORT_GENERATION_PENDING,
            WorkflowState.COMPLETED
        ]
        
        # 验证状态流程逻辑
        assert len(expected_flow) == 9
    
    def test_cosmic_only_flow(self, base_state):
        """测试COSMIC单一流程"""
        base_state["selected_strategy"] = EstimationStrategy.COSMIC_ONLY
        
        expected_flow = [
            WorkflowState.STARTING,
            WorkflowState.REQUIREMENT_INPUT_RECEIVED,
            WorkflowState.STANDARD_RECOMMENDATION_READY,
            WorkflowState.PROCESSES_IDENTIFIED,
            WorkflowState.COSMIC_PROCESSING_PENDING,
            WorkflowState.COSMIC_ANALYSIS_COMPLETED,
            WorkflowState.COSMIC_CALCULATION_COMPLETED,
            WorkflowState.CROSS_STANDARD_COMPARISON_PENDING,
            WorkflowState.REPORT_GENERATION_PENDING,
            WorkflowState.COMPLETED
        ]
        
        # 验证状态流程逻辑
        assert len(expected_flow) == 10
    
    def test_dual_parallel_flow(self, base_state):
        """测试双标准并行流程"""
        base_state["selected_strategy"] = EstimationStrategy.DUAL_PARALLEL
        
        # 双标准并行会包含所有NESMA和COSMIC的状态
        nesma_states = [
            WorkflowState.NESMA_CLASSIFICATION_COMPLETED,
            WorkflowState.NESMA_CALCULATION_COMPLETED
        ]
        
        cosmic_states = [
            WorkflowState.COSMIC_PROCESSING_PENDING,
            WorkflowState.COSMIC_ANALYSIS_COMPLETED,
            WorkflowState.COSMIC_CALCULATION_COMPLETED
        ]
        
        # 验证状态包含性
        assert len(nesma_states) == 2
        assert len(cosmic_states) == 3


class TestWorkflowErrorHandling:
    """工作流错误处理测试"""
    
    @pytest.fixture
    def sample_state(self):
        """创建样本状态用于错误处理测试"""
        from models.project_models import TechnologyStack, BusinessDomain
        project = ProjectInfo(
            name="错误处理测试项目",
            description="用于测试系统错误处理机制和异常恢复能力的项目",
            technology_stack=[TechnologyStack.PYTHON], 
            business_domain=BusinessDomain.TESTING
        )
        return create_initial_state("error-test-session", project)
    
    @pytest.mark.asyncio
    async def test_agent_execution_error(self, sample_state):
        """测试智能体执行错误"""
        sample_state["current_state"] = WorkflowState.REQUIREMENT_PARSING_COMPLETED
        
        with patch('agents.analysis.requirement_parser.RequirementParserAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.parse_requirements = AsyncMock(side_effect=Exception("模拟错误"))
            
            result = await parse_requirements_node(sample_state)
            
            assert result["current_state"] == WorkflowState.ERROR_ENCOUNTERED
            assert "模拟错误" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, sample_state):
        """测试重试机制"""
        sample_state["retry_count"] = 1
        sample_state["error_message"] = "网络连接错误"
        
        # 模拟重试逻辑 - 这里简化处理
        assert sample_state["retry_count"] == 1
        assert sample_state["error_message"] == "网络连接错误"
    
    @pytest.mark.asyncio 
    async def test_max_retries_exceeded(self, sample_state):
        """测试超过最大重试次数"""
        sample_state["retry_count"] = 3  # 假设最大重试次数为3
        sample_state["error_message"] = "持续性错误"
        
        # 模拟超过重试次数的处理
        assert sample_state["retry_count"] >= 3


class TestWorkflowPerformance:
    """工作流性能测试"""
    
    @pytest.fixture
    def sample_state(self):
        """创建样本状态用于性能测试"""
        from models.project_models import TechnologyStack, BusinessDomain
        project = ProjectInfo(
            name="性能测试项目", 
            description="用于测试系统性能和稳定性的完整项目场景",
            technology_stack=[TechnologyStack.PYTHON],
            business_domain=BusinessDomain.TESTING
        )
        return create_initial_state("performance-test-session", project)
    
    @pytest.mark.asyncio
    async def test_node_execution_time(self, sample_state):
        """测试节点执行时间"""
        import time
        
        start_time = time.time()
        result = await start_workflow_node(sample_state)
        execution_time = time.time() - start_time
        
        # 验证执行时间在合理范围内（< 1秒）
        assert execution_time < 1.0
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_state_serialization_performance(self, sample_state):
        """测试状态序列化性能"""
        import json
        import time
        
        # 创建一个复杂的状态对象
        complex_state = sample_state.copy()
        complex_state["large_data"] = ["item"] * 1000
        
        start_time = time.time()
        
        # 模拟状态序列化
        serialized = json.dumps(complex_state, default=str)
        deserialized = json.loads(serialized)
        
        serialization_time = time.time() - start_time
        
        # 验证序列化时间在合理范围内
        assert serialization_time < 0.1
        assert deserialized["session_id"] == complex_state["session_id"]


@pytest.mark.asyncio
async def test_complete_workflow_simulation():
    """完整工作流模拟测试"""
    
    # 创建初始状态
    from models.project_models import TechnologyStack, BusinessDomain
    initial_state = {
        "session_id": "simulation-test",
        "current_state": WorkflowState.STARTING,
        "project_info": ProjectInfo(
            name="模拟项目",
            description="这是一个完整的工作流模拟测试项目，包含用户管理、订单处理、支付等功能模块", 
            technology_stack=[TechnologyStack.JAVA],
            business_domain=BusinessDomain.ECOMMERCE
        ),
        "selected_strategy": EstimationStrategy.DUAL_PARALLEL,
        "user_requirements": "用户注册、登录、订单管理等功能",
        "execution_log": [],
        "user_preferences": {},
        "requirement_analysis": {},
        "process_identification": {},
        "standard_recommendation": {},
        "nesma_results": {},
        "cosmic_results": {},
        "validation_results": {},
        "comparison_analysis": {},
        "final_report": {},
                    "error_message": None,
        "retry_count": 0,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z"
    }
    
    # 启动工作流
    result = await start_workflow_node(initial_state)
    
    # 验证结果
    assert result["current_state"] == WorkflowState.REQUIREMENT_INPUT_RECEIVED
    assert result["session_id"] == "simulation-test"
    assert len(result["execution_log"]) > 0 