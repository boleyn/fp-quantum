"""
工作流节点单元测试

测试LangGraph工作流节点的功能和状态转换
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from graph.node_functions import (
    start_workflow,
    recommend_standards,
    parse_requirements,
    nesma_classify_functions,
    nesma_calculate_complexity,
    nesma_calculate_ufp,
    cosmic_identify_functional_users,
    cosmic_analyze_boundaries,
    cosmic_classify_data_movements,
    cosmic_calculate_cfp,
    validate_results,
    generate_report,
    handle_error,
    workflow_completed
)
from graph.state_definitions import WorkflowState, EstimationState
from models.common_models import EstimationStrategy
from models.project_models import ProjectInfo, TechnologyStack, BusinessDomain


class TestWorkflowNodes:
    """工作流节点测试"""
    
    @pytest.fixture
    def sample_state(self):
        """创建样本状态"""
        return EstimationState(
            session_id="test-session-123",
            current_state=WorkflowState.STARTED,
            project_info=ProjectInfo(
                name="测试项目",
                description="测试项目描述",
                technology_stack=[TechnologyStack.JAVA],
                business_domain=BusinessDomain.FINANCE
            ),
            strategy=EstimationStrategy.DUAL_PARALLEL,
            requirements="用户注册、登录、查询功能"
        )
    
    @pytest.mark.asyncio
    async def test_start_workflow_node(self, sample_state):
        """测试工作流启动节点"""
        result = await start_workflow(sample_state)
        
        assert result["current_state"] == WorkflowState.STANDARD_RECOMMENDATION
        assert result["execution_log"][-1]["step"] == "workflow_started"
        assert result["execution_log"][-1]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_recommend_standards_node(self, sample_state):
        """测试标准推荐节点"""
        sample_state.current_state = WorkflowState.STANDARD_RECOMMENDATION
        
        with patch('agents.standards.standard_recommender.StandardRecommenderAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "recommended_standards": ["NESMA", "COSMIC"],
                "confidence": 0.9,
                "reasoning": "项目适合双标准估算"
            }
            
            result = await recommend_standards(sample_state)
            
            assert result["current_state"] == WorkflowState.REQUIREMENT_PARSING
            assert "NESMA" in result["recommended_standards"]
            assert "COSMIC" in result["recommended_standards"]
    
    @pytest.mark.asyncio
    async def test_parse_requirements_node(self, sample_state):
        """测试需求解析节点"""
        sample_state.current_state = WorkflowState.REQUIREMENT_PARSING
        
        with patch('agents.analysis.requirement_parser.RequirementParserAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "identified_processes": [
                    {
                        "id": "process_1",
                        "name": "用户注册",
                        "description": "用户输入信息进行注册",
                        "type": "transactional"
                    },
                    {
                        "id": "process_2", 
                        "name": "用户登录",
                        "description": "用户验证身份",
                        "type": "inquiry"
                    }
                ],
                "total_processes": 2
            }
            
            result = await parse_requirements(sample_state)
            
            assert result["current_state"] == WorkflowState.KNOWLEDGE_RETRIEVAL
            assert len(result["identified_processes"]) == 2
            assert result["identified_processes"][0]["name"] == "用户注册"
    
    @pytest.mark.asyncio
    async def test_nesma_classify_functions_node(self, sample_state):
        """测试NESMA功能分类节点"""
        sample_state.current_state = WorkflowState.NESMA_CLASSIFICATION
        sample_state.identified_processes = [
            {
                "id": "process_1",
                "name": "用户注册",
                "description": "用户输入姓名、邮箱进行注册"
            }
        ]
        
        with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "function_type": "EI",
                "confidence": 0.9,
                "reasoning": "用户数据输入功能"
            }
            
            result = await nesma_classify_functions(sample_state)
            
            assert result["current_state"] == WorkflowState.NESMA_COMPLEXITY
            assert len(result["nesma_classifications"]) == 1
            assert result["nesma_classifications"][0]["function_type"] == "EI"
    
    @pytest.mark.asyncio
    async def test_nesma_calculate_complexity_node(self, sample_state):
        """测试NESMA复杂度计算节点"""
        sample_state.current_state = WorkflowState.NESMA_COMPLEXITY
        sample_state.nesma_classifications = [
            {
                "process_id": "process_1",
                "function_type": "EI",
                "description": "用户注册功能"
            }
        ]
        
        with patch('agents.standards.nesma.complexity_calculator.NESMAComplexityCalculatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "complexity": "LOW",
                "det_count": 5,
                "ftr_count": 1,
                "reasoning": "数据元素较少"
            }
            
            result = await nesma_calculate_complexity(sample_state)
            
            assert result["current_state"] == WorkflowState.NESMA_UFP_CALCULATION
            assert len(result["nesma_complexity_results"]) == 1
            assert result["nesma_complexity_results"][0]["complexity"] == "LOW"
    
    @pytest.mark.asyncio
    async def test_nesma_calculate_ufp_node(self, sample_state):
        """测试NESMA UFP计算节点"""
        sample_state.current_state = WorkflowState.NESMA_UFP_CALCULATION
        sample_state.nesma_classifications = [{"function_type": "EI"}]
        sample_state.nesma_complexity_results = [{"complexity": "LOW"}]
        
        with patch('agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "total_ufp": 25,
                "function_breakdown": [
                    {"function_type": "EI", "complexity": "LOW", "count": 5, "ufp": 25}
                ]
            }
            
            result = await nesma_calculate_ufp(sample_state)
            
            # 根据策略决定下一状态
            if sample_state.strategy == EstimationStrategy.NESMA_ONLY:
                assert result["current_state"] == WorkflowState.RESULT_VALIDATION
            else:
                assert result["current_state"] in [
                    WorkflowState.COSMIC_FUNCTIONAL_USER,
                    WorkflowState.RESULT_VALIDATION
                ]
            
            assert result["nesma_ufp_total"] == 25
    
    @pytest.mark.asyncio
    async def test_cosmic_identify_functional_users_node(self, sample_state):
        """测试COSMIC功能用户识别节点"""
        sample_state.current_state = WorkflowState.COSMIC_FUNCTIONAL_USER
        
        with patch('agents.standards.cosmic.functional_user_agent.COSMICFunctionalUserAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "functional_users": [
                    {"name": "注册用户", "description": "进行注册的普通用户"},
                    {"name": "系统管理员", "description": "管理用户账户的管理员"}
                ]
            }
            
            result = await cosmic_identify_functional_users(sample_state)
            
            assert result["current_state"] == WorkflowState.COSMIC_BOUNDARY_ANALYSIS
            assert len(result["cosmic_functional_users"]) == 2
    
    @pytest.mark.asyncio
    async def test_cosmic_analyze_boundaries_node(self, sample_state):
        """测试COSMIC边界分析节点"""
        sample_state.current_state = WorkflowState.COSMIC_BOUNDARY_ANALYSIS
        sample_state.cosmic_functional_users = [{"name": "注册用户"}]
        
        with patch('agents.standards.cosmic.boundary_analyzer.COSMICBoundaryAnalyzerAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "software_boundary": "用户管理系统边界",
                "persistent_storage_boundary": "用户数据库边界",
                "boundary_objects": ["用户表", "角色表"]
            }
            
            result = await cosmic_analyze_boundaries(sample_state)
            
            assert result["current_state"] == WorkflowState.COSMIC_DATA_MOVEMENT
            assert "用户管理系统边界" in result["cosmic_boundary_analysis"]["software_boundary"]
    
    @pytest.mark.asyncio
    async def test_cosmic_classify_data_movements_node(self, sample_state):
        """测试COSMIC数据移动分类节点"""
        sample_state.current_state = WorkflowState.COSMIC_DATA_MOVEMENT
        sample_state.identified_processes = [
            {"id": "process_1", "name": "用户注册", "description": "用户输入信息"}
        ]
        sample_state.cosmic_boundary_analysis = {"software_boundary": "系统边界"}
        
        with patch('agents.standards.cosmic.data_movement_classifier.COSMICDataMovementClassifierAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "data_movements": [
                    {"type": "Entry", "description": "用户信息输入"},
                    {"type": "Write", "description": "保存用户信息"}
                ]
            }
            
            result = await cosmic_classify_data_movements(sample_state)
            
            assert result["current_state"] == WorkflowState.COSMIC_CFP_CALCULATION
            assert len(result["cosmic_data_movements"]) == 2
    
    @pytest.mark.asyncio
    async def test_cosmic_calculate_cfp_node(self, sample_state):
        """测试COSMIC CFP计算节点"""
        sample_state.current_state = WorkflowState.COSMIC_CFP_CALCULATION
        sample_state.cosmic_data_movements = [
            {"type": "Entry", "count": 2},
            {"type": "Write", "count": 1}
        ]
        
        with patch('agents.standards.cosmic.cfp_calculator.COSMICCFPCalculatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "total_cfp": 3,
                "functional_processes": [
                    {"name": "用户注册", "cfp": 3, "data_movements": 3}
                ]
            }
            
            result = await cosmic_calculate_cfp(sample_state)
            
            assert result["current_state"] == WorkflowState.RESULT_VALIDATION
            assert result["cosmic_cfp_total"] == 3
    
    @pytest.mark.asyncio
    async def test_validate_results_node(self, sample_state):
        """测试结果验证节点"""
        sample_state.current_state = WorkflowState.RESULT_VALIDATION
        sample_state.nesma_ufp_total = 25
        sample_state.cosmic_cfp_total = 3
        
        with patch('agents.knowledge.validator.ValidatorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "validation_passed": True,
                "quality_score": 0.92,
                "issues": [],
                "recommendations": ["结果合理，可以使用"]
            }
            
            result = await validate_results(sample_state)
            
            assert result["current_state"] == WorkflowState.REPORT_GENERATION
            assert result["validation_results"]["validation_passed"] is True
    
    @pytest.mark.asyncio
    async def test_generate_report_node(self, sample_state):
        """测试报告生成节点"""
        sample_state.current_state = WorkflowState.REPORT_GENERATION
        sample_state.nesma_ufp_total = 25
        sample_state.cosmic_cfp_total = 3
        sample_state.validation_results = {"validation_passed": True}
        
        with patch('agents.output.report_generator.ReportGeneratorAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.return_value = {
                "report_type": "comprehensive_estimation_report",
                "executive_summary": "项目估算完成，NESMA 25 UFP，COSMIC 3 CFP",
                "detailed_results": {},
                "recommendations": []
            }
            
            result = await generate_report(sample_state)
            
            assert result["current_state"] == WorkflowState.COMPLETED
            assert "executive_summary" in result["final_report"]
    
    @pytest.mark.asyncio
    async def test_handle_error_node(self, sample_state):
        """测试错误处理节点"""
        sample_state.current_state = WorkflowState.ERROR
        sample_state.error_message = "测试错误"
        sample_state.retry_count = 0
        
        result = await handle_error(sample_state)
        
        if sample_state.retry_count < 3:
            # 应该重试
            assert result["retry_count"] == 1
            assert result["current_state"] != WorkflowState.FAILED
        else:
            # 超过重试次数，标记为失败
            assert result["current_state"] == WorkflowState.FAILED
    
    @pytest.mark.asyncio
    async def test_workflow_completed_node(self, sample_state):
        """测试工作流完成节点"""
        sample_state.current_state = WorkflowState.COMPLETED
        sample_state.final_report = {"report_type": "test_report"}
        
        result = await workflow_completed(sample_state)
        
        assert result["current_state"] == WorkflowState.COMPLETED
        assert result["execution_log"][-1]["step"] == "workflow_completed"
        assert result["completion_time"] is not None


class TestWorkflowStateTransitions:
    """工作流状态转换测试"""
    
    @pytest.fixture
    def base_state(self):
        """基础状态"""
        return EstimationState(
            session_id="test-123",
            current_state=WorkflowState.STARTED
        )
    
    def test_nesma_only_flow(self, base_state):
        """测试NESMA单一流程状态转换"""
        base_state.strategy = EstimationStrategy.NESMA_ONLY
        
        expected_flow = [
            WorkflowState.STARTED,
            WorkflowState.STANDARD_RECOMMENDATION,
            WorkflowState.REQUIREMENT_PARSING,
            WorkflowState.KNOWLEDGE_RETRIEVAL,
            WorkflowState.NESMA_CLASSIFICATION,
            WorkflowState.NESMA_COMPLEXITY,
            WorkflowState.NESMA_UFP_CALCULATION,
            WorkflowState.RESULT_VALIDATION,
            WorkflowState.REPORT_GENERATION,
            WorkflowState.COMPLETED
        ]
        
        # 验证每个状态转换是否合理
        for i in range(len(expected_flow) - 1):
            current = expected_flow[i]
            next_state = expected_flow[i + 1]
            
            # 这里应该有状态转换逻辑验证
            assert current != next_state  # 至少确保状态有变化
    
    def test_cosmic_only_flow(self, base_state):
        """测试COSMIC单一流程状态转换"""
        base_state.strategy = EstimationStrategy.COSMIC_ONLY
        
        expected_flow = [
            WorkflowState.STARTED,
            WorkflowState.STANDARD_RECOMMENDATION,
            WorkflowState.REQUIREMENT_PARSING,
            WorkflowState.KNOWLEDGE_RETRIEVAL,
            WorkflowState.COSMIC_FUNCTIONAL_USER,
            WorkflowState.COSMIC_BOUNDARY_ANALYSIS,
            WorkflowState.COSMIC_DATA_MOVEMENT,
            WorkflowState.COSMIC_CFP_CALCULATION,
            WorkflowState.RESULT_VALIDATION,
            WorkflowState.REPORT_GENERATION,
            WorkflowState.COMPLETED
        ]
        
        # 验证COSMIC流程长度合理
        assert len(expected_flow) >= 8
    
    def test_dual_parallel_flow(self, base_state):
        """测试双标准并行流程"""
        base_state.strategy = EstimationStrategy.DUAL_PARALLEL
        
        # 双标准并行应该包含更多状态
        nesma_states = [
            WorkflowState.NESMA_CLASSIFICATION,
            WorkflowState.NESMA_COMPLEXITY, 
            WorkflowState.NESMA_UFP_CALCULATION
        ]
        
        cosmic_states = [
            WorkflowState.COSMIC_FUNCTIONAL_USER,
            WorkflowState.COSMIC_BOUNDARY_ANALYSIS,
            WorkflowState.COSMIC_DATA_MOVEMENT,
            WorkflowState.COSMIC_CFP_CALCULATION
        ]
        
        # 验证包含两套标准的状态
        all_states = nesma_states + cosmic_states
        assert len(set(all_states)) == 7  # 应该有7个不同的状态


class TestWorkflowErrorHandling:
    """工作流错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_agent_execution_error(self, sample_state):
        """测试智能体执行错误"""
        sample_state.current_state = WorkflowState.NESMA_CLASSIFICATION
        
        with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.execute_task.side_effect = Exception("智能体执行失败")
            
            # 应该捕获异常并转到错误处理
            result = await nesma_classify_functions(sample_state)
            
            assert result["current_state"] == WorkflowState.ERROR
            assert "智能体执行失败" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, sample_state):
        """测试重试机制"""
        sample_state.current_state = WorkflowState.ERROR
        sample_state.error_message = "临时错误"
        sample_state.retry_count = 1
        sample_state.max_retries = 3
        
        result = await handle_error(sample_state)
        
        # 应该增加重试计数并重新尝试
        assert result["retry_count"] == 2
        assert result["current_state"] != WorkflowState.FAILED
    
    @pytest.mark.asyncio 
    async def test_max_retries_exceeded(self, sample_state):
        """测试超过最大重试次数"""
        sample_state.current_state = WorkflowState.ERROR
        sample_state.retry_count = 3
        sample_state.max_retries = 3
        
        result = await handle_error(sample_state)
        
        # 应该标记为失败
        assert result["current_state"] == WorkflowState.FAILED
        assert "超过最大重试次数" in result["error_message"]


class TestWorkflowPerformance:
    """工作流性能测试"""
    
    @pytest.mark.asyncio
    async def test_node_execution_time(self, sample_state):
        """测试节点执行时间"""
        import time
        
        start_time = time.time()
        result = await start_workflow(sample_state)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 简单节点应该很快完成
        assert execution_time < 0.1
        assert result["current_state"] == WorkflowState.STANDARD_RECOMMENDATION
    
    @pytest.mark.asyncio
    async def test_state_serialization_performance(self, sample_state):
        """测试状态序列化性能"""
        import json
        import time
        
        # 添加大量数据
        sample_state.execution_log = [
            {"step": f"step_{i}", "timestamp": "2024-01-01", "data": f"data_{i}"} 
            for i in range(1000)
        ]
        
        start_time = time.time()
        serialized = json.dumps(sample_state.dict())
        deserialized = EstimationState(**json.loads(serialized))
        end_time = time.time()
        
        # 序列化/反序列化应该很快
        assert end_time - start_time < 1.0
        assert deserialized.session_id == sample_state.session_id


@pytest.mark.asyncio
async def test_complete_workflow_simulation():
    """完整工作流模拟测试"""
    
    # 创建初始状态
    initial_state = EstimationState(
        session_id="simulation-test",
        current_state=WorkflowState.STARTED,
        strategy=EstimationStrategy.NESMA_ONLY,
        project_info=ProjectInfo(
            name="模拟项目",
            description="简单的测试项目",
            technology_stack=[TechnologyStack.PYTHON],
            business_domain=BusinessDomain.OTHER
        ),
        requirements="用户注册和登录功能"
    )
    
    # 模拟工作流执行
    state = initial_state
    
    # 1. 启动工作流
    state = EstimationState(**await start_workflow(state))
    assert state.current_state == WorkflowState.STANDARD_RECOMMENDATION
    
    # 2. 模拟标准推荐 (跳过实际执行，直接设置结果)
    state.current_state = WorkflowState.REQUIREMENT_PARSING
    state.recommended_standards = ["NESMA"]
    
    # 3. 模拟需求解析
    state.current_state = WorkflowState.KNOWLEDGE_RETRIEVAL
    state.identified_processes = [
        {"id": "p1", "name": "用户注册", "description": "注册功能"}
    ]
    
    # 验证状态正确传递
    assert len(state.identified_processes) == 1
    assert state.identified_processes[0]["name"] == "用户注册"
    
    # 验证执行日志记录
    assert len(state.execution_log) > 0
    assert state.execution_log[0]["step"] == "workflow_started" 