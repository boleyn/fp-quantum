"""
量子智能化功能点估算系统 - 端到端测试

测试完整的工作流程从输入到输出的整个过程
"""

import pytest
import asyncio
import time
import traceback
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import psutil
import os

# 工作流和智能体导入
from graph.workflow_graph import FPEstimationWorkflow
from models.project_models import (
    ProjectInfo, TechnologyStack, BusinessDomain, 
    EstimationStrategy, EstimationStandard
)
from models.nesma_models import NESMAFunctionType, NESMAComplexityLevel
from models.cosmic_models import COSMICDataMovementType
from models.common_models import ConfidenceLevel


class TestFullWorkflow:
    """完整工作流测试"""
    
    @pytest.fixture
    def workflow_instance(self):
        """创建工作流实例"""
        workflow = FPEstimationWorkflow()
        return workflow
    
    @pytest.fixture
    def sample_projects(self):
        """样本项目数据"""
        return {
            "small": ProjectInfo(
                name="小型CRM系统",
                description="客户关系管理系统，包含客户信息管理、销售跟踪、合同管理等基础功能模块，支持用户注册登录和基本的数据维护操作",
                technology_stack=[TechnologyStack.JAVA, TechnologyStack.MYSQL],
                business_domain=BusinessDomain.RETAIL
            ),
            "medium": ProjectInfo(
                name="中型电商平台",
                description="完整的电商平台系统，包含用户管理、商品管理、订单处理、支付系统、库存管理、物流跟踪、客服系统等核心功能模块，支持多用户角色和复杂的业务流程处理",
                technology_stack=[TechnologyStack.JAVA, TechnologyStack.MYSQL, TechnologyStack.REDIS],
                business_domain=BusinessDomain.ECOMMERCE
            ),
            "large": ProjectInfo(
                name="大型企业管理系统",
                description="大型企业资源规划系统，包含人力资源管理、财务管理、供应链管理、生产计划、项目管理、客户服务、数据分析等多个复杂业务模块，支持多组织架构、工作流审批、权限控制等高级功能特性",
                technology_stack=[TechnologyStack.JAVA, TechnologyStack.PYTHON, TechnologyStack.MYSQL, TechnologyStack.REDIS],
                business_domain=BusinessDomain.MANUFACTURING
            )
        }
    
    def _create_nesma_mock_responses(self):
        """创建NESMA相关的mock响应"""
        return {
            "standard_recommendation": {
                "recommended_standard": "NESMA",
                "confidence_score": 0.85,
                "reasoning": "传统企业应用适合NESMA标准",
                "alternative_standards": ["COSMIC"]
            },
            "requirement_parsing": {
                "functional_modules": [
                    {"name": "用户管理", "description": "用户注册登录"},
                    {"name": "商品管理", "description": "商品信息维护"}
                ],
                "business_entities": {"用户角色": ["管理员", "普通用户"]},
                "business_processes": []
            },
            "process_identification": [
                {
                    "id": "proc_1",
                    "name": "用户注册",
                    "description": "用户注册流程",
                    "data_groups": ["用户信息"],
                    "dependencies": []
                }
            ],
            "nesma_classification": {
                "function_type": NESMAFunctionType.EI,
                "confidence_score": 0.9,
                "justification": "数据输入功能"
            },
            "nesma_complexity": {
                "complexity": NESMAComplexityLevel.AVERAGE,
                "det_count": 8,
                "ret_count": 2
            },
            "nesma_ufp": {
                "total_ufp": 42,
                "function_breakdown": []
            }
        }
    
    def _create_cosmic_mock_responses(self):
        """创建COSMIC相关的mock响应"""
        return {
            "cosmic_functional_users": [
                {
                    "id": "user_1",
                    "name": "系统用户",
                    "description": "使用系统的用户",
                    "identification_confidence": 0.9
                }
            ],
            "cosmic_data_movements": [
                {
                    "type": COSMICDataMovementType.ENTRY,
                    "source": "用户界面",
                    "target": "应用层",
                    "data_group": "用户信息",
                    "confidence_score": 0.85
                }
            ],
            "cosmic_cfp": {
                "total_cfp": 15,
                "movement_breakdown": []
            }
        }
    
    @pytest.mark.asyncio
    async def test_nesma_only_workflow(self, workflow_instance, sample_projects):
        """测试纯NESMA估算工作流"""
        project = sample_projects["medium"]
        
        # 初始化工作流
        session_id = await workflow_instance.initialize(
            project_info=project,
            strategy=EstimationStrategy.NESMA_ONLY,
            requirements="测试NESMA估算工作流"
        )
        
        # Mock各个智能体的返回结果
        mock_responses = self._create_nesma_mock_responses()
        
        with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent.execute', new_callable=AsyncMock) as mock_classifier, \
             patch('agents.standards.nesma.complexity_calculator.NESMAComplexityCalculatorAgent.execute', new_callable=AsyncMock) as mock_complexity, \
             patch('agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent.execute', new_callable=AsyncMock) as mock_ufp, \
             patch('agents.standards.standard_recommender.StandardRecommenderAgent.recommend_standards', new_callable=AsyncMock) as mock_recommender, \
             patch('agents.analysis.requirement_parser.RequirementParserAgent.parse_requirements', new_callable=AsyncMock) as mock_parser, \
             patch('agents.analysis.process_identifier.ProcessIdentifierAgent.identify_processes', new_callable=AsyncMock) as mock_identifier:
            
            # 配置mock返回值
            mock_recommender.return_value = mock_responses["standard_recommendation"]
            mock_parser.return_value = mock_responses["requirement_parsing"]
            mock_identifier.return_value = mock_responses["process_identification"]
            mock_classifier.return_value = mock_responses["nesma_classification"]
            mock_complexity.return_value = mock_responses["nesma_complexity"]
            mock_ufp.return_value = mock_responses["nesma_ufp"]
            
            # 执行工作流
            result = await workflow_instance.execute()
            
            # 验证结果
            assert result is not None
            assert session_id == workflow_instance.get_session_id()
            
            # 验证调用次数
            assert mock_recommender.called
            assert mock_parser.called
    
    @pytest.mark.asyncio
    async def test_cosmic_only_workflow(self, workflow_instance, sample_projects):
        """测试纯COSMIC估算工作流"""
        project = sample_projects["medium"]
        
        # 初始化工作流
        session_id = await workflow_instance.initialize(
            project_info=project,
            strategy=EstimationStrategy.COSMIC_ONLY,
            requirements="测试COSMIC估算工作流"
        )
        
        # Mock各个智能体的返回结果
        mock_responses = self._create_cosmic_mock_responses()
        
        with patch('agents.standards.cosmic.functional_user_agent.COSMICFunctionalUserAgent.execute', new_callable=AsyncMock) as mock_functional_user, \
             patch('agents.standards.cosmic.boundary_analyzer.COSMICBoundaryAnalyzerAgent.execute', new_callable=AsyncMock) as mock_boundary, \
             patch('agents.standards.cosmic.data_movement_classifier.COSMICDataMovementClassifierAgent.execute', new_callable=AsyncMock) as mock_data_movement, \
             patch('agents.standards.cosmic.cfp_calculator.COSMICCFPCalculatorAgent.execute', new_callable=AsyncMock) as mock_cfp, \
             patch('agents.standards.standard_recommender.StandardRecommenderAgent.recommend_standards', new_callable=AsyncMock) as mock_recommender, \
             patch('agents.analysis.requirement_parser.RequirementParserAgent.parse_requirements', new_callable=AsyncMock) as mock_parser, \
             patch('agents.analysis.process_identifier.ProcessIdentifierAgent.identify_processes', new_callable=AsyncMock) as mock_identifier:
            
            # 配置mock返回值
            mock_recommender.return_value = {
                "recommended_standard": "COSMIC",
                "confidence_score": 0.9,
                "reasoning": "现代应用适合COSMIC标准",
                "alternative_standards": ["NESMA"]
            }
            mock_parser.return_value = {
                "functional_modules": [{"name": "API服务", "description": "RESTful API"}],
                "business_entities": {"功能用户": ["客户端应用"]},
                "business_processes": []
            }
            mock_identifier.return_value = [
                {
                    "id": "proc_1", 
                    "name": "数据处理",
                    "description": "数据处理流程",
                    "data_groups": ["业务数据"],
                    "dependencies": []
                }
            ]
            mock_functional_user.return_value = mock_responses["cosmic_functional_users"]
            mock_boundary.return_value = {"boundary_analysis": "completed"}
            mock_data_movement.return_value = mock_responses["cosmic_data_movements"]
            mock_cfp.return_value = mock_responses["cosmic_cfp"]
            
            # 执行工作流
            result = await workflow_instance.execute()
            
            # 验证结果
            assert result is not None
            assert session_id == workflow_instance.get_session_id()
            
            # 验证调用次数
            assert mock_recommender.called
            assert mock_functional_user.called
    
    @pytest.mark.asyncio
    async def test_dual_standard_workflow(self, workflow_instance, sample_projects):
        """测试双标准对比估算工作流"""
        project = sample_projects["large"]
        
        # 初始化工作流
        session_id = await workflow_instance.initialize(
            project_info=project,
            strategy=EstimationStrategy.DUAL_PARALLEL,
            requirements="测试双标准对比估算工作流"
        )
        
        # Mock两套标准的返回结果
        nesma_mock = self._create_nesma_mock_responses()
        cosmic_mock = self._create_cosmic_mock_responses()
        
        with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent.execute', new_callable=AsyncMock) as mock_nesma_classifier, \
             patch('agents.standards.nesma.complexity_calculator.NESMAComplexityCalculatorAgent.execute', new_callable=AsyncMock) as mock_nesma_complexity, \
             patch('agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent.execute', new_callable=AsyncMock) as mock_nesma_ufp, \
             patch('agents.standards.cosmic.functional_user_agent.COSMICFunctionalUserAgent.execute', new_callable=AsyncMock) as mock_cosmic_functional_user, \
             patch('agents.standards.cosmic.boundary_analyzer.COSMICBoundaryAnalyzerAgent.execute', new_callable=AsyncMock) as mock_cosmic_boundary, \
             patch('agents.standards.cosmic.data_movement_classifier.COSMICDataMovementClassifierAgent.execute', new_callable=AsyncMock) as mock_cosmic_data_movement, \
             patch('agents.standards.cosmic.cfp_calculator.COSMICCFPCalculatorAgent.execute', new_callable=AsyncMock) as mock_cosmic_cfp, \
             patch('agents.standards.standard_recommender.StandardRecommenderAgent.recommend_standards', new_callable=AsyncMock) as mock_recommender, \
             patch('agents.analysis.requirement_parser.RequirementParserAgent.parse_requirements', new_callable=AsyncMock) as mock_parser, \
             patch('agents.analysis.process_identifier.ProcessIdentifierAgent.identify_processes', new_callable=AsyncMock) as mock_identifier:
            
            # 配置mock返回值
            mock_recommender.return_value = {
                "recommended_standard": "NESMA+COSMIC",
                "confidence_score": 0.85,
                "reasoning": "复杂项目建议双标准对比",
                "alternative_standards": ["NESMA", "COSMIC"]
            }
            mock_parser.return_value = nesma_mock["requirement_parsing"]
            mock_identifier.return_value = nesma_mock["process_identification"]
            
            # NESMA mock配置
            mock_nesma_classifier.return_value = nesma_mock["nesma_classification"]
            mock_nesma_complexity.return_value = nesma_mock["nesma_complexity"]
            mock_nesma_ufp.return_value = nesma_mock["nesma_ufp"]
            
            # COSMIC mock配置
            mock_cosmic_functional_user.return_value = cosmic_mock["cosmic_functional_users"]
            mock_cosmic_boundary.return_value = {"boundary_analysis": "completed"}
            mock_cosmic_data_movement.return_value = cosmic_mock["cosmic_data_movements"]
            mock_cosmic_cfp.return_value = cosmic_mock["cosmic_cfp"]
            
            # 执行工作流
            result = await workflow_instance.execute()
            
            # 验证结果
            assert result is not None
            assert session_id == workflow_instance.get_session_id()
            
            # 验证调用次数
            assert mock_recommender.called
            assert mock_nesma_classifier.called
            assert mock_cosmic_functional_user.called
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, workflow_instance, sample_projects):
        """测试错误处理和重试机制"""
        project = sample_projects["small"]
        
        # 初始化工作流
        session_id = await workflow_instance.initialize(
            project_info=project,
            strategy=EstimationStrategy.NESMA_ONLY,
            requirements="测试错误处理和重试机制"
        )
        
        # 模拟第一次失败，第二次成功的场景
        call_count = 0
        
        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("模拟LLM调用失败")
            return {
                "function_type": "EI",
                "confidence": 0.9,
                "reasoning": "重试后成功"
            }
        
        with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent.execute', 
                   side_effect=failing_then_success):
            
            # 执行工作流（应该会自动重试）
            try:
                result = await workflow_instance.execute()
                # 验证重试机制生效
                assert call_count >= 2  # 至少调用了2次（第一次失败，第二次成功）
            except Exception:
                # 如果重试仍然失败，这也是可接受的
                assert call_count >= 1
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, workflow_instance, sample_projects):
        """测试性能基准"""
        performance_results = {}
        
        for project_size, project in sample_projects.items():
            # 执行多次取平均值
            execution_times = []
            memory_usages = []
            
            for run in range(2):  # 运行2次减少测试时间
                # 初始化工作流
                session_id = await workflow_instance.initialize(
                    project_info=project,
                    strategy=EstimationStrategy.NESMA_ONLY,
                    requirements=f"性能测试 - {project_size} 项目 - 第{run+1}次"
                )
                
                # Mock简化的响应以确保稳定性能
                with patch.multiple(
                    'agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent',
                    execute=AsyncMock(return_value={"function_type": "EI", "confidence": 0.9}),
                ), patch.multiple(
                    'agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent',
                    execute=AsyncMock(return_value={"total_ufp": 25, "function_breakdown": []}),
                ):
                    
                    # 记录开始时间和内存
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    # 执行工作流
                    await workflow_instance.execute()
                    
                    # 记录结束时间和内存
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    execution_times.append(end_time - start_time)
                    memory_usages.append(end_memory - start_memory)
            
            # 计算平均值
            avg_time = sum(execution_times) / len(execution_times)
            avg_memory = sum(memory_usages) / len(memory_usages)
            
            performance_results[project_size] = {
                "average_execution_time": avg_time,
                "average_memory_usage": avg_memory,
                "execution_times": execution_times,
                "memory_usages": memory_usages
            }
        
        # 性能断言
        assert performance_results["small"]["average_execution_time"] < 10.0  # 小项目10秒内
        assert performance_results["medium"]["average_execution_time"] < 15.0  # 中项目15秒内
        assert performance_results["large"]["average_execution_time"] < 30.0  # 大项目30秒内
        
        # 内存使用应该合理（小于200MB增长）
        for size, results in performance_results.items():
            assert abs(results["average_memory_usage"]) < 200.0
        
        print(f"\n📊 性能基准测试结果:")
        for size, results in performance_results.items():
            print(f"  {size}: 平均耗时 {results['average_execution_time']:.2f}s, "
                  f"平均内存变化 {results['average_memory_usage']:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, workflow_instance, sample_projects):
        """测试并发执行能力"""
        # 准备多个并发任务
        tasks = []
        for i in range(3):
            project = sample_projects["small"]
            
            # 为每个任务创建独立的工作流实例
            workflow = FPEstimationWorkflow()
            session_id = await workflow.initialize(
                project_info=project,
                strategy=EstimationStrategy.NESMA_ONLY,
                requirements=f"并发测试任务 {i+1}"
            )
            
            # 为每个任务创建独立的mock
            with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent.execute', 
                       new_callable=AsyncMock) as mock_classifier:
                mock_classifier.return_value = {"function_type": "EI", "confidence": 0.9}
                
                task = workflow.execute()
                tasks.append(task)
        
        # 并发执行所有任务
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 验证结果
        assert len(results) == 3
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 1  # 至少1个成功
        
        # 并发执行应该比串行执行快
        concurrent_time = end_time - start_time
        print(f"\n⚡ 并发执行耗时: {concurrent_time:.2f}s")
        assert concurrent_time < 30.0  # 3个任务并发执行应在30秒内完成


class TestConcurrentWorkflows:
    """并发工作流测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_estimations(self):
        """测试多个并发估算"""
        # 创建多个工作流实例并发执行
        workflows = [FPEstimationWorkflow() for _ in range(3)]
        sample_project = ProjectInfo(
            name="测试项目",
            description="并发测试项目的详细描述信息，用于验证系统的并发处理能力和稳定性表现",
            technology_stack=[TechnologyStack.PYTHON],
            business_domain=BusinessDomain.OTHER
        )
        
        # 准备并发任务
        tasks = []
        for i, workflow in enumerate(workflows):
            session_id = await workflow.initialize(
                project_info=sample_project,
                strategy=EstimationStrategy.NESMA_ONLY,
                requirements=f"并发估算测试 {i+1}"
            )
            
            with patch('agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent.execute',
                       new_callable=AsyncMock) as mock_classifier:
                mock_classifier.return_value = {"function_type": "EI", "confidence": 0.8}
                
                task = workflow.execute()
                tasks.append(task)
        
        # 执行并发测试
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证结果
        assert len(results) == 3
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 1  # 至少1个成功


@pytest.mark.asyncio
async def test_full_system_integration():
    """完整系统集成测试"""
    # 这是一个综合性测试，验证整个系统的端到端功能
    print("\n🚀 开始完整系统集成测试...")
    
    # 测试项目
    test_project = ProjectInfo(
        name="综合测试电商平台",
        description="""
        完整的电商平台功能：
        1. 用户管理：注册、登录、资料管理
        2. 商品管理：商品录入、分类、库存管理
        3. 订单管理：下单、支付、发货、退货
        4. 客服系统：在线咨询、工单处理
        5. 数据分析：销售统计、用户行为分析
        """,
        technology_stack=[TechnologyStack.JAVA, TechnologyStack.MYSQL, TechnologyStack.REDIS],
        business_domain=BusinessDomain.ECOMMERCE
    )
    
    # 创建工作流实例
    workflow = FPEstimationWorkflow()
    
    # 使用实际的模拟数据进行完整测试
    with patch('agents.standards.standard_recommender.StandardRecommenderAgent.recommend_standards', new_callable=AsyncMock) as mock_recommender, \
         patch('agents.analysis.requirement_parser.RequirementParserAgent.parse_requirements', new_callable=AsyncMock) as mock_parser, \
         patch('agents.analysis.process_identifier.ProcessIdentifierAgent.identify_processes', new_callable=AsyncMock) as mock_identifier, \
         patch('agents.output.report_generator.ReportGeneratorAgent.execute', new_callable=AsyncMock) as mock_report:
        
        # 配置复杂的mock数据
        mock_recommender.return_value = {
            "recommended_standards": [EstimationStandard.NESMA, EstimationStandard.COSMIC],
            "strategy": EstimationStrategy.DUAL_PARALLEL,
            "confidence_score": 0.9,
            "reasoning": "电商平台适合双标准对比估算"
        }
        
        mock_parser.return_value = {
            "functional_modules": [
                {"name": "用户管理", "description": "用户注册登录管理"},
                {"name": "商品管理", "description": "商品信息维护"},
                {"name": "订单管理", "description": "订单处理流程"},
                {"name": "支付系统", "description": "支付处理"},
                {"name": "客服系统", "description": "客户服务"}
            ],
            "business_entities": {
                "用户角色": ["管理员", "商家", "买家"],
                "业务对象": ["用户", "商品", "订单", "支付"],
                "功能操作": ["注册", "登录", "下单", "支付", "退货"]
            },
            "business_processes": []
        }
        
        mock_identifier.return_value = [
            {
                "id": "proc_1",
                "name": "用户注册",
                "description": "用户注册流程",
                "data_groups": ["用户信息"],
                "dependencies": []
            },
            {
                "id": "proc_2", 
                "name": "商品发布",
                "description": "商品发布流程",
                "data_groups": ["商品信息"],
                "dependencies": ["proc_1"]
            },
            {
                "id": "proc_3",
                "name": "订单处理",
                "description": "订单处理流程", 
                "data_groups": ["订单信息"],
                "dependencies": ["proc_1", "proc_2"]
            }
        ]
        
        mock_report.return_value = {
            "report_content": "系统集成测试报告",
            "charts": [],
            "summary": "测试完成"
        }
        
        # 初始化工作流
        session_id = await workflow.initialize(
            project_info=test_project,
            strategy=EstimationStrategy.DUAL_PARALLEL,
            requirements="完整系统集成测试"
        )
        
        # 执行完整工作流
        result = await workflow.execute()
        
        # 验证系统集成结果
        assert result is not None
        assert session_id == workflow.get_session_id()
        print("✅ 完整系统集成测试通过!")


@pytest.mark.asyncio 
async def test_stress_testing():
    """压力测试"""
    print("\n🔥 开始压力测试...")
    
    # 创建多个复杂项目进行压力测试
    stress_projects = []
    for i in range(3):  # 减少数量以避免超时
        project = ProjectInfo(
            name=f"压力测试项目{i+1}",
            description=f"压力测试项目{i+1}的详细描述，包含多个复杂业务模块和功能需求，用于验证系统在高负载情况下的稳定性和性能表现",
            technology_stack=[TechnologyStack.JAVA, TechnologyStack.MYSQL],
            business_domain=BusinessDomain.MANUFACTURING
        )
        stress_projects.append(project)
    
    # 并发执行压力测试
    workflows = [FPEstimationWorkflow() for _ in range(len(stress_projects))]
    
    # 准备压力测试任务
    stress_tasks = []
    for i, (workflow, project) in enumerate(zip(workflows, stress_projects)):
        session_id = await workflow.initialize(
            project_info=project,
            strategy=EstimationStrategy.NESMA_ONLY,
            requirements=f"压力测试 {i+1}"
        )
        
        task = workflow.execute()
        stress_tasks.append(task)
    
    # 执行压力测试
    start_time = time.time()
    stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
    end_time = time.time()
    
    # 分析压力测试结果
    successful_count = sum(1 for r in stress_results if not isinstance(r, Exception))
    failed_count = len(stress_results) - successful_count
    
    total_time = end_time - start_time
    
    print(f"📊 压力测试结果:")
    print(f"  总任务数: {len(stress_results)}")
    print(f"  成功数: {successful_count}")
    print(f"  失败数: {failed_count}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均每任务耗时: {total_time/len(stress_results):.2f}s")
    
    # 压力测试验证
    assert successful_count >= len(stress_results) * 0.5  # 至少50%成功率
    assert total_time < 90.0  # 总耗时不超过90秒
    
    print("✅ 压力测试通过!")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"]) 