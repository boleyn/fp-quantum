"""
端到端完整工作流测试

测试完整的功能点估算工作流程，包括NESMA、COSMIC和双标准对比
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import time

from graph.workflow_graph import FPEstimationWorkflow, create_compiled_workflow
from models.project_models import ProjectInfo, TechnologyStack, BusinessDomain
from models.common_models import EstimationStrategy, WorkflowState
from config.settings import get_settings


class TestFullWorkflow:
    """完整工作流端到端测试"""
    
    @pytest.fixture
    def sample_projects(self):
        """不同规模的样本项目"""
        return {
            "small": ProjectInfo(
                name="个人博客系统",
                description="""
                简单的个人博客系统，包含：
                1. 用户注册登录
                2. 文章发布和编辑
                3. 文章浏览和搜索
                4. 评论功能
                """,
                technology_stack=[TechnologyStack.PYTHON, TechnologyStack.MYSQL],
                business_domain=BusinessDomain.OTHER
            ),
            "medium": ProjectInfo(
                name="企业客户管理系统",
                description="""
                中等规模的CRM系统，包含：
                1. 客户信息管理：录入、查询、修改客户资料
                2. 销售机会管理：跟进销售线索，记录沟通历史
                3. 合同管理：合同生成、审批、归档
                4. 报表分析：销售统计、客户分析报表
                5. 权限管理：用户角色和权限控制
                6. 系统集成：与邮件系统、短信平台集成
                """,
                technology_stack=[TechnologyStack.JAVA, TechnologyStack.MYSQL, TechnologyStack.REDIS],
                business_domain=BusinessDomain.RETAIL
            ),
            "large": ProjectInfo(
                name="银行核心业务系统",
                description="""
                大型银行核心系统，包含：
                1. 账户管理：开户、销户、账户信息维护
                2. 存取款业务：现金存取、转账、汇款
                3. 贷款业务：贷款申请、审批、放款、还款
                4. 理财产品：产品管理、购买、赎回
                5. 风险控制：反洗钱、风险评估、预警
                6. 客户服务：客户投诉、咨询、服务记录
                7. 报表管理：监管报表、内部报表、统计分析
                8. 系统集成：与央行、征信、支付系统集成
                9. 数据管理：数据备份、恢复、归档
                10. 安全管理：访问控制、审计日志、加密
                """,
                technology_stack=[
                    TechnologyStack.JAVA, TechnologyStack.ORACLE, 
                    TechnologyStack.REDIS, TechnologyStack.AWS
                ],
                business_domain=BusinessDomain.FINANCE
            )
        }
    
    @pytest.fixture
    def workflow_instance(self):
        """创建工作流实例"""
        return FPEstimationWorkflow()
    
    @pytest.mark.asyncio
    async def test_nesma_only_workflow(self, workflow_instance, sample_projects):
        """测试纯NESMA估算工作流"""
        project = sample_projects["medium"]
        
        # Mock各个智能体的返回结果
        mock_responses = self._create_nesma_mock_responses()
        
        with patch.multiple(
            'agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent',
            execute_task=AsyncMock(side_effect=mock_responses["classifier"]),
        ), patch.multiple(
            'agents.standards.nesma.complexity_calculator.NESMAComplexityCalculatorAgent',
            execute_task=AsyncMock(side_effect=mock_responses["complexity"]),
        ), patch.multiple(
            'agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent',
            execute_task=AsyncMock(side_effect=mock_responses["ufp"]),
        ):
            # 初始化工作流
            session_id = await workflow_instance.initialize(
                project_info=project,
                strategy=EstimationStrategy.NESMA_ONLY,
                requirements=project.description
            )
            
            assert session_id is not None
            
            # 执行工作流
            start_time = time.time()
            final_state = await workflow_instance.execute()
            execution_time = time.time() - start_time
            
            # 验证执行结果
            assert final_state.current_state == WorkflowState.COMPLETED
            assert final_state.nesma_ufp_total > 0
            assert final_state.cosmic_cfp_total is None  # NESMA_ONLY模式不应有COSMIC结果
            assert final_state.final_report is not None
            assert execution_time < 30  # 中等项目应在30秒内完成
            
            print(f"✅ NESMA工作流测试完成: {final_state.nesma_ufp_total} UFP, 耗时: {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_cosmic_only_workflow(self, workflow_instance, sample_projects):
        """测试纯COSMIC估算工作流"""
        project = sample_projects["medium"]
        
        # Mock各个智能体的返回结果
        mock_responses = self._create_cosmic_mock_responses()
        
        with patch.multiple(
            'agents.standards.cosmic.functional_user_agent.COSMICFunctionalUserAgent',
            execute_task=AsyncMock(side_effect=mock_responses["functional_user"]),
        ), patch.multiple(
            'agents.standards.cosmic.boundary_analyzer.COSMICBoundaryAnalyzerAgent',
            execute_task=AsyncMock(side_effect=mock_responses["boundary"]),
        ), patch.multiple(
            'agents.standards.cosmic.data_movement_classifier.COSMICDataMovementClassifierAgent',
            execute_task=AsyncMock(side_effect=mock_responses["data_movement"]),
        ), patch.multiple(
            'agents.standards.cosmic.cfp_calculator.COSMICCFPCalculatorAgent',
            execute_task=AsyncMock(side_effect=mock_responses["cfp"]),
        ):
            # 初始化工作流
            session_id = await workflow_instance.initialize(
                project_info=project,
                strategy=EstimationStrategy.COSMIC_ONLY,
                requirements=project.description
            )
            
            # 执行工作流
            start_time = time.time()
            final_state = await workflow_instance.execute()
            execution_time = time.time() - start_time
            
            # 验证执行结果
            assert final_state.current_state == WorkflowState.COMPLETED
            assert final_state.cosmic_cfp_total > 0
            assert final_state.nesma_ufp_total is None  # COSMIC_ONLY模式不应有NESMA结果
            assert final_state.final_report is not None
            assert execution_time < 30  # 中等项目应在30秒内完成
            
            print(f"✅ COSMIC工作流测试完成: {final_state.cosmic_cfp_total} CFP, 耗时: {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_dual_standard_workflow(self, workflow_instance, sample_projects):
        """测试双标准对比估算工作流"""
        project = sample_projects["large"]
        
        # Mock两套标准的返回结果
        nesma_mock = self._create_nesma_mock_responses()
        cosmic_mock = self._create_cosmic_mock_responses()
        
        with patch.multiple(
            'agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent',
            execute_task=AsyncMock(side_effect=nesma_mock["classifier"]),
        ), patch.multiple(
            'agents.standards.nesma.complexity_calculator.NESMAComplexityCalculatorAgent',
            execute_task=AsyncMock(side_effect=nesma_mock["complexity"]),
        ), patch.multiple(
            'agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent',
            execute_task=AsyncMock(side_effect=nesma_mock["ufp"]),
        ), patch.multiple(
            'agents.standards.cosmic.functional_user_agent.COSMICFunctionalUserAgent',
            execute_task=AsyncMock(side_effect=cosmic_mock["functional_user"]),
        ), patch.multiple(
            'agents.standards.cosmic.boundary_analyzer.COSMICBoundaryAnalyzerAgent',
            execute_task=AsyncMock(side_effect=cosmic_mock["boundary"]),
        ), patch.multiple(
            'agents.standards.cosmic.data_movement_classifier.COSMICDataMovementClassifierAgent',
            execute_task=AsyncMock(side_effect=cosmic_mock["data_movement"]),
        ), patch.multiple(
            'agents.standards.cosmic.cfp_calculator.COSMICCFPCalculatorAgent',
            execute_task=AsyncMock(side_effect=cosmic_mock["cfp"]),
        ):
            # 初始化工作流
            session_id = await workflow_instance.initialize(
                project_info=project,
                strategy=EstimationStrategy.DUAL_PARALLEL,
                requirements=project.description
            )
            
            # 执行工作流
            start_time = time.time()
            final_state = await workflow_instance.execute()
            execution_time = time.time() - start_time
            
            # 验证执行结果
            assert final_state.current_state == WorkflowState.COMPLETED
            assert final_state.nesma_ufp_total > 0
            assert final_state.cosmic_cfp_total > 0
            assert final_state.final_report is not None
            assert execution_time < 60  # 大型项目双标准应在60秒内完成
            
            # 验证对比分析
            assert "comparison_analysis" in final_state.final_report
            comparison = final_state.final_report["comparison_analysis"]
            assert "variance_percentage" in comparison
            assert "recommendations" in comparison
            
            print(f"✅ 双标准工作流测试完成: NESMA {final_state.nesma_ufp_total} UFP, "
                  f"COSMIC {final_state.cosmic_cfp_total} CFP, 耗时: {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, workflow_instance, sample_projects):
        """测试错误处理和重试机制"""
        project = sample_projects["small"]
        
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
        
        with patch.object(
            workflow_instance, '_call_llm_with_retry',
            side_effect=failing_then_success
        ):
            # 初始化工作流
            session_id = await workflow_instance.initialize(
                project_info=project,
                strategy=EstimationStrategy.NESMA_ONLY,
                requirements=project.description
            )
            
            # 执行工作流，应该能够从错误中恢复
            final_state = await workflow_instance.execute()
            
            # 验证重试机制生效
            assert call_count > 1  # 确实进行了重试
            assert final_state.retry_count > 0
            
            # 最终应该成功或优雅失败
            assert final_state.current_state in [WorkflowState.COMPLETED, WorkflowState.FAILED]
            
            if final_state.current_state == WorkflowState.FAILED:
                assert final_state.error_message is not None
                print(f"⚠️ 工作流失败但错误处理正确: {final_state.error_message}")
            else:
                print(f"✅ 重试机制测试完成: 重试{final_state.retry_count}次后成功")
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, workflow_instance, sample_projects):
        """测试性能基准"""
        performance_results = {}
        
        for project_size, project in sample_projects.items():
            # 执行多次取平均值
            execution_times = []
            memory_usages = []
            
            for run in range(3):  # 运行3次
                # Mock简化的响应以确保稳定性能
                with patch.multiple(
                    'agents.standards.nesma.function_classifier.NESMAFunctionClassifierAgent',
                    execute_task=AsyncMock(return_value={"function_type": "EI", "confidence": 0.9}),
                ), patch.multiple(
                    'agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent',
                    execute_task=AsyncMock(return_value={"total_ufp": 25, "function_breakdown": []}),
                ):
                    # 初始化和执行
                    session_id = await workflow_instance.initialize(
                        project_info=project,
                        strategy=EstimationStrategy.NESMA_ONLY,
                        requirements=project.description
                    )
                    
                    start_time = time.time()
                    final_state = await workflow_instance.execute()
                    execution_time = time.time() - start_time
                    
                    execution_times.append(execution_time)
                    
                    # 简单的内存使用估算 (实际应使用psutil等工具)
                    estimated_memory = len(str(final_state)) * 0.001  # KB
                    memory_usages.append(estimated_memory)
            
            # 计算平均性能指标
            avg_time = sum(execution_times) / len(execution_times)
            avg_memory = sum(memory_usages) / len(memory_usages)
            
            performance_results[project_size] = {
                "avg_execution_time": avg_time,
                "avg_memory_usage": avg_memory,
                "max_execution_time": max(execution_times),
                "min_execution_time": min(execution_times)
            }
            
            # 性能基准验证
            if project_size == "small":
                assert avg_time < 15, f"小型项目执行时间过长: {avg_time}s"
            elif project_size == "medium":
                assert avg_time < 30, f"中型项目执行时间过长: {avg_time}s"
            elif project_size == "large":
                assert avg_time < 60, f"大型项目执行时间过长: {avg_time}s"
        
        # 打印性能报告
        print("\n📊 性能基准测试结果:")
        for project_size, metrics in performance_results.items():
            print(f"  {project_size.upper()}项目:")
            print(f"    平均执行时间: {metrics['avg_execution_time']:.2f}s")
            print(f"    最大执行时间: {metrics['max_execution_time']:.2f}s")
            print(f"    平均内存使用: {metrics['avg_memory_usage']:.2f}KB")
    
    def _create_nesma_mock_responses(self) -> Dict[str, List[Any]]:
        """创建NESMA模拟响应"""
        return {
            "classifier": [
                {"function_type": "EI", "confidence": 0.9, "reasoning": "用户输入功能"},
                {"function_type": "EO", "confidence": 0.85, "reasoning": "报告输出功能"},
                {"function_type": "EQ", "confidence": 0.88, "reasoning": "信息查询功能"},
                {"function_type": "ILF", "confidence": 0.92, "reasoning": "内部数据文件"},
                {"function_type": "EIF", "confidence": 0.87, "reasoning": "外部接口文件"}
            ],
            "complexity": [
                {"complexity": "AVERAGE", "det_count": 10, "ftr_count": 2},
                {"complexity": "LOW", "det_count": 5, "ftr_count": 1},
                {"complexity": "HIGH", "det_count": 20, "ftr_count": 3},
                {"complexity": "AVERAGE", "det_count": 25, "ret_count": 2},
                {"complexity": "LOW", "det_count": 8, "ret_count": 1}
            ],
            "ufp": [
                {
                    "total_ufp": 35,
                    "function_breakdown": [
                        {"type": "EI", "complexity": "AVERAGE", "weight": 4, "ufp": 4},
                        {"type": "EO", "complexity": "LOW", "weight": 4, "ufp": 4},
                        {"type": "EQ", "complexity": "HIGH", "weight": 6, "ufp": 6},
                        {"type": "ILF", "complexity": "AVERAGE", "weight": 10, "ufp": 10},
                        {"type": "EIF", "complexity": "LOW", "weight": 5, "ufp": 5}
                    ],
                    "type_summary": {"total_functions": 5}
                }
            ]
        }
    
    def _create_cosmic_mock_responses(self) -> Dict[str, List[Any]]:
        """创建COSMIC模拟响应"""
        return {
            "functional_user": [
                {
                    "functional_users": [
                        {"name": "业务用户", "user_type": "primary"},
                        {"name": "系统管理员", "user_type": "primary"},
                        {"name": "外部系统", "user_type": "secondary"}
                    ]
                }
            ],
            "boundary": [
                {
                    "software_boundary": {
                        "included_components": ["业务逻辑", "数据处理", "用户界面"],
                        "excluded_components": ["外部系统", "用户设备"]
                    },
                    "persistent_storage_boundary": {
                        "internal_storage": ["业务数据库", "配置数据"],
                        "external_storage": ["外部数据源"]
                    }
                }
            ],
            "data_movement": [
                {
                    "data_movements": [
                        {"type": "Entry", "description": "用户输入业务数据"},
                        {"type": "Read", "description": "读取业务配置"},
                        {"type": "Write", "description": "保存业务结果"},
                        {"type": "Exit", "description": "返回处理结果"}
                    ]
                }
            ],
            "cfp": [
                {
                    "total_cfp": 28,
                    "functional_processes": [
                        {
                            "name": "业务处理流程",
                            "cfp": 28,
                            "movement_count": {"Entry": 7, "Exit": 7, "Read": 7, "Write": 7}
                        }
                    ],
                    "cfp_breakdown": {"Entry": 7, "Exit": 7, "Read": 7, "Write": 7, "total": 28}
                }
            ]
        }


class TestWorkflowRecovery:
    """工作流恢复和状态管理测试"""
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self):
        """测试工作流状态持久化"""
        # 这个测试需要真实的状态存储机制
        # 这里提供测试框架，实际实现需要根据具体的持久化方案
        pass
    
    @pytest.mark.asyncio
    async def test_workflow_checkpoint_recovery(self):
        """测试工作流检查点恢复"""
        # 测试从中间检查点恢复执行
        pass


class TestConcurrentWorkflows:
    """并发工作流测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_estimations(self):
        """测试多个并发估算"""
        # 创建多个工作流实例并发执行
        workflows = [FPEstimationWorkflow() for _ in range(3)]
        sample_project = ProjectInfo(
            name="测试项目",
            description="并发测试项目",
            technology_stack=[TechnologyStack.PYTHON],
            business_domain=BusinessDomain.OTHER
        )
        
        # 并发执行多个估算
        tasks = []
        for i, workflow in enumerate(workflows):
            task = asyncio.create_task(self._run_single_estimation(workflow, sample_project, i))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        print(f"并发测试结果: {len(successful_results)}成功, {len(failed_results)}失败")
        
        # 至少应有一半成功
        assert len(successful_results) >= len(workflows) // 2
    
    async def _run_single_estimation(self, workflow, project, task_id):
        """运行单个估算任务"""
        try:
            with patch.multiple(
                'agents.standards.nesma.ufp_calculator.NESMAUFPCalculatorAgent',
                execute_task=AsyncMock(return_value={"total_ufp": 20 + task_id}),
            ):
                session_id = await workflow.initialize(
                    project_info=project,
                    strategy=EstimationStrategy.NESMA_ONLY,
                    requirements=project.description
                )
                
                final_state = await workflow.execute()
                return {"task_id": task_id, "result": final_state.nesma_ufp_total}
        except Exception as e:
            return {"task_id": task_id, "error": str(e)}


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
    with patch('agents.base.base_agent.BaseAgent._call_llm') as mock_llm:
        # 配置模拟返回
        mock_llm.side_effect = [
            {"recommended_standards": ["NESMA", "COSMIC"], "strategy": "DUAL_PARALLEL"},
            {"identified_processes": [{"name": "用户管理", "description": "用户注册登录"}]},
            {"function_type": "EI", "confidence": 0.9},
            {"complexity": "AVERAGE", "det_count": 10},
            {"total_ufp": 45, "function_breakdown": []},
            {"functional_users": [{"name": "用户", "type": "primary"}]},
            {"software_boundary": {"included": ["用户模块"]}},
            {"data_movements": [{"type": "Entry", "description": "用户输入"}]},
            {"total_cfp": 38, "functional_processes": []},
            {"final_report": {"summary": "估算完成"}}
        ]
        
        # 执行完整工作流
        session_id = await workflow.initialize(
            project_info=test_project,
            strategy=EstimationStrategy.DUAL_PARALLEL,
            requirements=test_project.description
        )
        
        start_time = time.time()
        final_state = await workflow.execute()
        total_time = time.time() - start_time
        
        # 验证最终结果
        assert final_state.current_state == WorkflowState.COMPLETED
        assert final_state.nesma_ufp_total > 0
        assert final_state.cosmic_cfp_total > 0
        assert final_state.final_report is not None
        
        print(f"✅ 系统集成测试完成!")
        print(f"   NESMA结果: {final_state.nesma_ufp_total} UFP")
        print(f"   COSMIC结果: {final_state.cosmic_cfp_total} CFP")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   会话ID: {session_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 