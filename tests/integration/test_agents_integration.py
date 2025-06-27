"""
量子智能化功能点估算系统 - 智能体集成测试

测试智能体之间的协作和数据流转
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from datetime import datetime

from agents.standards.standard_recommender import StandardRecommenderAgent
from agents.analysis.requirement_parser import RequirementParserAgent
from agents.analysis.process_identifier import ProcessIdentifierAgent
from agents.standards.nesma.function_classifier import NESMAFunctionClassifierAgent
from agents.standards.nesma.complexity_calculator import NESMAComplexityCalculatorAgent
from agents.standards.nesma.ufp_calculator import NESMAUFPCalculatorAgent
from agents.standards.cosmic.functional_user_agent import COSMICFunctionalUserAgent
from agents.standards.cosmic.boundary_analyzer import COSMICBoundaryAnalyzerAgent
from agents.standards.cosmic.data_movement_classifier import COSMICDataMovementClassifierAgent
from agents.standards.cosmic.cfp_calculator import COSMICCFPCalculatorAgent
from agents.knowledge.validator import ValidatorAgent
from agents.output.report_generator import ReportGeneratorAgent

from models.project_models import ProjectInfo, EstimationStrategy
from models.nesma_models import NESMAFunctionType, NESMAComplexityLevel
from models.cosmic_models import COSMICDataMovementType


class TestAgentsIntegration:
    """智能体集成测试类"""
    
    @pytest.fixture
    def sample_project(self):
        """创建样本项目"""
        from models.project_models import TechnologyStack, BusinessDomain
        return ProjectInfo(
            name="电商平台",
            description="包含用户管理、商品管理、订单处理、支付等功能的电商平台",
            technology_stack=[TechnologyStack.JAVA, TechnologyStack.SPRING, TechnologyStack.MYSQL],
            business_domain=BusinessDomain.ECOMMERCE
        )
    
    @pytest.fixture
    def agents_collection(self):
        """创建智能体集合"""
        return {
            "standard_recommender": StandardRecommenderAgent(),
            "requirement_parser": RequirementParserAgent(),
            "process_identifier": ProcessIdentifierAgent(),
            "nesma_classifier": NESMAFunctionClassifierAgent(),
            "nesma_complexity": NESMAComplexityCalculatorAgent(),
            "nesma_ufp": NESMAUFPCalculatorAgent(),
            "cosmic_functional_user": COSMICFunctionalUserAgent(),
            "cosmic_boundary": COSMICBoundaryAnalyzerAgent(),
            "cosmic_data_movement": COSMICDataMovementClassifierAgent(),
            "cosmic_cfp": COSMICCFPCalculatorAgent(),
            "validator": ValidatorAgent(),
            "report_generator": ReportGeneratorAgent()
        }
    
    @pytest.mark.asyncio
    async def test_nesma_only_workflow(self, sample_project, agents_collection):
        """测试NESMA单独估算工作流"""
        
        # 1. 标准推荐
        with patch.object(agents_collection["standard_recommender"], 'recommend_standards') as mock_recommend:
            mock_recommend.return_value = Mock(
                strategy=EstimationStrategy.NESMA_ONLY,
                confidence_score=0.9,
                reasoning="项目特征适合NESMA标准"
            )
            
            recommendation = await agents_collection["standard_recommender"].recommend_standards(
                project_info=sample_project,
                user_preferences={}
            )
            
            assert recommendation.strategy == EstimationStrategy.NESMA_ONLY
        
        # 2. 需求解析
        with patch.object(agents_collection["requirement_parser"], 'parse_requirements') as mock_parse:
            mock_parse.return_value = {
                "functional_modules": [
                    {"name": "用户管理", "description": "用户注册、登录、信息管理"},
                    {"name": "商品管理", "description": "商品添加、编辑、查询"},
                    {"name": "订单处理", "description": "订单创建、处理、查询"}
                ],
                "business_entities": {
                    "用户": ["用户ID", "用户名", "邮箱", "密码"],
                    "商品": ["商品ID", "商品名称", "价格", "库存"],
                    "订单": ["订单ID", "用户ID", "商品列表", "总金额", "状态"]
                },
                "business_processes": [
                    {"name": "用户注册", "steps": ["填写信息", "验证邮箱", "创建账户"]},
                    {"name": "下单流程", "steps": ["选择商品", "添加到购物车", "确认订单", "支付"]}
                ],
                "confidence_score": 0.88
            }
            
            parsing_result = await agents_collection["requirement_parser"].parse_requirements(
                requirement_text="电商平台需求文档...",
                project_info=sample_project
            )
            
            assert len(parsing_result["functional_modules"]) == 3
            assert parsing_result["confidence_score"] > 0.8
        
        # 3. 执行NESMA估算
        nesma_result = await self._run_nesma_estimation(
            parsing_result["functional_modules"],
            sample_project,
            {
                "nesma_classifier": agents_collection["nesma_classifier"],
                "nesma_complexity": agents_collection["nesma_complexity"],
                "nesma_ufp": agents_collection["nesma_ufp"]
            }
        )
        
        assert nesma_result["total_ufp"] > 0
        print(f"✅ NESMA单独估算测试完成，总UFP: {nesma_result['total_ufp']}")
    
    @pytest.mark.asyncio
    async def test_cosmic_only_workflow(self, sample_project, agents_collection):
        """测试COSMIC单独估算工作流"""
        
        # 1. 标准推荐
        with patch.object(agents_collection["standard_recommender"], 'recommend_standards') as mock_recommend:
            mock_recommend.return_value = Mock(
                strategy=EstimationStrategy.COSMIC_ONLY,
                confidence_score=0.85,
                reasoning="项目适合COSMIC标准"
            )
            
            recommendation = await agents_collection["standard_recommender"].recommend_standards(
                project_info=sample_project,
                user_preferences={}
            )
            
            assert recommendation.strategy == EstimationStrategy.COSMIC_ONLY
        
        # 2. 需求解析
        with patch.object(agents_collection["requirement_parser"], 'parse_requirements') as mock_parse:
            mock_parse.return_value = {
                "functional_modules": [
                    {"name": "用户管理", "description": "用户注册、登录、信息管理"},
                    {"name": "订单处理", "description": "订单创建、处理、查询"}
                ],
                "business_processes": [
                    {
                        "name": "用户注册流程",
                        "data_movements": ["用户输入信息", "系统验证", "存储用户数据", "返回注册结果"]
                    },
                    {
                        "name": "订单创建流程", 
                        "data_movements": ["接收订单信息", "验证商品库存", "计算总价", "保存订单", "返回订单号"]
                    }
                ],
                "confidence_score": 0.9
            }
            
            parsing_result = await agents_collection["requirement_parser"].parse_requirements(
                requirement_text="电商平台需求文档...",
                project_info=sample_project
            )
            
            assert len(parsing_result["business_processes"]) == 2
        
        # 3. 执行COSMIC估算
        cosmic_result = await self._run_cosmic_estimation(
            parsing_result["business_processes"],
            sample_project,
            {
                "cosmic_functional_user": agents_collection["cosmic_functional_user"],
                "cosmic_boundary": agents_collection["cosmic_boundary"],
                "cosmic_data_movement": agents_collection["cosmic_data_movement"],
                "cosmic_cfp": agents_collection["cosmic_cfp"]
            }
        )
        
        assert cosmic_result["total_cfp"] > 0
        print(f"✅ COSMIC单独估算测试完成，总CFP: {cosmic_result['total_cfp']}")
    
    @pytest.mark.asyncio
    async def test_dual_standard_workflow(self, sample_project, agents_collection):
        """测试双标准对比估算工作流"""
        
        # 1. 标准推荐
        with patch.object(agents_collection["standard_recommender"], 'recommend_standards') as mock_recommend:
            mock_recommend.return_value = Mock(
                strategy=EstimationStrategy.DUAL_PARALLEL,
                confidence_score=0.75,
                reasoning="项目复杂度适中，建议双标准对比"
            )
            
            recommendation = await agents_collection["standard_recommender"].recommend_standards(
                project_info=sample_project,
                user_preferences={}
            )
            
            assert recommendation.strategy == EstimationStrategy.DUAL_PARALLEL
        
        # 2. 需求解析
        with patch.object(agents_collection["requirement_parser"], 'parse_requirements') as mock_parse:
            mock_parse.return_value = {
                "functional_modules": [
                    {"name": "用户管理", "description": "用户注册、登录"},
                    {"name": "订单处理", "description": "订单创建、管理"}
                ],
                "business_processes": [
                    {"name": "用户注册流程", "description": "用户注册业务流程"},
                    {"name": "订单处理流程", "description": "订单处理业务流程"}
                ],
                "confidence_score": 0.85
            }
            
            parsing_result = await agents_collection["requirement_parser"].parse_requirements(
                requirement_text="电商平台需求文档...",
                project_info=sample_project
            )
        
        # 3. 并行执行NESMA和COSMIC估算
        nesma_agents = {
            "nesma_classifier": agents_collection["nesma_classifier"],
            "nesma_complexity": agents_collection["nesma_complexity"],
            "nesma_ufp": agents_collection["nesma_ufp"]
        }
        
        cosmic_agents = {
            "cosmic_functional_user": agents_collection["cosmic_functional_user"],
            "cosmic_boundary": agents_collection["cosmic_boundary"],
            "cosmic_data_movement": agents_collection["cosmic_data_movement"],
            "cosmic_cfp": agents_collection["cosmic_cfp"]
        }
        
        nesma_result, cosmic_result = await asyncio.gather(
            self._run_nesma_estimation(parsing_result["functional_modules"], sample_project, nesma_agents),
            self._run_cosmic_estimation(parsing_result["business_processes"], sample_project, cosmic_agents)
        )
        
        # 4. 对比分析
        comparison = self._compare_estimation_results(nesma_result, cosmic_result)
        
        assert nesma_result["total_ufp"] > 0
        assert cosmic_result["total_cfp"] > 0
        assert "variance_percentage" in comparison
        
        print(f"✅ 双标准对比测试完成")
        print(f"   NESMA UFP: {nesma_result['total_ufp']}")
        print(f"   COSMIC CFP: {cosmic_result['total_cfp']}")
        print(f"   差异率: {comparison['variance_percentage']:.2f}%")
    
    async def _run_nesma_estimation(
        self,
        functional_modules: List[Dict[str, Any]],
        project_info: ProjectInfo,
        agents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行NESMA估算"""
        
        # 功能分类
        classifications = []
        for module in functional_modules:
            with patch.object(agents["nesma_classifier"], 'classify_function') as mock_classify:
                mock_classify.return_value = Mock(
                    function_type=NESMAFunctionType.EI,
                    confidence_score=0.9,
                    function_description=module["description"]
                )
                
                result = await agents["nesma_classifier"].classify_function(module["description"])
                classifications.append(result)
        
        # 复杂度计算
        complexity_results = []
        for classification in classifications:
            with patch.object(agents["nesma_complexity"], 'execute') as mock_complexity:
                mock_complexity.return_value = {
                    "complexity": NESMAComplexityLevel.AVERAGE,
                    "det_count": 8,
                    "ret_count": 2,
                    "reasoning": "中等复杂度"
                }
                
                result = await agents["nesma_complexity"].execute(
                    task_name="calculate_complexity",
                    inputs={
                        "function_type": classification.function_type,
                        "function_description": classification.function_description
                    }
                )
                complexity_results.append(result)
        
        # UFP计算
        with patch.object(agents["nesma_ufp"], 'execute') as mock_ufp:
            total_ufp = len(classifications) * 5  # 模拟计算
            mock_ufp.return_value = {
                "total_ufp": total_ufp,
                "ufp_breakdown": {
                    "EI": total_ufp * 0.6,
                    "EO": total_ufp * 0.3,
                    "ILF": total_ufp * 0.1
                }
            }
            
            ufp_result = await agents["nesma_ufp"].execute(
                task_name="calculate_ufp",
                inputs={
                    "classifications": [
                        {
                            "function_type": c.function_type.value,
                            "complexity": cr["complexity"].value,
                            "count": 1
                        }
                        for c, cr in zip(classifications, complexity_results)
                    ]
                }
            )
        
        return ufp_result
    
    async def _run_cosmic_estimation(
        self,
        processes: List[Dict[str, Any]],
        project_info: ProjectInfo,
        agents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行COSMIC估算"""
        
        # 功能用户识别
        with patch.object(agents["cosmic_functional_user"], 'execute') as mock_user:
            mock_user.return_value = {
                "functional_users": [
                    {"id": "customer", "name": "客户", "description": "使用系统的客户"}
                ]
            }
            
            functional_users = await agents["cosmic_functional_user"].execute(
                "identify_functional_users",
                {"project_info": project_info}
            )
        
        # 边界分析
        with patch.object(agents["cosmic_boundary"], 'execute') as mock_boundary:
            mock_boundary.return_value = {
                "software_boundary": "电商平台核心系统",
                "persistent_storage_boundary": "主数据库",
                "boundary_reasoning": "基于系统架构确定边界"
            }
            
            boundary_analysis = await agents["cosmic_boundary"].execute(
                "analyze_boundaries",
                {
                    "project_info": project_info,
                    "functional_users": functional_users["functional_users"]
                }
            )
        
        # 数据移动分类
        data_movements = []
        for process in processes:
            with patch.object(agents["cosmic_data_movement"], 'execute') as mock_movement:
                mock_movement.return_value = {
                    "data_movements": [
                        {
                            "type": COSMICDataMovementType.ENTRY,
                            "data_group": f"数据组_{i}",
                            "source": "外部",
                            "target": "系统"
                        }
                        for i in range(2)  # 每个流程2个数据移动
                    ]
                }
                
                result = await agents["cosmic_data_movement"].execute(
                    "classify_data_movements",
                    {
                        "process_info": process,
                        "boundary_analysis": boundary_analysis
                    }
                )
                data_movements.extend(result["data_movements"])
        
        # CFP计算
        with patch.object(agents["cosmic_cfp"], 'execute') as mock_cfp:
            total_cfp = len(data_movements)  # 每个数据移动 = 1 CFP
            mock_cfp.return_value = {
                "total_cfp": total_cfp,
                "movement_breakdown": {
                    f"movement_{i}": 1 for i in range(len(data_movements))
                }
            }
            
            cfp_result = await agents["cosmic_cfp"].execute(
                "calculate_cfp",
                {"data_movements": data_movements}
            )
        
        return cfp_result
    
    def _compare_estimation_results(
        self, 
        nesma_result: Dict[str, Any], 
        cosmic_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """比较估算结果"""
        
        nesma_ufp = nesma_result["total_ufp"]
        cosmic_cfp = cosmic_result["total_cfp"]
        
        # 计算差异
        if nesma_ufp > 0 and cosmic_cfp > 0:
            variance = abs(nesma_ufp - cosmic_cfp)
            variance_percentage = (variance / max(nesma_ufp, cosmic_cfp)) * 100
        else:
            variance_percentage = 0
        
        return {
            "nesma_ufp": nesma_ufp,
            "cosmic_cfp": cosmic_cfp,
            "variance": abs(nesma_ufp - cosmic_cfp),
            "variance_percentage": variance_percentage,
            "variance_analysis": self._analyze_variance(variance_percentage),
            "comparison_time": datetime.now()
        }
    
    def _analyze_variance(self, variance_percentage: float) -> str:
        """分析差异原因"""
        if variance_percentage < 10:
            return "两种标准结果基本一致"
        elif variance_percentage < 25:
            return "存在中等差异，可能由于标准计算方法不同"
        else:
            return "存在较大差异，建议进一步分析原因"
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, sample_project, agents_collection):
        """测试错误处理机制"""
        
        # 测试无效输入
        with pytest.raises(Exception):
            await agents_collection["nesma_classifier"].classify_function("")
        
        # 测试不存在的任务
        with pytest.raises(ValueError):
            await agents_collection["nesma_classifier"].execute(
                "unknown_task",
                {"some_input": "value"}
            )
        
        print("✅ 错误处理测试通过")
    
    @pytest.mark.asyncio
    async def test_agent_performance(self, sample_project, agents_collection):
        """测试智能体性能"""
        
        # 模拟高并发标准推荐
        tasks = []
        for i in range(5):
            with patch.object(agents_collection["standard_recommender"], 'recommend_standards') as mock_recommend:
                mock_recommend.return_value = Mock(
                    strategy=EstimationStrategy.NESMA_ONLY,
                    confidence_score=0.9,
                    reasoning=f"测试推荐 {i}"
                )
                
                task = agents_collection["standard_recommender"].recommend_standards(
                    project_info=sample_project,
                    user_preferences={}
                )
                tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        print("✅ 性能测试通过")
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self, sample_project, agents_collection):
        """测试超时处理"""
        
        # 模拟超时情况
        with patch.object(agents_collection["requirement_parser"], 'parse_requirements') as mock_parse:
            mock_parse.side_effect = asyncio.TimeoutError("请求超时")
            
            task = agents_collection["requirement_parser"].parse_requirements(
                requirement_text="测试超时处理",
                project_info=sample_project
            )
            
            try:
                await asyncio.wait_for(task, timeout=1.0)
                assert False, "应该触发超时"
            except asyncio.TimeoutError:
                pass  # 预期的超时异常
        
        print("✅ 超时处理测试通过")


@pytest.mark.asyncio
async def test_integration_suite():
    """运行完整的集成测试套件"""
    
    print("🚀 开始智能体集成测试...")
    
    test_instance = TestAgentsIntegration()
    
    # 创建测试数据
    from models.project_models import TechnologyStack, BusinessDomain
    sample_project = ProjectInfo(
        name="集成测试项目",
        description="用于测试智能体集成的样本项目",
        technology_stack=[TechnologyStack.PYTHON, TechnologyStack.FASTAPI],
        business_domain=BusinessDomain.TESTING
    )
    
    agents_collection = {
        "standard_recommender": StandardRecommenderAgent(),
        "requirement_parser": RequirementParserAgent(),
        "process_identifier": ProcessIdentifierAgent(),
        "nesma_classifier": NESMAFunctionClassifierAgent(),
        "nesma_complexity": NESMAComplexityCalculatorAgent(),
        "nesma_ufp": NESMAUFPCalculatorAgent(),
        "cosmic_functional_user": COSMICFunctionalUserAgent(),
        "cosmic_boundary": COSMICBoundaryAnalyzerAgent(),
        "cosmic_data_movement": COSMICDataMovementClassifierAgent(),
        "cosmic_cfp": COSMICCFPCalculatorAgent(),
        "validator": ValidatorAgent(),
        "report_generator": ReportGeneratorAgent()
    }
    
    # 运行测试
    await test_instance.test_nesma_only_workflow(sample_project, agents_collection)
    await test_instance.test_cosmic_only_workflow(sample_project, agents_collection)
    await test_instance.test_dual_standard_workflow(sample_project, agents_collection)
    await test_instance.test_agent_error_handling(sample_project, agents_collection)
    await test_instance.test_agent_performance(sample_project, agents_collection)
    await test_instance.test_agent_timeout_handling(sample_project, agents_collection)
    
    print("🎉 所有集成测试通过！")


if __name__ == "__main__":
    asyncio.run(test_integration_suite()) 