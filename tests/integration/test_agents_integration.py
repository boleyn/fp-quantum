"""
量子智能化功能点估算系统 - 智能体集成测试

测试智能体间的协作流程，验证端到端估算能力
"""

import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from agents.standards.standard_recommender import StandardRecommenderAgent
from agents.analysis.requirement_parser import RequirementParserAgent
from agents.standards.nesma.function_classifier import NESMAFunctionClassifierAgent
from agents.standards.nesma.complexity_calculator import NESMAComplexityCalculatorAgent
from agents.standards.nesma.ufp_calculator import NESMAUFPCalculatorAgent
from agents.standards.cosmic.functional_user_agent import COSMICFunctionalUserAgent
from agents.standards.cosmic.data_movement_classifier import COSMICDataMovementClassifierAgent
from agents.standards.cosmic.boundary_analyzer import COSMICBoundaryAnalyzerAgent
from agents.standards.cosmic.cfp_calculator import COSMICCFPCalculatorAgent
from agents.output.report_generator import ReportGeneratorAgent

from models.project_models import ProjectInfo, EstimationStrategy
from models.nesma_models import NESMAFunctionType, NESMAComplexityLevel
from models.cosmic_models import COSMICDataMovementType


class TestAgentsIntegration:
    """智能体集成测试"""
    
    @pytest.fixture
    async def sample_project(self):
        """样本项目信息"""
        return ProjectInfo(
            name="电商平台用户管理系统",
            description="""
            开发一个电商平台的用户管理系统，包含以下功能：
            1. 用户注册：用户输入姓名、邮箱、密码等信息进行注册
            2. 用户登录：用户使用邮箱和密码登录系统
            3. 用户资料管理：用户可以查看和修改个人资料
            4. 用户列表查询：管理员可以查询系统中的用户列表
            5. 用户状态管理：管理员可以启用或禁用用户账户
            """,
            technology_stack=["Java", "Spring Boot", "MySQL", "Redis"],
            business_domain="电商"
        )
    
    @pytest.fixture
    async def agents_collection(self):
        """智能体集合"""
        return {
            "standard_recommender": StandardRecommenderAgent(),
            "requirement_parser": RequirementParserAgent(),
            "nesma_classifier": NESMAFunctionClassifierAgent(),
            "nesma_complexity": NESMAComplexityCalculatorAgent(),
            "nesma_ufp": NESMAUFPCalculatorAgent(),
            "cosmic_functional_user": COSMICFunctionalUserAgent(),
            "cosmic_data_movement": COSMICDataMovementClassifierAgent(),
            "cosmic_boundary": COSMICBoundaryAnalyzerAgent(),
            "cosmic_cfp": COSMICCFPCalculatorAgent(),
            "report_generator": ReportGeneratorAgent()
        }
    
    @pytest.mark.asyncio
    async def test_full_nesma_estimation_workflow(self, sample_project, agents_collection):
        """测试完整的NESMA估算流程"""
        
        # 1. 标准推荐
        recommendation = await agents_collection["standard_recommender"].execute_task(
            "recommend_standard",
            {"project_info": sample_project}
        )
        
        assert recommendation["recommended_standards"]
        assert "NESMA" in [std for std in recommendation["recommended_standards"]]
        
        # 2. 需求解析
        processes = await agents_collection["requirement_parser"].execute_task(
            "parse_requirements",
            {"requirements_text": sample_project.description}
        )
        
        assert len(processes["identified_processes"]) > 0
        
        # 3. NESMA功能分类
        classification_results = []
        for process in processes["identified_processes"]:
            result = await agents_collection["nesma_classifier"].execute_task(
                "classify_function",
                {"function_description": process["description"]}
            )
            classification_results.append(result)
        
        assert len(classification_results) > 0
        assert all(result["function_type"] in [t.value for t in NESMAFunctionType] 
                  for result in classification_results)
        
        # 4. NESMA复杂度计算
        complexity_results = []
        for classification in classification_results:
            result = await agents_collection["nesma_complexity"].execute_task(
                "calculate_complexity",
                {
                    "function_type": classification["function_type"],
                    "function_description": classification["function_description"]
                }
            )
            complexity_results.append(result)
        
        assert len(complexity_results) > 0
        assert all(result["complexity"] in [c.value for c in NESMAComplexityLevel] 
                  for result in complexity_results)
        
        # 5. NESMA UFP计算
        ufp_result = await agents_collection["nesma_ufp"].execute_task(
            "calculate_ufp",
            {
                "classifications": classification_results,
                "complexity_results": complexity_results,
                "project_info": sample_project
            }
        )
        
        assert ufp_result["total_ufp"] > 0
        assert "function_breakdown" in ufp_result
        
        # 6. 生成报告
        report = await agents_collection["report_generator"].execute_task(
            "generate_report",
            {
                "estimation_type": "NESMA",
                "results": ufp_result,
                "project_info": sample_project
            }
        )
        
        assert report["report_type"] == "NESMA_Estimation_Report"
        assert "executive_summary" in report
        
        print(f"✅ NESMA估算完成: {ufp_result['total_ufp']} UFP")
    
    @pytest.mark.asyncio
    async def test_full_cosmic_estimation_workflow(self, sample_project, agents_collection):
        """测试完整的COSMIC估算流程"""
        
        # 1. 标准推荐
        recommendation = await agents_collection["standard_recommender"].execute_task(
            "recommend_standard",
            {"project_info": sample_project}
        )
        
        assert recommendation["recommended_standards"]
        
        # 2. 需求解析
        processes = await agents_collection["requirement_parser"].execute_task(
            "parse_requirements",
            {"requirements_text": sample_project.description}
        )
        
        assert len(processes["identified_processes"]) > 0
        
        # 3. COSMIC功能用户识别
        functional_users = await agents_collection["cosmic_functional_user"].execute_task(
            "identify_functional_users",
            {"project_info": sample_project}
        )
        
        assert len(functional_users["functional_users"]) > 0
        
        # 4. COSMIC边界分析
        boundary_analysis = await agents_collection["cosmic_boundary"].execute_task(
            "analyze_boundaries",
            {
                "project_info": sample_project,
                "functional_users": functional_users["functional_users"]
            }
        )
        
        assert boundary_analysis["software_boundary"]
        assert boundary_analysis["persistent_storage_boundary"]
        
        # 5. COSMIC数据移动分类
        data_movements = []
        for process in processes["identified_processes"]:
            result = await agents_collection["cosmic_data_movement"].execute_task(
                "classify_data_movements",
                {
                    "process_description": process["description"],
                    "boundary_analysis": boundary_analysis
                }
            )
            data_movements.extend(result["data_movements"])
        
        assert len(data_movements) > 0
        assert all(movement["type"] in [t.value for t in COSMICDataMovementType] 
                  for movement in data_movements)
        
        # 6. COSMIC CFP计算
        cfp_result = await agents_collection["cosmic_cfp"].execute_task(
            "calculate_cfp",
            {
                "data_movements": data_movements,
                "project_info": sample_project,
                "boundary_analysis": boundary_analysis
            }
        )
        
        assert cfp_result["total_cfp"] > 0
        assert "type_statistics" in cfp_result
        
        # 7. 生成报告
        report = await agents_collection["report_generator"].execute_task(
            "generate_report",
            {
                "estimation_type": "COSMIC",
                "results": cfp_result,
                "project_info": sample_project
            }
        )
        
        assert report["report_type"] == "COSMIC_Estimation_Report"
        assert "executive_summary" in report
        
        print(f"✅ COSMIC估算完成: {cfp_result['total_cfp']} CFP")
    
    @pytest.mark.asyncio
    async def test_dual_standard_estimation(self, sample_project, agents_collection):
        """测试双标准估算"""
        
        # 1. 标准推荐 - 强制使用双标准
        recommendation = {
            "recommended_standards": ["NESMA", "COSMIC"],
            "strategy": EstimationStrategy.DUAL_PARALLEL
        }
        
        # 2. 需求解析
        processes = await agents_collection["requirement_parser"].execute_task(
            "parse_requirements",
            {"requirements_text": sample_project.description}
        )
        
        # 3. 并行执行NESMA和COSMIC估算
        nesma_task = self._run_nesma_estimation(
            processes["identified_processes"], 
            sample_project, 
            agents_collection
        )
        
        cosmic_task = self._run_cosmic_estimation(
            processes["identified_processes"],
            sample_project,
            agents_collection
        )
        
        nesma_result, cosmic_result = await asyncio.gather(nesma_task, cosmic_task)
        
        # 4. 比较分析
        comparison = self._compare_estimation_results(nesma_result, cosmic_result)
        
        assert comparison["nesma_ufp"] > 0
        assert comparison["cosmic_cfp"] > 0
        assert "variance_analysis" in comparison
        
        print(f"✅ 双标准估算完成:")
        print(f"   NESMA: {comparison['nesma_ufp']} UFP")
        print(f"   COSMIC: {comparison['cosmic_cfp']} CFP")
        print(f"   差异率: {comparison['variance_percentage']:.1f}%")
    
    async def _run_nesma_estimation(
        self, 
        processes: List[Dict[str, Any]], 
        project_info: ProjectInfo,
        agents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行NESMA估算"""
        
        # 功能分类
        classifications = []
        for process in processes:
            result = await agents["nesma_classifier"].execute_task(
                "classify_function",
                {"function_description": process["description"]}
            )
            classifications.append(result)
        
        # 复杂度计算
        complexity_results = []
        for classification in classifications:
            result = await agents["nesma_complexity"].execute_task(
                "calculate_complexity",
                {
                    "function_type": classification["function_type"],
                    "function_description": classification["function_description"]
                }
            )
            complexity_results.append(result)
        
        # UFP计算
        ufp_result = await agents["nesma_ufp"].execute_task(
            "calculate_ufp",
            {
                "classifications": classifications,
                "complexity_results": complexity_results,
                "project_info": project_info
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
        functional_users = await agents["cosmic_functional_user"].execute_task(
            "identify_functional_users",
            {"project_info": project_info}
        )
        
        # 边界分析
        boundary_analysis = await agents["cosmic_boundary"].execute_task(
            "analyze_boundaries",
            {
                "project_info": project_info,
                "functional_users": functional_users["functional_users"]
            }
        )
        
        # 数据移动分类
        data_movements = []
        for process in processes:
            result = await agents["cosmic_data_movement"].execute_task(
                "classify_data_movements",
                {
                    "process_description": process["description"],
                    "boundary_analysis": boundary_analysis
                }
            )
            data_movements.extend(result["data_movements"])
        
        # CFP计算
        cfp_result = await agents["cosmic_cfp"].execute_task(
            "calculate_cfp",
            {
                "data_movements": data_movements,
                "project_info": project_info,
                "boundary_analysis": boundary_analysis
            }
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
            await agents_collection["nesma_classifier"].execute_task(
                "classify_function",
                {"function_description": ""}
            )
        
        # 测试不存在的任务
        with pytest.raises(ValueError):
            await agents_collection["nesma_classifier"].execute_task(
                "unknown_task",
                {"some_input": "value"}
            )
        
        print("✅ 错误处理测试通过")
    
    @pytest.mark.asyncio
    async def test_agent_performance(self, sample_project, agents_collection):
        """测试智能体性能"""
        
        start_time = datetime.now()
        
        # 执行标准推荐任务
        recommendation = await agents_collection["standard_recommender"].execute_task(
            "recommend_standard",
            {"project_info": sample_project}
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 验证性能要求（应在30秒内完成）
        assert execution_time < 30, f"标准推荐耗时过长: {execution_time}秒"
        assert recommendation is not None
        
        print(f"✅ 性能测试通过: 标准推荐耗时 {execution_time:.2f}秒")
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, sample_project, agents_collection):
        """测试并发智能体执行"""
        
        # 创建多个并发任务
        tasks = []
        
        for i in range(5):
            task = agents_collection["requirement_parser"].execute_task(
                "parse_requirements",
                {"requirements_text": f"{sample_project.description} - 测试{i}"}
            )
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有任务都成功执行
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        print(f"✅ 并发测试通过: {len(successful_results)}/5 任务成功")


@pytest.mark.asyncio
async def test_integration_suite():
    """运行完整的集成测试套件"""
    
    print("🚀 开始智能体集成测试...")
    
    test_instance = TestAgentsIntegration()
    
    # 创建测试数据
    sample_project = ProjectInfo(
        name="集成测试项目",
        description="用于测试智能体集成的样本项目",
        technology_stack=["Python", "FastAPI"],
        business_domain="测试"
    )
    
    agents_collection = {
        "standard_recommender": StandardRecommenderAgent(),
        "requirement_parser": RequirementParserAgent(),
        "nesma_classifier": NESMAFunctionClassifierAgent(),
        "nesma_complexity": NESMAComplexityCalculatorAgent(),
        "nesma_ufp": NESMAUFPCalculatorAgent(),
        "cosmic_functional_user": COSMICFunctionalUserAgent(),
        "cosmic_data_movement": COSMICDataMovementClassifierAgent(),
        "cosmic_boundary": COSMICBoundaryAnalyzerAgent(),
        "cosmic_cfp": COSMICCFPCalculatorAgent(),
        "report_generator": ReportGeneratorAgent()
    }
    
    # 运行测试
    await test_instance.test_full_nesma_estimation_workflow(sample_project, agents_collection)
    await test_instance.test_full_cosmic_estimation_workflow(sample_project, agents_collection)
    await test_instance.test_dual_standard_estimation(sample_project, agents_collection)
    await test_instance.test_agent_error_handling(sample_project, agents_collection)
    await test_instance.test_agent_performance(sample_project, agents_collection)
    await test_instance.test_concurrent_agent_execution(sample_project, agents_collection)
    
    print("🎉 所有集成测试通过！")


if __name__ == "__main__":
    asyncio.run(test_integration_suite()) 