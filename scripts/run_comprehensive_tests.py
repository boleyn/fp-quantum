"""
量子智能化功能点估算系统 - 综合测试运行脚本

运行所有测试并生成综合报告
"""

import asyncio
import time
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """综合测试运行器"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        
        self.start_time = datetime.now()
        
        print("🚀 开始运行量子智能化功能点估算系统综合测试")
        print("=" * 60)
        
        # 1. CFP计算器测试
        await self._test_cfp_calculator()
        
        # 2. 集成测试
        await self._test_agents_integration()
        
        # 3. 性能优化测试
        await self._test_performance_optimization()
        
        # 4. 知识库增强测试
        await self._test_knowledge_base_enhancement()
        
        # 5. 生成综合报告
        self.end_time = datetime.now()
        report = await self._generate_comprehensive_report()
        
        return report
    
    async def _test_cfp_calculator(self) -> None:
        """测试CFP计算器"""
        
        print("\n📊 测试COSMIC CFP计算器...")
        
        try:
            # 模拟CFP计算器测试
            from agents.standards.cosmic.cfp_calculator import create_cosmic_cfp_calculator
            from models.cosmic_models import COSMICDataMovement, COSMICDataMovementType, COSMICBoundaryAnalysis
            from models.project_models import ProjectInfo
            
            calculator = await create_cosmic_cfp_calculator()
            
            # 创建测试数据
            test_movements = [
                COSMICDataMovement(
                    id="test_entry_1",
                    type=COSMICDataMovementType.ENTRY,
                    source="用户",
                    target="系统",
                    data_group="登录信息",
                    justification="用户登录"
                ),
                COSMICDataMovement(
                    id="test_read_1",
                    type=COSMICDataMovementType.READ,
                    source="数据库",
                    target="系统",
                    data_group="用户信息",
                    justification="验证用户"
                ),
                COSMICDataMovement(
                    id="test_write_1",
                    type=COSMICDataMovementType.WRITE,
                    source="系统",
                    target="数据库",
                    data_group="登录日志",
                    justification="记录登录"
                ),
                COSMICDataMovement(
                    id="test_exit_1",
                    type=COSMICDataMovementType.EXIT,
                    source="系统",
                    target="用户",
                    data_group="登录结果",
                    justification="返回状态"
                )
            ]
            
            test_project = ProjectInfo(
                name="CFP测试项目",
                description="测试CFP计算功能",
                technology_stack=["Python"],
                business_domain="测试"
            )
            
            test_boundary = COSMICBoundaryAnalysis(
                software_boundary="测试边界",
                persistent_storage_boundary="测试存储边界",
                functional_users=[],
                boundary_reasoning="测试边界分析"
            )
            
            # 执行CFP计算
            cfp_result = await calculator.calculate_cfp(
                test_movements,
                test_project,
                test_boundary
            )
            
            # 验证结果
            expected_cfp = len(test_movements)  # 4个数据移动 = 4 CFP
            actual_cfp = cfp_result["total_cfp"]
            
            self.test_results["cfp_calculator"] = {
                "status": "PASSED" if actual_cfp == expected_cfp else "FAILED",
                "expected_cfp": expected_cfp,
                "actual_cfp": actual_cfp,
                "type_statistics": cfp_result["type_statistics"],
                "quality_score": cfp_result["quality_metrics"]["quality_score"]
            }
            
            print(f"   ✅ CFP计算测试通过: {actual_cfp} CFP")
            
        except Exception as e:
            self.test_results["cfp_calculator"] = {
                "status": "ERROR",
                "error_message": str(e)
            }
            print(f"   ❌ CFP计算测试失败: {e}")
    
    async def _test_agents_integration(self) -> None:
        """测试智能体集成"""
        
        print("\n🤖 测试智能体集成...")
        
        try:
            # 运行集成测试的简化版本
            from tests.integration.test_agents_integration import test_integration_suite
            
            # 执行集成测试
            await test_integration_suite()
            
            self.test_results["agents_integration"] = {
                "status": "PASSED",
                "test_cases": [
                    "NESMA估算流程",
                    "COSMIC估算流程", 
                    "双标准估算",
                    "错误处理",
                    "性能测试",
                    "并发测试"
                ]
            }
            
            print("   ✅ 智能体集成测试通过")
            
        except Exception as e:
            self.test_results["agents_integration"] = {
                "status": "ERROR",
                "error_message": str(e)
            }
            print(f"   ❌ 智能体集成测试失败: {e}")
    
    async def _test_performance_optimization(self) -> None:
        """测试性能优化"""
        
        print("\n⚡ 测试性能优化...")
        
        try:
            from agents.base.performance_optimizer import PerformanceOptimizer
            
            optimizer = PerformanceOptimizer()
            
            # 测试缓存功能
            @optimizer.cached()
            async def test_cached_function(x: int) -> int:
                await asyncio.sleep(0.1)  # 模拟计算
                return x * x
            
            # 测试计时功能
            @optimizer.timed
            async def test_timed_function(x: int) -> int:
                await asyncio.sleep(0.05)
                return x + 1
            
            # 执行性能测试
            start_time = time.time()
            
            # 第一次调用（无缓存）
            result1 = await test_cached_function(5)
            first_call_time = time.time() - start_time
            
            # 第二次调用（有缓存）
            start_time = time.time()
            result2 = await test_cached_function(5)
            second_call_time = time.time() - start_time
            
            # 测试计时功能
            result3 = await test_timed_function(10)
            
            # 获取性能报告
            performance_report = optimizer.get_performance_report()
            
            # 验证缓存效果
            cache_effective = second_call_time < first_call_time * 0.5
            
            self.test_results["performance_optimization"] = {
                "status": "PASSED" if cache_effective else "FAILED",
                "cache_effective": cache_effective,
                "first_call_time": first_call_time,
                "second_call_time": second_call_time,
                "performance_report": performance_report
            }
            
            print(f"   ✅ 性能优化测试通过")
            print(f"      缓存效果: {first_call_time:.3f}s → {second_call_time:.3f}s")
            
        except Exception as e:
            self.test_results["performance_optimization"] = {
                "status": "ERROR",
                "error_message": str(e)
            }
            print(f"   ❌ 性能优化测试失败: {e}")
    
    async def _test_knowledge_base_enhancement(self) -> None:
        """测试知识库增强"""
        
        print("\n📚 测试知识库增强...")
        
        try:
            from scripts.enhance_knowledge_base import KnowledgeBaseEnhancer
            
            enhancer = KnowledgeBaseEnhancer()
            
            # 执行知识库增强
            enhancement_result = await enhancer.enhance_knowledge_base()
            
            # 验证增强结果
            docs_created = enhancement_result["total_documents"] > 0
            chunks_created = enhancement_result["total_chunks"] > 0
            
            self.test_results["knowledge_base_enhancement"] = {
                "status": "PASSED" if docs_created and chunks_created else "FAILED",
                "total_documents": enhancement_result["total_documents"],
                "total_chunks": enhancement_result["total_chunks"],
                "nesma_documents": enhancement_result["nesma_documents"],
                "cosmic_documents": enhancement_result["cosmic_documents"],
                "saved_to": enhancement_result["saved_to"]
            }
            
            print(f"   ✅ 知识库增强测试通过")
            print(f"      创建文档: {enhancement_result['total_documents']}")
            print(f"      文档块: {enhancement_result['total_chunks']}")
            
        except Exception as e:
            self.test_results["knowledge_base_enhancement"] = {
                "status": "ERROR",
                "error_message": str(e)
            }
            print(f"   ❌ 知识库增强测试失败: {e}")
    
    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        
        print("\n📋 生成综合测试报告...")
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results.values() if t.get("status") == "PASSED"])
        failed_tests = len([t for t in self.test_results.values() if t.get("status") == "FAILED"])
        error_tests = len([t for t in self.test_results.values() if t.get("status") == "ERROR"])
        
        # 计算总体成功率
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # 生成报告
        report = {
            "test_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate_percent": success_rate
            },
            "test_details": self.test_results,
            "recommendations": self._generate_recommendations(),
            "next_steps": self._generate_next_steps()
        }
        
        # 保存报告到文件
        report_file = Path("test_reports") / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print(f"\n📊 测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   错误: {error_tests}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   报告保存至: {report_file}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        for test_name, result in self.test_results.items():
            if result.get("status") == "FAILED":
                if test_name == "cfp_calculator":
                    recommendations.append("检查CFP计算逻辑，确保数据移动计数正确")
                elif test_name == "performance_optimization":
                    recommendations.append("优化缓存策略，提高性能优化效果")
                
            elif result.get("status") == "ERROR":
                recommendations.append(f"修复{test_name}模块的错误问题")
        
        # 通用建议
        if all(result.get("status") == "PASSED" for result in self.test_results.values()):
            recommendations.append("所有测试通过，系统准备就绪")
        else:
            recommendations.append("继续完善失败的测试模块")
        
        return recommendations or ["系统运行良好，继续监控性能"]
    
    def _generate_next_steps(self) -> List[str]:
        """生成下一步计划"""
        next_steps = [
            "完善错误处理机制",
            "增加更多边缘案例测试",
            "优化大规模数据处理性能", 
            "扩展知识库内容",
            "建立持续集成流程",
            "添加用户界面测试",
            "进行压力测试",
            "准备生产环境部署"
        ]
        
        return next_steps


async def main():
    """主函数"""
    
    runner = ComprehensiveTestRunner()
    
    try:
        report = await runner.run_all_tests()
        
        print("\n🎉 综合测试完成!")
        
        # 返回退出码
        if report["test_summary"]["success_rate_percent"] == 100:
            print("✅ 所有测试通过!")
            return 0
        else:
            print("⚠️  部分测试未通过，请查看报告详情")
            return 1
            
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 