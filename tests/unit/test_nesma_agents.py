"""
NESMA智能体单元测试

测试NESMA相关智能体的核心功能
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from agents.standards.nesma.function_classifier import NESMAFunctionClassifierAgent
from agents.standards.nesma.complexity_calculator import NESMAComplexityCalculatorAgent
from agents.standards.nesma.ufp_calculator import NESMAUFPCalculatorAgent

from models.nesma_models import (
    NESMAFunctionType, 
    NESMANESMAComplexityLevel, 
    NESMAFunction,
    NESMAComplexityAnalysis,
    NESMAUFPResult
)


class TestNESMAFunctionClassifier:
    """NESMA功能分类器测试"""
    
    @pytest.fixture
    def classifier(self):
        """创建分类器实例"""
        return NESMAFunctionClassifierAgent()
    
    @pytest.mark.asyncio
    async def test_classify_external_input(self, classifier):
        """测试外部输入(EI)分类"""
        with patch.object(classifier, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "function_type": "EI",
                "confidence": 0.9,
                "reasoning": "用户数据输入功能"
            }
            
            result = await classifier.classify_function(
                "用户注册：用户输入姓名、邮箱、密码等信息进行注册"
            )
            
            assert result["function_type"] == NESMAFunctionType.EI
            assert result["confidence"] >= 0.8
            assert "输入" in result["reasoning"]
    
    @pytest.mark.asyncio 
    async def test_classify_external_output(self, classifier):
        """测试外部输出(EO)分类"""
        with patch.object(classifier, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "function_type": "EO",
                "confidence": 0.85,
                "reasoning": "系统生成报告输出"
            }
            
            result = await classifier.classify_function(
                "生成用户活动报告：系统统计用户行为并生成月度报告"
            )
            
            assert result["function_type"] == NESMAFunctionType.EO
            assert result["confidence"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_classify_external_inquiry(self, classifier):
        """测试外部查询(EQ)分类"""
        with patch.object(classifier, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "function_type": "EQ", 
                "confidence": 0.88,
                "reasoning": "查询用户信息功能"
            }
            
            result = await classifier.classify_function(
                "查询用户信息：管理员根据用户ID查询用户详细信息"
            )
            
            assert result["function_type"] == NESMAFunctionType.EQ
            assert result["confidence"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_classify_internal_logical_file(self, classifier):
        """测试内部逻辑文件(ILF)分类"""
        with patch.object(classifier, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "function_type": "ILF",
                "confidence": 0.92,
                "reasoning": "系统维护的用户数据表"
            }
            
            result = await classifier.classify_function(
                "用户信息表：存储用户基本信息、账户状态等数据"
            )
            
            assert result["function_type"] == NESMAFunctionType.ILF
            assert result["confidence"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_classify_external_interface_file(self, classifier):
        """测试外部接口文件(EIF)分类"""
        with patch.object(classifier, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "function_type": "EIF",
                "confidence": 0.87,
                "reasoning": "外部系统提供的地址验证服务"
            }
            
            result = await classifier.classify_function(
                "地址验证服务：调用第三方API验证用户地址信息"
            )
            
            assert result["function_type"] == NESMAFunctionType.EIF
            assert result["confidence"] >= 0.8


class TestNESMAComplexityCalculator:
    """NESMA复杂度计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建复杂度计算器实例"""
        return NESMAComplexityCalculatorAgent()
    
    @pytest.mark.asyncio
    async def test_calculate_ei_complexity_low(self, calculator):
        """测试EI低复杂度计算"""
        with patch.object(calculator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "det_count": 5,
                "ftr_count": 1,
                "complexity": "LOW",
                "reasoning": "数据元素较少，引用文件类型单一"
            }
            
            result = await calculator.calculate_complexity(
                function_type=NESMAFunctionType.EI,
                function_description="简单的用户注册功能",
                detailed_analysis=True
            )
            
            assert result["complexity"] == NESMANESMAComplexityLevel.LOW
            assert result["det_count"] <= 15
            assert result["ftr_count"] <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_ei_complexity_high(self, calculator):
        """测试EI高复杂度计算"""
        with patch.object(calculator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "det_count": 25,
                "ftr_count": 3,
                "complexity": "HIGH", 
                "reasoning": "数据元素众多，涉及多个文件类型"
            }
            
            result = await calculator.calculate_complexity(
                function_type=NESMAFunctionType.EI,
                function_description="复杂的企业用户注册功能，包含多种验证",
                detailed_analysis=True
            )
            
            assert result["complexity"] == NESMANESMAComplexityLevel.HIGH
            assert result["det_count"] > 15
            assert result["ftr_count"] >= 2
    
    @pytest.mark.asyncio
    async def test_calculate_ilf_complexity_boundary(self, calculator):
        """测试ILF复杂度边界条件"""
        with patch.object(calculator, '_call_llm') as mock_llm:
            # 测试LOW-AVERAGE边界
            mock_llm.return_value = {
                "det_count": 19,  # 边界值
                "ret_count": 1,
                "complexity": "LOW",
                "reasoning": "刚好在LOW边界"
            }
            
            result = await calculator.calculate_complexity(
                function_type=NESMAFunctionType.ILF,
                function_description="用户基本信息表",
                detailed_analysis=True
            )
            
            assert result["complexity"] == NESMANESMAComplexityLevel.LOW
            assert result["det_count"] == 19
            assert result["ret_count"] == 1
    
    @pytest.mark.asyncio
    async def test_calculate_eq_complexity_average(self, calculator):
        """测试EQ平均复杂度计算"""
        with patch.object(calculator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "det_count": 8,
                "ftr_count": 2, 
                "complexity": "AVERAGE",
                "reasoning": "中等数量的数据元素和文件类型"
            }
            
            result = await calculator.calculate_complexity(
                function_type=NESMAFunctionType.EQ,
                function_description="用户信息查询功能",
                detailed_analysis=True
            )
            
            assert result["complexity"] == NESMANESMAComplexityLevel.AVERAGE
            assert 6 <= result["det_count"] <= 19
            assert 2 <= result["ftr_count"] <= 3


class TestNESMAUFPCalculator:
    """NESMA UFP计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建UFP计算器实例"""
        return NESMAUFPCalculatorAgent()
    
    def test_standard_weight_table(self, calculator):
        """测试NESMA标准权重表的准确性"""
        # NESMA官方权重表
        expected_weights = {
            NESMAFunctionType.ILF: {
                NESMANESMAComplexityLevel.LOW: 7,
                NESMANESMAComplexityLevel.AVERAGE: 10, 
                NESMANESMAComplexityLevel.HIGH: 15
            },
            NESMAFunctionType.EIF: {
                NESMANESMAComplexityLevel.LOW: 5,
                NESMANESMAComplexityLevel.AVERAGE: 7,
                NESMANESMAComplexityLevel.HIGH: 10
            },
            NESMAFunctionType.EI: {
                NESMANESMAComplexityLevel.LOW: 3,
                NESMANESMAComplexityLevel.AVERAGE: 4,
                NESMANESMAComplexityLevel.HIGH: 6
            },
            NESMAFunctionType.EO: {
                NESMANESMAComplexityLevel.LOW: 4,
                NESMANESMAComplexityLevel.AVERAGE: 5,
                NESMANESMAComplexityLevel.HIGH: 7
            },
            NESMAFunctionType.EQ: {
                NESMANESMAComplexityLevel.LOW: 3,
                NESMANESMAComplexityLevel.AVERAGE: 4,
                NESMANESMAComplexityLevel.HIGH: 6
            }
        }
        
        # 验证权重表
        for func_type, complexities in expected_weights.items():
            for complexity, expected_weight in complexities.items():
                actual_weight = calculator.get_weight(func_type, complexity)
                assert actual_weight == expected_weight, \
                    f"权重不匹配: {func_type}-{complexity} 期望{expected_weight}, 实际{actual_weight}"
    
    @pytest.mark.asyncio
    async def test_calculate_ufp_simple_project(self, calculator):
        """测试简单项目的UFP计算"""
        functions = [
            {
                "type": NESMAFunctionType.EI,
                "complexity": NESMANESMAComplexityLevel.LOW,
                "count": 2,
                "description": "用户注册和登录"
            },
            {
                "type": NESMAFunctionType.EQ,
                "complexity": NESMANESMAComplexityLevel.AVERAGE,
                "count": 1,
                "description": "用户信息查询"
            },
            {
                "type": NESMAFunctionType.ILF,
                "complexity": NESMANESMAComplexityLevel.LOW,
                "count": 1,
                "description": "用户信息表"
            }
        ]
        
        result = await calculator.calculate_ufp(functions)
        
        # 手动计算验证: 2*3 + 1*4 + 1*7 = 6 + 4 + 7 = 17 UFP
        expected_ufp = 17
        assert result["total_ufp"] == expected_ufp
        assert len(result["function_breakdown"]) == 3
    
    @pytest.mark.asyncio
    async def test_calculate_ufp_complex_project(self, calculator):
        """测试复杂项目的UFP计算"""
        functions = [
            {
                "type": NESMAFunctionType.EI,
                "complexity": NESMANESMAComplexityLevel.HIGH,
                "count": 3,
                "description": "复杂数据输入"
            },
            {
                "type": NESMAFunctionType.EO,
                "complexity": NESMANESMAComplexityLevel.AVERAGE,
                "count": 2, 
                "description": "报告生成"
            },
            {
                "type": NESMAFunctionType.EQ,
                "complexity": NESMANESMAComplexityLevel.LOW,
                "count": 5,
                "description": "基本查询"
            },
            {
                "type": NESMAFunctionType.ILF,
                "complexity": NESMANESMAComplexityLevel.HIGH,
                "count": 2,
                "description": "核心数据文件"
            },
            {
                "type": NESMAFunctionType.EIF,
                "complexity": NESMANESMAComplexityLevel.AVERAGE,
                "count": 1,
                "description": "外部接口"
            }
        ]
        
        result = await calculator.calculate_ufp(functions)
        
        # 手动计算: 3*6 + 2*5 + 5*3 + 2*15 + 1*7 = 18 + 10 + 15 + 30 + 7 = 80 UFP
        expected_ufp = 80
        assert result["total_ufp"] == expected_ufp
        assert result["total_ufp"] > 50  # 复杂项目应该超过50 UFP
    
    @pytest.mark.asyncio
    async def test_ufp_breakdown_accuracy(self, calculator):
        """测试UFP分解的准确性"""
        functions = [
            {
                "type": NESMAFunctionType.EI,
                "complexity": NESMANESMAComplexityLevel.AVERAGE,
                "count": 1,
                "description": "测试输入"
            }
        ]
        
        result = await calculator.calculate_ufp(functions)
        breakdown = result["function_breakdown"]
        
        assert len(breakdown) == 1
        assert breakdown[0]["type"] == NESMAFunctionType.EI
        assert breakdown[0]["complexity"] == NESMANESMAComplexityLevel.AVERAGE
        assert breakdown[0]["weight"] == 4  # EI-AVERAGE权重
        assert breakdown[0]["ufp"] == 4     # 1*4=4
        assert breakdown[0]["count"] == 1


@pytest.mark.asyncio
async def test_nesma_agent_integration():
    """NESMA智能体端到端集成测试"""
    
    # 准备测试数据
    project_description = """
    电商平台订单管理系统：
    1. 用户下单 - 输入商品信息、收货地址等
    2. 订单查询 - 查看订单状态和详情
    3. 订单修改 - 修改订单商品或地址
    4. 库存扣减 - 自动扣减商品库存
    5. 订单数据存储 - 维护订单信息表
    """
    
    # 创建智能体实例
    classifier = NESMAFunctionClassifierAgent()
    complexity_calc = NESMAComplexityCalculatorAgent()
    ufp_calc = NESMAUFPCalculatorAgent()
    
    # Mock LLM调用结果
    classification_results = [
        {"function_type": "EI", "confidence": 0.9, "description": "用户下单"},
        {"function_type": "EQ", "confidence": 0.85, "description": "订单查询"}, 
        {"function_type": "EI", "confidence": 0.8, "description": "订单修改"},
        {"function_type": "EO", "confidence": 0.87, "description": "库存扣减"},
        {"function_type": "ILF", "confidence": 0.92, "description": "订单数据存储"}
    ]
    
    complexity_results = [
        {"complexity": "AVERAGE", "det_count": 12, "ftr_count": 2},
        {"complexity": "LOW", "det_count": 8, "ftr_count": 1},
        {"complexity": "AVERAGE", "det_count": 10, "ftr_count": 2},
        {"complexity": "LOW", "det_count": 5, "ftr_count": 1},
        {"complexity": "AVERAGE", "det_count": 25, "ret_count": 2}
    ]
    
    # 模拟分类过程
    with patch.object(classifier, '_call_llm') as mock_classifier:
        mock_classifier.side_effect = classification_results
        
        with patch.object(complexity_calc, '_call_llm') as mock_complexity:
            mock_complexity.side_effect = complexity_results
            
            # 执行完整流程
            functions = []
            for i, desc in enumerate(["用户下单", "订单查询", "订单修改", "库存扣减", "订单数据存储"]):
                # 分类
                classification = await classifier.classify_function(desc)
                
                # 复杂度计算
                complexity = await complexity_calc.calculate_complexity(
                    classification["function_type"],
                    desc
                )
                
                functions.append({
                    "type": classification["function_type"],
                    "complexity": complexity["complexity"],
                    "count": 1,
                    "description": desc
                })
            
            # UFP计算
            final_result = await ufp_calc.calculate_ufp(functions)
            
            # 验证结果
            assert final_result["total_ufp"] > 0
            assert len(final_result["function_breakdown"]) == 5
            
            # 验证UFP合理性 (一般电商订单系统应该在20-40 UFP范围)
            assert 15 <= final_result["total_ufp"] <= 50
            
            print(f"✅ NESMA集成测试完成，总UFP: {final_result['total_ufp']}")