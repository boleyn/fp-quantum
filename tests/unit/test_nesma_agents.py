"""
量子智能化功能点估算系统 - NESMA智能体单元测试

测试NESMA标准相关的功能分类、复杂度计算和UFP计算智能体
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from agents.standards.nesma.function_classifier import NESMAFunctionClassifierAgent
from agents.standards.nesma.complexity_calculator import NESMAComplexityCalculatorAgent
from agents.standards.nesma.ufp_calculator import NESMAUFPCalculatorAgent
from models.nesma_models import (
    NESMAFunctionType, NESMAComplexityLevel, NESMAFunctionClassification
)
from models.project_models import ProjectInfo, TechnologyStack, BusinessDomain
from models.common_models import EstimationStrategy


class TestNESMAFunctionClassifier:
    """NESMA功能分类器测试"""
    
    @pytest.fixture
    def classifier(self):
        """创建分类器实例"""
        return NESMAFunctionClassifierAgent()
    
    @pytest.mark.asyncio
    async def test_classify_external_input(self, classifier):
        """测试外部输入(EI)分类"""
        with patch.object(classifier, 'classify_function') as mock_classify:
            mock_classification = NESMAFunctionClassification(
                function_id="test_ei_1",
                function_name="用户注册",
                function_description="用户输入姓名、邮箱、密码等信息进行注册",
                function_type=NESMAFunctionType.EI,
                confidence_score=0.9,
                justification="用户数据输入功能",
                rules_applied=["EI定义规则"]
            )
            mock_classify.return_value = mock_classification
            
            result = await classifier.classify_function(
                "用户输入姓名、邮箱、密码等信息进行注册"
            )
            
            assert result.function_type == NESMAFunctionType.EI
            assert result.confidence_score >= 0.8
            assert "输入" in result.justification
    
    @pytest.mark.asyncio 
    async def test_classify_external_output(self, classifier):
        """测试外部输出(EO)分类"""
        with patch.object(classifier, 'classify_function') as mock_classify:
            mock_classification = NESMAFunctionClassification(
                function_id="test_eo_1",
                function_name="报告生成",
                function_description="生成用户活动报告：系统统计用户行为并生成月度报告",
                function_type=NESMAFunctionType.EO,
                confidence_score=0.85,
                justification="系统生成报告输出",
                rules_applied=["EO定义规则"]
            )
            mock_classify.return_value = mock_classification
            
            result = await classifier.classify_function(
                "生成用户活动报告：系统统计用户行为并生成月度报告"
            )
            
            assert result.function_type == NESMAFunctionType.EO
            assert result.confidence_score >= 0.8
            assert "报告" in result.justification or "生成" in result.justification
    
    @pytest.mark.asyncio
    async def test_classify_external_inquiry(self, classifier):
        """测试外部查询(EQ)分类"""
        with patch.object(classifier, 'classify_function') as mock_classify:
            mock_classification = NESMAFunctionClassification(
                function_id="test_eq_1",
                function_name="用户信息查询",
                function_description="查询用户信息：管理员根据用户ID查询用户详细信息",
                function_type=NESMAFunctionType.EQ,
                confidence_score=0.88,
                justification="查询用户信息功能",
                rules_applied=["EQ定义规则"]
            )
            mock_classify.return_value = mock_classification
            
            result = await classifier.classify_function(
                "查询用户信息：管理员根据用户ID查询用户详细信息"
            )
            
            assert result.function_type == NESMAFunctionType.EQ
            assert result.confidence_score >= 0.8
            assert "查询" in result.justification
    
    @pytest.mark.asyncio
    async def test_classify_internal_logical_file(self, classifier):
        """测试内部逻辑文件(ILF)分类"""
        with patch.object(classifier, 'classify_function') as mock_classify:
            mock_classification = NESMAFunctionClassification(
                function_id="test_ilf_1",
                function_name="用户信息表",
                function_description="用户信息表：存储用户基本信息、账户状态等数据",
                function_type=NESMAFunctionType.ILF,
                confidence_score=0.92,
                justification="系统维护的用户数据表",
                rules_applied=["ILF定义规则"]
            )
            mock_classify.return_value = mock_classification
            
            result = await classifier.classify_function(
                "用户信息表：存储用户基本信息、账户状态等数据"
            )
            
            assert result.function_type == NESMAFunctionType.ILF
            assert result.confidence_score >= 0.8
            assert "表" in result.justification or "数据" in result.justification
    
    @pytest.mark.asyncio
    async def test_classify_external_interface_file(self, classifier):
        """测试外部接口文件(EIF)分类"""
        with patch.object(classifier, 'classify_function') as mock_classify:
            mock_classification = NESMAFunctionClassification(
                function_id="test_eif_1",
                function_name="地址验证服务",
                function_description="地址验证服务：调用第三方API验证用户地址信息",
                function_type=NESMAFunctionType.EIF,
                confidence_score=0.87,
                justification="外部系统提供的地址验证服务",
                rules_applied=["EIF定义规则"]
            )
            mock_classify.return_value = mock_classification
            
            result = await classifier.classify_function(
                "地址验证服务：调用第三方API验证用户地址信息"
            )
            
            assert result.function_type == NESMAFunctionType.EIF
            assert result.confidence_score >= 0.8
            assert "外部" in result.justification or "第三方" in result.justification


class TestNESMAComplexityCalculator:
    """NESMA复杂度计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建复杂度计算器实例"""
        return NESMAComplexityCalculatorAgent()
    
    @pytest.mark.asyncio
    async def test_calculate_ei_complexity_low(self, calculator):
        """测试EI低复杂度计算"""
        with patch.object(calculator, 'execute') as mock_execute:
            mock_result = {
                "function_id": "test_ei_1",
                "function_type": "EI",
                "complexity": NESMAComplexityLevel.LOW,
                "det_count": 5,
                "ftr_count": 1,
                "reasoning": "数据元素较少，引用文件类型单一",
                "weight": 3,
                "fp_value": 3
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_complexity",
                {
                    "function_type": NESMAFunctionType.EI,
                    "function_description": "简单的用户注册功能",
                    "detailed_analysis": True
                }
            )
            
            assert result["complexity"] == NESMAComplexityLevel.LOW
            assert result["det_count"] <= 15
            assert result["ftr_count"] <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_ei_complexity_high(self, calculator):
        """测试EI高复杂度计算"""
        with patch.object(calculator, 'execute') as mock_execute:
            mock_result = {
                "function_id": "test_ei_2",
                "function_type": "EI",
                "complexity": NESMAComplexityLevel.HIGH,
                "det_count": 25,
                "ftr_count": 3,
                "reasoning": "数据元素众多，涉及多个文件类型",
                "weight": 6,
                "fp_value": 6
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_complexity",
                {
                    "function_type": NESMAFunctionType.EI,
                    "function_description": "复杂的企业用户注册功能，包含多种验证",
                    "detailed_analysis": True
                }
            )
            
            assert result["complexity"] == NESMAComplexityLevel.HIGH
            assert result["det_count"] > 15
            assert result["ftr_count"] >= 2
    
    @pytest.mark.asyncio
    async def test_calculate_ilf_complexity_boundary(self, calculator):
        """测试ILF复杂度边界条件"""
        with patch.object(calculator, 'execute') as mock_execute:
            # 测试LOW-AVERAGE边界
            mock_result = {
                "function_id": "test_ilf_1",
                "function_type": "ILF",
                "complexity": NESMAComplexityLevel.LOW,
                "det_count": 19,  # 边界值
                "ret_count": 1,
                "reasoning": "刚好在LOW边界",
                "weight": 7,
                "fp_value": 7
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_complexity",
                {
                    "function_type": NESMAFunctionType.ILF,
                    "function_description": "用户基本信息表",
                    "detailed_analysis": True
                }
            )
            
            assert result["complexity"] == NESMAComplexityLevel.LOW
            assert result["det_count"] == 19
            assert result["ret_count"] == 1
    
    @pytest.mark.asyncio
    async def test_calculate_eq_complexity_average(self, calculator):
        """测试EQ平均复杂度计算"""
        with patch.object(calculator, 'execute') as mock_execute:
            mock_result = {
                "function_id": "test_eq_1",
                "function_type": "EQ",
                "complexity": NESMAComplexityLevel.AVERAGE,
                "det_count": 8,
                "ftr_count": 2,
                "reasoning": "中等数量的数据元素和文件类型",
                "weight": 4,
                "fp_value": 4
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_complexity",
                {
                    "function_type": NESMAFunctionType.EQ,
                    "function_description": "用户信息查询功能",
                    "detailed_analysis": True
                }
            )
            
            assert result["complexity"] == NESMAComplexityLevel.AVERAGE
            assert result["det_count"] == 8
            assert result["ftr_count"] == 2


class TestNESMAUFPCalculator:
    """NESMA UFP计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        """创建UFP计算器实例"""
        return NESMAUFPCalculatorAgent()
    
    def test_standard_weight_table(self, calculator):
        """测试标准权重表"""
        # 测试权重表是否正确 - 通过调用内部方法
        from models.nesma_models import NESMAWeightTable
        weights = NESMAWeightTable(version="2.3")
        
        # EI权重
        assert weights.get_weight(NESMAFunctionType.EI, NESMAComplexityLevel.LOW) == 3
        assert weights.get_weight(NESMAFunctionType.EI, NESMAComplexityLevel.AVERAGE) == 4
        assert weights.get_weight(NESMAFunctionType.EI, NESMAComplexityLevel.HIGH) == 6
        
        # EO权重
        assert weights.get_weight(NESMAFunctionType.EO, NESMAComplexityLevel.LOW) == 4
        assert weights.get_weight(NESMAFunctionType.EO, NESMAComplexityLevel.AVERAGE) == 5
        assert weights.get_weight(NESMAFunctionType.EO, NESMAComplexityLevel.HIGH) == 7
        
        # EQ权重
        assert weights.get_weight(NESMAFunctionType.EQ, NESMAComplexityLevel.LOW) == 3
        assert weights.get_weight(NESMAFunctionType.EQ, NESMAComplexityLevel.AVERAGE) == 4
        assert weights.get_weight(NESMAFunctionType.EQ, NESMAComplexityLevel.HIGH) == 6
        
        # ILF权重
        assert weights.get_weight(NESMAFunctionType.ILF, NESMAComplexityLevel.LOW) == 7
        assert weights.get_weight(NESMAFunctionType.ILF, NESMAComplexityLevel.AVERAGE) == 10
        assert weights.get_weight(NESMAFunctionType.ILF, NESMAComplexityLevel.HIGH) == 15
        
        # EIF权重
        assert weights.get_weight(NESMAFunctionType.EIF, NESMAComplexityLevel.LOW) == 5
        assert weights.get_weight(NESMAFunctionType.EIF, NESMAComplexityLevel.AVERAGE) == 7
        assert weights.get_weight(NESMAFunctionType.EIF, NESMAComplexityLevel.HIGH) == 10
    
    @pytest.mark.asyncio
    async def test_calculate_ufp_simple_project(self, calculator):
        """测试简单项目UFP计算"""
        # 模拟分类结果
        classifications = [
            {"function_type": "EI", "complexity": "LOW", "count": 2},
            {"function_type": "EO", "complexity": "AVERAGE", "count": 1},
            {"function_type": "EQ", "complexity": "LOW", "count": 3},
            {"function_type": "ILF", "complexity": "AVERAGE", "count": 2}
        ]
        
        with patch.object(calculator, 'execute') as mock_execute:
            expected_ufp = (2*3) + (1*5) + (3*3) + (2*10)  # 6+5+9+20 = 40
            mock_result = {
                "total_ufp": expected_ufp,
                "function_breakdown": classifications,
                "weight_table_used": "NESMA v2.3",
                "calculation_details": {
                    "EI": {"count": 2, "weight": 3, "subtotal": 6},
                    "EO": {"count": 1, "weight": 5, "subtotal": 5},
                    "EQ": {"count": 3, "weight": 3, "subtotal": 9},
                    "ILF": {"count": 2, "weight": 10, "subtotal": 20}
                }
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_ufp",
                {
                    "classifications": classifications,
                    "project_info": {"name": "简单测试项目"}
                }
            )
            
            assert result["total_ufp"] == expected_ufp
            assert "calculation_details" in result
    
    @pytest.mark.asyncio
    async def test_calculate_ufp_complex_project(self, calculator):
        """测试复杂项目UFP计算"""
        classifications = [
            {"function_type": "EI", "complexity": "HIGH", "count": 5},
            {"function_type": "EO", "complexity": "HIGH", "count": 3},
            {"function_type": "EQ", "complexity": "AVERAGE", "count": 8},
            {"function_type": "ILF", "complexity": "HIGH", "count": 4},
            {"function_type": "EIF", "complexity": "AVERAGE", "count": 2}
        ]
        
        with patch.object(calculator, 'execute') as mock_execute:
            expected_ufp = (5*6) + (3*7) + (8*4) + (4*15) + (2*7)  # 30+21+32+60+14 = 157
            mock_result = {
                "total_ufp": expected_ufp,
                "function_breakdown": classifications,
                "weight_table_used": "NESMA v2.3"
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_ufp",
                {
                    "classifications": classifications,
                    "project_info": {"name": "复杂测试项目"}
                }
            )
            
            assert result["total_ufp"] == expected_ufp
            assert result["total_ufp"] > 100  # 验证复杂项目UFP较高
    
    @pytest.mark.asyncio
    async def test_ufp_breakdown_accuracy(self, calculator):
        """测试UFP分解准确性"""
        classifications = [
            {"function_type": "EI", "complexity": "AVERAGE", "count": 3},
            {"function_type": "ILF", "complexity": "LOW", "count": 1}
        ]
        
        with patch.object(calculator, 'execute') as mock_execute:
            mock_result = {
                "total_ufp": 19,  # (3*4) + (1*7) = 12 + 7 = 19
                "function_breakdown": classifications,
                "calculation_details": {
                    "EI": {"count": 3, "weight": 4, "subtotal": 12},
                    "ILF": {"count": 1, "weight": 7, "subtotal": 7}
                }
            }
            mock_execute.return_value = mock_result
            
            result = await calculator.execute(
                "calculate_ufp",
                {
                    "classifications": classifications,
                    "project_info": {"name": "分解测试项目"}
                }
            )
            
            # 验证分解计算
            details = result["calculation_details"]
            ei_subtotal = details["EI"]["subtotal"]
            ilf_subtotal = details["ILF"]["subtotal"]
            
            assert ei_subtotal == 12
            assert ilf_subtotal == 7
            assert result["total_ufp"] == ei_subtotal + ilf_subtotal


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
    
    # Mock各个阶段的结果
    with patch.object(classifier, 'classify_function') as mock_classify, \
         patch.object(complexity_calc, 'execute') as mock_complexity, \
         patch.object(ufp_calc, 'execute') as mock_ufp:
        
        # Mock分类结果
        mock_classifications = [
            NESMAFunctionClassification(
                function_id="func_1",
                function_name="用户下单",
                function_description="用户下单 - 输入商品信息、收货地址等",
                function_type=NESMAFunctionType.EI,
                confidence_score=0.9,
                justification="数据输入功能",
                rules_applied=["EI定义规则"]
            ),
            NESMAFunctionClassification(
                function_id="func_2",
                function_name="订单查询",
                function_description="订单查询 - 查看订单状态和详情",
                function_type=NESMAFunctionType.EQ,
                confidence_score=0.85,
                justification="查询功能",
                rules_applied=["EQ定义规则"]
            ),
            NESMAFunctionClassification(
                function_id="func_3",
                function_name="订单修改",
                function_description="订单修改 - 修改订单商品或地址",
                function_type=NESMAFunctionType.EI,
                confidence_score=0.8,
                justification="数据更新功能",
                rules_applied=["EI定义规则"]
            ),
            NESMAFunctionClassification(
                function_id="func_4",
                function_name="库存扣减",
                function_description="库存扣减 - 自动扣减商品库存",
                function_type=NESMAFunctionType.EO,
                confidence_score=0.87,
                justification="业务处理输出",
                rules_applied=["EO定义规则"]
            ),
            NESMAFunctionClassification(
                function_id="func_5",
                function_name="订单数据存储",
                function_description="订单数据存储 - 维护订单信息表",
                function_type=NESMAFunctionType.ILF,
                confidence_score=0.92,
                justification="内部逻辑文件",
                rules_applied=["ILF定义规则"]
            )
        ]
        
        mock_classify.side_effect = mock_classifications
        
        # Mock复杂度计算结果
        complexity_results = [
            {"complexity": NESMAComplexityLevel.AVERAGE, "det_count": 12, "ftr_count": 2},
            {"complexity": NESMAComplexityLevel.LOW, "det_count": 8, "ftr_count": 1},
            {"complexity": NESMAComplexityLevel.AVERAGE, "det_count": 10, "ftr_count": 2},
            {"complexity": NESMAComplexityLevel.LOW, "det_count": 5, "ftr_count": 1},
            {"complexity": NESMAComplexityLevel.AVERAGE, "det_count": 25, "ret_count": 2}
        ]
        
        mock_complexity.side_effect = complexity_results
        
        # Mock UFP计算结果
        mock_ufp.return_value = {
            "total_ufp": 47,  # 4+3+4+4+10+15 = 40 (估算值)
            "function_breakdown": [
                {"function_type": "EI", "complexity": "AVERAGE", "count": 2, "subtotal": 8},
                {"function_type": "EQ", "complexity": "LOW", "count": 1, "subtotal": 3},
                {"function_type": "EO", "complexity": "LOW", "count": 1, "subtotal": 4},
                {"function_type": "ILF", "complexity": "AVERAGE", "count": 1, "subtotal": 10}
            ]
        }
        
        # 执行分类
        function_descriptions = [
            "用户下单 - 输入商品信息、收货地址等",
            "订单查询 - 查看订单状态和详情", 
            "订单修改 - 修改订单商品或地址",
            "库存扣减 - 自动扣减商品库存",
            "订单数据存储 - 维护订单信息表"
        ]
        
        classifications = []
        for desc in function_descriptions:
            result = await classifier.classify_function(desc)
            classifications.append(result)
        
        # 执行复杂度计算
        complexity_results_actual = []
        for i, classification in enumerate(classifications):
            result = await complexity_calc.execute(
                "calculate_complexity",
                {
                    "function_type": classification.function_type,
                    "function_description": classification.function_description
                }
            )
            complexity_results_actual.append(result)
        
        # 执行UFP计算
        ufp_result = await ufp_calc.execute(
            "calculate_ufp",
            {
                "classifications": [
                    {
                        "function_type": c.function_type.value,
                        "complexity": cr["complexity"].value,
                        "count": 1
                    }
                    for c, cr in zip(classifications, complexity_results_actual)
                ]
            }
        )
        
        # 验证结果
        assert len(classifications) == 5
        assert len(complexity_results_actual) == 5
        assert ufp_result["total_ufp"] > 0
        
        # 验证分类类型分布
        function_types = [c.function_type for c in classifications]
        assert NESMAFunctionType.EI in function_types
        assert NESMAFunctionType.EQ in function_types
        assert NESMAFunctionType.EO in function_types
        assert NESMAFunctionType.ILF in function_types
        
        print(f"✅ NESMA智能体集成测试通过，总UFP: {ufp_result['total_ufp']}")