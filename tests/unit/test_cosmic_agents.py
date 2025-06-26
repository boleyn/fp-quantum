"""
量子智能化功能点估算系统 - COSMIC智能体单元测试

测试COSMIC标准相关的所有智能体功能
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from agents.standards.cosmic.functional_user_agent import COSMICFunctionalUserAgent
from agents.standards.cosmic.data_movement_classifier import COSMICDataMovementClassifierAgent
from agents.standards.cosmic.boundary_analyzer import COSMICBoundaryAnalyzerAgent
from agents.standards.cosmic.cfp_calculator import COSMICCFPCalculatorAgent
from models.cosmic_models import (
    COSMICDataMovementType,
    COSMICBoundaryType,
    COSMICFunctionalUser,
    COSMICDataMovement,
    COSMICBoundaryAnalysis,
    COSMICEstimationResult
)
from models.project_models import ProjectInfo, TechnologyStack, BusinessDomain


class TestCOSMICAgents:
    """COSMIC智能体测试"""
    
    @pytest.fixture
    def sample_project(self):
        return ProjectInfo(
            name="测试系统",
            description="测试描述",
            technology_stack=[TechnologyStack.JAVA],
            business_domain=BusinessDomain.ECOMMERCE
        )
    
    @pytest.mark.asyncio
    async def test_functional_user_agent(self, sample_project):
        """测试功能用户识别"""
        agent = COSMICFunctionalUserAgent()
        
        with patch.object(agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "functional_users": [
                    {"id": "user1", "name": "客户", "description": "终端用户"}
                ]
            }
            
            result = await agent.execute_task(
                "identify_functional_users",
                {"project_info": sample_project}
            )
            
            assert len(result["functional_users"]) == 1


class TestCOSMICFunctionalUserAgent:
    """COSMIC功能用户识别智能体测试"""
    
    @pytest.fixture
    def functional_user_agent(self):
        """创建功能用户识别智能体"""
        return COSMICFunctionalUserAgent()
    
    @pytest.fixture
    def sample_project_info(self):
        """样本项目信息"""
        return ProjectInfo(
            name="电商平台订单管理系统",
            description="""
            构建一个电商平台的订单管理系统，主要功能包括：
            1. 客户下单：客户在前端选择商品并创建订单
            2. 订单处理：系统处理订单信息并保存到数据库
            3. 库存管理：系统调用库存服务检查和更新库存
            4. 支付处理：系统调用第三方支付接口处理支付
            5. 订单查询：客户和管理员可以查询订单状态
            """,
            technology_stack=[TechnologyStack.JAVA, TechnologyStack.MYSQL],
            business_domain=BusinessDomain.ECOMMERCE
        )
    
    @pytest.mark.asyncio
    async def test_identify_primary_functional_users(self, functional_user_agent, sample_project_info):
        """测试识别主要功能用户"""
        with patch.object(functional_user_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "functional_users": [
                    {
                        "name": "普通用户",
                        "description": "使用系统注册、登录、购买商品的终端用户",
                        "user_type": "primary",
                        "interactions": ["注册", "登录", "浏览商品", "下单", "支付"]
                    },
                    {
                        "name": "管理员",
                        "description": "系统管理员，负责用户管理、商品管理等",
                        "user_type": "primary", 
                        "interactions": ["用户管理", "商品管理", "订单管理", "系统配置"]
                    }
                ]
            }
            
            result = await functional_user_agent.execute_task(
                "identify_functional_users",
                {"project_info": sample_project_info}
            )
            
            assert len(result["functional_users"]) == 2
            assert all(user["user_type"] == "primary" for user in result["functional_users"])
            assert "普通用户" in [user["name"] for user in result["functional_users"]]
            assert "管理员" in [user["name"] for user in result["functional_users"]]
    
    @pytest.mark.asyncio
    async def test_identify_secondary_functional_users(self, functional_user_agent, sample_project_info):
        """测试识别次要功能用户"""
        with patch.object(functional_user_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "functional_users": [
                    {
                        "name": "支付系统",
                        "description": "第三方支付系统，处理支付请求和回调",
                        "user_type": "secondary",
                        "interactions": ["接收支付请求", "返回支付结果", "发送支付通知"]
                    },
                    {
                        "name": "物流系统", 
                        "description": "外部物流系统，接收发货信息",
                        "user_type": "secondary",
                        "interactions": ["接收发货通知", "返回物流状态"]
                    }
                ]
            }
            
            result = await functional_user_agent.execute_task(
                "identify_functional_users",
                {"project_info": sample_project_info}
            )
            
            assert len(result["functional_users"]) == 2
            assert all(user["user_type"] == "secondary" for user in result["functional_users"])
            assert "支付系统" in [user["name"] for user in result["functional_users"]]
    
    @pytest.mark.asyncio
    async def test_validate_functional_user_types(self, functional_user_agent, sample_project_info):
        """测试功能用户类型验证"""
        with patch.object(functional_user_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "functional_users": [
                    {
                        "name": "客户", 
                        "description": "发起业务事务的人类用户",
                        "user_type": "primary",
                        "interactions": ["业务操作"]
                    },
                    {
                        "name": "数据库",
                        "description": "存储业务数据的持久存储",
                        "user_type": "storage",
                        "interactions": ["数据读写"]
                    }
                ]
            }
            
            result = await functional_user_agent.execute_task(
                "identify_functional_users",
                {"project_info": sample_project_info}
            )
            
            # 验证功能用户类型的正确性
            primary_users = [u for u in result["functional_users"] if u["user_type"] == "primary"]
            storage_users = [u for u in result["functional_users"] if u["user_type"] == "storage"]
            
            assert len(primary_users) >= 1  # 至少要有一个主要功能用户
            assert len(storage_users) >= 0  # 存储类用户可选


class TestCOSMICBoundaryAnalyzer:
    """COSMIC边界分析器测试"""
    
    @pytest.fixture
    def boundary_agent(self):
        """创建边界分析器智能体"""
        return COSMICBoundaryAnalyzerAgent()
    
    @pytest.fixture
    def sample_functional_users(self):
        """样本功能用户"""
        return [
            {
                "id": "customer",
                "name": "客户",
                "description": "电商平台的终端用户",
                "boundary_definition": "通过Web界面与系统交互"
            },
            {
                "id": "payment_service",
                "name": "支付服务",
                "description": "第三方支付处理系统",
                "boundary_definition": "通过API接口与系统交互"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_analyze_software_boundary(self, boundary_agent, sample_functional_users):
        """测试软件边界分析"""
        with patch.object(boundary_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "software_boundary": "订单管理系统包含订单处理、库存管理、支付集成等核心功能模块",
                "persistent_storage_boundary": "订单数据库、用户数据库作为持久存储边界",
                "boundary_components": [
                    {
                        "component": "订单处理模块",
                        "type": "core_function",
                        "description": "处理订单创建、修改、取消等核心业务逻辑"
                    },
                    {
                        "component": "数据访问层",
                        "type": "data_interface",
                        "description": "与数据库交互的接口层"
                    },
                    {
                        "component": "API网关",
                        "type": "external_interface",
                        "description": "与外部系统交互的接口"
                    }
                ],
                "boundary_reasoning": "基于功能职责和数据流动确定软件边界",
                "data_flow_analysis": "分析了数据在系统内外的流动路径"
            }
            
            project_info = ProjectInfo(
                name="订单管理系统",
                description="处理电商订单的核心系统",
                technology_stack=[TechnologyStack.JAVA],
                business_domain=BusinessDomain.ECOMMERCE
            )
            
            result = await boundary_agent.execute_task(
                "analyze_boundaries",
                {
                    "project_info": project_info,
                    "functional_users": sample_functional_users
                }
            )
            
            assert "software_boundary" in result
            assert "persistent_storage_boundary" in result
            assert len(result["boundary_components"]) >= 3
            assert "boundary_reasoning" in result
    
    @pytest.mark.asyncio
    async def test_persistent_storage_identification(self, boundary_agent):
        """测试持久存储识别"""
        test_cases = [
            {
                "system_type": "数据库系统",
                "expected_storage": ["关系数据库", "缓存系统", "文件存储"],
                "description": "包含多种存储类型的系统"
            },
            {
                "system_type": "微服务系统",
                "expected_storage": ["服务A数据库", "服务B数据库", "共享缓存"],
                "description": "微服务架构的存储边界"
            }
        ]
        
        for case in test_cases:
            with patch.object(boundary_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = {
                    "software_boundary": f"{case['system_type']}的软件边界",
                    "persistent_storage_boundary": f"识别的持久存储：{', '.join(case['expected_storage'])}",
                    "boundary_components": [],
                    "storage_types": case["expected_storage"],
                    "boundary_reasoning": f"针对{case['system_type']}的存储边界分析"
                }
                
                project_info = ProjectInfo(
                    name=case["system_type"],
                    description=case["description"],
                    technology_stack=[TechnologyStack.JAVA],
                    business_domain=BusinessDomain.OTHER
                )
                
                result = await boundary_agent.execute_task(
                    "analyze_boundaries",
                    {"project_info": project_info, "functional_users": []}
                )
                
                assert "persistent_storage_boundary" in result
                assert len(result.get("storage_types", [])) == len(case["expected_storage"])


class TestCOSMICDataMovementClassifier:
    """COSMIC数据移动分类器测试"""
    
    @pytest.fixture
    def movement_agent(self):
        """创建数据移动分类器智能体"""
        return COSMICDataMovementClassifierAgent()
    
    @pytest.fixture
    def sample_boundary_analysis(self):
        """样本边界分析结果"""
        return {
            "software_boundary": "订单管理系统核心功能",
            "persistent_storage_boundary": "订单数据库、用户数据库",
            "boundary_components": []
        }
    
    @pytest.fixture
    def sample_processes(self):
        """样本业务流程"""
        return [
            {
                "name": "客户下单",
                "description": "客户通过Web界面输入订单信息，系统处理并保存到数据库",
                "expected_movements": [
                    {"type": "Entry", "description": "客户订单信息进入系统"},
                    {"type": "Write", "description": "订单数据写入订单数据库"}
                ]
            },
            {
                "name": "订单查询",
                "description": "客户查询订单状态，系统从数据库读取并显示",
                "expected_movements": [
                    {"type": "Entry", "description": "查询请求进入系统"},
                    {"type": "Read", "description": "从订单数据库读取数据"},
                    {"type": "Exit", "description": "订单信息返回给客户"}
                ]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_classify_data_movements(self, movement_agent, sample_boundary_analysis, sample_processes):
        """测试数据移动分类"""
        for process in sample_processes:
            with patch.object(movement_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = {
                    "data_movements": [
                        {
                            "id": f"movement_{i}",
                            "type": movement["type"],
                            "source": "客户" if movement["type"] == "Entry" else "系统",
                            "target": "系统" if movement["type"] == "Entry" else "数据库",
                            "data_group": "订单信息",
                            "description": movement["description"],
                            "justification": f"{movement['type']}类型的数据移动识别"
                        }
                        for i, movement in enumerate(process["expected_movements"])
                    ],
                    "classification_reasoning": f"基于{process['name']}流程分析识别数据移动",
                    "boundary_validation": "所有数据移动都符合边界定义"
                }
                
                result = await movement_agent.execute_task(
                    "classify_data_movements",
                    {
                        "process_description": process["description"],
                        "boundary_analysis": sample_boundary_analysis
                    }
                )
                
                assert len(result["data_movements"]) == len(process["expected_movements"])
                
                classified_types = [mov["type"] for mov in result["data_movements"]]
                expected_types = [mov["type"] for mov in process["expected_movements"]]
                assert classified_types == expected_types
                
                assert all("justification" in mov for mov in result["data_movements"])
    
    @pytest.mark.asyncio
    async def test_all_movement_types(self, movement_agent, sample_boundary_analysis):
        """测试所有数据移动类型"""
        movement_scenarios = [
            {
                "type": "Entry",
                "scenario": "用户登录输入用户名密码",
                "expected_source": "用户",
                "expected_target": "认证系统"
            },
            {
                "type": "Exit", 
                "scenario": "系统生成PDF报表并下载",
                "expected_source": "报表生成器",
                "expected_target": "用户"
            },
            {
                "type": "Read",
                "scenario": "系统查询用户权限信息",
                "expected_source": "权限数据库",
                "expected_target": "权限管理模块"
            },
            {
                "type": "Write",
                "scenario": "系统保存用户操作日志",
                "expected_source": "日志收集器",
                "expected_target": "日志数据库"
            }
        ]
        
        for scenario in movement_scenarios:
            with patch.object(movement_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = {
                    "data_movements": [{
                        "id": "test_movement",
                        "type": scenario["type"],
                        "source": scenario["expected_source"],
                        "target": scenario["expected_target"],
                        "data_group": "测试数据组",
                        "description": scenario["scenario"],
                        "justification": f"{scenario['type']}类型数据移动"
                    }],
                    "classification_reasoning": f"识别{scenario['type']}类型数据移动",
                    "boundary_validation": "符合COSMIC边界规则"
                }
                
                result = await movement_agent.execute_task(
                    "classify_data_movements",
                    {
                        "process_description": scenario["scenario"],
                        "boundary_analysis": sample_boundary_analysis
                    }
                )
                
                movement = result["data_movements"][0]
                assert movement["type"] == scenario["type"]
                assert movement["source"] == scenario["expected_source"]
                assert movement["target"] == scenario["expected_target"]


class TestCOSMICCFPCalculator:
    """COSMIC CFP计算器测试"""
    
    @pytest.fixture
    def cfp_agent(self):
        """创建CFP计算器智能体"""
        return COSMICCFPCalculatorAgent()
    
    @pytest.fixture
    def sample_data_movements(self):
        """样本数据移动列表"""
        return [
            {
                "id": "entry_1",
                "type": "Entry",
                "source": "客户",
                "target": "订单系统",
                "data_group": "订单信息",
                "description": "客户输入订单信息"
            },
            {
                "id": "write_1",
                "type": "Write",
                "source": "订单系统",
                "target": "订单数据库",
                "data_group": "订单数据",
                "description": "保存订单到数据库"
            },
            {
                "id": "read_1",
                "type": "Read",
                "source": "库存数据库",
                "target": "库存系统",
                "data_group": "库存信息",
                "description": "读取商品库存"
            },
            {
                "id": "exit_1",
                "type": "Exit",
                "source": "订单系统",
                "target": "客户",
                "data_group": "订单确认",
                "description": "返回订单确认信息"
            },
            {
                "id": "entry_2",
                "type": "Entry",
                "source": "支付网关",
                "target": "支付系统",
                "data_group": "支付结果",
                "description": "接收支付结果"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_calculate_total_cfp(self, cfp_agent, sample_data_movements):
        """测试CFP总计算"""
        expected_cfp = len(sample_data_movements)  # 每个数据移动 = 1 CFP
        
        with patch.object(cfp_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "total_cfp": expected_cfp,
                "movement_breakdown": [
                    {
                        "movement_id": movement["id"],
                        "type": movement["type"],
                        "data_group": movement["data_group"],
                        "cfp_value": 1,  # 每个数据移动1 CFP
                        "description": movement["description"]
                    }
                    for movement in sample_data_movements
                ],
                "type_summary": {
                    "Entry": {"count": 2, "total_cfp": 2},
                    "Exit": {"count": 1, "total_cfp": 1},
                    "Read": {"count": 1, "total_cfp": 1},
                    "Write": {"count": 1, "total_cfp": 1}
                },
                "calculation_details": {
                    "cosmic_version": "4.0.1",
                    "calculation_method": "1_CFP_per_movement",
                    "total_movements": expected_cfp
                },
                "functional_processes": [
                    {
                        "process_name": "客户下单流程",
                        "movements": ["entry_1", "write_1", "exit_1"],
                        "cfp_subtotal": 3
                    },
                    {
                        "process_name": "库存检查流程", 
                        "movements": ["read_1"],
                        "cfp_subtotal": 1
                    },
                    {
                        "process_name": "支付处理流程",
                        "movements": ["entry_2"],
                        "cfp_subtotal": 1
                    }
                ]
            }
            
            result = await cfp_agent.execute_task(
                "calculate_cfp",
                {
                    "data_movements": sample_data_movements,
                    "functional_processes": []
                }
            )
            
            assert result["total_cfp"] == expected_cfp
            assert len(result["movement_breakdown"]) == len(sample_data_movements)
            assert result["type_summary"]["Entry"]["count"] == 2
            assert result["type_summary"]["Exit"]["count"] == 1
            assert result["type_summary"]["Read"]["count"] == 1
            assert result["type_summary"]["Write"]["count"] == 1
            assert "calculation_details" in result
    
    @pytest.mark.asyncio
    async def test_functional_process_grouping(self, cfp_agent):
        """测试功能过程分组"""
        movements_by_process = {
            "用户注册流程": [
                {"id": "reg_entry", "type": "Entry", "data_group": "用户信息"},
                {"id": "reg_write", "type": "Write", "data_group": "用户数据"},
                {"id": "reg_exit", "type": "Exit", "data_group": "注册结果"}
            ],
            "用户查询流程": [
                {"id": "query_entry", "type": "Entry", "data_group": "查询条件"},
                {"id": "query_read", "type": "Read", "data_group": "用户数据"},
                {"id": "query_exit", "type": "Exit", "data_group": "查询结果"}
            ]
        }
        
        all_movements = []
        expected_processes = []
        
        for process_name, movements in movements_by_process.items():
            all_movements.extend(movements)
            expected_processes.append({
                "process_name": process_name,
                "movements": [mov["id"] for mov in movements],
                "cfp_subtotal": len(movements)
            })
        
        with patch.object(cfp_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "total_cfp": len(all_movements),
                "movement_breakdown": [
                    {
                        "movement_id": mov["id"],
                        "type": mov["type"],
                        "data_group": mov["data_group"],
                        "cfp_value": 1
                    }
                    for mov in all_movements
                ],
                "type_summary": {
                    "Entry": {"count": 2, "total_cfp": 2},
                    "Exit": {"count": 2, "total_cfp": 2},
                    "Read": {"count": 1, "total_cfp": 1},
                    "Write": {"count": 1, "total_cfp": 1}
                },
                "functional_processes": expected_processes,
                "calculation_details": {"cosmic_version": "4.0.1"}
            }
            
            result = await cfp_agent.execute_task(
                "calculate_cfp",
                {
                    "data_movements": all_movements,
                    "functional_processes": []
                }
            )
            
            assert len(result["functional_processes"]) == 2
            for process in result["functional_processes"]:
                assert process["process_name"] in movements_by_process.keys()
                assert process["cfp_subtotal"] == 3  # 每个流程3个数据移动


@pytest.mark.asyncio
async def test_cosmic_agent_integration():
    """测试COSMIC智能体集成"""
    
    # 创建智能体实例
    functional_user_agent = COSMICFunctionalUserAgent()
    boundary_agent = COSMICBoundaryAnalyzerAgent()
    movement_agent = COSMICDataMovementClassifierAgent()
    cfp_agent = COSMICCFPCalculatorAgent()
    
    # 样本项目信息
    project_info = ProjectInfo(
        name="电商订单系统",
        description="处理电商订单的完整流程",
        technology_stack=[TechnologyStack.JAVA],
        business_domain=BusinessDomain.ECOMMERCE
    )
    
    # 1. 功能用户识别
    with patch.object(functional_user_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "functional_users": [
                {"id": "customer", "name": "客户", "description": "下单用户", "boundary_definition": "Web界面"},
                {"id": "admin", "name": "管理员", "description": "系统管理员", "boundary_definition": "管理后台"}
            ],
            "identification_reasoning": "基于业务流程识别功能用户"
        }
        
        functional_users_result = await functional_user_agent.execute_task(
            "identify_functional_users",
            {"project_info": project_info}
        )
    
    # 2. 边界分析
    with patch.object(boundary_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "software_boundary": "订单管理系统核心模块",
            "persistent_storage_boundary": "订单数据库和用户数据库",
            "boundary_components": [],
            "boundary_reasoning": "基于功能职责划分边界"
        }
        
        boundary_result = await boundary_agent.execute_task(
            "analyze_boundaries",
            {
                "project_info": project_info,
                "functional_users": functional_users_result["functional_users"]
            }
        )
    
    # 3. 数据移动分类
    test_processes = [
        "客户下单：输入订单信息，系统保存到数据库",
        "订单查询：查询订单状态并显示结果"
    ]
    
    all_movements = []
    for process_desc in test_processes:
        with patch.object(movement_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            if "下单" in process_desc:
                movements = [
                    {"id": "order_entry", "type": "Entry", "data_group": "订单信息"},
                    {"id": "order_write", "type": "Write", "data_group": "订单数据"}
                ]
            else:
                movements = [
                    {"id": "query_entry", "type": "Entry", "data_group": "查询条件"},
                    {"id": "query_read", "type": "Read", "data_group": "订单数据"},
                    {"id": "query_exit", "type": "Exit", "data_group": "查询结果"}
                ]
            
            mock_llm.return_value = {
                "data_movements": movements,
                "classification_reasoning": f"分析{process_desc}的数据移动"
            }
            
            movement_result = await movement_agent.execute_task(
                "classify_data_movements",
                {
                    "process_description": process_desc,
                    "boundary_analysis": boundary_result
                }
            )
            all_movements.extend(movement_result["data_movements"])
    
    # 4. CFP计算
    with patch.object(cfp_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
        expected_cfp = len(all_movements)
        mock_llm.return_value = {
            "total_cfp": expected_cfp,
            "movement_breakdown": [
                {"movement_id": mov["id"], "type": mov["type"], "cfp_value": 1}
                for mov in all_movements
            ],
            "type_summary": {
                "Entry": {"count": 2, "total_cfp": 2},
                "Exit": {"count": 1, "total_cfp": 1},
                "Read": {"count": 1, "total_cfp": 1},
                "Write": {"count": 1, "total_cfp": 1}
            },
            "calculation_details": {"cosmic_version": "4.0.1"}
        }
        
        cfp_result = await cfp_agent.execute_task(
            "calculate_cfp",
            {
                "data_movements": all_movements,
                "functional_processes": []
            }
        )
    
    # 验证集成结果
    assert len(functional_users_result["functional_users"]) == 2
    assert "software_boundary" in boundary_result
    assert len(all_movements) == 5  # 2 + 3 个数据移动
    assert cfp_result["total_cfp"] == 5
    
    print(f"✅ COSMIC智能体集成测试通过，总CFP: {cfp_result['total_cfp']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 