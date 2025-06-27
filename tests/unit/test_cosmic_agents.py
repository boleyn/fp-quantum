"""
量子智能化功能点估算系统 - COSMIC智能体单元测试

测试COSMIC标准相关的所有智能体功能
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from datetime import datetime

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
from models.common_models import ConfidenceLevel


class TestCOSMICAgents:
    """COSMIC智能体基础测试"""
    
    @pytest.fixture
    def sample_project(self):
        return ProjectInfo(
            name="测试系统",
            description="这是一个用于测试功能点估算的电商系统，包含订单管理、用户管理、支付处理等核心功能模块",
            technology_stack=[TechnologyStack.JAVA],
            business_domain=BusinessDomain.ECOMMERCE
        )
    
    @pytest.fixture
    def functional_user_agent(self):
        return COSMICFunctionalUserAgent()
    
    @pytest.fixture
    def boundary_agent(self):
        return COSMICBoundaryAnalyzerAgent()
    
    @pytest.fixture
    def movement_agent(self):
        return COSMICDataMovementClassifierAgent()
    
    @pytest.fixture
    def cfp_agent(self):
        return COSMICCFPCalculatorAgent()
    
    def test_functional_user_agent(self, functional_user_agent):
        """测试功能用户智能体实例化"""
        assert functional_user_agent.agent_id == "cosmic_functional_user_agent"
        assert functional_user_agent.specialty == "cosmic_functional_user_identification"
        assert "功能用户识别" in functional_user_agent._get_capabilities()


class TestCOSMICFunctionalUserAgent:
    """COSMIC功能用户智能体测试"""
    
    @pytest.fixture
    def functional_user_agent(self):
        return COSMICFunctionalUserAgent()
    
    @pytest.fixture 
    def sample_project_info(self):
        return ProjectInfo(
            name="电商订单系统",
            description="这是一个完整的电商订单管理系统，包含用户注册登录、商品浏览、订单创建、支付处理、物流跟踪等功能模块",
            technology_stack=[TechnologyStack.JAVA],
            business_domain=BusinessDomain.ECOMMERCE
        )
    
    @pytest.mark.asyncio
    async def test_identify_primary_functional_users(self, functional_user_agent, sample_project_info):
        """测试识别主要功能用户"""
        with patch.object(functional_user_agent, 'execute') as mock_execute:
            # 模拟识别结果
            mock_execute.return_value = [
                COSMICFunctionalUser(
                    user_id="customer",
                    name="客户",
                    description="电商平台的终端用户",
                    user_type="人员",
                    boundary_definition="通过Web界面与系统交互",
                    interaction_scope="Web前端界面",
                    identification_confidence=0.9,
                    identification_reasoning="基于用户需求文档识别的主要用户"
                ),
                COSMICFunctionalUser(
                    user_id="payment_service",
                    name="支付服务",
                    description="第三方支付处理系统",
                    user_type="系统",
                    boundary_definition="通过API接口与系统交互",
                    interaction_scope="API接口",
                    identification_confidence=0.8,
                    identification_reasoning="基于系统架构分析识别的外部系统"
                )
            ]
            
            # 调用测试方法
            result = await functional_user_agent.execute("identify_functional_users", {
                "project_info": sample_project_info
            })
            
            # 验证结果
            assert len(result) == 2
            assert any(user.user_id == "customer" for user in result)
            assert any(user.user_id == "payment_service" for user in result)
    
    @pytest.mark.asyncio
    async def test_identify_secondary_functional_users(self, functional_user_agent, sample_project_info):
        """测试识别次要功能用户"""
        with patch.object(functional_user_agent, 'execute') as mock_execute:
            mock_execute.return_value = [
                COSMICFunctionalUser(
                    user_id="admin",
                    name="系统管理员",
                    description="管理系统配置和用户权限",
                    user_type="人员",
                    boundary_definition="通过管理后台界面操作",
                    interaction_scope="管理后台界面",
                    identification_confidence=0.9,
                    identification_reasoning="基于管理功能需求识别的重要用户"
                ),
                COSMICFunctionalUser(
                    user_id="logistics_system",
                    name="物流系统",
                    description="处理订单配送和跟踪",
                    user_type="外部系统",
                    boundary_definition="通过API接口交换物流信息",
                    interaction_scope="RESTful API接口",
                    identification_confidence=0.8,
                    identification_reasoning="外部集成系统，处理物流相关业务"
                )
            ]
            
            result = await functional_user_agent.execute(
                project_info=sample_project_info,
                user_type="secondary"
            )
            
            assert len(result) == 2
            assert result[0].name == "系统管理员"
            assert result[0].user_type == "人员"
            assert result[1].name == "物流系统"
            assert result[1].user_type == "外部系统"
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_functional_user_types(self, functional_user_agent, sample_project_info):
        """测试功能用户类型验证"""
        with patch.object(functional_user_agent, 'execute') as mock_execute:
            mock_functional_users = [
                COSMICFunctionalUser(
                    user_id="customer",
                    name="客户",
                    description="电商平台用户",
                    user_type="人员",
                    boundary_definition="Web界面交互",
                    interaction_scope="Web前端界面",
                    identification_confidence=0.95,
                    identification_reasoning="基于用户需求分析识别的主要用户类型"
                )
            ]
            
            mock_execute.return_value = {
                "is_valid": True,
                "confidence_score": 0.95,
                "validation_issues": [],
                "suggestions": []
            }
            
            result = await functional_user_agent.execute(
                functional_users=mock_functional_users,
                project_description=sample_project_info.description
            )
            
            assert result["is_valid"] == True
            assert result["confidence_score"] > 0.9


class TestCOSMICBoundaryAnalyzer:
    """COSMIC边界分析器测试"""
    
    @pytest.fixture
    def boundary_agent(self):
        return COSMICBoundaryAnalyzerAgent()
    
    @pytest.fixture
    def sample_functional_users(self):
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
        with patch.object(boundary_agent, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {
                "boundary_definition": "订单管理系统核心功能",
                "software_components": [
                    {"name": "订单服务", "type": "业务逻辑"},
                    {"name": "用户服务", "type": "业务逻辑"}
                ],
                "validation_result": {
                    "is_valid": True,
                    "issues": []
                }
            }
            
            project_info = ProjectInfo(
                name="电商系统",
                description="电商订单管理系统，提供完整的订单处理流程包括订单创建、支付、配送跟踪等功能",
                technology_stack=[TechnologyStack.JAVA],
                business_domain=BusinessDomain.ECOMMERCE
            )
            
            result = await boundary_agent.execute("analyze_software_boundary", {
                "project_info": project_info,
                "functional_users": sample_functional_users
            })
            
            assert "boundary_definition" in result
            assert "software_components" in result
            assert result["validation_result"]["is_valid"] == True
    
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
            with patch.object(boundary_agent, 'execute', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = {
                    "storage_components": [
                        {"name": storage, "type": "持久存储"}
                        for storage in case["expected_storage"]
                    ],
                    "storage_boundary": f"{case['system_type']}的存储边界",
                    "access_patterns": []
                }
                
                project_info = ProjectInfo(
                    name=case["system_type"],
                    description=case["description"] + "，需要分析其持久存储边界和访问模式",
                    technology_stack=[TechnologyStack.JAVA],
                    business_domain=BusinessDomain.ECOMMERCE
                )
                
                result = await boundary_agent.execute("analyze_storage_boundary", {
                    "project_info": project_info,
                    "architecture_info": {}
                })
                
                assert "storage_components" in result
                assert len(result["storage_components"]) == len(case["expected_storage"])


class TestCOSMICDataMovementClassifier:
    """COSMIC数据移动分类器测试"""
    
    @pytest.fixture
    def movement_agent(self):
        return COSMICDataMovementClassifierAgent()
    
    @pytest.fixture
    def sample_boundary_analysis(self):
        return {
            "software_boundary": "订单管理系统核心功能",
            "persistent_storage_boundary": "订单数据库、用户数据库",
            "boundary_components": []
        }
    
    @pytest.fixture
    def sample_processes(self):
        return [
            {
                "name": "订单创建",
                "description": "客户通过Web界面输入订单信息，系统处理并保存到数据库",
                "expected_movements": [
                    {"type": "Entry", "description": "客户订单信息进入系统"},
                    {"type": "Write", "description": "订单数据写入数据库"},
                    {"type": "Exit", "description": "订单确认信息返回给客户"}
                ]
            },
            {
                "name": "订单查询", 
                "description": "客户查询订单状态，系统从数据库读取并返回信息",
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
            with patch.object(movement_agent, 'execute', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = {
                    "data_movements": [
                        {
                            "id": f"{process['name']}_movement_{i}",
                            "type": movement["type"],
                            "description": movement["description"],
                            "data_group": f"数据组{i+1}",
                            "source": "测试源",
                            "target": "测试目标"
                        }
                        for i, movement in enumerate(process["expected_movements"])
                    ],
                    "classification_confidence": 0.9,
                    "validation_issues": []
                }
                
                result = await movement_agent.execute("classify_data_movements", {
                    "process_info": process,
                    "boundary_analysis": sample_boundary_analysis
                })
                
                assert "data_movements" in result
                assert len(result["data_movements"]) == len(process["expected_movements"])
                assert result["classification_confidence"] > 0.8
    
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
            with patch.object(movement_agent, 'execute', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = {
                    "data_movements": [
                        {
                            "id": f"{scenario['type']}_test",
                            "type": scenario["type"],
                            "scenario": scenario["scenario"],
                            "source": scenario["expected_source"],
                            "target": scenario["expected_target"],
                            "data_group": "测试数据组"
                        }
                    ],
                    "movement_analysis": {
                        "type_distribution": {scenario["type"]: 1},
                        "complexity_score": 0.5
                    }
                }
                
                result = await movement_agent.execute("classify_data_movements", {
                    "process_info": {"scenario": scenario["scenario"]},
                    "boundary_analysis": sample_boundary_analysis
                })
                
                movements = result["data_movements"]
                assert len(movements) == 1
                assert movements[0]["type"] == scenario["type"]
                assert movements[0]["source"] == scenario["expected_source"]
                assert movements[0]["target"] == scenario["expected_target"]


class TestCOSMICCFPCalculator:
    """COSMIC CFP计算器测试"""
    
    @pytest.fixture
    def cfp_agent(self):
        return COSMICCFPCalculatorAgent()
    
    @pytest.fixture
    def sample_data_movements(self):
        return [
            {
                "id": "entry_1",
                "type": "Entry",
                "description": "客户输入订单信息",
                "data_group": "订单信息",
                "source": "客户",
                "target": "订单系统"
            },
            {
                "id": "write_1", 
                "type": "Write",
                "description": "订单数据写入数据库",
                "data_group": "订单数据",
                "source": "订单系统",
                "target": "订单数据库"
            },
            {
                "id": "exit_1",
                "type": "Exit", 
                "description": "订单确认信息返回",
                "data_group": "确认信息",
                "source": "订单系统",
                "target": "客户"
            },
            {
                "id": "entry_2",
                "type": "Entry",
                "description": "接收支付结果",
                "data_group": "支付结果",
                "source": "支付网关",
                "target": "订单系统"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_calculate_total_cfp(self, cfp_agent, sample_data_movements):
        """测试CFP总计算"""
        expected_cfp = len(sample_data_movements)  # 每个数据移动 = 1 CFP
        
        with patch.object(cfp_agent, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {
                "total_cfp": expected_cfp,
                "movement_breakdown": {
                    movement["id"]: 1 for movement in sample_data_movements
                },
                "calculation_details": {
                    "total_movements": len(sample_data_movements),
                    "cfp_per_movement": 1
                }
            }
            
            result = await cfp_agent.execute("calculate_cfp", {
                "data_movements": sample_data_movements
            })
            
            assert result["total_cfp"] == expected_cfp
            assert len(result["movement_breakdown"]) == len(sample_data_movements)
    
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
        
        with patch.object(cfp_agent, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {
                "functional_processes": expected_processes,
                "total_cfp": len(all_movements),
                "process_distribution": {
                    process["process_name"]: process["cfp_subtotal"]
                    for process in expected_processes
                }
            }
            
            result = await cfp_agent.execute("calculate_cfp", {
                "data_movements": all_movements,
                "group_by_process": True
            })
            
            assert "functional_processes" in result
            assert len(result["functional_processes"]) == 2
            assert result["total_cfp"] == 6  # 3 + 3 movements


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
        description="处理电商订单的完整流程，包含用户管理、商品管理、订单处理、支付集成等核心业务功能",
        technology_stack=[TechnologyStack.JAVA],
        business_domain=BusinessDomain.ECOMMERCE
    )
    
    # 1. 功能用户识别
    with patch.object(functional_user_agent, 'execute', new_callable=AsyncMock) as mock_user_agent:
        mock_user_agent.return_value = [
            COSMICFunctionalUser(
                user_id="customer",
                name="客户",
                description="电商用户",
                user_type="人员",
                boundary_definition="Web界面",
                interaction_scope="前端Web界面",
                identification_confidence=0.9,
                identification_reasoning="基于电商系统需求分析识别的主要用户"
            )
        ]
        
        functional_users = await functional_user_agent.execute("identify_functional_users", {
            "project_info": project_info
        })
        
        assert len(functional_users) == 1
        assert functional_users[0].user_id == "customer"
    
    # 2. 边界分析
    with patch.object(boundary_agent, 'execute', new_callable=AsyncMock) as mock_boundary_agent:
        mock_boundary_analysis = {
            "boundary_definition": "电商订单系统边界",
            "software_components": [{"name": "订单服务", "type": "核心服务"}],
            "validation_result": {"is_valid": True}
        }
        mock_boundary_agent.return_value = mock_boundary_analysis
        
        boundary_analysis = await boundary_agent.execute("analyze_software_boundary", {
            "project_info": project_info,
            "functional_users": functional_users
        })
        
        assert "boundary_definition" in boundary_analysis
    
    # 3. 数据移动分类
    with patch.object(movement_agent, 'execute', new_callable=AsyncMock) as mock_movement_agent:
        mock_movements = {
            "data_movements": [
                {"id": "entry_1", "type": "Entry", "data_group": "订单数据"},
                {"id": "write_1", "type": "Write", "data_group": "订单数据"}
            ]
        }
        mock_movement_agent.return_value = mock_movements
        
        data_movements = await movement_agent.execute("classify_data_movements", {
            "process_info": {"name": "订单处理"},
            "boundary_analysis": boundary_analysis
        })
        
        assert len(data_movements["data_movements"]) == 2
    
    # 4. CFP计算
    with patch.object(cfp_agent, 'execute', new_callable=AsyncMock) as mock_cfp_agent:
        mock_cfp_result = {
            "total_cfp": 2,
            "movement_breakdown": {"entry_1": 1, "write_1": 1}
        }
        mock_cfp_agent.return_value = mock_cfp_result
        
        cfp_result = await cfp_agent.execute("calculate_cfp", {
            "data_movements": data_movements["data_movements"]
        })
        
        assert cfp_result["total_cfp"] == 2
        assert "movement_breakdown" in cfp_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 