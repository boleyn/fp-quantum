"""
量子智能化功能点估算系统 - COSMIC数据移动分类器智能体

基于COSMIC v4.0+标准分类四种数据移动类型：Entry、Exit、Read、Write
"""

import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import SpecializedAgent
from agents.knowledge.rule_retriever import RuleRetrieverAgent
from models.cosmic_models import (
    COSMICDataMovementType, COSMICDataMovement, 
    COSMICFunctionalUser, COSMICBoundaryAnalysis
)
from models.project_models import ProcessDetails
from models.common_models import ConfidenceLevel
from config.settings import get_settings

logger = logging.getLogger(__name__)


class COSMICDataMovementClassifierAgent(SpecializedAgent):
    """COSMIC数据移动分类器智能体"""
    
    def __init__(
        self,
        rule_retriever: Optional[RuleRetrieverAgent] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        super().__init__(
            agent_id="cosmic_data_movement_classifier",
            specialty="cosmic_data_movement_classification",
            llm=llm
        )
        
        self.rule_retriever = rule_retriever
        self.settings = get_settings()
        
        # COSMIC数据移动分类规则
        self.data_movement_rules = self._load_data_movement_rules()
        self.classification_history: List[COSMICDataMovement] = []
        
    def _load_data_movement_rules(self) -> Dict[str, Any]:
        """加载COSMIC数据移动分类规则"""
        return {
            "Entry": {
                "定义": "将数据从功能用户移动到软件内部",
                "特征": [
                    "数据从外部进入软件",
                    "通常对应用户输入或外部系统发送的数据",
                    "数据移动的起点是功能用户",
                    "数据移动的终点是软件内部"
                ],
                "识别关键词": [
                    "输入", "提交", "发送", "传入", "接收",
                    "录入", "导入", "上传", "创建", "添加"
                ],
                "典型场景": [
                    "用户填写表单并提交",
                    "外部系统发送数据",
                    "文件上传操作",
                    "API接收外部数据"
                ]
            },
            "Exit": {
                "定义": "将数据从软件内部移动到功能用户",
                "特征": [
                    "数据从软件发送到外部",
                    "通常对应输出、显示或发送给外部系统的数据",
                    "数据移动的起点是软件内部",
                    "数据移动的终点是功能用户"
                ],
                "识别关键词": [
                    "输出", "显示", "发送", "传出", "生成",
                    "导出", "下载", "打印", "推送", "通知"
                ],
                "典型场景": [
                    "向用户显示查询结果",
                    "生成并发送报告",
                    "向外部系统发送数据",
                    "API返回数据给调用方"
                ]
            },
            "Read": {
                "定义": "从持久存储读取数据供功能过程使用",
                "特征": [
                    "从数据库或文件系统读取数据",
                    "数据移动的起点是持久存储",
                    "数据移动的终点是软件内部处理逻辑",
                    "通常用于查询、验证或计算"
                ],
                "识别关键词": [
                    "查询", "读取", "检索", "获取", "查找",
                    "加载", "访问", "调用", "验证", "计算"
                ],
                "典型场景": [
                    "查询数据库获取用户信息",
                    "读取配置文件",
                    "加载历史数据进行分析",
                    "验证用户权限"
                ]
            },
            "Write": {
                "定义": "将数据写入持久存储",
                "特征": [
                    "向数据库或文件系统写入数据",
                    "数据移动的起点是软件内部处理逻辑",
                    "数据移动的终点是持久存储",
                    "通常用于保存、更新或删除数据"
                ],
                "识别关键词": [
                    "保存", "存储", "写入", "更新", "修改",
                    "删除", "插入", "创建", "备份", "记录"
                ],
                "典型场景": [
                    "保存用户注册信息",
                    "更新订单状态",
                    "删除过期数据",
                    "记录操作日志"
                ]
            }
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "数据移动类型分类",
            "数据移动路径分析",
            "功能过程分解",
            "分类验证",
            "CFP预计算"
        ]
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据移动分类任务"""
        if task_name == "classify_data_movements":
            return await self.classify_data_movements(
                inputs["process_details"],
                inputs["functional_users"],
                inputs["boundary_analysis"]
            )
        elif task_name == "analyze_single_process":
            return await self.analyze_single_process(
                inputs["process_detail"],
                inputs["functional_users"],
                inputs["boundary_analysis"]
            )
        elif task_name == "validate_data_movements":
            return await self.validate_data_movements(
                inputs["data_movements"],
                inputs["process_details"]
            )
        elif task_name == "optimize_data_movement_identification":
            return await self.optimize_data_movement_identification(
                inputs["data_movements"],
                inputs["feedback"]
            )
        else:
            raise ValueError(f"未知任务: {task_name}")
    
    async def classify_data_movements(
        self,
        process_details: List[ProcessDetails],
        functional_users: List[COSMICFunctionalUser],
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> List[COSMICDataMovement]:
        """分类所有功能过程的数据移动"""
        
        all_data_movements = []
        
        for process in process_details:
            try:
                # 分析单个过程的数据移动
                process_movements = await self.analyze_single_process(
                    process,
                    functional_users,
                    boundary_analysis
                )
                
                all_data_movements.extend(process_movements)
                
            except Exception as e:
                logger.error(f"分析过程 {process.name} 时出错: {e}")
                # 使用保守估算
                fallback_movements = self._create_fallback_movements(process, functional_users)
                all_data_movements.extend(fallback_movements)
        
        # 记录分类历史
        self.classification_history.extend(all_data_movements)
        
        return all_data_movements
    
    async def analyze_single_process(
        self,
        process_detail: ProcessDetails,
        functional_users: List[COSMICFunctionalUser],
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> List[COSMICDataMovement]:
        """分析单个功能过程的数据移动"""
        
        # 1. 获取相关COSMIC规则
        cosmic_rules = await self._retrieve_cosmic_movement_rules(process_detail.description)
        
        # 2. 使用LLM分析数据移动
        identified_movements = await self._llm_identify_data_movements(
            process_detail,
            functional_users,
            boundary_analysis,
            cosmic_rules
        )
        
        # 3. 验证和完善分类结果
        validated_movements = await self._validate_and_refine_movements(
            identified_movements,
            process_detail,
            functional_users,
            boundary_analysis
        )
        
        return validated_movements
    
    async def validate_data_movements(
        self,
        data_movements: List[COSMICDataMovement],
        process_details: List[ProcessDetails]
    ) -> Dict[str, Any]:
        """验证数据移动分类结果"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "validation_issues": [],
            "suggestions": [],
            "statistics": {}
        }
        
        # 验证移动完整性
        completeness_issues = self._check_movement_completeness(data_movements, process_details)
        validation_result["validation_issues"].extend(completeness_issues)
        
        # 验证分类准确性
        accuracy_issues = await self._check_movement_accuracy(data_movements)
        validation_result["validation_issues"].extend(accuracy_issues)
        
        # 验证数据一致性
        consistency_issues = self._check_movement_consistency(data_movements)
        validation_result["validation_issues"].extend(consistency_issues)
        
        # 生成统计信息
        validation_result["statistics"] = self._generate_movement_statistics(data_movements)
        
        # 计算整体验证分数
        if validation_result["validation_issues"]:
            validation_result["is_valid"] = False
            validation_result["confidence_score"] = max(0.1,
                1.0 - len(validation_result["validation_issues"]) * 0.1
            )
        
        # 生成改进建议
        if not validation_result["is_valid"]:
            validation_result["suggestions"] = self._generate_movement_suggestions(
                validation_result["validation_issues"]
            )
        
        return validation_result
    
    async def optimize_data_movement_identification(
        self,
        data_movements: List[COSMICDataMovement],
        feedback: Dict[str, Any]
    ) -> List[COSMICDataMovement]:
        """基于反馈优化数据移动识别"""
        
        optimized_movements = []
        
        for movement in data_movements:
            # 检查是否有针对此移动的反馈
            movement_feedback = feedback.get(movement.id, {})
            
            if movement_feedback:
                # 根据反馈调整分类
                optimized_movement = await self._apply_movement_feedback(
                    movement,
                    movement_feedback
                )
                optimized_movements.append(optimized_movement)
            else:
                optimized_movements.append(movement)
        
        return optimized_movements
    
    async def _retrieve_cosmic_movement_rules(self, process_description: str) -> List[str]:
        """检索相关的COSMIC数据移动规则"""
        
        if not self.rule_retriever:
            return []
        
        query = f"COSMIC data movement Entry Exit Read Write classification {process_description}"
        
        try:
            retrieved_rules = await self.rule_retriever.execute_task(
                "retrieve_rules",
                {
                    "query": query,
                    "source_type": "COSMIC",
                    "max_results": 5
                }
            )
            
            return retrieved_rules.get("rules", [])
            
        except Exception as e:
            logger.warning(f"检索COSMIC数据移动规则失败: {e}")
            return []
    
    async def _llm_identify_data_movements(
        self,
        process_detail: ProcessDetails,
        functional_users: List[COSMICFunctionalUser],
        boundary_analysis: COSMICBoundaryAnalysis,
        cosmic_rules: List[str]
    ) -> List[COSMICDataMovement]:
        """使用LLM识别数据移动"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是COSMIC数据移动分类专家，需要识别功能过程中的所有数据移动。

COSMIC数据移动类型：
1. Entry：数据从功能用户移动到软件内部
2. Exit：数据从软件内部移动到功能用户
3. Read：从持久存储读取数据
4. Write：向持久存储写入数据

分类规则：
{data_movement_rules}

相关COSMIC规则：
{cosmic_rules}

软件边界：
{software_boundary}

持久存储边界：
{storage_boundary}

功能用户：
{functional_users}

请仔细分析功能过程，识别所有数据移动。每个数据移动必须明确指定：
- 类型（Entry/Exit/Read/Write）
- 源头（数据来源）
- 目标（数据去向）
- 数据组（移动的数据类型）

返回JSON格式：
{{
  "data_movements": [
    {{
      "id": "movement_id",
      "type": "Entry|Exit|Read|Write",
      "source": "数据源头",
      "target": "数据目标",
      "data_group": "数据组名称",
      "justification": "分类理由"
    }}
  ]
}}"""),
            ("human", """功能过程：
名称：{process_name}
描述：{process_description}
数据组：{data_groups}
依赖：{dependencies}

请识别此功能过程中的所有数据移动。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                data_movement_rules=self._format_movement_rules(),
                cosmic_rules="\n".join(cosmic_rules),
                software_boundary=boundary_analysis.software_boundary,
                storage_boundary=boundary_analysis.persistent_storage_boundary,
                functional_users="\n".join([f"- {u.name}: {u.description}" for u in functional_users]),
                process_name=process_detail.name,
                process_description=process_detail.description,
                data_groups=", ".join(process_detail.data_groups),
                dependencies=", ".join(process_detail.dependencies)
            )
        )
        
        # 解析LLM响应
        return await self._parse_data_movements_response(response.content, process_detail.id)
    
    async def _validate_and_refine_movements(
        self,
        identified_movements: List[COSMICDataMovement],
        process_detail: ProcessDetails,
        functional_users: List[COSMICFunctionalUser],
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> List[COSMICDataMovement]:
        """验证和完善数据移动分类"""
        
        validated_movements = []
        
        for movement in identified_movements:
            # 验证移动定义的合理性
            if self._is_valid_data_movement(movement, functional_users, boundary_analysis):
                validated_movements.append(movement)
            else:
                # 尝试修正移动分类
                corrected_movement = await self._correct_data_movement(
                    movement,
                    process_detail,
                    functional_users,
                    boundary_analysis
                )
                if corrected_movement:
                    validated_movements.append(corrected_movement)
        
        # 检查是否遗漏重要数据移动
        additional_movements = await self._identify_missing_movements(
            validated_movements,
            process_detail,
            functional_users
        )
        validated_movements.extend(additional_movements)
        
        return validated_movements
    
    def _format_movement_rules(self) -> str:
        """格式化数据移动规则"""
        formatted_rules = ""
        for movement_type, rules in self.data_movement_rules.items():
            formatted_rules += f"\n{movement_type}：\n"
            formatted_rules += f"定义：{rules['定义']}\n"
            formatted_rules += f"特征：{'; '.join(rules['特征'])}\n"
            formatted_rules += f"关键词：{', '.join(rules['识别关键词'])}\n"
        return formatted_rules
    
    async def _parse_data_movements_response(
        self, 
        response_content: str, 
        process_id: str
    ) -> List[COSMICDataMovement]:
        """解析LLM响应的数据移动"""
        
        try:
            import json
            # 尝试提取JSON
            if "{" in response_content and "}" in response_content:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                json_str = response_content[start:end]
                data = json.loads(json_str)
                
                movements = []
                for movement_data in data.get("data_movements", []):
                    movement = COSMICDataMovement(
                        id=movement_data.get("id", f"{process_id}_movement_{len(movements)}"),
                        type=COSMICDataMovementType(movement_data.get("type", "Entry")),
                        source=movement_data.get("source", "未知源头"),
                        target=movement_data.get("target", "未知目标"),
                        data_group=movement_data.get("data_group", "未知数据组"),
                        justification=movement_data.get("justification", "自动识别")
                    )
                    movements.append(movement)
                
                return movements
                
        except Exception as e:
            logger.warning(f"解析数据移动响应失败: {e}")
        
        # 回退方案：基于文本解析
        return self._parse_movements_from_text(response_content, process_id)
    
    def _parse_movements_from_text(self, text: str, process_id: str) -> List[COSMICDataMovement]:
        """从文本中解析数据移动"""
        
        movements = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(movement_type in line for movement_type in ["Entry", "Exit", "Read", "Write"]):
                # 尝试提取移动类型
                movement_type = None
                for mt in ["Entry", "Exit", "Read", "Write"]:
                    if mt in line:
                        movement_type = COSMICDataMovementType(mt)
                        break
                
                if movement_type:
                    movement = COSMICDataMovement(
                        id=f"{process_id}_movement_{len(movements)}",
                        type=movement_type,
                        source="待确认",
                        target="待确认",
                        data_group="待确认",
                        justification=f"从文本识别: {line}"
                    )
                    movements.append(movement)
        
        # 如果没有识别到任何移动，创建默认移动
        if not movements:
            movements = self._create_default_movements(process_id)
        
        return movements
    
    def _create_default_movements(self, process_id: str) -> List[COSMICDataMovement]:
        """创建默认数据移动"""
        return [
            COSMICDataMovement(
                id=f"{process_id}_entry_default",
                type=COSMICDataMovementType.ENTRY,
                source="功能用户",
                target="软件系统",
                data_group="输入数据",
                justification="默认Entry移动"
            ),
            COSMICDataMovement(
                id=f"{process_id}_read_default",
                type=COSMICDataMovementType.READ,
                source="持久存储",
                target="软件系统",
                data_group="存储数据",
                justification="默认Read移动"
            ),
            COSMICDataMovement(
                id=f"{process_id}_write_default",
                type=COSMICDataMovementType.WRITE,
                source="软件系统",
                target="持久存储",
                data_group="处理数据",
                justification="默认Write移动"
            ),
            COSMICDataMovement(
                id=f"{process_id}_exit_default",
                type=COSMICDataMovementType.EXIT,
                source="软件系统",
                target="功能用户",
                data_group="输出数据",
                justification="默认Exit移动"
            )
        ]
    
    def _is_valid_data_movement(
        self,
        movement: COSMICDataMovement,
        functional_users: List[COSMICFunctionalUser],
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> bool:
        """验证数据移动定义是否有效"""
        
        # 基本验证
        if not movement.source or not movement.target or not movement.data_group:
            return False
        
        # 验证移动类型与源头/目标的一致性
        if movement.type == COSMICDataMovementType.ENTRY:
            # Entry应该从功能用户到软件
            user_names = [user.name for user in functional_users]
            return any(user_name in movement.source for user_name in user_names)
        
        elif movement.type == COSMICDataMovementType.EXIT:
            # Exit应该从软件到功能用户
            user_names = [user.name for user in functional_users]
            return any(user_name in movement.target for user_name in user_names)
        
        elif movement.type == COSMICDataMovementType.READ:
            # Read应该从持久存储到软件
            return "存储" in movement.source or "数据库" in movement.source
        
        elif movement.type == COSMICDataMovementType.WRITE:
            # Write应该从软件到持久存储
            return "存储" in movement.target or "数据库" in movement.target
        
        return True
    
    async def _correct_data_movement(
        self,
        movement: COSMICDataMovement,
        process_detail: ProcessDetails,
        functional_users: List[COSMICFunctionalUser],
        boundary_analysis: COSMICBoundaryAnalysis
    ) -> Optional[COSMICDataMovement]:
        """修正数据移动分类"""
        
        # 基于移动类型修正源头和目标
        corrected_source = movement.source
        corrected_target = movement.target
        
        if movement.type == COSMICDataMovementType.ENTRY:
            if not any(user.name in movement.source for user in functional_users):
                corrected_source = functional_users[0].name if functional_users else "功能用户"
            if "软件" not in movement.target:
                corrected_target = "软件系统"
        
        elif movement.type == COSMICDataMovementType.EXIT:
            if "软件" not in movement.source:
                corrected_source = "软件系统"
            if not any(user.name in movement.target for user in functional_users):
                corrected_target = functional_users[0].name if functional_users else "功能用户"
        
        elif movement.type == COSMICDataMovementType.READ:
            if "存储" not in movement.source and "数据库" not in movement.source:
                corrected_source = "持久存储"
            if "软件" not in movement.target:
                corrected_target = "软件系统"
        
        elif movement.type == COSMICDataMovementType.WRITE:
            if "软件" not in movement.source:
                corrected_source = "软件系统"
            if "存储" not in movement.target and "数据库" not in movement.target:
                corrected_target = "持久存储"
        
        return COSMICDataMovement(
            id=movement.id,
            type=movement.type,
            source=corrected_source,
            target=corrected_target,
            data_group=movement.data_group,
            justification=f"已修正：{movement.justification}"
        )
    
    async def _identify_missing_movements(
        self,
        identified_movements: List[COSMICDataMovement],
        process_detail: ProcessDetails,
        functional_users: List[COSMICFunctionalUser]
    ) -> List[COSMICDataMovement]:
        """识别可能遗漏的数据移动"""
        
        additional_movements = []
        existing_types = {movement.type for movement in identified_movements}
        
        # 检查是否缺少常见的移动类型
        if COSMICDataMovementType.ENTRY not in existing_types:
            # 大多数功能过程都需要Entry
            entry_movement = COSMICDataMovement(
                id=f"{process_detail.id}_missing_entry",
                type=COSMICDataMovementType.ENTRY,
                source=functional_users[0].name if functional_users else "功能用户",
                target="软件系统",
                data_group="输入数据",
                justification="补充遗漏的Entry移动"
            )
            additional_movements.append(entry_movement)
        
        if COSMICDataMovementType.READ not in existing_types:
            # 大多数功能过程都需要读取数据
            read_movement = COSMICDataMovement(
                id=f"{process_detail.id}_missing_read",
                type=COSMICDataMovementType.READ,
                source="持久存储",
                target="软件系统",
                data_group="业务数据",
                justification="补充遗漏的Read移动"
            )
            additional_movements.append(read_movement)
        
        return additional_movements
    
    def _create_fallback_movements(
        self, 
        process: ProcessDetails, 
        functional_users: List[COSMICFunctionalUser]
    ) -> List[COSMICDataMovement]:
        """创建保守的数据移动估算"""
        
        # 使用保守的4个数据移动（每种类型一个）
        return [
            COSMICDataMovement(
                id=f"{process.id}_fallback_entry",
                type=COSMICDataMovementType.ENTRY,
                source=functional_users[0].name if functional_users else "用户",
                target="软件系统",
                data_group="输入数据",
                justification="保守估算 - Entry"
            ),
            COSMICDataMovement(
                id=f"{process.id}_fallback_read",
                type=COSMICDataMovementType.READ,
                source="持久存储",
                target="软件系统",
                data_group="业务数据",
                justification="保守估算 - Read"
            ),
            COSMICDataMovement(
                id=f"{process.id}_fallback_write",
                type=COSMICDataMovementType.WRITE,
                source="软件系统",
                target="持久存储",
                data_group="处理结果",
                justification="保守估算 - Write"
            ),
            COSMICDataMovement(
                id=f"{process.id}_fallback_exit",
                type=COSMICDataMovementType.EXIT,
                source="软件系统",
                target=functional_users[0].name if functional_users else "用户",
                data_group="输出数据",
                justification="保守估算 - Exit"
            )
        ]
    
    def _check_movement_completeness(
        self,
        data_movements: List[COSMICDataMovement],
        process_details: List[ProcessDetails]
    ) -> List[Dict[str, str]]:
        """检查数据移动完整性"""
        issues = []
        
        # 检查每个过程是否都有数据移动
        processes_with_movements = set()
        for movement in data_movements:
            # 从movement ID中提取process ID
            if "_" in movement.id:
                process_id = movement.id.split("_")[0]
                processes_with_movements.add(process_id)
        
        for process in process_details:
            if process.id not in processes_with_movements:
                issues.append({
                    "type": "missing_process_movements",
                    "message": f"过程 {process.name} 没有数据移动"
                })
        
        # 检查数据移动类型分布
        movement_types = [movement.type for movement in data_movements]
        type_counts = {
            COSMICDataMovementType.ENTRY: movement_types.count(COSMICDataMovementType.ENTRY),
            COSMICDataMovementType.EXIT: movement_types.count(COSMICDataMovementType.EXIT),
            COSMICDataMovementType.READ: movement_types.count(COSMICDataMovementType.READ),
            COSMICDataMovementType.WRITE: movement_types.count(COSMICDataMovementType.WRITE)
        }
        
        if type_counts[COSMICDataMovementType.ENTRY] == 0:
            issues.append({
                "type": "no_entry_movements",
                "message": "没有Entry类型的数据移动，这通常是不合理的"
            })
        
        if type_counts[COSMICDataMovementType.READ] == 0:
            issues.append({
                "type": "no_read_movements", 
                "message": "没有Read类型的数据移动，大多数应用都需要读取数据"
            })
        
        return issues
    
    async def _check_movement_accuracy(
        self, 
        data_movements: List[COSMICDataMovement]
    ) -> List[Dict[str, str]]:
        """检查数据移动分类准确性"""
        issues = []
        
        for movement in data_movements:
            # 检查Entry移动的合理性
            if movement.type == COSMICDataMovementType.ENTRY:
                if "软件" in movement.source or "系统" in movement.source:
                    issues.append({
                        "type": "invalid_entry_source",
                        "message": f"Entry移动 {movement.id} 的源头不应该是软件系统"
                    })
            
            # 检查Exit移动的合理性
            elif movement.type == COSMICDataMovementType.EXIT:
                if "存储" in movement.target or "数据库" in movement.target:
                    issues.append({
                        "type": "invalid_exit_target",
                        "message": f"Exit移动 {movement.id} 的目标不应该是存储系统"
                    })
            
            # 检查Read移动的合理性
            elif movement.type == COSMICDataMovementType.READ:
                if "用户" in movement.source:
                    issues.append({
                        "type": "invalid_read_source",
                        "message": f"Read移动 {movement.id} 的源头不应该是用户"
                    })
            
            # 检查Write移动的合理性
            elif movement.type == COSMICDataMovementType.WRITE:
                if "用户" in movement.target:
                    issues.append({
                        "type": "invalid_write_target",
                        "message": f"Write移动 {movement.id} 的目标不应该是用户"
                    })
        
        return issues
    
    def _check_movement_consistency(
        self, 
        data_movements: List[COSMICDataMovement]
    ) -> List[Dict[str, str]]:
        """检查数据移动一致性"""
        issues = []
        
        # 检查ID唯一性
        movement_ids = [movement.id for movement in data_movements]
        if len(movement_ids) != len(set(movement_ids)):
            issues.append({
                "type": "duplicate_movement_ids",
                "message": "存在重复的数据移动ID"
            })
        
        # 检查数据组命名一致性
        data_groups = [movement.data_group for movement in data_movements]
        if len(set(data_groups)) == len(data_groups):
            # 如果所有数据组都不同，可能存在命名不一致
            if len(data_groups) > 5:
                issues.append({
                    "type": "inconsistent_data_group_naming",
                    "message": "数据组命名可能不一致，建议统一命名规范"
                })
        
        return issues
    
    def _generate_movement_statistics(
        self, 
        data_movements: List[COSMICDataMovement]
    ) -> Dict[str, Any]:
        """生成数据移动统计信息"""
        
        # 按类型统计
        type_counts = {}
        for movement_type in COSMICDataMovementType:
            type_counts[movement_type.value] = len([
                m for m in data_movements if m.type == movement_type
            ])
        
        # 数据组统计
        data_groups = [movement.data_group for movement in data_movements]
        unique_data_groups = len(set(data_groups))
        
        # 源头和目标统计
        sources = [movement.source for movement in data_movements]
        targets = [movement.target for movement in data_movements]
        unique_sources = len(set(sources))
        unique_targets = len(set(targets))
        
        return {
            "total_movements": len(data_movements),
            "type_distribution": type_counts,
            "unique_data_groups": unique_data_groups,
            "unique_sources": unique_sources,
            "unique_targets": unique_targets,
            "cfp_estimate": len(data_movements)  # 每个数据移动 = 1 CFP
        }
    
    def _generate_movement_suggestions(
        self, 
        validation_issues: List[Dict[str, str]]
    ) -> List[str]:
        """生成数据移动改进建议"""
        suggestions = []
        
        issue_types = [issue["type"] for issue in validation_issues]
        
        if "missing_process_movements" in issue_types:
            suggestions.append("为所有功能过程识别数据移动，确保完整性")
        
        if "no_entry_movements" in issue_types:
            suggestions.append("添加Entry类型数据移动，分析用户输入或外部数据接收")
        
        if "no_read_movements" in issue_types:
            suggestions.append("添加Read类型数据移动，分析数据库查询和数据读取操作")
        
        if any("invalid_" in issue_type for issue_type in issue_types):
            suggestions.append("重新检查数据移动的源头和目标，确保符合COSMIC定义")
        
        if "duplicate_" in str(issue_types):
            suggestions.append("检查并修正重复的ID或命名冲突")
        
        if "inconsistent_" in str(issue_types):
            suggestions.append("统一数据组命名规范，保持一致性")
        
        return suggestions or ["数据移动分类验证通过，无需调整"]
    
    async def _apply_movement_feedback(
        self,
        movement: COSMICDataMovement,
        feedback: Dict[str, Any]
    ) -> COSMICDataMovement:
        """应用反馈调整数据移动"""
        
        # 根据反馈类型调整移动
        if feedback.get("correct_type"):
            movement.type = COSMICDataMovementType(feedback["correct_type"])
        
        if feedback.get("correct_source"):
            movement.source = feedback["correct_source"]
        
        if feedback.get("correct_target"):
            movement.target = feedback["correct_target"]
        
        if feedback.get("correct_data_group"):
            movement.data_group = feedback["correct_data_group"]
        
        # 更新理由
        movement.justification = f"基于反馈调整：{feedback.get('reason', '用户修正')}"
        
        return movement
    
    def get_classification_history(self) -> List[COSMICDataMovement]:
        """获取分类历史"""
        return self.classification_history.copy()
    
    def get_movement_rules(self) -> Dict[str, Any]:
        """获取数据移动规则"""
        return self.data_movement_rules.copy()


async def create_cosmic_data_movement_classifier(
    rule_retriever: Optional[RuleRetrieverAgent] = None,
    llm: Optional[BaseLanguageModel] = None
) -> COSMICDataMovementClassifierAgent:
    """创建COSMIC数据移动分类器智能体"""
    return COSMICDataMovementClassifierAgent(rule_retriever=rule_retriever, llm=llm)


if __name__ == "__main__":
    async def main():
        # 测试COSMIC数据移动分类器
        classifier = await create_cosmic_data_movement_classifier()
        
        # 测试数据
        test_process = ProcessDetails(
            id="process_001",
            name="用户注册",
            description="用户填写注册信息，系统验证并保存到数据库，然后发送确认邮件",
            data_groups=["用户信息", "验证数据"],
            dependencies=[]
        )
        
        test_users = [
            COSMICFunctionalUser(
                id="user_001",
                name="注册用户",
                description="需要注册账号的新用户",
                boundary_definition="用户位于软件边界外"
            )
        ]
        
        test_boundary = COSMICBoundaryAnalysis(
            software_boundary="用户注册系统",
            persistent_storage_boundary="用户数据库",
            functional_users=test_users,
            boundary_reasoning="测试边界"
        )
        
        # 分析数据移动
        movements = await classifier.analyze_single_process(
            test_process,
            test_users,
            test_boundary
        )
        
        print(f"识别到 {len(movements)} 个数据移动:")
        for movement in movements:
            print(f"- {movement.type.value}: {movement.source} -> {movement.target} ({movement.data_group})")
        
    asyncio.run(main()) 