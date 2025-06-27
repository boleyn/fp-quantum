"""
量子智能化功能点估算系统 - 流程识别智能体

从需求解析结果中识别独立的功能流程
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import time
import re

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base.base_agent import SpecializedAgent
from models.project_models import ProjectInfo, ProcessDetails
from models.common_models import ConfidenceLevel, ValidationResult
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ProcessIdentifierAgent(SpecializedAgent):
    """流程识别智能体"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        super().__init__(
            agent_id="process_identifier",
            specialty="business_process_analysis",
            llm=llm
        )
        
        self.settings = get_settings()
        
        # 流程识别模式和规则
        self.process_patterns = self._load_process_patterns()
        self.boundary_rules = self._load_boundary_rules()
        
    def _load_process_patterns(self) -> Dict[str, List[str]]:
        """加载流程识别模式"""
        return {
            "触发词": [
                "当", "如果", "在...情况下", "需要", "要求",
                "用户", "系统", "管理员", "操作员"
            ],
            "动作词": [
                "创建", "新增", "添加", "录入", "输入",
                "查询", "搜索", "检索", "浏览", "查看",
                "修改", "编辑", "更新", "变更", "调整",
                "删除", "移除", "取消", "撤销", "清除",
                "审核", "审批", "确认", "验证", "检查",
                "生成", "产生", "创建", "输出", "导出",
                "发送", "传输", "通知", "提醒", "推送"
            ],
            "结果词": [
                "完成", "成功", "失败", "错误",
                "保存", "存储", "记录", "更新",
                "通知", "提醒", "反馈", "确认"
            ],
            "数据对象": [
                "信息", "数据", "记录", "文件", "报表",
                "订单", "用户", "产品", "客户", "项目",
                "账户", "权限", "配置", "参数", "设置"
            ]
        }
    
    def _load_boundary_rules(self) -> Dict[str, Any]:
        """加载流程边界识别规则"""
        return {
            "独立性原则": [
                "每个流程有明确的开始和结束",
                "流程内部逻辑完整，可独立执行",
                "流程有明确的输入和输出",
                "流程处理特定的业务场景"
            ],
            "边界标识": [
                "用户操作触发点",
                "系统响应结束点",
                "数据状态变化点",
                "业务规则检查点"
            ],
            "分割标准": [
                "不同的业务目标",
                "不同的用户角色",
                "不同的数据主体",
                "不同的处理逻辑"
            ],
            "合并条件": [
                "相同的业务目标",
                "连续的操作序列",
                "紧密的数据关联",
                "不可分割的事务"
            ]
        }
    
    def _get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        return [
            "功能流程边界识别",
            "业务流程分解",
            "流程依赖关系分析",
            "数据流分析",
            "流程完整性验证"
        ]
    
    async def identify_processes(
        self,
        requirement_analysis: Any,
        project_info: ProjectInfo
    ) -> List[ProcessDetails]:
        """识别功能流程"""
        
        logger.info(f"🔍 开始识别功能流程...")
        
        start_time = time.time()
        
        try:
            # 1. 从需求分析结果提取流程信息
            raw_processes = await self._extract_process_candidates(requirement_analysis)
            
            # 2. 应用流程边界规则
            bounded_processes = await self._apply_boundary_rules(raw_processes, project_info)
            
            # 3. 优化和合并流程
            optimized_processes = await self._optimize_processes(bounded_processes)
            
            # 4. 验证流程完整性
            validated_processes = await self._validate_processes(optimized_processes, requirement_analysis)
            
            # 5. 生成流程详情
            detailed_processes = await self._generate_process_details(validated_processes)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ 流程识别完成，识别出 {len(detailed_processes)} 个流程，耗时 {processing_time:.2f} 秒")
            
            return detailed_processes
            
        except Exception as e:
            logger.error(f"❌ 流程识别失败: {str(e)}")
            raise
    
    async def _extract_process_candidates(
        self, 
        requirement_analysis: Any
    ) -> List[Dict[str, Any]]:
        """从需求分析结果提取流程候选"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是业务流程分析专家，需要从需求分析结果中识别独立的功能流程。

一个独立的功能流程应该具备：
1. 明确的业务目标
2. 清晰的触发条件
3. 完整的处理步骤
4. 明确的输出结果
5. 涉及的数据组

请识别每个候选流程，并提供详细分析。

重要：输出必须是有效的JSON数组格式，每个流程对象包含以下字段：
- name: 流程名称（不能为空或"未知"）
- business_goal: 业务目标
- trigger_conditions: 触发条件列表
- main_steps: 主要步骤列表
- input_data: 输入数据列表
- output_results: 输出结果列表
- involved_roles: 涉及角色列表"""),
            ("human", """需求分析结果：
{requirement_analysis}

请识别其中的功能流程候选，严格按照以下JSON格式返回：

```json
[
  {{
    "name": "具体的流程名称",
    "business_goal": "明确的业务目标描述",
    "trigger_conditions": ["触发条件1", "触发条件2"],
    "main_steps": ["步骤1", "步骤2", "步骤3"],
    "input_data": ["输入数据1", "输入数据2"],
    "output_results": ["输出结果1", "输出结果2"],
    "involved_roles": ["角色1", "角色2"]
  }}
]
```

注意：
1. 确保JSON格式正确
2. 流程名称必须具体且有意义，不能是"未知"或空值
3. 每个字段都要有具体内容，不能为空""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                requirement_analysis=self._format_requirement_analysis(requirement_analysis)
            )
        )
        
        return await self._parse_process_candidates(response.content)
    
    async def _apply_boundary_rules(
        self,
        raw_processes: List[Dict[str, Any]],
        project_info: ProjectInfo
    ) -> List[Dict[str, Any]]:
        """应用流程边界规则"""
        
        logger.info(f"🔍 开始应用边界规则，候选流程数量: {len(raw_processes)}")
        
        bounded_processes = []
        
        for i, process in enumerate(raw_processes):
            logger.debug(f"处理第 {i+1} 个流程: {process.get('name', '未知')}")
            
            # 检查流程独立性
            if self._check_process_independence(process):
                # 应用边界标识规则
                bounded_process = await self._identify_process_boundaries(process, project_info)
                bounded_processes.append(bounded_process)
                logger.info(f"✅ 流程 '{process.get('name', '未知')}' 通过边界规则检查")
            else:
                logger.warning(f"❌ 流程 '{process.get('name', '未知')}' 不满足独立性原则，跳过")
        
        logger.info(f"✅ 边界规则应用完成，有效流程数量: {len(bounded_processes)}")
        return bounded_processes
    
    def _check_process_independence(self, process: Dict[str, Any]) -> bool:
        """检查流程独立性"""
        
        logger.debug(f"🔍 检查流程独立性: {process}")
        
        # 检查流程名称
        name = process.get("name") or process.get("流程名称") or process.get("process_name")
        if not name or name.strip() == "" or name == "未知":
            logger.warning(f"流程名称为空或未知: {name}")
            return False
        
        # 检查是否有明确的业务目标（兼容多种字段名）
        business_goal = (process.get("business_goal") or 
                        process.get("业务目标") or 
                        process.get("description") or
                        process.get("功能描述") or
                        process.get("目标"))
        if not business_goal:
            logger.warning(f"流程 '{name}' 缺少业务目标")
            return False
        
        # 检查是否有触发条件（兼容多种字段名）
        trigger_conditions = (process.get("trigger_conditions") or 
                             process.get("触发条件") or 
                             process.get("triggers") or
                             process.get("input") or
                             process.get("输入条件"))
        if not trigger_conditions:
            logger.warning(f"流程 '{name}' 缺少触发条件")
            return False
        
        # 检查是否有处理步骤（兼容多种字段名）
        main_steps = (process.get("main_steps") or 
                     process.get("主要步骤") or 
                     process.get("steps") or
                     process.get("处理步骤") or
                     process.get("流程"))
        if not main_steps:
            logger.warning(f"流程 '{name}' 缺少处理步骤")
            return False
        
        # 检查是否有输出结果（兼容多种字段名）
        output_results = (process.get("output_results") or 
                         process.get("输出结果") or 
                         process.get("output") or
                         process.get("结果") or
                         process.get("outputs"))
        if not output_results:
            logger.warning(f"流程 '{name}' 缺少输出结果")
            return False
        
        logger.info(f"✅ 流程 '{name}' 通过独立性检查")
        return True
    
    async def _identify_process_boundaries(
        self,
        process: Dict[str, Any],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """识别流程边界"""
        
        # 识别流程开始边界
        start_boundary = self._identify_start_boundary(process)
        
        # 识别流程结束边界
        end_boundary = self._identify_end_boundary(process)
        
        # 识别数据边界
        data_boundary = await self._identify_data_boundary(process, project_info)
        
        process["boundaries"] = {
            "start": start_boundary,
            "end": end_boundary,
            "data": data_boundary
        }
        
        return process
    
    def _identify_start_boundary(self, process: Dict[str, Any]) -> Dict[str, Any]:
        """识别流程开始边界"""
        
        trigger_conditions = process.get("trigger_conditions", [])
        
        return {
            "type": "user_action",
            "description": "用户操作触发",
            "conditions": trigger_conditions,
            "entry_point": "用户界面或API调用"
        }
    
    def _identify_end_boundary(self, process: Dict[str, Any]) -> Dict[str, Any]:
        """识别流程结束边界"""
        
        output_results = process.get("output_results", [])
        
        return {
            "type": "system_response",
            "description": "系统响应完成",
            "results": output_results,
            "exit_point": "用户界面反馈或数据持久化"
        }
    
    async def _identify_data_boundary(
        self,
        process: Dict[str, Any],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """识别数据边界"""
        
        input_data = process.get("input_data", [])
        output_results = process.get("output_results", [])
        
        # 分析数据流动
        data_flow = await self._analyze_data_flow(input_data, output_results, project_info)
        
        return {
            "input_sources": input_data,
            "output_targets": output_results,
            "data_flow": data_flow,
            "storage_requirements": self._analyze_storage_requirements(data_flow)
        }
    
    async def _analyze_data_flow(
        self,
        input_data: List[str],
        output_results: List[str],
        project_info: ProjectInfo
    ) -> Dict[str, Any]:
        """分析数据流动"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是数据流分析专家，需要分析业务流程中的数据流动。

分析要点：
1. 数据来源和去向
2. 数据转换和处理
3. 数据依赖关系
4. 存储需求"""),
            ("human", """项目信息：{project_info}

输入数据：{input_data}
输出结果：{output_results}

请分析数据流动模式，返回JSON格式的分析结果。""")
        ])
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                project_info=project_info.description,
                input_data=input_data,
                output_results=output_results
            )
        )
        
        return await self._parse_data_flow_analysis(response.content)
    
    def _analyze_storage_requirements(self, data_flow: Dict[str, Any]) -> List[str]:
        """分析存储需求"""
        
        requirements = []
        
        # 基于数据流分析存储需求
        if data_flow.get("persistent_data"):
            requirements.append("持久化存储")
        
        if data_flow.get("temporary_data"):
            requirements.append("临时存储")
        
        if data_flow.get("cached_data"):
            requirements.append("缓存存储")
        
        return requirements
    
    async def _optimize_processes(
        self,
        bounded_processes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """优化和合并流程"""
        
        optimized = []
        
        # 检查是否有可以合并的流程
        merged_groups = self._identify_mergeable_processes(bounded_processes)
        
        for group in merged_groups:
            if len(group) == 1:
                optimized.append(group[0])
            else:
                # 合并流程组
                merged_process = await self._merge_process_group(group)
                optimized.append(merged_process)
        
        return optimized
    
    def _identify_mergeable_processes(
        self,
        processes: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """识别可合并的流程"""
        
        groups = []
        processed = set()
        
        for i, process1 in enumerate(processes):
            if i in processed:
                continue
            
            group = [process1]
            processed.add(i)
            
            for j, process2 in enumerate(processes):
                if j <= i or j in processed:
                    continue
                
                if self._should_merge_processes(process1, process2):
                    group.append(process2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _should_merge_processes(
        self,
        process1: Dict[str, Any],
        process2: Dict[str, Any]
    ) -> bool:
        """判断是否应该合并两个流程"""
        
        # 检查业务目标相似性
        goal1 = process1.get("business_goal", "").lower()
        goal2 = process2.get("business_goal", "").lower()
        
        if self._calculate_text_similarity(goal1, goal2) > 0.8:
            return True
        
        # 检查数据关联性
        data1 = set(process1.get("input_data", []) + process1.get("output_results", []))
        data2 = set(process2.get("input_data", []) + process2.get("output_results", []))
        
        overlap = len(data1 & data2) / max(len(data1 | data2), 1)
        if overlap > 0.6:
            return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _merge_process_group(
        self,
        process_group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """合并流程组"""
        
        if len(process_group) == 1:
            return process_group[0]
        
        # 合并流程信息
        merged_name = " & ".join([p.get("name", "未知") for p in process_group])
        merged_goal = self._merge_business_goals([p.get("business_goal", "") for p in process_group])
        merged_steps = self._merge_process_steps([p.get("main_steps", []) for p in process_group])
        
        return {
            "name": merged_name,
            "business_goal": merged_goal,
            "trigger_conditions": self._merge_lists([p.get("trigger_conditions", []) for p in process_group]),
            "main_steps": merged_steps,
            "input_data": self._merge_lists([p.get("input_data", []) for p in process_group]),
            "output_results": self._merge_lists([p.get("output_results", []) for p in process_group]),
            "involved_roles": self._merge_lists([p.get("involved_roles", []) for p in process_group]),
            "boundaries": self._merge_boundaries([p.get("boundaries", {}) for p in process_group])
        }
    
    def _merge_business_goals(self, goals: List[str]) -> str:
        """合并业务目标"""
        unique_goals = list(set([goal.strip() for goal in goals if goal.strip()]))
        return "；".join(unique_goals)
    
    def _merge_process_steps(self, step_lists: List[List[str]]) -> List[str]:
        """合并流程步骤"""
        all_steps = []
        for steps in step_lists:
            all_steps.extend(steps)
        
        # 去重并保持顺序
        seen = set()
        unique_steps = []
        for step in all_steps:
            if step not in seen:
                unique_steps.append(step)
                seen.add(step)
        
        return unique_steps
    
    def _merge_lists(self, list_of_lists: List[List[str]]) -> List[str]:
        """合并字符串列表"""
        all_items = []
        for lst in list_of_lists:
            all_items.extend(lst)
        
        return list(set(all_items))  # 去重
    
    def _merge_boundaries(self, boundary_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并边界信息"""
        
        if not boundary_list:
            return {}
        
        merged = {
            "start": boundary_list[0].get("start", {}),
            "end": boundary_list[-1].get("end", {}),
            "data": {}
        }
        
        # 合并数据边界
        all_input_sources = []
        all_output_targets = []
        
        for boundary in boundary_list:
            data_boundary = boundary.get("data", {})
            all_input_sources.extend(data_boundary.get("input_sources", []))
            all_output_targets.extend(data_boundary.get("output_targets", []))
        
        merged["data"] = {
            "input_sources": list(set(all_input_sources)),
            "output_targets": list(set(all_output_targets))
        }
        
        return merged
    
    async def _validate_processes(
        self,
        processes: List[Dict[str, Any]],
        requirement_analysis: Any
    ) -> List[Dict[str, Any]]:
        """验证流程完整性"""
        
        validated_processes = []
        
        for process in processes:
            validation_result = await self._validate_single_process(process, requirement_analysis)
            
            if validation_result["is_valid"]:
                process["validation"] = validation_result
                validated_processes.append(process)
            else:
                logger.warning(f"流程 '{process.get('name', '未知')}' 验证失败: {validation_result['issues']}")
        
        return validated_processes
    
    async def _validate_single_process(
        self,
        process: Dict[str, Any],
        requirement_analysis: Any
    ) -> Dict[str, Any]:
        """验证单个流程"""
        
        validation_result = {
            "is_valid": True,
            "confidence_score": 1.0,
            "issues": [],
            "suggestions": []
        }
        
        # 检查流程完整性
        completeness_issues = self._check_process_completeness(process)
        validation_result["issues"].extend(completeness_issues)
        
        # 检查与需求的一致性
        consistency_issues = self._check_requirement_consistency(process, requirement_analysis)
        validation_result["issues"].extend(consistency_issues)
        
        # 检查流程合理性
        rationality_issues = await self._check_process_rationality(process)
        validation_result["issues"].extend(rationality_issues)
        
        # 计算验证分数
        if validation_result["issues"]:
            validation_result["is_valid"] = len(validation_result["issues"]) <= 2
            validation_result["confidence_score"] = max(0.1,
                1.0 - len(validation_result["issues"]) * 0.15
            )
        
        return validation_result
    
    def _check_process_completeness(self, process: Dict[str, Any]) -> List[str]:
        """检查流程完整性"""
        
        issues = []
        
        required_fields = ["name", "business_goal", "trigger_conditions", "main_steps", "output_results"]
        
        for field in required_fields:
            if not process.get(field):
                issues.append(f"缺少必要字段: {field}")
        
        # 检查步骤数量
        steps = process.get("main_steps", [])
        if len(steps) < 2:
            issues.append("流程步骤过少，可能不完整")
        
        return issues
    
    def _check_requirement_consistency(
        self,
        process: Dict[str, Any],
        requirement_analysis: Any
    ) -> List[str]:
        """检查与需求的一致性"""
        
        issues = []
        
        # 检查功能模块一致性
        if hasattr(requirement_analysis, 'functional_modules'):
            functional_modules = requirement_analysis.functional_modules
        elif isinstance(requirement_analysis, dict):
            functional_modules = requirement_analysis.get("functional_modules", [])
        else:
            functional_modules = []
        
        module_names = [m.get("模块名称", "") for m in functional_modules]
        
        process_name = process.get("name", "")
        if not any(name in process_name or process_name in name for name in module_names):
            issues.append("流程与功能模块不匹配")
        
        return issues
    
    async def _check_process_rationality(self, process: Dict[str, Any]) -> List[str]:
        """检查流程合理性"""
        
        issues = []
        
        # 检查输入输出逻辑
        input_data = process.get("input_data", [])
        output_results = process.get("output_results", [])
        
        if not input_data and output_results:
            issues.append("有输出但无输入，可能逻辑不合理")
        
        # 检查步骤逻辑
        main_steps = process.get("main_steps", [])
        if len(main_steps) > 10:
            issues.append("流程步骤过多，建议拆分")
        
        return issues
    
    async def _generate_process_details(
        self,
        validated_processes: List[Dict[str, Any]]
    ) -> List[ProcessDetails]:
        """生成流程详情"""
        
        process_details = []
        
        for i, process in enumerate(validated_processes):
            detail = ProcessDetails(
                id=f"process_{i+1}",
                name=process.get("name", f"流程{i+1}"),
                description=process.get("business_goal", ""),
                data_groups=process.get("input_data", []) + process.get("output_results", []),
                dependencies=self._extract_dependencies(process, validated_processes),
                inputs=process.get("input_data", []),
                outputs=process.get("output_results", []),
                business_rules=process.get("business_rules", []),
                complexity_indicators=process.get("complexity_indicators", {}),
                metadata={
                    "trigger_conditions": process.get("trigger_conditions", []),
                    "main_steps": process.get("main_steps", []),
                    "involved_roles": process.get("involved_roles", []),
                    "boundaries": process.get("boundaries", {}),
                    "validation": process.get("validation", {})
                }
            )
            
            process_details.append(detail)
        
        return process_details
    
    def _extract_dependencies(
        self,
        current_process: Dict[str, Any],
        all_processes: List[Dict[str, Any]]
    ) -> List[str]:
        """提取流程依赖关系"""
        
        dependencies = []
        current_inputs = set(current_process.get("input_data", []))
        
        for process in all_processes:
            if process == current_process:
                continue
            
            process_outputs = set(process.get("output_results", []))
            
            # 如果当前流程的输入依赖其他流程的输出
            if current_inputs & process_outputs:
                dependencies.append(process.get("name", "未知流程"))
        
        return dependencies
    
    # 辅助方法
    def _format_requirement_analysis(self, requirement_analysis: Any) -> str:
        """格式化需求分析结果"""
        
        formatted = []
        
        # 处理RequirementAnalysis对象或字典
        if hasattr(requirement_analysis, 'functional_modules'):
            functional_modules = requirement_analysis.functional_modules
            business_entities = requirement_analysis.business_entities
        elif isinstance(requirement_analysis, dict):
            functional_modules = requirement_analysis.get("functional_modules", [])
            business_entities = requirement_analysis.get("business_entities", {})
        else:
            functional_modules = []
            business_entities = {}
        
        if functional_modules:
            formatted.append("功能模块：")
            for module in functional_modules:
                formatted.append(f"- {module.get('模块名称', '未知')}: {module.get('功能描述', '')}")
        
        if business_entities:
            formatted.append("\n业务实体：")
            for category, entities in business_entities.items():
                formatted.append(f"- {category}: {', '.join(entities)}")
        
        return "\n".join(formatted)
    
    async def _parse_process_candidates(self, response_content: str) -> List[Dict[str, Any]]:
        """解析流程候选"""
        try:
            import json
            logger.debug(f"🔍 LLM原始响应内容: {response_content}")
            
            # 简化的JSON解析，实际应该更健壮
            if "```json" in response_content:
                json_part = response_content.split("```json")[1].split("```")[0]
                parsed_result = json.loads(json_part)
                logger.info(f"✅ 成功解析JSON，获得 {len(parsed_result)} 个流程候选")
                return parsed_result
            else:
                parsed_result = json.loads(response_content)
                logger.info(f"✅ 成功解析JSON，获得 {len(parsed_result)} 个流程候选")
                return parsed_result
        except Exception as e:
            logger.error(f"❌ 解析流程候选失败: {e}")
            logger.error(f"响应内容: {response_content[:500]}...")
            return []
    
    async def _parse_data_flow_analysis(self, response_content: str) -> Dict[str, Any]:
        """解析数据流分析"""
        try:
            import json
            if "```json" in response_content:
                json_part = response_content.split("```json")[1].split("```")[0]
                return json.loads(json_part)
            else:
                return json.loads(response_content)
        except Exception as e:
            logger.error(f"解析数据流分析失败: {e}")
            return {}
    
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行流程识别任务"""
        if task_name == "identify_processes":
            processes = await self.identify_processes(
                inputs["requirement_analysis"],
                inputs["project_info"]
            )
            return {
                "processes": processes,
                "process_count": len(processes),
                "task_status": "completed"
            }
        elif task_name == "extract_process_candidates":
            candidates = await self._extract_process_candidates(
                inputs["requirement_analysis"]
            )
            return {
                "candidates": candidates,
                "task_status": "completed"
            }
        elif task_name == "validate_processes":
            validated = await self._validate_processes(
                inputs["processes"],
                inputs["requirement_analysis"]
            )
            return {
                "validated_processes": validated,
                "task_status": "completed"
            }
        else:
            raise ValueError(f"未知任务: {task_name}")


if __name__ == "__main__":
    # 测试流程识别智能体
    async def test_process_identifier():
        agent = ProcessIdentifierAgent()
        
        # 测试需求分析结果
        test_requirement_analysis = {
            "functional_modules": [
                {
                    "模块名称": "用户管理",
                    "功能描述": "用户注册、登录、信息维护"
                },
                {
                    "模块名称": "订单处理",
                    "功能描述": "订单创建、查询、修改、取消"
                }
            ],
            "business_entities": {
                "用户角色": ["客户", "管理员"],
                "业务对象": ["用户", "订单", "产品"]
            }
        }
        
        project_info = ProjectInfo(
            name="电商系统",
            description="在线购物平台",
            technology_stack=["Python", "Django", "MySQL"],
            business_domain="电商"
        )
        
        processes = await agent.identify_processes(test_requirement_analysis, project_info)
        
        print(f"识别出 {len(processes)} 个功能流程：")
        for process in processes:
            print(f"- {process.name}: {process.description}")
    
    asyncio.run(test_process_identifier()) 