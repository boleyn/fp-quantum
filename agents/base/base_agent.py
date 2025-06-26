"""
量子智能化功能点估算系统 - 基础智能体类

提供所有智能体的通用功能和接口定义
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config.settings import get_settings
from models.common_models import ProcessingStatus, ExecutionLog, ErrorInfo

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """智能体配置"""
    agent_id: str = Field(..., description="智能体ID")
    name: str = Field(..., description="智能体名称")
    description: str = Field(default="", description="智能体描述")
    max_retries: int = Field(default=3, description="最大重试次数")
    timeout_seconds: int = Field(default=300, description="超时时间(秒)")
    enable_logging: bool = Field(default=True, description="是否启用日志")
    enable_cache: bool = Field(default=True, description="是否启用缓存")


class BaseAgent(ABC):
    """基础智能体类"""
    
    def __init__(
        self,
        agent_id: str,
        llm: Optional[BaseLanguageModel] = None,
        config: Optional[AgentConfig] = None
    ):
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())
        self.settings = get_settings()
        
        # 配置
        self.config = config or AgentConfig(
            agent_id=agent_id,
            name=agent_id.replace("_", " ").title()
        )
        
        # LLM配置
        if llm:
            self.llm = llm
        else:
            self.llm = self._create_default_llm()
        
        # 执行状态
        self.status = ProcessingStatus.READY
        self.execution_logs: List[ExecutionLog] = []
        self.error_info: Optional[ErrorInfo] = None
        
        # 统计信息
        self.total_executions = 0
        self.successful_executions = 0
        self.total_execution_time = 0.0
        
        # 缓存
        self.cache: Dict[str, Any] = {} if self.config.enable_cache else None
        
        logger.info(f"🤖 智能体 {self.agent_id} 初始化完成")
    
    def _create_default_llm(self) -> BaseLanguageModel:
        """创建默认LLM"""
        return ChatOpenAI(
            model=self.settings.llm.worker_model,
            api_key=self.settings.llm.deepseek_api_key,
            base_url=self.settings.llm.deepseek_api_base,
            temperature=0.1,
            max_tokens=4000,
            timeout=self.config.timeout_seconds
        )
    
    async def initialize(self):
        """初始化智能体（子类可重写）"""
        self.status = ProcessingStatus.READY
        self._log_execution("INITIALIZE", {"status": "completed"})
        logger.info(f"✅ 智能体 {self.agent_id} 初始化完成")
    
    async def execute(
        self,
        task_name: str,
        inputs: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """执行任务的通用包装器"""
        start_time = time.time()
        self.total_executions += 1
        
        try:
            # 检查缓存
            if use_cache and self.cache is not None:
                cache_key = self._generate_cache_key(task_name, inputs)
                if cache_key in self.cache:
                    logger.info(f"📋 使用缓存结果: {task_name}")
                    self._log_execution(task_name, {"cache_hit": True})
                    return self.cache[cache_key]
            
            # 设置状态
            self.status = ProcessingStatus.PROCESSING
            
            # 执行具体任务
            result = await self._execute_task(task_name, inputs)
            
            # 缓存结果
            if use_cache and self.cache is not None and result:
                cache_key = self._generate_cache_key(task_name, inputs)
                self.cache[cache_key] = result
            
            # 更新统计
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.successful_executions += 1
            self.status = ProcessingStatus.COMPLETED
            
            # 记录执行日志
            self._log_execution(task_name, {
                "status": "success",
                "execution_time": execution_time,
                "result_size": len(str(result))
            })
            
            return result
            
        except Exception as e:
            # 错误处理
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.status = ProcessingStatus.ERROR
            
            error_info = ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                occurred_at=datetime.now()
            )
            self.error_info = error_info
            
            self._log_execution(task_name, {
                "status": "error",
                "execution_time": execution_time,
                "error": str(e)
            })
            
            logger.error(f"❌ 智能体 {self.agent_id} 执行任务 {task_name} 失败: {str(e)}")
            raise
    
    @abstractmethod
    async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体任务（子类必须实现）"""
        pass
    
    def _generate_cache_key(self, task_name: str, inputs: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        input_str = str(sorted(inputs.items()))
        hash_obj = hashlib.md5(f"{task_name}_{input_str}".encode())
        return hash_obj.hexdigest()
    
    def _log_execution(self, step_name: str, details: Dict[str, Any]):
        """记录执行日志"""
        if not self.config.enable_logging:
            return
        
        log_entry = ExecutionLog(
            step_name=step_name,
            agent_id=self.agent_id,
            session_id=self.session_id,
            timestamp=datetime.now(),
            details=details
        )
        
        self.execution_logs.append(log_entry)
        
        # 限制日志数量
        if len(self.execution_logs) > 1000:
            self.execution_logs = self.execution_logs[-500:]
    
    async def retry_with_backoff(
        self,
        func,
        max_retries: Optional[int] = None,
        backoff_factor: float = 2.0,
        *args,
        **kwargs
    ):
        """带退避的重试机制"""
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"❌ 重试 {max_retries} 次后仍失败: {str(e)}")
                    raise
                
                wait_time = backoff_factor ** attempt
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败，{wait_time}秒后重试: {str(e)}")
                await asyncio.sleep(wait_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        success_rate = (
            self.successful_executions / self.total_executions 
            if self.total_executions > 0 else 0.0
        )
        
        avg_execution_time = (
            self.total_execution_time / self.total_executions 
            if self.total_executions > 0 else 0.0
        )
        
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time,
            "cache_size": len(self.cache) if self.cache else 0,
            "log_entries": len(self.execution_logs),
            "last_error": self.error_info.model_dump() if self.error_info else None
        }
    
    def clear_cache(self):
        """清除缓存"""
        if self.cache:
            self.cache.clear()
            logger.info(f"🗑️ {self.agent_id} 缓存已清除")
    
    def export_logs(self) -> List[Dict[str, Any]]:
        """导出执行日志"""
        return [log.model_dump() for log in self.execution_logs]
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试LLM连接
            test_response = await self.llm.ainvoke("测试连接")
            llm_healthy = bool(test_response.content)
        except Exception as e:
            llm_healthy = False
            logger.warning(f"⚠️ LLM健康检查失败: {str(e)}")
        
        stats = self.get_statistics()
        
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "llm_healthy": llm_healthy,
            "cache_enabled": self.cache is not None,
            "uptime_seconds": sum(log.details.get("execution_time", 0) for log in self.execution_logs),
            "last_activity": (
                self.execution_logs[-1].timestamp.isoformat() 
                if self.execution_logs else None
            ),
            "performance": {
                "success_rate": stats["success_rate"],
                "average_execution_time": stats["average_execution_time"]
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id='{self.agent_id}', status='{self.status.value}')"


class SpecializedAgent(BaseAgent):
    """专业化智能体基类"""
    
    def __init__(
        self,
        agent_id: str,
        specialty: str,
        llm: Optional[BaseLanguageModel] = None,
        config: Optional[AgentConfig] = None
    ):
        super().__init__(agent_id, llm, config)
        self.specialty = specialty
        self.expertise_areas: List[str] = []
        self.knowledge_sources: List[str] = []
    
    def add_expertise_area(self, area: str):
        """添加专业领域"""
        if area not in self.expertise_areas:
            self.expertise_areas.append(area)
    
    def add_knowledge_source(self, source: str):
        """添加知识源"""
        if source not in self.knowledge_sources:
            self.knowledge_sources.append(source)
    
    def get_expertise_info(self) -> Dict[str, Any]:
        """获取专业信息"""
        return {
            "specialty": self.specialty,
            "expertise_areas": self.expertise_areas,
            "knowledge_sources": self.knowledge_sources,
            "capabilities": self._get_capabilities()
        }
    
    @abstractmethod
    def _get_capabilities(self) -> List[str]:
        """获取能力列表（子类实现）"""
        pass


if __name__ == "__main__":
    # 测试基础智能体
    class TestAgent(BaseAgent):
        async def _execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
            # 模拟任务执行
            await asyncio.sleep(0.1)
            return {"result": f"Processed {task_name} with {len(inputs)} inputs"}
        
        def _get_capabilities(self) -> List[str]:
            return ["测试功能", "演示功能"]
    
    async def main():
        # 创建测试智能体
        agent = TestAgent("test_agent")
        await agent.initialize()
        
        # 执行测试任务
        result = await agent.execute("test_task", {"input": "test_data"})
        print(f"执行结果: {result}")
        
        # 获取统计信息
        stats = agent.get_statistics()
        print(f"统计信息: {stats}")
        
        # 健康检查
        health = await agent.health_check()
        print(f"健康状态: {health}")
    
    asyncio.run(main())