#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - FastAPI应用

提供RESTful API接口
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import get_settings
from models import (
    ProjectInfo, EstimationStrategy,
    NESMAFunctionClassification, COSMICDataMovement
)
from graph.state_definitions import WorkflowState
from graph.workflow_graph import FPEstimationWorkflow

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 启动量子智能化功能点估算系统 API")
    yield
    logger.info("🛑 关闭量子智能化功能点估算系统 API")


# 创建FastAPI应用
app = FastAPI(
    title="量子智能化功能点估算系统",
    description="基于AI的NESMA和COSMIC功能点估算平台",
    version=settings.app_version,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class EstimationRequest(BaseModel):
    """估算请求模型"""
    project_name: str
    description: str
    technology_stack: List[str]
    business_domain: str
    strategy: Optional[EstimationStrategy] = EstimationStrategy.DUAL_PARALLEL
    requirements: Optional[str] = None


class EstimationResponse(BaseModel):
    """估算响应模型"""
    session_id: str
    status: WorkflowState
    project_info: ProjectInfo
    strategy: EstimationStrategy
    nesma_results: Optional[Dict[str, Any]] = None
    cosmic_results: Optional[Dict[str, Any]] = None
    comparison_analysis: Optional[Dict[str, Any]] = None
    final_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class SessionStatus(BaseModel):
    """会话状态模型"""
    session_id: str
    status: WorkflowState
    progress: float
    current_step: str
    execution_log: List[str]
    error_message: Optional[str] = None


# 工作流实例管理
active_workflows: Dict[str, FPEstimationWorkflow] = {}


@app.get("/")
async def root():
    """根端点"""
    return {
        "name": "量子智能化功能点估算系统",
        "version": settings.app_version,
        "status": "运行中",
        "endpoints": {
            "估算": "/estimate",
            "状态查询": "/status/{session_id}",
            "健康检查": "/health",
            "API文档": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": "2024-12-19T10:00:00Z",
        "version": settings.app_version,
        "services": {
            "api": "运行中",
            "llm": "就绪",
            "database": "连接中",
            "vector_store": "就绪"
        }
    }


@app.post("/test/simple-estimate")
async def test_simple_estimate():
    """简单的估算测试端点，用于调试"""
    logger.info("🧪 开始简单估算测试...")
    
    try:
        # 创建测试项目信息
        project_info = ProjectInfo(
            name="测试项目",
            description="用户注册、登录、个人信息管理、密码修改功能",
            technology_stack=["Python", "PostgreSQL"],
            business_domain="其他"
        )
        
        logger.info(f"📋 创建测试项目: {project_info.name}")
        
        # 创建工作流实例
        workflow = FPEstimationWorkflow()
        session_id = await workflow.initialize(
            project_info=project_info,
            strategy=EstimationStrategy.NESMA_ONLY,  # 只使用NESMA进行测试
            requirements=project_info.description
        )
        
        logger.info(f"🆔 会话ID: {session_id}")
        
        # 执行工作流
        logger.info("⚡ 开始执行工作流...")
        final_state = await workflow.execute()
        
        logger.info(f"✅ 工作流执行完成，状态: {final_state.get('current_state')}")
        
        # 构建响应
        response = {
            "session_id": session_id,
            "status": final_state.get("current_state", "UNKNOWN"),
            "project_info": {
                "name": project_info.name,
                "description": project_info.description,
                "technology_stack": project_info.technology_stack,
                "business_domain": project_info.business_domain
            },
            "nesma_results": final_state.get("nesma_results"),
            "cosmic_results": final_state.get("cosmic_results"),
            "error_message": final_state.get("error_message"),
            "execution_log": final_state.get("execution_log", [])
        }
        
        logger.info(f"📊 返回响应: {response}")
        return response
        
    except Exception as e:
        logger.error(f"❌ 简单估算测试失败: {str(e)}")
        logger.exception("详细错误信息:")
        return {
            "error": str(e),
            "status": "FAILED",
            "message": "简单估算测试失败"
        }


@app.post("/estimate", response_model=EstimationResponse)
async def create_estimation(
    request: EstimationRequest, 
    background_tasks: BackgroundTasks
):
    """创建新的功能点估算任务"""
    try:
        # 创建项目信息
        project_info = ProjectInfo(
            name=request.project_name,
            description=request.description,
            technology_stack=request.technology_stack,
            business_domain=request.business_domain
        )
        
        # 创建工作流实例
        workflow = FPEstimationWorkflow()
        session_id = await workflow.initialize(
            project_info=project_info,
            strategy=request.strategy,
            requirements=request.requirements or request.description
        )
        
        # 保存到活跃工作流
        active_workflows[session_id] = workflow
        
        # 后台执行工作流
        background_tasks.add_task(execute_workflow, session_id, workflow)
        
        return EstimationResponse(
            session_id=session_id,
            status=WorkflowState.STARTING,
            project_info=project_info,
            strategy=request.strategy
        )
        
    except Exception as e:
        logger.error(f"创建估算任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建估算任务失败: {str(e)}")


@app.get("/status/{session_id}", response_model=SessionStatus)
async def get_estimation_status(session_id: str):
    """获取估算任务状态"""
    try:
        if session_id not in active_workflows:
            raise HTTPException(status_code=404, detail="未找到指定的估算会话")
        
        workflow = active_workflows[session_id]
        state = await workflow.get_current_state()
        
        current_status = state.get("current_state", WorkflowState.STARTING)
        
        # 转换execution_log为字符串列表
        execution_log_raw = state.get("execution_log", [])
        execution_log = []
        for log_entry in execution_log_raw:
            if isinstance(log_entry, dict):
                # 转换字典为可读字符串
                timestamp = log_entry.get("timestamp", "")
                action = log_entry.get("action", "")
                status = log_entry.get("status", "")
                log_str = f"[{timestamp}] {action}: {status}"
                execution_log.append(log_str)
            elif isinstance(log_entry, str):
                execution_log.append(log_entry)
            else:
                execution_log.append(str(log_entry))
        
        return SessionStatus(
            session_id=session_id,
            status=current_status,
            progress=calculate_progress(current_status),
            current_step=current_status.value if hasattr(current_status, 'value') else str(current_status),
            execution_log=execution_log,
            error_message=state.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@app.get("/result/{session_id}", response_model=EstimationResponse)
async def get_estimation_result(session_id: str):
    """获取估算结果"""
    try:
        if session_id not in active_workflows:
            raise HTTPException(status_code=404, detail="未找到指定的估算会话")
        
        workflow = active_workflows[session_id]
        state = await workflow.get_current_state()
        
        current_status = state.get("current_state", WorkflowState.STARTING)
        
        if current_status not in [WorkflowState.COMPLETED, WorkflowState.ERROR_ENCOUNTERED]:
            raise HTTPException(status_code=202, detail="估算任务尚未完成")
        
        return EstimationResponse(
            session_id=session_id,
            status=current_status,
            project_info=state.get("project_info"),
            strategy=state.get("selected_strategy"),
            nesma_results={
                "classifications": [c.dict() for c in state.get("nesma_classifications", [])],
                "complexity_results": [r.dict() for r in state.get("nesma_complexity_results", [])],
                "ufp_total": state.get("nesma_ufp_total")
            } if state.get("nesma_classifications") else None,
            cosmic_results={
                "functional_users": [u.dict() for u in state.get("cosmic_functional_users", [])],
                "data_movements": [m.dict() for m in state.get("cosmic_data_movements", [])],
                "cfp_total": state.get("cosmic_cfp_total")
            } if state.get("cosmic_data_movements") else None,
            comparison_analysis=state.get("comparison_analysis"),
            final_report=state.get("final_report"),
            error_message=state.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")


@app.delete("/session/{session_id}")
async def cancel_estimation(session_id: str):
    """取消估算任务"""
    try:
        if session_id not in active_workflows:
            raise HTTPException(status_code=404, detail="未找到指定的估算会话")
        
        workflow = active_workflows[session_id]
        await workflow.cancel()
        
        # 从活跃工作流中移除
        del active_workflows[session_id]
        
        return {"message": f"估算任务 {session_id} 已取消"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@app.get("/sessions")
async def list_active_sessions():
    """列出活跃的估算会话"""
    try:
        sessions = []
        for session_id, workflow in active_workflows.items():
            state = await workflow.get_current_state()
            current_status = state.get("current_state", WorkflowState.STARTING)
            project_info = state.get("project_info")
            
            sessions.append({
                "session_id": session_id,
                "project_name": project_info.name if project_info else "未知项目",
                "status": current_status.value if hasattr(current_status, 'value') else str(current_status),
                "progress": calculate_progress(current_status)
            })
        
        return {"active_sessions": sessions, "total": len(sessions)}
        
    except Exception as e:
        logger.error(f"列出会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出会话失败: {str(e)}")


async def execute_workflow(session_id: str, workflow: FPEstimationWorkflow):
    """后台执行工作流"""
    try:
        logger.info(f"开始执行工作流: {session_id}")
        
        # 执行工作流
        final_state = await workflow.execute()
        
        final_status = final_state.get("current_state", "UNKNOWN") if isinstance(final_state, dict) else "COMPLETED"
        logger.info(f"工作流执行完成: {session_id}, 状态: {final_status}")
        
    except Exception as e:
        logger.error(f"工作流执行失败: {session_id}, 错误: {e}")
        # 工作流内部会处理错误状态


def calculate_progress(status: WorkflowState) -> float:
    """计算进度百分比"""
    progress_map = {
        WorkflowState.STARTING: 0.0,
        WorkflowState.REQUIREMENT_INPUT_RECEIVED: 5.0,
        WorkflowState.STANDARD_IDENTIFICATION_PENDING: 10.0,
        WorkflowState.STANDARD_RECOMMENDATION_READY: 15.0,
        WorkflowState.STANDARD_ROUTING_COMPLETED: 20.0,
        WorkflowState.PROCESS_IDENTIFICATION_PENDING: 25.0,
        WorkflowState.PROCESSES_IDENTIFIED: 30.0,
        WorkflowState.NESMA_PROCESSING_PENDING: 40.0,
        WorkflowState.NESMA_CLASSIFICATION_COMPLETED: 50.0,
        WorkflowState.NESMA_CALCULATION_COMPLETED: 60.0,
        WorkflowState.COSMIC_PROCESSING_PENDING: 40.0,
        WorkflowState.COSMIC_ANALYSIS_COMPLETED: 70.0,
        WorkflowState.COSMIC_CALCULATION_COMPLETED: 75.0,
        WorkflowState.CROSS_STANDARD_COMPARISON_PENDING: 80.0,
        WorkflowState.REPORT_GENERATION_PENDING: 90.0,
        WorkflowState.COMPLETED: 100.0,
        WorkflowState.ERROR_ENCOUNTERED: 100.0,
        WorkflowState.TERMINATED: 100.0,
    }
    
    return progress_map.get(status, 0.0)


# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误，请稍后重试"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 