#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 安装验证脚本

验证系统安装和配置的完整性
"""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def print_header():
    """打印验证头部"""
    header = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║            系统安装验证 - 量子智能化功能点估算系统                 ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(header, style="bold blue")


async def check_dependencies() -> Dict[str, Any]:
    """检查依赖包"""
    console.print("[yellow]🔍 检查系统依赖...[/yellow]")
    
    results = {
        "status": "success",
        "dependencies": {},
        "errors": []
    }
    
    required_packages = [
        ("langchain", "LangChain核心"),
        ("langchain_community", "LangChain社区组件"),
        ("langchain_openai", "LangChain OpenAI集成"),
        ("langgraph", "LangGraph工作流引擎"),
        ("fastapi", "FastAPI Web框架"),
        ("uvicorn", "ASGI服务器"),
        ("pydantic", "数据验证"),
        ("motor", "MongoDB异步驱动"),
        ("rich", "终端美化"),
        ("typer", "CLI框架")
    ]
    
    for package, description in required_packages:
        try:
            __import__(package)
            results["dependencies"][package] = {"status": "✅", "description": description}
        except ImportError as e:
            results["dependencies"][package] = {"status": "❌", "description": description, "error": str(e)}
            results["errors"].append(f"缺少依赖: {package}")
            results["status"] = "error"
    
    return results


async def check_configuration() -> Dict[str, Any]:
    """检查配置文件"""
    console.print("[yellow]⚙️ 检查配置文件...[/yellow]")
    
    results = {
        "status": "success",
        "config_files": {},
        "errors": []
    }
    
    config_files = [
        ("config/settings.py", "主配置文件"),
        ("pyproject.toml", "项目配置"),
        ("env.example", "环境变量示例"),
    ]
    
    for file_path, description in config_files:
        path = Path(file_path)
        if path.exists():
            results["config_files"][file_path] = {"status": "✅", "description": description}
        else:
            results["config_files"][file_path] = {"status": "❌", "description": description}
            results["errors"].append(f"配置文件缺失: {file_path}")
            results["status"] = "warning"
    
    # 检查设置导入
    try:
        from config.settings import get_settings
        settings = get_settings()
        results["config_files"]["settings_import"] = {"status": "✅", "description": "配置导入正常"}
    except Exception as e:
        results["config_files"]["settings_import"] = {"status": "❌", "description": "配置导入失败", "error": str(e)}
        results["errors"].append(f"配置导入失败: {str(e)}")
        results["status"] = "error"
    
    return results


async def check_core_modules() -> Dict[str, Any]:
    """检查核心模块"""
    console.print("[yellow]🧠 检查核心模块...[/yellow]")
    
    results = {
        "status": "success",
        "modules": {},
        "errors": []
    }
    
    core_modules = [
        ("models", "数据模型"),
        ("agents.base.base_agent", "基础智能体"),
        ("agents.orchestrator.workflow_orchestrator", "工作流编排器"),
        ("agents.analysis.requirement_parser", "需求解析器"),
        ("agents.analysis.process_identifier", "流程识别器"),
        ("agents.analysis.comparison_analyzer", "对比分析器"),
        ("agents.knowledge.validator", "质量验证器"),
        ("graph.state_definitions", "状态定义"),
        ("graph.node_functions", "节点函数"),
        ("graph.workflow_graph", "工作流图"),
        ("knowledge_base.rag_chains", "RAG链"),
    ]
    
    for module_name, description in core_modules:
        try:
            __import__(module_name)
            results["modules"][module_name] = {"status": "✅", "description": description}
        except ImportError as e:
            results["modules"][module_name] = {"status": "❌", "description": description, "error": str(e)}
            results["errors"].append(f"模块导入失败: {module_name}")
            results["status"] = "error"
        except Exception as e:
            results["modules"][module_name] = {"status": "⚠️", "description": description, "error": str(e)}
            results["errors"].append(f"模块运行异常: {module_name}")
            if results["status"] != "error":
                results["status"] = "warning"
    
    return results


async def check_workflow_functionality() -> Dict[str, Any]:
    """检查工作流功能"""
    console.print("[yellow]🔄 检查工作流功能...[/yellow]")
    
    results = {
        "status": "success",
        "workflow_components": {},
        "errors": []
    }
    
    try:
        # 检查工作流图创建
        from graph.workflow_graph import FPEstimationWorkflow
        workflow = FPEstimationWorkflow()
        results["workflow_components"]["workflow_creation"] = {"status": "✅", "description": "工作流创建成功"}
        
        # 检查状态定义
        from graph.state_definitions import WorkflowState, WorkflowGraphState
        results["workflow_components"]["state_definitions"] = {"status": "✅", "description": "状态定义正常"}
        
        # 检查节点函数
        from graph.node_functions import start_workflow_node
        results["workflow_components"]["node_functions"] = {"status": "✅", "description": "节点函数可用"}
        
    except Exception as e:
        results["workflow_components"]["workflow_test"] = {"status": "❌", "description": "工作流测试失败", "error": str(e)}
        results["errors"].append(f"工作流功能异常: {str(e)}")
        results["status"] = "error"
    
    return results


async def check_knowledge_base() -> Dict[str, Any]:
    """检查知识库"""
    console.print("[yellow]📚 检查知识库...[/yellow]")
    
    results = {
        "status": "success",
        "knowledge_components": {},
        "errors": []
    }
    
    # 检查知识库目录
    kb_dirs = [
        ("knowledge_base/documents/nesma", "NESMA文档目录"),
        ("knowledge_base/documents/cosmic", "COSMIC文档目录"),
        ("knowledge_base/documents/common", "通用文档目录"),
    ]
    
    for dir_path, description in kb_dirs:
        path = Path(dir_path)
        if path.exists():
            results["knowledge_components"][dir_path] = {"status": "✅", "description": description}
        else:
            results["knowledge_components"][dir_path] = {"status": "⚠️", "description": f"{description} (未创建)"}
            if results["status"] != "error":
                results["status"] = "warning"
    
    # 检查RAG组件
    try:
        from knowledge_base.rag_chains import create_rag_chain
        results["knowledge_components"]["rag_chains"] = {"status": "✅", "description": "RAG链可用"}
    except Exception as e:
        results["knowledge_components"]["rag_chains"] = {"status": "❌", "description": "RAG链异常", "error": str(e)}
        results["errors"].append(f"RAG链异常: {str(e)}")
        results["status"] = "error"
    
    return results


async def check_api_functionality() -> Dict[str, Any]:
    """检查API功能"""
    console.print("[yellow]🌐 检查API功能...[/yellow]")
    
    results = {
        "status": "success",
        "api_components": {},
        "errors": []
    }
    
    try:
        from api.main import app
        results["api_components"]["fastapi_app"] = {"status": "✅", "description": "FastAPI应用创建成功"}
    except Exception as e:
        results["api_components"]["fastapi_app"] = {"status": "❌", "description": "FastAPI应用创建失败", "error": str(e)}
        results["errors"].append(f"API应用异常: {str(e)}")
        results["status"] = "error"
    
    return results


def create_results_table(all_results: Dict[str, Dict]) -> Table:
    """创建结果表格"""
    table = Table(title="验证结果汇总")
    table.add_column("检查项", style="cyan")
    table.add_column("状态", style="magenta")
    table.add_column("详情", style="green")
    
    for check_name, result in all_results.items():
        status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "warning" else "❌"
        error_count = len(result.get("errors", []))
        detail = f"{error_count} 个问题" if error_count > 0 else "正常"
        
        table.add_row(check_name, status_icon, detail)
    
    return table


async def main():
    """主验证函数"""
    print_header()
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # 执行各项检查
        checks = [
            ("依赖检查", check_dependencies),
            ("配置检查", check_configuration),
            ("核心模块检查", check_core_modules),
            ("工作流功能检查", check_workflow_functionality),
            ("知识库检查", check_knowledge_base),
            ("API功能检查", check_api_functionality),
        ]
        
        for check_name, check_func in checks:
            task = progress.add_task(f"执行 {check_name}...", total=None)
            try:
                result = await check_func()
                all_results[check_name] = result
                progress.remove_task(task)
            except Exception as e:
                all_results[check_name] = {
                    "status": "error",
                    "errors": [f"检查执行失败: {str(e)}"]
                }
                progress.remove_task(task)
    
    # 显示结果
    console.print("\n")
    console.print(create_results_table(all_results))
    
    # 显示详细错误
    all_errors = []
    for check_name, result in all_results.items():
        if result["status"] == "error":
            for error in result.get("errors", []):
                all_errors.append(f"{check_name}: {error}")
    
    if all_errors:
        console.print("\n")
        error_panel = Panel(
            "\n".join(all_errors),
            title="⚠️ 需要解决的问题",
            border_style="red"
        )
        console.print(error_panel)
    
    # 总体状态
    overall_status = "success"
    if any(r["status"] == "error" for r in all_results.values()):
        overall_status = "error"
    elif any(r["status"] == "warning" for r in all_results.values()):
        overall_status = "warning"
    
    if overall_status == "success":
        console.print("\n[bold green]🎉 系统验证通过！所有组件运行正常。[/bold green]")
        console.print("[green]✅ 可以开始使用功能点估算系统[/green]")
    elif overall_status == "warning":
        console.print("\n[bold yellow]⚠️ 系统基本正常，但有一些警告。[/bold yellow]")
        console.print("[yellow]💡 建议解决警告问题以获得最佳体验[/yellow]")
    else:
        console.print("\n[bold red]❌ 系统验证失败！请解决错误后重试。[/bold red]")
        sys.exit(1)
    
    # 使用建议
    console.print("\n[blue]💡 下一步建议：[/blue]")
    console.print("1. 运行 'python main.py setup-kb' 初始化知识库")
    console.print("2. 使用 'python main.py estimate' 开始功能点估算")
    console.print("3. 启动 'python main.py server' 运行API服务")


if __name__ == "__main__":
    # Windows兼容性
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]验证已中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]验证执行失败: {str(e)}[/red]")
        traceback.print_exc()
        sys.exit(1)
