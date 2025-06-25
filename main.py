#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 主程序入口

提供命令行接口和程序启动入口
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import get_settings
from models import ProjectInfo, TechnologyStack, BusinessDomain

# 初始化控制台和应用
console = Console()
app = typer.Typer(
    name="fp-quantum",
    help="量子智能化功能点估算系统",
    add_completion=False,
)


def print_banner():
    """打印系统横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        量子智能化功能点估算系统                                   ║
    ║        FP-Quantum Estimation System                           ║
    ║                                                               ║
    ║        🤖 AI-Powered Function Point Estimation                ║
    ║        📊 NESMA & COSMIC Standards Support                    ║
    ║        🚀 Powered by LangGraph & DeepSeek                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


@app.command("version")
def show_version():
    """显示版本信息"""
    settings = get_settings()
    console.print(f"[bold green]{settings.app_name}[/bold green]")
    console.print(f"版本: [bold]{settings.app_version}[/bold]")
    console.print(f"环境: [bold]{settings.environment}[/bold]")


@app.command("config")
def show_config():
    """显示配置信息"""
    settings = get_settings()
    
    # 创建配置表格
    table = Table(title="系统配置")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="magenta")
    
    table.add_row("应用名称", settings.app_name)
    table.add_row("版本", settings.app_version)
    table.add_row("环境", settings.environment)
    table.add_row("调试模式", str(settings.debug))
    table.add_row("编排者模型", settings.llm.orchestrator_model)
    table.add_row("执行者模型", settings.llm.worker_model)
    table.add_row("向量模型", settings.llm.embedding_model)
    table.add_row("向量存储", settings.vector_store.provider)
    table.add_row("PostgreSQL", f"{settings.database.postgres_host}:{settings.database.postgres_port}")
    table.add_row("MongoDB", f"{settings.database.mongodb_host}:{settings.database.mongodb_port}")
    
    console.print(table)


@app.command("estimate")
def run_estimation(
    project_name: str = typer.Option(..., "--name", "-n", help="项目名称"),
    description: str = typer.Option(..., "--desc", "-d", help="项目描述"),
    tech_stack: List[str] = typer.Option([], "--tech", "-t", help="技术栈 (可多次指定)"),
    domain: str = typer.Option(..., "--domain", help="业务领域"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="交互式模式"),
):
    """运行功能点估算"""
    try:
        # 验证技术栈
        valid_tech_stack = []
        for tech in tech_stack:
            try:
                valid_tech_stack.append(TechnologyStack(tech))
            except ValueError:
                console.print(f"[red]错误: 无效的技术栈 '{tech}'[/red]")
                console.print(f"[yellow]可用技术栈: {[t.value for t in TechnologyStack]}[/yellow]")
                return
        
        # 验证业务领域
        try:
            business_domain = BusinessDomain(domain)
        except ValueError:
            console.print(f"[red]错误: 无效的业务领域 '{domain}'[/red]")
            console.print(f"[yellow]可用业务领域: {[d.value for d in BusinessDomain]}[/yellow]")
            return
        
        # 创建项目信息
        project_info = ProjectInfo(
            name=project_name,
            description=description,
            technology_stack=valid_tech_stack,
            business_domain=business_domain,
        )
        
        console.print(Panel(
            f"[bold]项目信息[/bold]\n"
            f"名称: {project_info.name}\n"
            f"描述: {project_info.description}\n"
            f"技术栈: {[t.value for t in project_info.technology_stack]}\n"
            f"业务领域: {project_info.business_domain.value}",
            title="估算准备",
            border_style="green"
        ))
        
        if interactive:
            # 交互式模式
            run_interactive_estimation(project_info)
        else:
            # 自动模式
            run_automatic_estimation(project_info)
            
    except Exception as e:
        console.print(f"[red]估算失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


def run_interactive_estimation(project_info: ProjectInfo):
    """运行交互式估算"""
    console.print("[yellow]🤖 启动交互式估算模式...[/yellow]")
    
    # TODO: 实现交互式估算逻辑
    console.print("[green]✅ 交互式估算功能即将推出![/green]")


def run_automatic_estimation(project_info: ProjectInfo):
    """运行自动估算"""
    console.print("[yellow]🚀 启动自动估算模式...[/yellow]")
    
    # TODO: 实现自动估算逻辑
    console.print("[green]✅ 自动估算功能即将推出![/green]")


@app.command("server")
def start_server(
    host: str = typer.Option("0.0.0.0", "--host", help="服务器地址"),
    port: int = typer.Option(8000, "--port", help="服务器端口"),
    reload: bool = typer.Option(False, "--reload", help="自动重载"),
):
    """启动API服务器"""
    try:
        import uvicorn
        from api.main import app as api_app
        
        console.print(f"[green]🚀 启动API服务器 http://{host}:{port}[/green]")
        
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        console.print("[red]错误: 缺少API依赖。请安装web额外依赖: pip install .[web][/red]")
    except Exception as e:
        console.print(f"[red]启动服务器失败: {str(e)}[/red]")


@app.command("setup-kb")
def setup_knowledge_base(
    nesma_path: Optional[Path] = typer.Option(None, "--nesma", help="NESMA文档路径"),
    cosmic_path: Optional[Path] = typer.Option(None, "--cosmic", help="COSMIC文档路径"),
    force: bool = typer.Option(False, "--force", help="强制重建"),
):
    """设置知识库"""
    console.print("[yellow]📚 开始设置知识库...[/yellow]")
    
    try:
        # TODO: 实现知识库设置逻辑
        from scripts.setup_knowledge_base import main as setup_main
        
        # 异步运行设置
        asyncio.run(setup_main())
        
        console.print("[green]✅ 知识库设置完成![/green]")
    except Exception as e:
        console.print(f"[red]知识库设置失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


@app.command("validate")
def validate_system():
    """验证系统配置和依赖"""
    console.print("[yellow]🔍 验证系统配置...[/yellow]")
    
    issues = []
    
    try:
        # 验证配置
        settings = get_settings()
        console.print("[green]✅ 配置文件加载成功[/green]")
        
        # 验证API密钥
        if not settings.llm.deepseek_api_key:
            issues.append("DeepSeek API密钥未设置")
        
        if not settings.llm.bge_m3_api_key:
            issues.append("BGE-M3 API密钥未设置")
        
        # 验证数据库配置
        if not settings.database.postgres_password:
            issues.append("PostgreSQL密码未设置")
            
        if not settings.database.mongodb_password:
            issues.append("MongoDB密码未设置")
        
        # TODO: 验证数据库连接
        # TODO: 验证API连通性
        # TODO: 验证知识库状态
        
        if issues:
            console.print("[yellow]⚠️  发现以下问题:[/yellow]")
            for issue in issues:
                console.print(f"  • {issue}")
        else:
            console.print("[green]✅ 系统验证通过![/green]")
            
    except Exception as e:
        console.print(f"[red]系统验证失败: {str(e)}[/red]")


@app.command("demo")
def run_demo():
    """运行演示示例"""
    console.print("[yellow]🎭 运行演示示例...[/yellow]")
    
    # 创建演示项目
    demo_project = ProjectInfo(
        name="电商平台功能点估算演示",
        description="这是一个演示用的电商平台项目，包含用户管理、商品管理、订单处理等核心功能模块。",
        technology_stack=[
            TechnologyStack.JAVA,
            TechnologyStack.REACT,
            TechnologyStack.MYSQL,
            TechnologyStack.REDIS
        ],
        business_domain=BusinessDomain.ECOMMERCE,
        expected_duration_days=180,
        team_size=8
    )
    
    console.print(Panel(
        f"[bold]演示项目[/bold]\n"
        f"名称: {demo_project.name}\n"
        f"描述: {demo_project.description}\n"
        f"技术栈: {[t.value for t in demo_project.technology_stack]}\n"
        f"业务领域: {demo_project.business_domain.value}\n"
        f"预期周期: {demo_project.expected_duration_days} 天\n"
        f"团队规模: {demo_project.team_size} 人",
        title="演示项目信息",
        border_style="blue"
    ))
    
    console.print("[green]✅ 演示项目创建成功! 可使用 estimate 命令进行估算。[/green]")


def main():
    """主程序入口"""
    print_banner()
    app()


if __name__ == "__main__":
    main() 