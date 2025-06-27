#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 主程序入口

提供命令行接口和程序启动入口
"""

import asyncio
import logging
from typing import List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings
from graph.state_definitions import WorkflowState, WorkflowGraphState
from graph.workflow_graph import create_function_point_workflow
from knowledge_base.auto_setup import ensure_knowledge_base_ready
from models import ProjectInfo, TechnologyStack, BusinessDomain
from models.common_models import EstimationStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fp_quantum.log')
    ]
)

logger = logging.getLogger(__name__)

# 初始化控制台和应用
console = Console()
app = typer.Typer(
    name="fp-quantum",
    help="量子智能化功能点估算系统",
    add_completion=False,
)

# 全局知识库状态缓存
_kb_initialized = False


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


async def ensure_system_ready():
    """确保系统就绪 - 包括知识库检查和初始化"""
    global _kb_initialized
    
    # 如果已经初始化过，直接返回
    if _kb_initialized:
        return True
    
    try:
        console.print("[yellow]🔍 检查系统状态...[/yellow]")
        
        # 1. 检查配置
        settings = get_settings()
        console.print("[green]✅ 配置加载完成[/green]")
        
        # 2. 检查并初始化知识库
        console.print("[yellow]📚 检查知识库状态...[/yellow]")
        kb_ready = await ensure_knowledge_base_ready()
        
        if kb_ready:
            console.print("[green]✅ 知识库就绪[/green]")
            _kb_initialized = True
            return True
        else:
            console.print("[red]❌ 知识库初始化失败[/red]")
            return False
        
    except Exception as e:
        console.print(f"[red]❌ 系统初始化失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()
        return False


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
        # 🔥 首先确保系统就绪（包括知识库）
        async def check_and_run():
            if not await ensure_system_ready():
                console.print("[red]❌ 系统未就绪，无法执行估算[/red]")
                return
            
            # 系统就绪后继续执行估算逻辑
            await run_estimation_logic()
        
        async def run_estimation_logic():
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
                await run_interactive_estimation(project_info)
            else:
                # 自动模式
                await run_automatic_estimation(project_info)
                
        asyncio.run(check_and_run())
            
    except Exception as e:
        console.print(f"[red]估算失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


async def run_interactive_estimation(project_info: ProjectInfo):
    """运行交互式估算"""
    console.print("[yellow]🤖 启动交互式估算模式...[/yellow]")
    
    try:
        # 获取用户需求输入
        requirements = typer.prompt("\n请输入详细的项目需求描述")
        
        # 创建工作流实例
        from graph.workflow_graph import FPEstimationWorkflow
        workflow = FPEstimationWorkflow()
        
        # 异步执行估算
        session_id = await workflow.initialize(
            project_info=project_info,
            strategy=EstimationStrategy.DUAL_PARALLEL,
            requirements=requirements
        )
        
        console.print(f"[green]📋 会话ID: {session_id}[/green]")
        console.print("[yellow]⏳ 正在执行估算，请稍候...[/yellow]")
        
        # 执行工作流
        final_state = await workflow.execute()
        
        # 显示结果
        if final_state.current_state == WorkflowState.COMPLETED:
            console.print("[green]✅ 估算完成![/green]")
            
            # 显示NESMA结果
            if final_state.nesma_ufp_total:
                console.print(f"[blue]📊 NESMA UFP总计: {final_state.nesma_ufp_total}[/blue]")
            
            # 显示COSMIC结果
            if final_state.cosmic_cfp_total:
                console.print(f"[blue]📊 COSMIC CFP总计: {final_state.cosmic_cfp_total}[/blue]")
            
            # 显示报告
            if final_state.final_report:
                console.print("[green]📄 详细报告已生成[/green]")
                
        else:
            console.print(f"[red]❌ 估算失败: {final_state.error_message}[/red]")
        
    except Exception as e:
        console.print(f"[red]交互式估算失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


async def run_automatic_estimation(project_info: ProjectInfo):
    """运行自动估算模式"""
    console.print("[yellow]🔄 启动自动估算模式...[/yellow]")
    
    try:
        # 创建工作流实例并执行
        workflow = await create_function_point_workflow()
        
        console.print("[yellow]⏳ 正在执行估算，请稍候...[/yellow]")
        
        # 生成一个会话ID
        import uuid
        from datetime import datetime
        session_id = str(uuid.uuid4())
        
        # 创建工作流状态 - 使用Pydantic模型
        initial_state = WorkflowGraphState(
            session_id=session_id,
            project_info=project_info,
            user_requirements=project_info.description
        )
        
        result = await workflow.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )
        
        # 显示结果 - 完全使用Pydantic模型，不再使用dict
        console.print("[green]✅ 自动估算完成![/green]")
        
        # LangGraph返回的是AddableValuesDict，需要转换为WorkflowGraphState
        try:
            # 将AddableValuesDict转换为字典，然后创建Pydantic模型
            result_dict = dict(result) if hasattr(result, '__iter__') else result
            
            # 创建WorkflowGraphState实例来确保类型安全
            workflow_state = WorkflowGraphState(**result_dict)
            
            # 使用Pydantic模型属性访问 - 完全避免dict操作
            if workflow_state.nesma_results and workflow_state.nesma_results.total_ufp:
                console.print(f"[blue]📊 NESMA UFP总计: {workflow_state.nesma_results.total_ufp}[/blue]")
            
            if workflow_state.cosmic_results and workflow_state.cosmic_results.total_cfp:
                console.print(f"[blue]📊 COSMIC CFP总计: {workflow_state.cosmic_results.total_cfp}[/blue]")
            
            # 检查最终报告 - 使用Pydantic模型属性
            if workflow_state.final_report:
                console.print("[green]�� 详细报告已生成[/green]")
                console.print(f"[cyan]📋 报告摘要:[/cyan]")
                
                # 获取估算策略 - 使用Pydantic模型属性
                selected_strategy = workflow_state.selected_strategy
                console.print(f"  估算策略: {selected_strategy or 'unknown'}")
                
                # 显示各种格式的报告
                successful_reports = []
                
                if workflow_state.final_report.markdown:
                    if workflow_state.final_report.markdown.error:
                        console.print(f"  ❌ MARKDOWN格式: 生成失败 - {workflow_state.final_report.markdown.error}")
                    else:
                        successful_reports.append("markdown")
                        file_path = workflow_state.final_report.markdown.file_path
                        if file_path:
                            console.print(f"  ✅ MARKDOWN格式: {file_path}")
                        else:
                            console.print(f"  ✅ MARKDOWN格式: 内容已生成")
                
                if workflow_state.final_report.excel:
                    if workflow_state.final_report.excel.error:
                        console.print(f"  ❌ EXCEL格式: 生成失败 - {workflow_state.final_report.excel.error}")
                    else:
                        successful_reports.append("excel")
                        file_path = workflow_state.final_report.excel.file_path or "未知路径"
                        console.print(f"  ✅ EXCEL格式: {file_path}")
                
                if workflow_state.final_report.word:
                    if workflow_state.final_report.word.error:
                        console.print(f"  ❌ WORD格式: 生成失败 - {workflow_state.final_report.word.error}")
                    else:
                        successful_reports.append("word")
                        file_path = workflow_state.final_report.word.file_path or "未知路径"
                        console.print(f"  ✅ WORD格式: {file_path}")
                
                if successful_reports:
                    console.print(f"  📊 成功生成 {len(successful_reports)} 种格式的报告")
                
                # 显示NESMA和COSMIC结果（如果报告中有）
                if (workflow_state.final_report.markdown and 
                    workflow_state.final_report.markdown.content and 
                    "NESMA UFP" in workflow_state.final_report.markdown.content):
                    import re
                    nesma_match = re.search(r'NESMA.*?(\d+)', workflow_state.final_report.markdown.content)
                    if nesma_match:
                        console.print(f"  📊 NESMA UFP: {nesma_match.group(1)}")
                
                if (workflow_state.final_report.markdown and 
                    workflow_state.final_report.markdown.content and 
                    "COSMIC CFP" in workflow_state.final_report.markdown.content):
                    import re
                    cosmic_match = re.search(r'COSMIC.*?(\d+)', workflow_state.final_report.markdown.content)
                    if cosmic_match:
                        console.print(f"  📊 COSMIC CFP: {cosmic_match.group(1)}")
        
        except Exception as conversion_error:
            console.print(f"[red]结果转换失败: {str(conversion_error)}[/red]")
            # 降级处理：直接使用字典访问（仅作为最后的备选方案）
            try:
                result_dict = dict(result) if hasattr(result, '__iter__') else result
                
                # 检查NESMA结果
                nesma_results = result_dict.get("nesma_results")
                if nesma_results and isinstance(nesma_results, dict) and nesma_results.get("total_ufp"):
                    console.print(f"[blue]📊 NESMA UFP总计: {nesma_results['total_ufp']}[/blue]")
                
                # 检查COSMIC结果
                cosmic_results = result_dict.get("cosmic_results")
                if cosmic_results and isinstance(cosmic_results, dict) and cosmic_results.get("total_cfp"):
                    console.print(f"[blue]📊 COSMIC CFP总计: {cosmic_results['total_cfp']}[/blue]")
                
                # 检查最终报告
                final_report = result_dict.get("final_report")
                if final_report and isinstance(final_report, dict):
                    console.print("[green]📄 详细报告已生成[/green]")
                    console.print(f"[cyan]📋 报告摘要:[/cyan]")
                    
                    # 获取估算策略
                    selected_strategy = result_dict.get("selected_strategy")
                    console.print(f"  估算策略: {selected_strategy or 'unknown'}")
                    
                    # 显示各种格式的报告
                    successful_reports = []
                    for format_type, report_data in final_report.items():
                        if isinstance(report_data, dict):
                            if "error" in report_data:
                                console.print(f"  ❌ {format_type.upper()}格式: 生成失败 - {report_data['error']}")
                            else:
                                successful_reports.append(format_type)
                                if format_type == "markdown":
                                    file_path = report_data.get("file_path", "未知路径")
                                    if file_path:
                                        console.print(f"  ✅ {format_type.upper()}格式: {file_path}")
                                    else:
                                        console.print(f"  ✅ {format_type.upper()}格式: 内容已生成")
                                else:
                                    file_path = report_data.get("file_path", "未知路径")
                                    console.print(f"  ✅ {format_type.upper()}格式: {file_path}")
                    
                    if successful_reports:
                        console.print(f"  📊 成功生成 {len(successful_reports)} 种格式的报告")
                elif final_report:
                    console.print("[green]📄 详细报告已生成[/green]")
                    console.print(f"[cyan]📋 报告摘要:[/cyan]")
                    console.print(f"  报告类型: {type(final_report).__name__}")
                    console.print(f"  报告内容: {str(final_report)[:100]}...")
            
            except Exception as fallback_error:
                console.print(f"[red]降级处理也失败: {str(fallback_error)}[/red]")
        
    except Exception as e:
        console.print(f"[red]自动估算失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


@app.command("api")
@app.command("server")
def start_server(
    host: str = typer.Option("0.0.0.0", "--host", help="服务器地址"),
    port: int = typer.Option(8000, "--port", help="服务器端口"),
    reload: bool = typer.Option(False, "--reload", help="自动重载"),
):
    """启动API服务器"""
    try:
        # 🔥 启动服务器前确保系统就绪
        async def check_and_start():
            if not await ensure_system_ready():
                console.print("[red]❌ 系统未就绪，无法启动服务器[/red]")
                return
            
            # 系统就绪后启动服务器
            console.print(f"[green]🚀 启动API服务器 http://{host}:{port}[/green]")
            import uvicorn
            from api.main import app as api_app
            uvicorn.run(api_app, host=host, port=port, reload=reload)
        
        asyncio.run(check_and_start())
        
    except Exception as e:
        console.print(f"[red]服务器启动失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


@app.command("setup-kb")
def setup_knowledge_base(
    force: bool = typer.Option(False, "--force", help="强制重建"),
    check_only: bool = typer.Option(False, "--check-only", help="仅检查状态"),
):
    """设置或检查知识库"""
    try:
        async def run_setup():
            if check_only:
                console.print("[yellow]🔍 检查知识库状态...[/yellow]")
                kb_ready = await ensure_knowledge_base_ready()
                if kb_ready:
                    console.print("[green]✅ 知识库状态正常[/green]")
                else:
                    console.print("[red]❌ 知识库状态异常[/red]")
                return
            
            console.print("[yellow]🔧 设置知识库...[/yellow]")
            
            from knowledge_base.auto_setup import IncrementalKnowledgeBaseManager
            
            manager = IncrementalKnowledgeBaseManager()
            try:
                result = await manager.auto_update_knowledge_base()
                
                # 显示详细结果
                console.print(f"[cyan]📋 处理摘要:[/cyan]")
                console.print(f"  状态: {result.get('status', 'unknown')}")
                console.print(f"  耗时: {result.get('duration', 0):.2f} 秒")
                
                changes = result.get('changes', {})
                console.print(f"[cyan]📊 文件变化:[/cyan]")
                console.print(f"  新增: {changes.get('new_files', 0)} 个文件")
                console.print(f"  修改: {changes.get('modified_files', 0)} 个文件")
                console.print(f"  删除: {changes.get('deleted_files', 0)} 个文件")
                console.print(f"  未变化: {changes.get('unchanged_files', 0)} 个文件")
                console.print(f"  总处理: {result.get('total_processed', 0)} 个文件")
                
                if result.get('file_counts'):
                    console.print(f"[cyan]📁 文件分布:[/cyan]")
                    for category, count in result['file_counts'].items():
                        console.print(f"  {category}: {count} 个文件")
                
                if result['status'] in ['success', 'check_completed']:
                    console.print("[green]✅ 知识库管理完成![/green]")
                    # 更新全局状态
                    global _kb_initialized
                    _kb_initialized = True
                else:
                    console.print("[red]❌ 知识库管理失败![/red]")
                    
            finally:
                await manager.close()
        
        asyncio.run(run_setup())
        
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
    try:
        # 🔥 演示前确保系统就绪
        async def check_and_demo():
            if not await ensure_system_ready():
                console.print("[red]❌ 系统未就绪，无法运行演示[/red]")
                return
            
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
            
            # 运行演示估算
            await run_automatic_estimation(demo_project)
        
        asyncio.run(check_and_demo())
        
    except Exception as e:
        console.print(f"[red]演示运行失败: {str(e)}[/red]")
        if get_settings().debug:
            console.print_exception()


async def initialize_system():
    """系统初始化 - 向后兼容的函数"""
    return await ensure_system_ready()


async def main():
    """主程序 - 测试模式"""
    # 初始化系统
    if not await ensure_system_ready():
        logger.error("❌ 系统初始化失败，程序退出")
        return
    
    logger.info("✅ 系统初始化完成，开始功能点估算...")
    
    # 创建示例项目信息
    project_info = ProjectInfo(
        name="电商平台功能模块",
        description="""
        需要开发一个电商平台的核心功能模块，包括：
        1. 用户管理模块：用户注册、登录、个人信息管理
        2. 商品管理模块：商品信息维护、分类管理、库存管理
        3. 订单处理模块：订单创建、修改、状态跟踪
        4. 支付系统：多种支付方式集成、支付状态管理
        5. 报表统计：销售报表、用户统计、库存报表
        """,
        technology_stack=["Java", "Spring Boot", "MySQL", "Redis", "React"],
        business_domain="电商",
        complexity_level="中等"
    )
    
    try:
        # 创建工作流实例
        workflow = await create_function_point_workflow()
        
        # 执行估算
        logger.info("🔄 开始执行功能点估算...")
        
        result = await workflow.ainvoke({
            "project_info": project_info,
            "current_state": "STARTING",
            "execution_log": [],
            "retry_count": 0,
            "max_retries": 3
        })
        
        # 输出结果
        logger.info("✅ 功能点估算完成")
        
        if result.get("final_report"):
            report = result["final_report"]
            print("\n" + "="*50)
            print("📊 功能点估算报告")
            print("="*50)
            print(f"项目名称: {project_info.name}")
            print(f"估算策略: {result.get('selected_strategy', 'unknown')}")
            
            if result.get("nesma_ufp_total"):
                print(f"NESMA估算结果: {result['nesma_ufp_total']} UFP")
            
            if result.get("cosmic_cfp_total"):
                print(f"COSMIC估算结果: {result['cosmic_cfp_total']} CFP")
                
            print("="*50)
        else:
            logger.warning("⚠️ 未生成最终报告")
            
    except Exception as e:
        logger.error(f"❌ 功能点估算执行失败: {e}")
        raise


def main():
    """主程序入口"""
    print_banner()
    app()


if __name__ == "__main__":
    asyncio.run(main()) 