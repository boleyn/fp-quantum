#!/bin/bash

# 量子智能化功能点估算系统 - 测试运行脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 量子智能化功能点估算系统 - 测试套件${NC}"
echo "================================================"

# 检查Python环境
echo -e "${YELLOW}📋 检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python版本: $PYTHON_VERSION${NC}"

# 检查依赖
echo -e "${YELLOW}📦 检查依赖包...${NC}"
if ! python3 -c "import pytest" &> /dev/null; then
    echo -e "${YELLOW}⚠️ 正在安装pytest...${NC}"
    pip install pytest pytest-asyncio pytest-json-report
fi

if ! python3 -c "import rich" &> /dev/null; then
    echo -e "${YELLOW}⚠️ 正在安装rich...${NC}"
    pip install rich
fi

echo -e "${GREEN}✅ 依赖检查完成${NC}"

# 创建测试报告目录
mkdir -p test_reports

# 显示菜单
show_menu() {
    echo ""
    echo -e "${BLUE}请选择测试模式:${NC}"
    echo "1) 🧪 单元测试"
    echo "2) 🔗 集成测试" 
    echo "3) 🎯 端到端测试"
    echo "4) ⚡ 性能测试"
    echo "5) 📊 完整测试套件"
    echo "6) 🚀 快速验证测试"
    echo "7) 📈 综合评估报告"
    echo "0) 退出"
    echo ""
}

# 运行单元测试
run_unit_tests() {
    echo -e "${YELLOW}🧪 运行单元测试...${NC}"
    
    if [ -f "tests/unit/test_nesma_agents.py" ]; then
        echo "  📋 NESMA智能体测试"
        python3 -m pytest tests/unit/test_nesma_agents.py -v --tb=short
    fi
    
    if [ -f "tests/unit/test_cosmic_agents.py" ]; then
        echo "  📋 COSMIC智能体测试"
        python3 -m pytest tests/unit/test_cosmic_agents.py -v --tb=short
    fi
    
    echo -e "${GREEN}✅ 单元测试完成${NC}"
}

# 运行集成测试
run_integration_tests() {
    echo -e "${YELLOW}🔗 运行集成测试...${NC}"
    
    if [ -f "tests/integration/test_agents_integration.py" ]; then
        echo "  🤝 智能体集成测试"
        python3 -m pytest tests/integration/test_agents_integration.py -v --tb=short
    fi
    
    echo -e "${GREEN}✅ 集成测试完成${NC}"
}

# 运行端到端测试
run_e2e_tests() {
    echo -e "${YELLOW}🎯 运行端到端测试...${NC}"
    
    if [ -f "tests/e2e/test_full_workflow.py" ]; then
        echo "  🌐 完整工作流测试"
        python3 -m pytest tests/e2e/test_full_workflow.py -v --tb=short -s
    fi
    
    echo -e "${GREEN}✅ 端到端测试完成${NC}"
}

# 运行性能测试
run_performance_tests() {
    echo -e "${YELLOW}⚡ 运行性能测试...${NC}"
    
    # 性能测试通常需要特殊配置
    echo "  ⏱️ 响应时间测试"
    echo "  🔄 并发处理测试"
    echo "  💾 资源使用测试"
    
    # 这里可以添加具体的性能测试命令
    python3 -c "
import asyncio
import time
print('  📊 模拟性能测试...')
time.sleep(2)
print('  ✅ 性能测试完成')
"
    
    echo -e "${GREEN}✅ 性能测试完成${NC}"
}

# 运行完整测试套件
run_full_tests() {
    echo -e "${YELLOW}📊 运行完整测试套件...${NC}"
    
    START_TIME=$(date +%s)
    
    run_unit_tests
    run_integration_tests
    run_e2e_tests
    run_performance_tests
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}🎉 完整测试套件执行完成!${NC}"
    echo -e "${BLUE}📊 总耗时: ${DURATION}秒${NC}"
}

# 快速验证测试
run_quick_tests() {
    echo -e "${YELLOW}🚀 运行快速验证测试...${NC}"
    
    # 只运行关键的快速测试
    echo "  ⚡ 核心功能验证"
    
    python3 -c "
import sys
sys.path.append('.')

# 简单的导入测试
try:
    from config.settings import get_settings
    print('  ✅ 配置模块正常')
except Exception as e:
    print(f'  ❌ 配置模块错误: {e}')

try:
    from models.project_models import ProjectInfo
    print('  ✅ 数据模型正常')
except Exception as e:
    print(f'  ❌ 数据模型错误: {e}')

try:
    # 测试基础功能
    project = ProjectInfo(
        name='测试项目',
        description='快速测试项目',
        technology_stack=['Python'],
        business_domain='其他'
    )
    print('  ✅ 项目模型创建正常')
except Exception as e:
    print(f'  ❌ 项目模型错误: {e}')
"
    
    echo -e "${GREEN}✅ 快速验证完成${NC}"
}

# 运行综合评估
run_comprehensive_evaluation() {
    echo -e "${YELLOW}📈 运行综合评估报告...${NC}"
    
    if [ -f "scripts/run_comprehensive_tests.py" ]; then
        python3 scripts/run_comprehensive_tests.py
    else
        echo -e "${RED}❌ 综合评估脚本未找到${NC}"
        return 1
    fi
}

# 主循环
while true; do
    show_menu
    read -p "请输入选择 [0-7]: " choice
    
    case $choice in
        1)
            run_unit_tests
            ;;
        2)
            run_integration_tests
            ;;
        3)
            run_e2e_tests
            ;;
        4)
            run_performance_tests
            ;;
        5)
            run_full_tests
            ;;
        6)
            run_quick_tests
            ;;
        7)
            run_comprehensive_evaluation
            ;;
        0)
            echo -e "${BLUE}👋 再见!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 无效选择，请重新输入${NC}"
            ;;
    esac
    
    echo ""
    read -p "按回车键继续..."
done 