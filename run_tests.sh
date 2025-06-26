#!/bin/bash

# 量子智能化功能点估算系统 - 测试运行脚本
# 提供交互式菜单来运行不同类型的测试

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 输出带颜色的文本
echo_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 显示标题
show_banner() {
    echo_color $CYAN "════════════════════════════════════════════════════════════════"
    echo_color $CYAN "        量子智能化功能点估算系统 - 测试运行器"
    echo_color $CYAN "        Quantum Function Point Estimation System - Test Runner"
    echo_color $CYAN "════════════════════════════════════════════════════════════════"
    echo
}

# 显示菜单
show_menu() {
    echo_color $BLUE "请选择要执行的测试类型："
    echo
    echo_color $GREEN "  1. 🔬 单元测试 (Unit Tests)"
    echo_color $GREEN "  2. 🔗 集成测试 (Integration Tests)"
    echo_color $GREEN "  3. 🎯 端到端测试 (End-to-End Tests)"
    echo_color $GREEN "  4. ⚡ 性能测试 (Performance Tests)"
    echo_color $GREEN "  5. 📚 知识库测试 (Knowledge Base Tests)"
    echo_color $GREEN "  6. 🚀 完整测试套件 (All Tests)"
    echo_color $GREEN "  7. 📊 综合评估报告 (Comprehensive Assessment)"
    echo_color $GREEN "  8. 🔧 快速验证测试 (Quick Validation)"
    echo
    echo_color $YELLOW "  0. 退出 (Exit)"
    echo
}

# 运行单元测试
run_unit_tests() {
    echo_color $BLUE "🔬 正在运行单元测试..."
    python -m pytest tests/unit/ -v --tb=short
    echo_color $GREEN "✅ 单元测试完成"
}

# 运行集成测试
run_integration_tests() {
    echo_color $BLUE "🔗 正在运行集成测试..."
    python -m pytest tests/integration/ -v --tb=short
    echo_color $GREEN "✅ 集成测试完成"
}

# 运行端到端测试
run_e2e_tests() {
    echo_color $BLUE "🎯 正在运行端到端测试..."
    python -m pytest tests/e2e/ -v --tb=short -s
    echo_color $GREEN "✅ 端到端测试完成"
}

# 运行性能测试
run_performance_tests() {
    echo_color $BLUE "⚡ 正在运行性能测试..."
    python scripts/run_comprehensive_tests.py performance
    echo_color $GREEN "✅ 性能测试完成"
}

# 运行知识库测试
run_knowledge_base_tests() {
    echo_color $BLUE "📚 正在运行知识库测试..."
    python scripts/test_knowledge_base.py
    echo_color $GREEN "✅ 知识库测试完成"
}

# 运行完整测试套件
run_all_tests() {
    echo_color $BLUE "🚀 正在运行完整测试套件..."
    python scripts/run_comprehensive_tests.py all --performance
    echo_color $GREEN "✅ 完整测试套件完成"
}

# 运行综合评估
run_comprehensive_assessment() {
    echo_color $BLUE "📊 正在进行综合评估..."
    python scripts/run_comprehensive_tests.py all --performance --output assessment_report.json
    echo_color $GREEN "✅ 综合评估完成，报告已保存到 assessment_report.json"
}

# 快速验证测试
run_quick_validation() {
    echo_color $BLUE "🔧 正在进行快速验证..."
    
    # 检查依赖
    echo_color $YELLOW "检查Python依赖..."
    python -c "import agents, graph, models, knowledge_base; print('✅ 核心模块导入成功')"
    
    # 运行关键测试
    echo_color $YELLOW "运行关键测试用例..."
    python -m pytest tests/unit/test_nesma_agents.py::TestNESMAFunctionClassifier::test_classify_external_input -v
    python -m pytest tests/unit/test_cosmic_agents.py::TestCOSMICFunctionalUserAgent -v -k "test_identify_functional_users"
    
    echo_color $GREEN "✅ 快速验证完成"
}

# 检查环境
check_environment() {
    echo_color $YELLOW "检查运行环境..."
    
    # 检查Python版本
    python_version=$(python --version 2>&1)
    echo_color $CYAN "Python版本: $python_version"
    
    # 检查虚拟环境
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo_color $GREEN "✅ 虚拟环境: $VIRTUAL_ENV"
    else
        echo_color $YELLOW "⚠️  未检测到虚拟环境"
    fi
    
    # 检查必要文件
    required_files=("pyproject.toml" "main.py" "tests/" "agents/" "graph/")
    for file in "${required_files[@]}"; do
        if [[ -e "$file" ]]; then
            echo_color $GREEN "✅ $file 存在"
        else
            echo_color $RED "❌ $file 缺失"
            exit 1
        fi
    done
    
    echo_color $GREEN "✅ 环境检查通过"
    echo
}

# 主函数
main() {
    show_banner
    check_environment
    
    while true; do
        show_menu
        echo -n "请输入选项 (0-8): "
        read choice
        echo
        
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
                run_knowledge_base_tests
                ;;
            6)
                run_all_tests
                ;;
            7)
                run_comprehensive_assessment
                ;;
            8)
                run_quick_validation
                ;;
            0)
                echo_color $BLUE "感谢使用测试运行器！"
                exit 0
                ;;
            *)
                echo_color $RED "无效选项，请重新选择。"
                ;;
        esac
        
        echo
        echo_color $YELLOW "按 Enter 键继续..."
        read
        echo
    done
}

# 如果直接运行脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 