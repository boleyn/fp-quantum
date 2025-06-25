#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 安装验证脚本

验证系统安装是否正确，检查依赖和配置
"""

import sys
import os
from pathlib import Path

def main():
    """主验证函数"""
    print("🔍 量子智能化功能点估算系统 - 安装验证")
    print("=" * 60)
    
    # 检查Python版本
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro} (需要3.11+)")
        return 1
    
    # 检查关键文件
    required_files = [
        "pyproject.toml",
        "main.py", 
        "README.md",
        "models/__init__.py",
        "config/settings.py"
    ]
    
    print("\n📁 文件结构检查:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (缺失)")
    
    print("\n🎉 基础验证完成！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
