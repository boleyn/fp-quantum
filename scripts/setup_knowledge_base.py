#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 知识库设置脚本

使用增量更新机制管理知识库：
- 检测文件变化（新增、修改、删除）
- 自动更新PgVector向量存储
- MongoDB记录文档状态
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.auto_setup import IncrementalKnowledgeBaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('knowledge_base_setup.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    logger.info("🚀 启动知识库设置脚本...")
    
    manager = IncrementalKnowledgeBaseManager()
    
    try:
        # 执行增量更新
        result = await manager.auto_update_knowledge_base()
        
        # 输出结果
        print("\n" + "="*60)
        print("📊 知识库更新结果")
        print("="*60)
        print(f"状态: {result['status']}")
        print(f"消息: {result['message']}")
        print(f"耗时: {result.get('duration', 0):.2f} 秒")
        
        if result.get('changes'):
            changes = result['changes']
            print(f"\n变化统计:")
            print(f"  新增文件: {changes['new_files']}")
            print(f"  修改文件: {changes['modified_files']}")
            print(f"  删除文件: {changes['deleted_files']}")
            print(f"  未变化文件: {changes['unchanged_files']}")
            print(f"  总处理数: {result.get('total_processed', 0)}")
        
        print(f"\n支持类别: {result.get('categories', [])}")
        print("="*60)
        
        if result['status'] == 'success':
            print("✅ 知识库更新成功！")
            return 0
        else:
            print("❌ 知识库更新失败！")
            return 1
            
    except Exception as e:
        logger.error(f"❌ 脚本执行失败: {e}")
        print(f"❌ 执行失败: {e}")
        return 1
        
    finally:
        await manager.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 