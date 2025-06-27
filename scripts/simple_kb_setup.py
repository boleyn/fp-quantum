#!/usr/bin/env python3
"""
简单的知识库设置脚本
快速设置向量数据库用于测试
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_knowledge_base_main(
    nesma_path: Path = None,
    cosmic_path: Path = None,
    force_rebuild: bool = False
):
    """主要的知识库设置函数"""
    
    logger.info("🚀 开始设置知识库...")
    
    # 设置默认路径
    if not nesma_path:
        nesma_path = Path("knowledge_base/documents/nesma")
    if not cosmic_path:
        cosmic_path = Path("knowledge_base/documents/cosmic")
    
    try:
        # 检查文档是否存在
        nesma_docs = list(nesma_path.glob("*.pdf")) if nesma_path.exists() else []
        cosmic_docs = list(cosmic_path.glob("*.pdf")) if cosmic_path.exists() else []
        common_docs = list(Path("knowledge_base/documents/common").glob("*.md"))
        
        logger.info(f"📚 发现文档: NESMA {len(nesma_docs)} 份, COSMIC {len(cosmic_docs)} 份, 通用 {len(common_docs)} 份")
        
        # 创建知识库条目
        knowledge_entries = []
        
        # 添加NESMA规则
        nesma_rules = [
            {
                "id": "nesma_ei_rules",
                "content": """
NESMA外部输入(EI)识别规则：
1. 处理来自外部的数据输入
2. 可能导致内部逻辑文件的更新
3. 包含业务逻辑处理
4. 数据跨越系统边界

复杂度计算规则：
- DET(数据元素类型): 用户可识别的输入字段数量
- FTR(文件类型引用): 被引用或更新的逻辑文件数量

复杂度矩阵(EI):
- DET 1-14, FTR 0-1: Low
- DET 1-14, FTR 2: Low  
- DET 1-14, FTR 3+: Average
- DET 15-25, FTR 0-1: Low
- DET 15-25, FTR 2: Average
- DET 15-25, FTR 3+: High
- DET 26+, FTR 0-1: Average
- DET 26+, FTR 2: High
- DET 26+, FTR 3+: High
                """,
                "metadata": {"type": "nesma_rules", "function_type": "EI"}
            },
            {
                "id": "nesma_eo_rules", 
                "content": """
NESMA外部输出(EO)识别规则：
1. 向外部用户展示数据
2. 包含计算或数据处理逻辑
3. 从内部逻辑文件读取数据
4. 数据跨越系统边界输出

复杂度计算规则：
- DET: 用户可识别的输出字段数量
- FTR: 被读取的逻辑文件数量

复杂度矩阵(EO):
- DET 1-19, FTR 0-1: Low
- DET 1-19, FTR 2: Low
- DET 1-19, FTR 3+: Average
- DET 20-25, FTR 0-1: Low
- DET 20-25, FTR 2: Average  
- DET 20-25, FTR 3+: High
- DET 26+, FTR 0-1: Average
- DET 26+, FTR 2: High
- DET 26+, FTR 3+: High
                """,
                "metadata": {"type": "nesma_rules", "function_type": "EO"}
            },
            {
                "id": "nesma_eq_rules",
                "content": """
NESMA外部查询(EQ)识别规则：
1. 从外部发起的数据查询
2. 不更新内部逻辑文件
3. 简单的数据检索，无复杂计算
4. 输入和输出都跨越系统边界

复杂度计算规则：
- DET: 输入和输出字段总数
- FTR: 被查询的逻辑文件数量

复杂度矩阵(EQ):
- DET 1-19, FTR 0-1: Low
- DET 1-19, FTR 2: Low
- DET 1-19, FTR 3+: Average
- DET 20-25, FTR 0-1: Low
- DET 20-25, FTR 2: Average
- DET 20-25, FTR 3+: High
- DET 26+, FTR 0-1: Average
- DET 26+, FTR 2: High
- DET 26+, FTR 3+: High
                """,
                "metadata": {"type": "nesma_rules", "function_type": "EQ"}
            },
            {
                "id": "nesma_ilf_rules",
                "content": """
NESMA内部逻辑文件(ILF)识别规则：
1. 由应用程序维护的数据组
2. 存储在应用程序边界内
3. 用户可识别的业务数据
4. 通过EI功能进行维护

复杂度计算规则：
- DET: 用户可识别的数据字段数量
- RET: 记录元素类型数量(子组数量)

复杂度矩阵(ILF):
- DET 1-19, RET 1: Low
- DET 1-19, RET 2-5: Low
- DET 1-19, RET 6+: Average
- DET 20-50, RET 1: Low
- DET 20-50, RET 2-5: Average
- DET 20-50, RET 6+: High
- DET 51+, RET 1: Average
- DET 51+, RET 2-5: High
- DET 51+, RET 6+: High
                """,
                "metadata": {"type": "nesma_rules", "function_type": "ILF"}
            },
            {
                "id": "nesma_eif_rules",
                "content": """
NESMA外部接口文件(EIF)识别规则：
1. 由其他应用程序维护的数据组
2. 被当前应用程序引用
3. 存储在应用程序边界外
4. 用户可识别的业务数据

复杂度计算规则：
- DET: 用户可识别的数据字段数量
- RET: 记录元素类型数量(子组数量)

复杂度矩阵(EIF):
- DET 1-19, RET 1: Low
- DET 1-19, RET 2-5: Low
- DET 1-19, RET 6+: Average
- DET 20-50, RET 1: Low
- DET 20-50, RET 2-5: Average
- DET 20-50, RET 6+: High
- DET 51+, RET 1: Average
- DET 51+, RET 2-5: High
- DET 51+, RET 6+: High
                """,
                "metadata": {"type": "nesma_rules", "function_type": "EIF"}
            }
        ]
        
        knowledge_entries.extend(nesma_rules)
        
        # 添加通用知识
        general_knowledge = [
            {
                "id": "fp_estimation_basics",
                "content": """
功能点估算基础知识：

1. 功能点估算的目的：
   - 衡量软件规模
   - 项目工作量估算
   - 成本预测
   - 质量基准

2. 主要估算标准：
   - NESMA: 荷兰软件度量协会标准
   - COSMIC: ISO/IEC 19761国际标准
   - IFPUG: 国际功能点用户组标准

3. 估算流程：
   - 需求分析
   - 功能识别
   - 功能分类
   - 复杂度计算
   - 功能点计算
                """,
                "metadata": {"type": "general_knowledge", "topic": "fp_basics"}
            }
        ]
        
        knowledge_entries.extend(general_knowledge)
        
        logger.info(f"✅ 创建了 {len(knowledge_entries)} 条知识库条目")
        
        # 这里可以添加向量数据库存储逻辑
        # 但为了简单起见，我们先创建一个内存知识库
        
        # 保存到文件供系统使用
        import json
        kb_file = Path("knowledge_base/simple_kb.json")
        kb_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_entries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 知识库已保存到: {kb_file}")
        
        return {
            "status": "success",
            "entries_count": len(knowledge_entries),
            "file_path": str(kb_file)
        }
        
    except Exception as e:
        logger.error(f"❌ 知识库设置失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_knowledge_base_main(force_rebuild=True)) 