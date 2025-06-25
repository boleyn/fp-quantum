#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 知识库初始化脚本

整合文档加载、向量化、存储的完整知识库设置流程
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_base.loaders.pdf_loader import load_knowledge_base_pdfs
from knowledge_base.vector_stores.mongodb_atlas import setup_mongodb_vector_stores
from knowledge_base.embeddings.embedding_models import get_default_embedding_model, test_embedding_model
from config.settings import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查前置条件...")
    
    settings = get_settings()
    
    # 检查必要的配置
    checks = {
        "MongoDB连接": bool(settings.database.mongodb_url),
        "嵌入模型API": bool(settings.llm.bge_m3_api_key or settings.llm.openai_api_key),
        "NESMA文档目录": settings.knowledge_base.nesma_docs_path.exists(),
        "COSMIC文档目录": settings.knowledge_base.cosmic_docs_path.exists(),
        "通用文档目录": settings.knowledge_base.common_docs_path.exists()
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n⚠️ 部分前置条件不满足，但仍可以继续...")
        
    # 测试嵌入模型
    print("\n🧪 测试嵌入模型...")
    model_working = await test_embedding_model()
    if not model_working:
        print("❌ 嵌入模型测试失败，请检查API配置")
        return False
    
    return True


async def create_document_directories():
    """创建文档目录"""
    print("📁 确保文档目录存在...")
    
    settings = get_settings()
    directories = [
        settings.knowledge_base.nesma_docs_path,
        settings.knowledge_base.cosmic_docs_path,
        settings.knowledge_base.common_docs_path
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")


async def load_and_process_documents():
    """加载和处理文档"""
    print("📚 加载知识库文档...")
    
    try:
        documents_by_type = await load_knowledge_base_pdfs()
        
        total_docs = sum(len(docs) for docs in documents_by_type.values())
        print(f"\n📊 文档加载完成，总计 {total_docs} 个文档块:")
        
        for source_type, docs in documents_by_type.items():
            print(f"  {source_type.upper()}: {len(docs)} 个文档块")
            
            # 显示示例文档
            if docs:
                sample_doc = docs[0]
                print(f"    示例: {sample_doc.metadata.get('file_name', 'unknown')}")
                print(f"    内容预览: {sample_doc.page_content[:100]}...")
        
        return documents_by_type
        
    except Exception as e:
        logger.error(f"❌ 文档加载失败: {str(e)}")
        return {}


async def setup_vector_storage(documents_by_type: Dict[str, List]):
    """设置向量存储"""
    print("\n🔗 设置向量存储...")
    
    if not documents_by_type:
        print("⚠️ 没有文档可供向量化，跳过向量存储设置")
        return None
    
    try:
        # 获取嵌入模型
        embeddings = get_default_embedding_model()
        
        # 设置MongoDB Atlas向量存储
        vector_manager = await setup_mongodb_vector_stores(
            documents_by_type, 
            embeddings
        )
        
        print("✅ 向量存储设置完成")
        
        # 获取统计信息
        print("\n📊 向量存储统计:")
        for source_type in documents_by_type.keys():
            try:
                stats = await vector_manager.get_collection_stats(source_type)
                print(f"  {source_type.upper()}:")
                print(f"    文档数量: {stats['total_documents']}")
                print(f"    集合名称: {stats['collection_name']}")
                if stats['standards']:
                    for standard, info in stats['standards'].items():
                        print(f"    {standard}: {info['count']} 个文档")
            except Exception as e:
                print(f"    ❌ 获取统计失败: {str(e)}")
        
        return vector_manager
        
    except Exception as e:
        logger.error(f"❌ 向量存储设置失败: {str(e)}")
        return None


async def test_retrieval_system(vector_manager):
    """测试检索系统"""
    print("\n🔍 测试检索系统...")
    
    if not vector_manager:
        print("⚠️ 向量管理器不可用，跳过检索测试")
        return
    
    try:
        from knowledge_base.retrievers.semantic_retriever import create_knowledge_retrievers
        
        # 创建检索器
        retrievers = await create_knowledge_retrievers(vector_manager)
        multi_retriever = retrievers["multi_source"]
        
        # 测试查询
        test_queries = [
            "功能点分类规则",
            "复杂度计算方法",
            "数据移动类型"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n  测试查询 {i}: {query}")
            
            try:
                result = await multi_retriever.adaptive_retrieve(query, min_chunks=1)
                
                print(f"    📊 检索到 {len(result.retrieved_chunks)} 个结果")
                print(f"    ⏱️ 耗时: {result.retrieval_time_ms}ms")
                
                if result.retrieved_chunks:
                    best_chunk = result.retrieved_chunks[0]
                    print(f"    🎯 最佳匹配 (分数: {best_chunk.relevance_score:.3f})")
                    print(f"       来源: {best_chunk.source_type.value}")
                
            except Exception as e:
                print(f"    ❌ 查询失败: {str(e)}")
        
        print("\n✅ 检索系统测试完成")
        
    except Exception as e:
        logger.error(f"❌ 检索系统测试失败: {str(e)}")


async def generate_setup_report(vector_manager, documents_by_type):
    """生成设置报告"""
    print("\n📋 生成设置报告...")
    
    report = {
        "setup_time": asyncio.get_event_loop().time(),
        "document_stats": {
            source_type: len(docs) 
            for source_type, docs in documents_by_type.items()
        },
        "total_documents": sum(len(docs) for docs in documents_by_type.values()),
        "vector_storage": "MongoDB Atlas" if vector_manager else "未设置",
        "embedding_model": "BGE-M3 (默认)",
        "status": "完成"
    }
    
    print("📊 设置报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 保存报告到文件
    import json
    report_file = Path("knowledge_base_setup_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 报告已保存到: {report_file}")
    
    return report


async def main():
    """主函数"""
    print("🚀 开始知识库初始化...")
    print("=" * 60)
    
    try:
        # 1. 检查前置条件
        if not await check_prerequisites():
            print("❌ 前置条件检查失败，终止初始化")
            return
        
        # 2. 创建目录
        await create_document_directories()
        
        # 3. 加载文档
        documents_by_type = await load_and_process_documents()
        
        # 4. 设置向量存储
        vector_manager = await setup_vector_storage(documents_by_type)
        
        # 5. 测试检索系统
        await test_retrieval_system(vector_manager)
        
        # 6. 生成报告
        report = await generate_setup_report(vector_manager, documents_by_type)
        
        # 7. 清理
        if vector_manager:
            await vector_manager.close()
        
        print("\n" + "=" * 60)
        print("✅ 知识库初始化完成！")
        
        # 使用建议
        print("\n💡 使用建议:")
        print("1. 将PDF文档放入对应的文档目录:")
        print(f"   - NESMA: knowledge_base/documents/nesma/")
        print(f"   - COSMIC: knowledge_base/documents/cosmic/")
        print(f"   - 通用: knowledge_base/documents/common/")
        print("2. 重新运行此脚本来更新知识库")
        print("3. 使用 main.py estimate 命令开始功能点估算")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断，知识库初始化已取消")
    except Exception as e:
        logger.error(f"❌ 知识库初始化失败: {str(e)}")
        print(f"\n❌ 初始化失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # 设置事件循环策略 (Windows兼容性)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 