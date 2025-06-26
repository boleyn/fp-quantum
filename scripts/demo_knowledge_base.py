#!/usr/bin/env python3
"""
量子智能化功能点估算系统 - 知识库演示脚本

展示基于PgVector的知识库功能，包括：
- 文档加载和处理
- 向量存储和检索
- RAG链构建和查询
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from knowledge_base.loaders.pdf_loader import EnhancedPDFLoader
from knowledge_base.vector_stores.pgvector_store import PgVectorStore, create_pgvector_store
from knowledge_base.rag_chains import RAGChainBuilder
from knowledge_base.embeddings.embedding_models import get_embedding_model
from config.settings import get_settings

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base import (
    quick_setup_rag,
    RAGChainFactory,
    get_available_models,
    get_supported_formats,
    KNOWLEDGE_BASE_CONFIG
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseDemo:
    """知识库演示类"""
    
    def __init__(self):
        self.rag_system = None
        
    async def run_complete_demo(self):
        """运行完整演示"""
        
        print("🚀 欢迎使用量子智能化功能点估算知识库演示！")
        print("="*80)
        
        # 显示系统信息
        await self.show_system_info()
        
        # 演示快速设置
        await self.demo_quick_setup()
        
        # 演示文档加载
        await self.demo_document_loading()
        
        # 演示检索功能
        await self.demo_retrieval_features()
        
        # 演示RAG查询
        await self.demo_rag_queries()
        
        # 演示高级功能
        await self.demo_advanced_features()
        
        print("\n🎉 演示完成！感谢使用量子智能化功能点估算知识库。")
    
    async def show_system_info(self):
        """显示系统信息"""
        
        print("\n📋 系统信息")
        print("-" * 40)
        
        print(f"支持的文档格式 ({len(get_supported_formats())} 种):")
        for fmt in get_supported_formats():
            print(f"  • {fmt}")
        
        print(f"\n可用的嵌入模型 ({len(get_available_models())} 个):")
        for model in get_available_models():
            print(f"  • {model}")
        
        print("\n默认配置:")
        for key, value in KNOWLEDGE_BASE_CONFIG.items():
            print(f"  • {key}: {value}")
    
    async def demo_quick_setup(self):
        """演示快速设置"""
        
        print("\n🔧 演示1: 快速设置RAG系统")
        print("-" * 40)
        
        try:
            print("正在创建测试文档...")
            
            # 创建演示文档目录
            demo_dir = Path("demo_docs")
            demo_dir.mkdir(exist_ok=True)
            
            # 创建NESMA演示文档
            nesma_dir = demo_dir / "nesma"
            nesma_dir.mkdir(exist_ok=True)
            
            nesma_content = """
# NESMA功能点分析方法

## 功能类型分类

### 1. 内部逻辑文件 (ILF)
ILF是应用程序内部维护的用户可识别的数据组。
- 必须是用户可识别的
- 必须由应用程序内部维护
- 包含控制信息

### 2. 外部接口文件 (EIF)
EIF是其他应用程序维护的用户可识别的数据组。
- 被测量应用程序引用
- 由其他应用程序维护

### 3. 外部输入 (EI)
EI是从外部进入应用程序边界的数据。
- 维护一个或多个ILF
- 或改变系统行为

### 4. 外部输出 (EO)
EO是从应用程序内部发送到外部的数据。
- 包含计算、导出或处理的数据
- 改变系统行为

### 5. 外部查询 (EQ)
EQ是检索数据的请求。
- 不包含计算、导出或处理
- 不改变系统行为

## 复杂度计算

### DET (数据元素类型)
- 用户可识别、不重复的字段
- 控制信息不计算在内

### RET (记录元素类型)
- 用户可识别的子群组
- 每个ILF/EIF至少包含一个RET

### 复杂度矩阵
| 功能类型 | DET数量 | RET数量 | 复杂度 |
|---------|---------|---------|--------|
| ILF/EIF | 1-19    | 1       | Low    |
| ILF/EIF | 20-50   | 1       | Average|
| ILF/EIF | 51+     | 1       | High   |
"""
            
            (nesma_dir / "nesma_guide.md").write_text(nesma_content, encoding='utf-8')
            
            # 创建COSMIC演示文档
            cosmic_dir = demo_dir / "cosmic"
            cosmic_dir.mkdir(exist_ok=True)
            
            cosmic_content = """
# COSMIC功能点分析方法

## 基本概念

### 功能用户 (Functional User)
功能用户是发送或接收数据的用户、系统或设备。

### 软件边界 (Software Boundary)
软件边界定义了被测量软件的范围。

### 数据移动 (Data Movement)
COSMIC基于数据移动进行计量，包括四种类型：

## 数据移动类型

### 1. Entry (入口)
Entry将数据从功能用户移动到功能过程。
- 数据跨越软件边界进入
- 用于后续处理

### 2. Exit (出口)
Exit将数据从功能过程移动到功能用户。
- 数据跨越软件边界输出
- 向功能用户提供信息

### 3. Read (读取)
Read将数据从持久存储移动到功能过程。
- 从数据存储检索数据
- 用于功能过程处理

### 4. Write (写入)
Write将数据从功能过程移动到持久存储。
- 将数据存储到持久存储
- 保持数据状态

## CFP计算
每个数据移动等于1 CFP (COSMIC功能点)。
总CFP = Entry + Exit + Read + Write的数量

## 边界识别原则
1. 明确定义软件边界
2. 识别所有功能用户
3. 确定持久存储边界
4. 分析数据移动路径
"""
            
            (cosmic_dir / "cosmic_guide.md").write_text(cosmic_content, encoding='utf-8')
            
            print("✅ 演示文档创建完成")
            
            # 快速设置RAG系统
            print("正在设置RAG系统...")
            
            document_paths = {
                "nesma": str(nesma_dir),
                "cosmic": str(cosmic_dir)
            }
            
            self.rag_system = await quick_setup_rag(
                document_paths=document_paths,
                embedding_model="bge_m3",
                vector_store="pgvector",
                include_web=False
            )
            
            print("✅ RAG系统设置完成！")
            print(f"   - 向量存储类型: pgvector")
            print(f"   - 嵌入模型: bge_m3")
            print(f"   - 混合检索: 启用")
            
        except Exception as e:
            print(f"❌ 快速设置失败: {e}")
    
    async def demo_document_loading(self):
        """演示文档加载"""
        
        print("\n📚 演示2: 文档加载功能")
        print("-" * 40)
        
        try:
            from knowledge_base.loaders.custom_loaders import FunctionPointDocumentLoader
            
            # 演示不同格式的文档加载
            loader = FunctionPointDocumentLoader()
            
            # 加载目录
            demo_dir = Path("demo_docs")
            if demo_dir.exists():
                print("正在加载文档目录...")
                documents = loader.load_directory(demo_dir)
                print(f"✅ 从目录加载了 {len(documents)} 个文档块")
                
                # 显示文档统计
                doc_stats = {}
                for doc in documents:
                    source_type = doc.metadata.get('source_type', 'unknown')
                    doc_stats[source_type] = doc_stats.get(source_type, 0) + 1
                
                print("文档统计:")
                for source_type, count in doc_stats.items():
                    print(f"  • {source_type}: {count} 个文档块")
            
            # 演示单文件加载
            test_file = demo_dir / "nesma" / "nesma_guide.md"
            if test_file.exists():
                print(f"\n正在加载单个文件: {test_file.name}")
                single_docs = loader.load_file(test_file)
                print(f"✅ 从单文件加载了 {len(single_docs)} 个文档块")
                
                # 显示第一个文档块的信息
                if single_docs:
                    first_doc = single_docs[0]
                    print(f"首个文档块预览 ({len(first_doc.page_content)} 字符):")
                    print(f"   {first_doc.page_content[:200]}...")
                    print(f"   元数据: {first_doc.metadata}")
            
        except Exception as e:
            print(f"❌ 文档加载演示失败: {e}")
    
    async def demo_retrieval_features(self):
        """演示检索功能"""
        
        print("\n🔍 演示3: 检索功能")
        print("-" * 40)
        
        if not self.rag_system:
            print("❌ RAG系统未初始化，跳过检索演示")
            return
        
        try:
            # 演示语义检索
            print("1. 语义检索演示")
            semantic_retriever = self.rag_system.get_retriever("nesma_semantic")
            if semantic_retriever:
                results = semantic_retriever.search("什么是内部逻辑文件", k=3)
                print(f"   ✅ 语义检索返回 {len(results)} 个结果")
                if results:
                    print(f"   最相关结果: {results[0][0].page_content[:100]}...")
            
            # 演示关键词检索
            print("\n2. 关键词检索演示")
            keyword_retriever = self.rag_system.get_retriever("nesma_keyword")
            if keyword_retriever:
                results = keyword_retriever.search_function_rules("ILF", k=3)
                print(f"   ✅ 关键词检索返回 {len(results)} 个结果")
                if results:
                    print(f"   最相关结果: {results[0][0].page_content[:100]}...")
            
            # 演示混合检索
            print("\n3. 混合检索演示")
            hybrid_retriever = self.rag_system.get_retriever("nesma_hybrid")
            if hybrid_retriever:
                results = hybrid_retriever.hybrid_search("DET RET 复杂度计算", k=3)
                print(f"   ✅ 混合检索返回 {len(results)} 个结果")
                if results:
                    print(f"   最相关结果: {results[0][0].page_content[:100]}...")
            
        except Exception as e:
            print(f"❌ 检索功能演示失败: {e}")
    
    async def demo_rag_queries(self):
        """演示RAG查询"""
        
        print("\n🤖 演示4: RAG智能问答")
        print("-" * 40)
        
        if not self.rag_system:
            print("❌ RAG系统未初始化，跳过RAG查询演示")
            return
        
        # 模拟查询（由于没有真实的LLM，这里展示查询结构）
        test_questions = [
            "什么是ILF？它有什么特征？",
            "COSMIC中的Entry数据移动是什么意思？",
            "如何计算功能点的复杂度？",
            "DET和RET在复杂度计算中的作用是什么？"
        ]
        
        print("演示查询列表:")
        for i, question in enumerate(test_questions, 1):
            print(f"   {i}. {question}")
        
        print("\n注意: 完整的RAG查询需要配置大语言模型(LLM)")
        print("在生产环境中，这些查询将返回基于知识库的智能回答。")
        
        try:
            # 展示检索到的相关文档（不调用LLM）
            sample_question = test_questions[0]
            print(f"\n示例: 针对问题 '{sample_question}' 的文档检索:")
            
            # 使用混合检索获取相关文档
            hybrid_retriever = self.rag_system.get_retriever("nesma_hybrid")
            if hybrid_retriever:
                results = hybrid_retriever.hybrid_search(sample_question, k=2)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"   相关文档 {i} (相似度: {score:.3f}):")
                    print(f"   {doc.page_content[:200]}...")
                    print(f"   来源: {doc.metadata.get('source', '未知')}")
                    print()
            
        except Exception as e:
            print(f"❌ RAG查询演示失败: {e}")
    
    async def demo_advanced_features(self):
        """演示高级功能"""
        
        print("\n🎯 演示5: 高级功能")
        print("-" * 40)
        
        try:
            # 1. 向量存储统计
            print("1. 向量存储统计信息")
            from knowledge_base.vector_stores.pgvector_store import PgVectorStore
            from knowledge_base.embeddings.embedding_models import get_embedding_model
            
            embeddings = get_embedding_model("bge_m3")
            pgvector_store = PgVectorStore()
            
            # 显示支持的源类型
            source_types = pgvector_store.source_types
            print(f"   支持的源类型: {source_types}")
            
            for source_type in source_types[:2]:  # 只显示前2个
                try:
                    stats = await pgvector_store.get_collection_stats(source_type)
                    print(f"   源类型 '{source_type}': {stats.get('status', '未知')} 状态")
                except Exception:
                    print(f"   源类型 '{source_type}': 未连接")
            
            # 2. 嵌入模型比较
            print("\n2. 嵌入模型信息")
            available_models = get_available_models()
            current_model = get_embedding_model("bge_m3")
            
            # 测试嵌入生成
            test_text = "功能点估算是软件项目管理的重要技术"
            embedding = current_model.embed_query(test_text)
            print(f"   当前模型: bge_m3")
            print(f"   嵌入维度: {len(embedding)}")
            print(f"   测试文本: {test_text}")
            
            # 3. 文档格式支持
            print("\n3. 文档格式支持")
            supported_formats = get_supported_formats()
            print(f"   支持 {len(supported_formats)} 种格式:")
            for fmt in supported_formats[:5]:  # 只显示前5种
                print(f"   • {fmt}")
            
            # 4. 配置信息
            print("\n4. 系统配置")
            print(f"   默认块大小: {KNOWLEDGE_BASE_CONFIG['default_chunk_size']}")
            print(f"   块重叠大小: {KNOWLEDGE_BASE_CONFIG['default_chunk_overlap']}")
            print(f"   批处理大小: {KNOWLEDGE_BASE_CONFIG['batch_size']}")
            print(f"   支持语言: {', '.join(KNOWLEDGE_BASE_CONFIG['supported_languages'])}")
            
        except Exception as e:
            print(f"❌ 高级功能演示失败: {e}")
    
    def cleanup_demo_files(self):
        """清理演示文件"""
        
        print("\n🧹 清理演示文件...")
        
        try:
            import shutil
            
            # 清理演示文档
            demo_dir = Path("demo_docs")
            if demo_dir.exists():
                shutil.rmtree(demo_dir)
                print("✅ 演示文档已清理")
            
            # PgVector数据存储在PostgreSQL中，无需清理本地文件
            print("注意: PgVector数据存储在PostgreSQL数据库中")
            
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")


async def interactive_demo():
    """交互式演示"""
    
    demo = KnowledgeBaseDemo()
    
    while True:
        print("\n" + "="*60)
        print("🤖 量子智能化功能点估算知识库 - 交互式演示")
        print("="*60)
        print("1. 完整演示")
        print("2. 系统信息")
        print("3. 快速设置")
        print("4. 文档加载")
        print("5. 检索功能")
        print("6. 高级功能")
        print("7. 清理文件")
        print("0. 退出")
        
        try:
            choice = input("\n请选择功能 (0-7): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                await demo.run_complete_demo()
            elif choice == "2":
                await demo.show_system_info()
            elif choice == "3":
                await demo.demo_quick_setup()
            elif choice == "4":
                await demo.demo_document_loading()
            elif choice == "5":
                await demo.demo_retrieval_features()
            elif choice == "6":
                await demo.demo_advanced_features()
            elif choice == "7":
                demo.cleanup_demo_files()
            else:
                print("❌ 无效选择，请重试")
                
        except KeyboardInterrupt:
            print("\n👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"❌ 执行失败: {e}")
        
        input("\n按回车键继续...")
    
    # 询问是否清理
    try:
        cleanup = input("\n是否清理演示文件? (y/N): ").strip().lower()
        if cleanup in ['y', 'yes']:
            demo.cleanup_demo_files()
    except:
        pass


async def main():
    """主函数"""
    
    print("🚀 启动知识库演示程序...")
    
    try:
        # 检查是否为交互模式
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            await interactive_demo()
        else:
            # 运行完整演示
            demo = KnowledgeBaseDemo()
            await demo.run_complete_demo()
            
            # 询问是否清理
            try:
                cleanup = input("\n是否清理演示文件? (y/N): ").strip().lower()
                if cleanup in ['y', 'yes']:
                    demo.cleanup_demo_files()
            except KeyboardInterrupt:
                pass
        
    except KeyboardInterrupt:
        print("\n👋 演示被用户中断，感谢使用！")
    except Exception as e:
        print(f"\n❌ 演示执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 