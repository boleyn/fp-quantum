#!/usr/bin/env python3
"""
简化的知识库设置脚本
基于现有文档进行基础处理
"""

import asyncio
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SimpleKnowledgeBaseSetup:
    def __init__(self):
        self.base_dir = Path("knowledge_base")
        
        # 文档处理顺序（简化版）
        self.processing_plan = [
            {
                "category": "nesma",
                "filename": "NESMA_FPA_Method_v2.3.pdf",
                "title": "NESMA功能点分析方法v2.3",
                "priority": "high"
            },
            {
                "category": "cosmic", 
                "filename": "COSMIC度量手册V5.0-part-1-原则、定义与规则.pdf",
                "title": "COSMIC度量手册-原则与规则",
                "priority": "high"
            },
            {
                "category": "cosmic",
                "filename": "COSMIC度量手册V5.0-part-2-指南.pdf", 
                "title": "COSMIC度量手册-实施指南",
                "priority": "high"
            },
            {
                "category": "cosmic",
                "filename": "COSMIC度量手册V5.0-part-3-案例.pdf",
                "title": "COSMIC度量手册-案例集",
                "priority": "medium"
            }
        ]
    
    async def process_documents(self):
        """处理文档并生成分块"""
        
        print("🚀 开始处理知识库文档...")
        
        # 中英文优化的分词器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", "!", "?", "；", ";", "：", ":"]
        )
        
        all_processed_docs = []
        
        for item in self.processing_plan:
            category = item["category"]
            filename = item["filename"]
            title = item["title"]
            
            file_path = self.base_dir / "documents" / category / filename
            
            if not file_path.exists():
                print(f"⚠️ 文件不存在，跳过: {filename}")
                continue
            
            print(f"🔄 处理文档: {title}")
            
            try:
                # 使用简单的PDF加载器
                loader = PyPDFLoader(str(file_path))
                
                # 加载文档
                docs = loader.load()
                print(f"   📄 加载了 {len(docs)} 页内容")
                
                # 文本分块
                split_docs = text_splitter.split_documents(docs)
                print(f"   ✂️ 分割为 {len(split_docs)} 个文档块")
                
                # 添加元数据
                for doc in split_docs:
                    doc.metadata.update({
                        "source_category": category,
                        "document_title": title,
                        "filename": filename,
                        "priority": item["priority"],
                        "chunk_id": f"{category}_{len(all_processed_docs)}"
                    })
                
                all_processed_docs.extend(split_docs)
                print(f"   ✅ 完成处理: {title}")
                
            except Exception as e:
                print(f"   ❌ 处理失败 {title}: {e}")
                continue
        
        print(f"\n📊 文档处理完成!")
        print(f"   总文档块数: {len(all_processed_docs)}")
        
        # 统计各类文档
        categories = {}
        for doc in all_processed_docs:
            cat = doc.metadata.get("source_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        print("   分类统计:")
        for cat, count in categories.items():
            print(f"     {cat.upper()}: {count} 个文档块")
        
        # 保存示例到文件
        await self.save_sample_chunks(all_processed_docs[:5])
        
        return all_processed_docs
    
    async def save_sample_chunks(self, sample_docs):
        """保存示例文档块到文件"""
        
        sample_file = self.base_dir / "sample_chunks.txt"
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("知识库文档块示例\n")
            f.write("=" * 50 + "\n\n")
            
            for i, doc in enumerate(sample_docs, 1):
                f.write(f"文档块 {i}:\n")
                f.write(f"来源: {doc.metadata.get('document_title', '未知')}\n")
                f.write(f"类别: {doc.metadata.get('source_category', '未知')}\n")
                f.write(f"内容预览:\n{doc.page_content[:500]}...\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"📄 示例文档块已保存到: {sample_file}")
    
    async def test_basic_search(self, processed_docs):
        """测试基础搜索功能"""
        
        print("\n🔍 测试基础文本搜索...")
        
        test_keywords = [
            "功能点",
            "复杂度", 
            "数据移动",
            "ILF",
            "Entry"
        ]
        
        search_results = {}
        
        for keyword in test_keywords:
            matches = []
            for doc in processed_docs:
                if keyword.lower() in doc.page_content.lower():
                    matches.append({
                        "title": doc.metadata.get("document_title", "未知"),
                        "category": doc.metadata.get("source_category", "未知"),
                        "content_preview": doc.page_content[:200] + "..."
                    })
            
            search_results[keyword] = matches[:3]  # 只取前3个结果
            print(f"   关键词 '{keyword}': 找到 {len(matches)} 个匹配")
        
        # 保存搜索结果示例
        await self.save_search_results(search_results)
        
        return search_results
    
    async def save_search_results(self, results):
        """保存搜索结果示例"""
        
        results_file = self.base_dir / "search_test_results.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("基础搜索测试结果\n")
            f.write("=" * 50 + "\n\n")
            
            for keyword, matches in results.items():
                f.write(f"关键词: {keyword}\n")
                f.write(f"匹配数量: {len(matches)}\n\n")
                
                for i, match in enumerate(matches, 1):
                    f.write(f"  结果 {i}:\n")
                    f.write(f"    文档: {match['title']}\n") 
                    f.write(f"    类别: {match['category']}\n")
                    f.write(f"    内容: {match['content_preview']}\n\n")
                
                f.write("-" * 30 + "\n\n")
        
        print(f"🔍 搜索结果已保存到: {results_file}")

async def main():
    """主函数"""
    
    print("🚀 量子智能化功能点估算系统 - 简化知识库设置")
    print("=" * 60)
    
    setup = SimpleKnowledgeBaseSetup()
    
    try:
        # 处理文档
        processed_docs = await setup.process_documents()
        
        if processed_docs:
            # 测试基础搜索
            search_results = await setup.test_basic_search(processed_docs)
            
            print("\n" + "=" * 60)
            print("✅ 知识库基础设置完成!")
            print("\n💡 后续建议:")
            print("1. 检查生成的sample_chunks.txt文件，验证文档解析质量")
            print("2. 查看search_test_results.txt，了解基础搜索效果")
            print("3. 考虑设置向量存储以支持语义搜索")
            print("4. 开始开发功能点估算智能体")
            
        else:
            print("❌ 没有成功处理任何文档，请检查文档路径和格式")
            
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        
    print("\n🎉 处理完成!")

if __name__ == "__main__":
    asyncio.run(main()) 