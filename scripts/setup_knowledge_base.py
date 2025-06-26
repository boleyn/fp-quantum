#!/usr/bin/env python3
"""
知识库初始化脚本 - 基于现有标准文档
针对已有的NESMA和COSMIC文档进行优化处理
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import os
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, base_dir: str = "knowledge_base"):
        self.base_dir = Path(base_dir)
        self.documents_dir = self.base_dir / "documents"
        
        # 文档映射
        self.document_mapping = {
            "nesma": {
                "NESMA_FPA_Method_v2.3.pdf": {
                    "title": "NESMA功能点分析方法v2.3",
                    "type": "official_standard",
                    "language": "英文",
                    "priority": "high",
                    "description": "NESMA官方功能点估算标准文档"
                }
            },
            "cosmic": {
                "COSMIC度量手册V5.0-part-1-原则、定义与规则.pdf": {
                    "title": "COSMIC度量手册-原则与规则",
                    "type": "official_standard",
                    "language": "中文",
                    "priority": "high",
                    "description": "COSMIC v5.0核心理论和定义"
                },
                "COSMIC度量手册V5.0-part-2-指南.pdf": {
                    "title": "COSMIC度量手册-实施指南",
                    "type": "implementation_guide",
                    "language": "中文", 
                    "priority": "high",
                    "description": "COSMIC v5.0实施操作指南"
                },
                "COSMIC度量手册V5.0-part-3-案例.pdf": {
                    "title": "COSMIC度量手册-案例集",
                    "type": "case_studies",
                    "language": "中文",
                    "priority": "medium",
                    "description": "COSMIC v5.0实际应用案例"
                },
                "COSMIC早期软件规模度量指南-实践级-Early-Software-Sizing（Practitioners.pdf": {
                    "title": "COSMIC早期估算-实践级",
                    "type": "early_sizing_guide",
                    "language": "中文",
                    "priority": "medium",
                    "description": "早期阶段功能点估算实践指南"
                },
                "COSMIC早期软件规模度量指南-–-专家级V2-Early-Software-Sizing（Experts.pdf": {
                    "title": "COSMIC早期估算-专家级",
                    "type": "advanced_guide",
                    "language": "中文",
                    "priority": "medium",
                    "description": "高级早期功能点估算指南"
                }
            },
            "common": {
                "NESMA_FPA_Method_v2.3.pdf": {
                    "title": "NESMA参考文档",
                    "type": "reference",
                    "language": "英文",
                    "priority": "medium",
                    "description": "通用NESMA参考文档"
                },
                "工作量拆分讲解V2.pptx": {
                    "title": "工作量拆分培训",
                    "type": "training_material",
                    "language": "中文",
                    "priority": "medium",
                    "description": "功能点工作量拆分培训材料"
                }
            }
        }
    
    def analyze_existing_documents(self) -> Dict[str, Any]:
        """分析现有文档"""
        
        logger.info("🔍 分析现有文档资源...")
        
        analysis = {
            "total_documents": 0,
            "by_category": {},
            "by_language": {"中文": 0, "英文": 0},
            "by_priority": {"high": 0, "medium": 0, "low": 0},
            "missing_files": [],
            "processing_plan": []
        }
        
        for category, docs in self.document_mapping.items():
            category_path = self.documents_dir / category
            category_info = {
                "found": 0,
                "missing": 0,
                "files": []
            }
            
            for filename, metadata in docs.items():
                file_path = category_path / filename
                if file_path.exists():
                    category_info["found"] += 1
                    category_info["files"].append({
                        "filename": filename,
                        "size": f"{file_path.stat().st_size / 1024 / 1024:.1f}MB",
                        "metadata": metadata
                    })
                    
                    # 统计
                    analysis["total_documents"] += 1
                    analysis["by_language"][metadata["language"]] += 1
                    analysis["by_priority"][metadata["priority"]] += 1
                    
                else:
                    category_info["missing"] += 1
                    analysis["missing_files"].append(f"{category}/{filename}")
            
            analysis["by_category"][category] = category_info
        
        return analysis
    
    def create_processing_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建文档处理计划"""
        
        plan = []
        
        # 高优先级文档优先处理
        high_priority_docs = []
        medium_priority_docs = []
        
        for category, info in analysis["by_category"].items():
            for file_info in info["files"]:
                if file_info["metadata"]["priority"] == "high":
                    high_priority_docs.append({
                        "category": category,
                        "file": file_info,
                        "processing_order": 1
                    })
                else:
                    medium_priority_docs.append({
                        "category": category,
                        "file": file_info,
                        "processing_order": 2
                    })
        
        plan.extend(high_priority_docs)
        plan.extend(medium_priority_docs)
        
        return plan
    
    def validate_document_quality(self, file_path: Path) -> Dict[str, Any]:
        """验证文档质量"""
        
        validation = {
            "readable": False,
            "size_ok": False,
            "format_supported": False,
            "estimated_pages": 0,
            "issues": []
        }
        
        try:
            # 检查文件大小
            size_mb = file_path.stat().st_size / 1024 / 1024
            if size_mb > 0.1:  # 至少100KB
                validation["size_ok"] = True
            else:
                validation["issues"].append("文件过小，可能损坏")
            
            # 检查文件格式
            if file_path.suffix.lower() in ['.pdf', '.pptx', '.docx']:
                validation["format_supported"] = True
            else:
                validation["issues"].append("不支持的文件格式")
            
            # 估算页面数（基于文件大小）
            if file_path.suffix.lower() == '.pdf':
                validation["estimated_pages"] = max(1, int(size_mb * 50))  # 粗略估算
            
            validation["readable"] = validation["size_ok"] and validation["format_supported"]
            
        except Exception as e:
            validation["issues"].append(f"文件访问错误: {e}")
        
        return validation
    
    def generate_enhancement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成知识库增强建议"""
        
        suggestions = []
        
        # 基于分析结果生成建议
        if analysis["total_documents"] >= 6:
            suggestions.append("✅ 核心文档齐全，可以开始RAG系统开发")
        
        if analysis["by_language"]["中文"] > analysis["by_language"]["英文"]:
            suggestions.append("🌏 中文文档丰富，建议优化中文分词和语义理解")
        
        if analysis["missing_files"]:
            suggestions.append(f"📥 建议补充缺失文档: {', '.join(analysis['missing_files'])}")
        
        # 按文档类型提供建议
        cosmic_docs = analysis["by_category"].get("cosmic", {}).get("found", 0)
        nesma_docs = analysis["by_category"].get("nesma", {}).get("found", 0)
        
        if cosmic_docs >= 4:
            suggestions.append("🎯 COSMIC文档完整，可以优先开发COSMIC估算模块")
        
        if nesma_docs >= 1:
            suggestions.append("📊 NESMA基础文档可用，建议补充更多实践案例")
        
        # 处理优化建议
        suggestions.extend([
            "🔧 建议使用unstructured库进行PDF解析，支持表格和图片提取",
            "📝 建议创建中英文术语对照表，提高检索准确性",
            "🎨 建议设置智能分块策略，按章节和主题分割文档",
            "🔍 建议配置多查询检索器，提高知识检索召回率",
            "📈 建议建立质量评估机制，监控RAG检索效果"
        ])
        
        return suggestions
    
    def create_setup_script(self, processing_plan: List[Dict[str, Any]]) -> str:
        """创建设置脚本"""
        
        script_content = f'''#!/usr/bin/env python3
"""
自动生成的知识库设置脚本
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import asyncio
from pathlib import Path
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AutoKnowledgeBaseSetup:
    def __init__(self):
        self.base_dir = Path("knowledge_base")
        self.processing_order = {processing_plan}
    
    async def setup_documents(self):
        """按计划处理文档"""
        
        # 中英文优化的分词器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", "。", ".", "!", "?", "；", ";"]
        )
        
        processed_docs = []
        
        for item in self.processing_order:
            category = item["category"]
            file_info = item["file"]
            
            file_path = self.base_dir / "documents" / category / file_info["filename"]
            
            if not file_path.exists():
                continue
            
            print(f"🔄 处理文档: {{file_info['metadata']['title']}}")
            
            # 选择合适的加载器
            if file_path.suffix.lower() == '.pdf':
                loader = UnstructuredPDFLoader(
                    str(file_path),
                    mode="elements",
                    strategy="hi_res"  # 高分辨率处理
                )
            elif file_path.suffix.lower() == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file_path))
            else:
                continue
            
            # 加载和分块
            docs = await loader.aload()
            split_docs = text_splitter.split_documents(docs)
            
            # 添加元数据
            for doc in split_docs:
                doc.metadata.update({{
                    "source_category": category,
                    "document_type": file_info["metadata"]["type"],
                    "language": file_info["metadata"]["language"],
                    "priority": file_info["metadata"]["priority"],
                    "title": file_info["metadata"]["title"]
                }})
            
            processed_docs.extend(split_docs)
            print(f"✅ 完成: {{len(split_docs)}} 个文档块")
        
        return processed_docs

if __name__ == "__main__":
    setup = AutoKnowledgeBaseSetup()
    asyncio.run(setup.setup_documents())
'''
        
        return script_content

async def main():
    """主函数"""
    
    print("🚀 量子智能化功能点估算系统 - 知识库分析")
    print("=" * 60)
    
    builder = KnowledgeBaseBuilder()
    
    # 分析现有文档
    print("\n📊 文档资源分析:")
    analysis = builder.analyze_existing_documents()
    
    print(f"📚 总文档数: {analysis['total_documents']}")
    print(f"🌏 语言分布: 中文 {analysis['by_language']['中文']} 份, 英文 {analysis['by_language']['英文']} 份")
    print(f"⭐ 优先级分布: 高 {analysis['by_priority']['high']} 份, 中 {analysis['by_priority']['medium']} 份")
    
    print("\n📂 分类详情:")
    for category, info in analysis["by_category"].items():
        print(f"  {category.upper()}: {info['found']} 份文档")
        for file_info in info["files"]:
            print(f"    ✅ {file_info['metadata']['title']} ({file_info['size']})")
    
    if analysis["missing_files"]:
        print(f"\n❌ 缺失文档: {len(analysis['missing_files'])} 份")
        for missing in analysis["missing_files"]:
            print(f"    ❌ {missing}")
    
    # 文档质量验证
    print("\n🔍 文档质量验证:")
    quality_issues = []
    
    for category, info in analysis["by_category"].items():
        for file_info in info["files"]:
            file_path = builder.documents_dir / category / file_info["filename"]
            validation = builder.validate_document_quality(file_path)
            
            if validation["readable"]:
                print(f"    ✅ {file_info['filename']}: 质量良好")
            else:
                print(f"    ❌ {file_info['filename']}: {', '.join(validation['issues'])}")
                quality_issues.extend(validation["issues"])
    
    # 生成处理计划
    print("\n📋 文档处理计划:")
    processing_plan = builder.create_processing_plan(analysis)
    
    for i, item in enumerate(processing_plan, 1):
        priority_icon = "🔥" if item["file"]["metadata"]["priority"] == "high" else "📄"
        print(f"  {i}. {priority_icon} {item['file']['metadata']['title']}")
    
    # 生成增强建议
    print("\n💡 知识库增强建议:")
    suggestions = builder.generate_enhancement_suggestions(analysis)
    
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    # 创建自动设置脚本
    print("\n🔧 生成自动化设置脚本...")
    script_content = builder.create_setup_script(processing_plan)
    
    script_path = builder.base_dir / "auto_setup.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 自动设置脚本已生成: {script_path}")
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📊 知识库状态总结:")
    print(f"✅ 可用文档: {analysis['total_documents']} 份")
    print(f"🎯 标准覆盖: COSMIC完整, NESMA基础")
    print(f"🌐 语言支持: 中英文双语")
    print(f"🔧 质量状态: {'良好' if not quality_issues else '需要检查'}")
    
    if analysis["total_documents"] >= 5:
        print("\n🎉 恭喜！您的知识库资源充足，可以开始AI系统开发！")
        print("\n🔥 推荐后续行动:")
        print("1. 运行 python knowledge_base/auto_setup.py 初始化向量存储")
        print("2. 测试RAG检索功能，验证知识库效果")
        print("3. 开始开发COSMIC估算智能体（文档最完整）")
        print("4. 逐步完善NESMA估算功能")
    else:
        print("\n⚠️ 建议补充更多文档资源后再开始开发")

if __name__ == "__main__":
    asyncio.run(main()) 