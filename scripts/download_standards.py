#!/usr/bin/env python3
"""
功能点估算标准文档下载脚本

由于NESMA和COSMIC官方标准文档通常是付费的，本脚本提供多种获取途径和替代资源
"""

import os
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StandardDocumentDownloader:
    """标准文档下载器"""
    
    def __init__(self, base_dir: str = "knowledge_base/documents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.base_dir / "nesma").mkdir(exist_ok=True)
        (self.base_dir / "cosmic").mkdir(exist_ok=True)
        (self.base_dir / "common").mkdir(exist_ok=True)
        (self.base_dir / "supplementary").mkdir(exist_ok=True)
    
    def get_official_sources_info(self) -> Dict[str, Any]:
        """获取官方文档来源信息"""
        return {
            "NESMA": {
                "official_website": "https://www.nesma.org/",
                "documents": {
                    "NESMA FPA Method v2.3": {
                        "description": "NESMA功能点分析方法官方标准",
                        "url": "https://www.nesma.org/products/",
                        "price": "付费文档",
                        "language": "英文",
                        "status": "✅ 已获得"
                    },
                    "NESMA Desktop Reference": {
                        "description": "NESMA桌面参考手册",
                        "url": "https://www.nesma.org/products/",
                        "price": "付费文档",
                        "language": "英文",
                        "status": "❌ 建议补充"
                    }
                }
            },
            "COSMIC": {
                "official_website": "https://cosmic-sizing.org/",
                "documents": {
                    "COSMIC Measurement Manual v5.0": {
                        "description": "COSMIC度量手册v5.0完整版",
                        "parts": [
                            "Part 1: 原则、定义与规则 ✅ 已获得",
                            "Part 2: 指南 ✅ 已获得", 
                            "Part 3: 案例 ✅ 已获得"
                        ],
                        "url": "https://cosmic-sizing.org/publications/",
                        "price": "免费",
                        "language": "多语言",
                        "status": "✅ 完整获得"
                    },
                    "COSMIC Early Software Sizing Guide": {
                        "description": "COSMIC早期软件规模度量指南",
                        "levels": [
                            "实践级 ✅ 已获得",
                            "专家级V2 ✅ 已获得"
                        ],
                        "url": "https://cosmic-sizing.org/publications/",
                        "price": "免费",
                        "language": "中英文",
                        "status": "✅ 完整获得"
                    }
                }
            }
        }
    
    def get_supplementary_resources(self) -> List[Dict[str, str]]:
        """获取补充资源信息"""
        return [
            {
                "name": "ISO/IEC 14143-1:2007",
                "description": "软件测量 - 功能规模测量 - 第1部分：概念定义",
                "type": "ISO标准",
                "source": "ISO官网",
                "url": "https://www.iso.org/standard/44553.html",
                "price": "付费",
                "importance": "高 - 功能点估算理论基础"
            },
            {
                "name": "ISO/IEC 29881:2010", 
                "description": "软件和系统工程 - FiSMA 1.1功能规模测量方法",
                "type": "ISO标准",
                "source": "ISO官网", 
                "url": "https://www.iso.org/standard/45746.html",
                "price": "付费",
                "importance": "中 - 替代功能点方法"
            },
            {
                "name": "IFPUG计数实践手册",
                "description": "国际功能点用户组计数实践手册",
                "type": "行业标准",
                "source": "IFPUG官网",
                "url": "https://www.ifpug.org/",
                "price": "付费",
                "importance": "高 - 经典功能点方法"
            },
            {
                "name": "学术论文和案例研究",
                "description": "功能点估算相关的学术研究论文",
                "type": "学术资源",
                "source": "IEEE Xplore, ACM Digital Library",
                "url": "多个学术数据库",
                "price": "部分免费",
                "importance": "中 - 研究前沿和案例"
            },
            {
                "name": "行业最佳实践报告",
                "description": "软件工程行业的功能点估算最佳实践",
                "type": "行业报告",
                "source": "咨询公司、行业协会",
                "url": "多个来源",
                "price": "部分免费",
                "importance": "中 - 实践经验"
            }
        ]
    
    def get_free_alternatives(self) -> List[Dict[str, str]]:
        """获取免费替代资源"""
        return [
            {
                "name": "COSMIC官方培训材料",
                "description": "COSMIC官网提供的免费培训幻灯片和资料",
                "url": "https://cosmic-sizing.org/training-material/",
                "type": "培训材料",
                "language": "英文",
                "format": "PDF, PPT"
            },
            {
                "name": "功能点估算开源工具",
                "description": "GitHub上的功能点估算工具和示例",
                "url": "https://github.com/search?q=function+point+estimation",
                "type": "开源软件",
                "language": "多语言",
                "format": "源代码, 文档"
            },
            {
                "name": "大学课程资料",
                "description": "软件工程课程中的功能点估算教学材料",
                "url": "各大学开放课程网站",
                "type": "教学资料", 
                "language": "多语言",
                "format": "PDF, 视频"
            },
            {
                "name": "技术博客和文章",
                "description": "技术专家撰写的功能点估算实践文章",
                "url": "Medium, CSDN, 博客园等技术平台",
                "type": "实践文章",
                "language": "中英文",
                "format": "网页, PDF"
            }
        ]
    
    async def create_knowledge_enhancement_plan(self):
        """创建知识库增强计划"""
        
        plan = {
            "current_status": "🎉 已拥有核心标准文档，基础完备",
            "immediate_actions": [
                "✅ 验证现有PDF文档的完整性和可读性",
                "✅ 设置文档处理管道，确保能正确解析中英文内容", 
                "✅ 建立基础的RAG检索系统",
                "✅ 测试现有文档的查询效果"
            ],
            "short_term_enhancements": [
                "📥 下载COSMIC官方免费培训材料作为补充",
                "📝 收集中文功能点估算实践案例",
                "🔄 添加行业特定的功能点估算指南",
                "📊 收集不同技术栈的功能点估算参考数据"
            ],
            "long_term_goals": [
                "💰 考虑购买NESMA Desktop Reference等高级文档", 
                "🎓 收集学术论文和最新研究成果",
                "🏢 建立企业级功能点估算知识库",
                "🌐 支持多行业、多技术栈的估算标准"
            ]
        }
        
        return plan
    
    def download_free_cosmic_materials(self):
        """下载免费的COSMIC补充材料"""
        
        cosmic_free_urls = [
            {
                "name": "COSMIC介绍材料",
                "url": "https://cosmic-sizing.org/wp-content/uploads/2019/04/Introduction-to-COSMIC.pdf",
                "filename": "COSMIC_Introduction.pdf"
            },
            {
                "name": "COSMIC快速参考指南",
                "url": "https://cosmic-sizing.org/wp-content/uploads/2019/04/COSMIC-Quick-Reference.pdf", 
                "filename": "COSMIC_Quick_Reference.pdf"
            }
        ]
        
        downloaded = []
        for material in cosmic_free_urls:
            try:
                response = requests.get(material["url"], timeout=30)
                if response.status_code == 200:
                    file_path = self.base_dir / "cosmic" / material["filename"]
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    downloaded.append(material["name"])
                    logger.info(f"✅ 下载成功: {material['name']}")
            except Exception as e:
                logger.warning(f"❌ 下载失败 {material['name']}: {e}")
        
        return downloaded
    
    def create_supplementary_documents(self):
        """创建补充文档"""
        
        # 创建功能点估算快速参考手册
        quick_ref_content = """
# 功能点估算快速参考手册

## NESMA功能类型

### 数据功能
- **ILF (内部逻辑文件)**: 应用程序内部维护的逻辑相关数据组
- **EIF (外部接口文件)**: 其他应用程序维护但本应用程序引用的逻辑相关数据组

### 事务功能  
- **EI (外部输入)**: 从应用程序边界外部输入数据或控制信息的基本流程
- **EO (外部输出)**: 向应用程序边界外部发送数据的基本流程
- **EQ (外部查询)**: 向应用程序边界外部发送数据的基本流程，无派生数据

## COSMIC数据移动类型

- **Entry**: 数据从功能用户移动到被测软件内部
- **Exit**: 数据从被测软件内部移动到功能用户
- **Read**: 数据从持久存储移动到被测软件的功能流程
- **Write**: 数据从被测软件的功能流程移动到持久存储

## 复杂度权重表

### NESMA权重
| 功能类型 | 低复杂度 | 平均复杂度 | 高复杂度 |
|---------|---------|-----------|---------|
| ILF     | 7       | 10        | 15      |
| EIF     | 5       | 7         | 10      |
| EI      | 3       | 4         | 6       |
| EO      | 4       | 5         | 7       |
| EQ      | 3       | 4         | 6       |

### COSMIC计算
- 每个数据移动 = 1 CFP (COSMIC功能点)
        """
        
        ref_path = self.base_dir / "common" / "功能点估算快速参考.md"
        with open(ref_path, 'w', encoding='utf-8') as f:
            f.write(quick_ref_content)
        
        logger.info(f"✅ 创建快速参考手册: {ref_path}")
        
        # 创建中文术语对照表
        glossary_content = """
# 功能点估算术语对照表

## NESMA术语对照
- Function Point (FP) / 功能点
- Unadjusted Function Point (UFP) / 未调整功能点
- Internal Logical File (ILF) / 内部逻辑文件
- External Interface File (EIF) / 外部接口文件
- External Input (EI) / 外部输入
- External Output (EO) / 外部输出
- External Inquiry (EQ) / 外部查询
- Data Element Type (DET) / 数据元素类型
- Record Element Type (RET) / 记录元素类型

## COSMIC术语对照
- COSMIC Function Point (CFP) / COSMIC功能点
- Functional User / 功能用户
- Functional Process / 功能流程
- Data Movement / 数据移动
- Software Boundary / 软件边界
- Persistent Storage / 持久存储
        """
        
        glossary_path = self.base_dir / "common" / "术语对照表.md"
        with open(glossary_path, 'w', encoding='utf-8') as f:
            f.write(glossary_content)
        
        logger.info(f"✅ 创建术语对照表: {glossary_path}")

def main():
    """主函数"""
    print("🚀 量子智能化功能点估算系统 - 标准文档资源分析")
    
    downloader = StandardDocumentDownloader()
    
    # 显示官方文档来源信息
    print("\n📋 官方标准文档评估:")
    sources_info = downloader.get_official_sources_info()
    
    for standard, info in sources_info.items():
        print(f"\n### {standard}")
        print(f"官方网站: {info['official_website']}")
        for doc_name, doc_info in info['documents'].items():
            print(f"- {doc_name}: {doc_info['status']}")
    
    # 显示补充资源建议
    print("\n📚 建议补充的资源:")
    supplementary = downloader.get_supplementary_resources()
    for resource in supplementary:
        print(f"- {resource['name']}: {resource['importance']}")
    
    # 显示免费替代资源
    print("\n🆓 免费替代资源:")
    free_resources = downloader.get_free_alternatives()
    for resource in free_resources:
        print(f"- {resource['name']}: {resource['type']}")
    
    # 尝试下载免费材料
    print("\n⬇️ 尝试下载免费补充材料...")
    downloaded = downloader.download_free_cosmic_materials()
    if downloaded:
        print(f"✅ 成功下载: {', '.join(downloaded)}")
    else:
        print("ℹ️ 未找到可用的免费下载链接")
    
    # 创建补充文档
    print("\n📝 创建补充文档...")
    downloader.create_supplementary_documents()
    
    print("\n🎉 知识库资源分析完成!")
    print("\n💡 建议:")
    print("1. 您的核心标准文档已经很完整，可以直接开始开发")
    print("2. 建议重点完善中文实践案例和行业特定指南")
    print("3. 可以考虑从学术数据库收集最新研究成果")
    print("4. 建立持续更新机制，跟踪标准版本更新")

if __name__ == "__main__":
    main() 