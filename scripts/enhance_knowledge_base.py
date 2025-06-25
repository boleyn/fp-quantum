"""
量子智能化功能点估算系统 - 知识库增强脚本

添加NESMA和COSMIC标准文档，完善知识库内容
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from knowledge_base.embeddings.embedding_models import get_embedding_model
from knowledge_base.vector_stores.mongodb_atlas import setup_mongodb_vector

logger = logging.getLogger(__name__)


class KnowledgeBaseEnhancer:
    """知识库增强器"""
    
    def __init__(self):
        self.embeddings = get_embedding_model("bge_m3")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", ".", "!", "?", "；", ";"]
        )
        
    async def create_nesma_knowledge_documents(self) -> List[Dict[str, Any]]:
        """创建NESMA知识文档"""
        
        nesma_documents = [
            {
                "title": "NESMA功能类型分类规则",
                "content": """
                NESMA v2.3+ 功能类型分类详细规则
                
                ## 内部逻辑文件 (ILF - Internal Logical File)
                
                **定义**: 由应用程序维护的用户可识别的逻辑相关数据组
                
                **识别规则**:
                1. 数据必须由该应用程序维护（增删改）
                2. 数据必须对用户有意义
                3. 数据必须构成逻辑相关的组
                4. 必须通过应用程序边界的一个或多个基本过程进行维护
                
                **常见示例**:
                - 客户主数据文件
                - 产品目录
                - 订单记录
                - 用户账户信息
                
                **排除条件**:
                - 纯技术文件（如日志、配置文件）
                - 临时工作文件
                - 由其他应用维护的文件
                
                ## 外部接口文件 (EIF - External Interface File)
                
                **定义**: 由其他应用程序维护，但被估算应用程序引用的逻辑相关数据组
                
                **识别规则**:
                1. 数据由其他应用程序维护
                2. 数据被估算应用程序引用或使用
                3. 数据对用户有意义
                4. 数据构成逻辑相关的组
                
                **常见示例**:
                - 外部系统的客户数据
                - 第三方产品目录
                - 共享的代码表
                - 外部汇率数据
                
                **重要区别**: EIF不被该应用程序维护，只能读取或引用
                
                ## 外部输入 (EI - External Input)
                
                **定义**: 从应用程序边界外部进入，用于维护一个或多个ILF和/或改变系统行为的基本过程
                
                **识别规则**:
                1. 数据或控制信息来自应用程序边界外部
                2. 过程更新一个或多个ILF
                3. 或改变应用程序的行为
                4. 具有唯一的处理逻辑
                
                **常见示例**:
                - 用户注册
                - 数据录入屏幕
                - 批量数据导入
                - 系统配置更新
                
                **处理规则**:
                - 相同数据通过不同界面输入 = 多个EI
                - 批量和在线输入同一数据 = 2个EI
                
                ## 外部输出 (EO - External Output)
                
                **定义**: 向应用程序边界外部发送数据或控制信息的基本过程，包含额外处理逻辑
                
                **识别规则**:
                1. 数据或信息发送到应用程序边界外部
                2. 包含数学计算、导出数据或创建额外信息
                3. 或改变应用程序行为
                4. 具有唯一的处理逻辑
                
                **常见示例**:
                - 包含计算的报表
                - 数据导出功能
                - 仪表板（带计算）
                - 统计分析报告
                
                **重要特征**: 必须包含派生数据或额外处理逻辑
                
                ## 外部查询 (EQ - External Inquiry)
                
                **定义**: 向应用程序边界外部发送数据或控制信息的基本过程，仅包含检索逻辑
                
                **识别规则**:
                1. 数据发送到应用程序边界外部
                2. 仅包含检索和显示逻辑
                3. 不包含数学计算或派生数据
                4. 不更新ILF
                
                **常见示例**:
                - 简单查询屏幕
                - 数据列表显示
                - 记录详情查看
                - 基本搜索功能
                
                **EO vs EQ区别**:
                - EO: 包含计算、派生或复杂处理
                - EQ: 仅检索和显示原始数据
                
                ## 分类决策树
                
                1. 是否为数据文件？
                   - 是 → 2
                   - 否 → 5
                
                2. 数据由本应用维护？
                   - 是 → ILF
                   - 否 → 3
                
                3. 数据被本应用引用？
                   - 是 → EIF
                   - 否 → 不计算
                
                4. 是否为事务处理？
                   - 是 → 5
                   - 否 → 不计算
                
                5. 是否更新数据或改变行为？
                   - 是 → EI
                   - 否 → 6
                
                6. 是否包含计算或派生数据？
                   - 是 → EO
                   - 否 → EQ
                """
            },
            {
                "title": "NESMA复杂度计算方法",
                "content": """
                NESMA复杂度计算详细方法
                
                ## 数据元素类型 (DET - Data Element Type)
                
                **定义**: 用户可识别的、不可重复的字段
                
                **计算规则**:
                1. 每个唯一的用户可识别字段 = 1 DET
                2. 重复字段只计算一次
                3. 技术字段不计算（如ID、时间戳）
                4. 操作按钮/命令每个功能计1 DET
                
                **ILF/EIF的DET计算**:
                - 包含该文件中所有用户可识别字段
                - 外键引用计1 DET
                - 重复组中的字段按唯一类型计算
                
                **EI的DET计算**:
                - 输入字段数量
                - 文件操作命令（增、删、改各1个）
                - 不包括系统生成字段
                
                **EO/EQ的DET计算**:
                - 输出字段数量
                - 输入参数字段数量
                - 查询条件字段数量
                
                ## 记录元素类型 (RET - Record Element Type)
                
                **定义**: 逻辑相关的数据元素子组
                
                **计算规则**:
                1. 主记录类型 = 1 RET
                2. 每个重复子组 = 1 RET
                3. 每个不同的记录格式 = 1 RET
                
                **ILF/EIF的RET计算**:
                - 主记录 = 1 RET
                - 每个重复组 = 1 RET
                - 每个子实体类型 = 1 RET
                
                **EI的RET计算**:
                - 每个被维护的ILF = 1 RET
                - 不同的输入记录格式各计1 RET
                
                **EO/EQ的RET计算**:
                - 每个被引用的ILF/EIF = 1 RET
                - 不同的输出记录格式各计1 RET
                
                ## 复杂度等级矩阵
                
                ### ILF复杂度矩阵
                | DET范围 | 1-19 DET | 20-50 DET | 51+ DET |
                |---------|----------|-----------|---------|
                | 1 RET   | Low      | Low       | Average |
                | 2-5 RET | Low      | Average   | High    |
                | 6+ RET  | Average  | High      | High    |
                
                ### EIF复杂度矩阵
                | DET范围 | 1-19 DET | 20-50 DET | 51+ DET |
                |---------|----------|-----------|---------|
                | 1 RET   | Low      | Low       | Average |
                | 2-5 RET | Low      | Average   | High    |
                | 6+ RET  | Average  | High      | High    |
                
                ### EI复杂度矩阵
                | DET范围 | 1-14 DET | 15-19 DET | 20+ DET |
                |---------|----------|-----------|---------|
                | 1 RET   | Low      | Low       | Average |
                | 2 RET   | Low      | Average   | High    |
                | 3+ RET  | Average  | High      | High    |
                
                ### EO复杂度矩阵
                | DET范围 | 1-19 DET | 20-50 DET | 51+ DET |
                |---------|----------|-----------|---------|
                | 1 RET   | Low      | Low       | Average |
                | 2-3 RET | Low      | Average   | High    |
                | 4+ RET  | Average  | High      | High    |
                
                ### EQ复杂度矩阵
                | DET范围 | 1-19 DET | 20-50 DET | 51+ DET |
                |---------|----------|-----------|---------|
                | 1 RET   | Low      | Low       | Average |
                | 2-3 RET | Low      | Average   | High    |
                | 4+ RET  | Average  | High      | High    |
                
                ## 计算示例
                
                **用户注册功能（EI）**:
                - DET: 姓名、邮箱、密码、确认密码、手机号 = 5 DET
                - RET: 更新用户表 = 1 RET
                - 复杂度: 5 DET + 1 RET = Low
                
                **销售报表功能（EO）**:
                - DET: 产品名、销量、金额、日期、客户数 = 5 DET
                - RET: 订单表、产品表 = 2 RET
                - 复杂度: 5 DET + 2 RET = Low
                
                **客户信息文件（ILF）**:
                - DET: 客户号、姓名、地址、电话、邮箱等 = 15 DET
                - RET: 主记录 = 1 RET
                - 复杂度: 15 DET + 1 RET = Low
                """
            },
            {
                "title": "NESMA权重表和UFP计算",
                "content": """
                NESMA权重表和未调整功能点计算
                
                ## 标准权重表
                
                ### 数据功能权重
                | 功能类型 | Low | Average | High |
                |---------|-----|---------|------|
                | ILF     | 7   | 10      | 15   |
                | EIF     | 5   | 7       | 10   |
                
                ### 事务功能权重
                | 功能类型 | Low | Average | High |
                |---------|-----|---------|------|
                | EI      | 3   | 4       | 6    |
                | EO      | 4   | 5       | 7    |
                | EQ      | 3   | 4       | 6    |
                
                ## UFP计算公式
                
                UFP = Σ(功能类型数量 × 对应权重)
                
                **详细计算**:
                - ILF_UFP = (ILF_Low × 7) + (ILF_Average × 10) + (ILF_High × 15)
                - EIF_UFP = (EIF_Low × 5) + (EIF_Average × 7) + (EIF_High × 10)
                - EI_UFP = (EI_Low × 3) + (EI_Average × 4) + (EI_High × 6)
                - EO_UFP = (EO_Low × 4) + (EO_Average × 5) + (EO_High × 7)
                - EQ_UFP = (EQ_Low × 3) + (EQ_Average × 4) + (EQ_High × 6)
                
                总UFP = ILF_UFP + EIF_UFP + EI_UFP + EO_UFP + EQ_UFP
                
                ## 计算示例
                
                **电商系统估算**:
                
                ### 数据功能
                - ILF: 客户文件(Low,7)、产品文件(Average,10)、订单文件(High,15)
                - EIF: 外部支付系统(Low,5)、物流接口(Average,7)
                
                ### 事务功能
                - EI: 用户注册(Low,3)、下单(Average,4)、支付(High,6)
                - EO: 销售报表(Average,5)、发票生成(High,7)
                - EQ: 产品查询(Low,3)、订单查询(Average,4)
                
                ### UFP计算
                - 数据功能: (7+10+15) + (5+7) = 32 + 12 = 44 UFP
                - 事务功能: (3+4+6) + (5+7) + (3+4) = 13 + 12 + 7 = 32 UFP
                - 总计: 44 + 32 = 76 UFP
                
                ## 质量检查
                
                **合理性检查**:
                1. 数据功能占比通常在30-70%
                2. 事务功能占比通常在30-70%
                3. 平均复杂度分布应合理
                
                **常见错误**:
                1. 遗漏数据文件
                2. 重复计算功能
                3. 复杂度评估偏差
                4. 边界识别错误
                
                **最佳实践**:
                1. 详细记录每个功能的识别依据
                2. 进行多轮评审
                3. 对比同类项目
                4. 记录假设和排除项
                """
            }
        ]
        
        return nesma_documents
    
    async def create_cosmic_knowledge_documents(self) -> List[Dict[str, Any]]:
        """创建COSMIC知识文档"""
        
        cosmic_documents = [
            {
                "title": "COSMIC基本概念和原则",
                "content": """
                COSMIC v4.0+ 基本概念和测量原则
                
                ## 核心概念
                
                ### 功能用户 (Functional User)
                **定义**: 发送数据到软件或从软件接收数据的用户类型
                
                **特征**:
                - 可以是人类用户、设备、其他软件
                - 通过软件边界与软件交互
                - 具有特定的功能需求
                
                **识别规则**:
                1. 必须是数据的发送者或接收者
                2. 必须位于软件边界之外
                3. 必须有明确的功能目的
                
                **常见类型**:
                - 最终用户：直接使用软件的人
                - 系统管理员：配置和维护软件的人
                - 外部系统：与软件交互的其他软件
                - 设备：传感器、打印机等硬件设备
                
                ### 软件边界 (Software Boundary)
                **定义**: 区分被测量软件与其环境的概念界限
                
                **确定原则**:
                1. 基于用户视角定义
                2. 包含所有被测量的功能
                3. 排除不属于测量范围的组件
                
                **边界内容**:
                - 应用程序代码
                - 配置数据
                - 业务规则
                - 用户界面逻辑
                
                **边界外内容**:
                - 功能用户
                - 持久存储
                - 操作系统
                - 中间件（通常）
                
                ### 持久存储 (Persistent Storage)
                **定义**: 软件边界外的数据存储，在软件执行间保持数据
                
                **特征**:
                - 独立于软件生命周期
                - 可被多个软件访问
                - 包含结构化数据
                
                **常见形式**:
                - 数据库
                - 文件系统
                - 配置文件
                - 缓存系统
                
                ## 测量原则
                
                ### 功能过程 (Functional Process)
                **定义**: 由唯一的功能用户触发的最小功能单元
                
                **识别规则**:
                1. 由一个功能用户触发
                2. 包含有意义的结果
                3. 是自包含的功能单元
                4. 能独立执行
                
                **组成部分**:
                - 触发事件
                - 处理逻辑
                - 数据移动
                - 输出结果
                
                ### 数据组 (Data Group)
                **定义**: 在功能过程中作为整体移动的相关数据属性集合
                
                **识别规则**:
                1. 逻辑相关的数据属性
                2. 在功能过程中一起移动
                3. 对功能用户有意义
                
                **粒度原则**:
                - 最小有意义的数据集合
                - 避免过于细粒度
                - 避免过于粗粒度
                
                ## 数据移动类型
                
                ### Entry (输入)
                **定义**: 数据组从功能用户移动到被测量软件内部
                
                **特征**:
                - 起点：功能用户
                - 终点：软件内部
                - 目的：提供数据给软件处理
                
                **常见场景**:
                - 用户输入表单数据
                - 外部系统发送数据
                - 文件上传
                - API调用参数
                
                ### Exit (输出)
                **定义**: 数据组从被测量软件内部移动到功能用户
                
                **特征**:
                - 起点：软件内部
                - 终点：功能用户
                - 目的：向功能用户提供信息
                
                **常见场景**:
                - 显示查询结果
                - 生成报表
                - 发送通知
                - API返回数据
                
                ### Read (读取)
                **定义**: 数据组从持久存储移动到被测量软件内部
                
                **特征**:
                - 起点：持久存储
                - 终点：软件内部
                - 目的：获取处理所需数据
                
                **常见场景**:
                - 查询数据库
                - 读取配置文件
                - 加载用户信息
                - 访问缓存数据
                
                ### Write (写入)
                **定义**: 数据组从被测量软件内部移动到持久存储
                
                **特征**:
                - 起点：软件内部
                - 终点：持久存储
                - 目的：保存或更新数据
                
                **常见场景**:
                - 保存用户输入
                - 更新记录状态
                - 写入日志
                - 缓存计算结果
                
                ## 测量规则
                
                ### CFP计算公式
                CFP = Entry数量 + Exit数量 + Read数量 + Write数量
                
                **基本原则**:
                - 1个数据移动 = 1 CFP
                - 所有数据移动等权重
                - 按功能过程分组计算
                
                ### 边界一致性
                **软件边界一致性**:
                - Entry/Exit必须穿越软件边界
                - 边界内的数据传递不计算
                
                **存储边界一致性**:
                - Read/Write必须穿越存储边界
                - 软件内部数据操作不计算
                
                ### 聚合级别
                **功能过程级别**:
                - 按单个功能过程识别数据移动
                - 确保完整性和一致性
                
                **应用程序级别**:
                - 汇总所有功能过程的CFP
                - 避免重复计算
                """
            },
            {
                "title": "COSMIC数据移动识别指南",
                "content": """
                COSMIC数据移动识别详细指南
                
                ## 识别步骤
                
                ### 步骤1：识别功能过程
                **方法**:
                1. 分析用户需求
                2. 识别触发事件
                3. 确定处理边界
                4. 验证功能完整性
                
                **示例 - 用户登录过程**:
                - 触发者：用户
                - 输入：用户名、密码
                - 处理：验证凭据
                - 输出：登录结果
                
                ### 步骤2：识别数据组
                **原则**:
                1. 逻辑相关性
                2. 移动一致性
                3. 用户可识别性
                
                **粒度控制**:
                - 过细：每个字段单独成组
                - 适中：相关字段组合
                - 过粗：整个记录作为一组
                
                ### 步骤3：跟踪数据移动
                **系统化方法**:
                1. 绘制数据流图
                2. 标识边界穿越点
                3. 分类移动类型
                4. 验证移动必要性
                
                ## Entry识别详解
                
                ### 典型Entry场景
                **用户界面输入**:
                - 表单提交：用户填写的表单数据
                - 文件上传：用户选择的文件内容
                - 选择操作：下拉选择、复选框状态
                
                **系统接口输入**:
                - API调用：外部系统传入的参数
                - 消息队列：接收的消息内容
                - 文件导入：批量导入的数据
                
                **设备输入**:
                - 传感器数据：温度、压力等测量值
                - 扫描设备：条码、二维码内容
                - GPS设备：位置坐标信息
                
                ### Entry识别规则
                1. **边界穿越验证**
                   - 数据必须从外部进入软件
                   - 不包括软件内部的数据传递
                
                2. **功能用户验证**
                   - 必须来自已识别的功能用户
                   - 验证用户的合法性
                
                3. **数据组完整性**
                   - 确保数据组的逻辑完整性
                   - 避免拆分相关数据
                
                ## Exit识别详解
                
                ### 典型Exit场景
                **用户界面输出**:
                - 查询结果：显示的数据列表
                - 报表生成：格式化的报告内容
                - 状态信息：操作结果提示
                
                **系统接口输出**:
                - API响应：返回给调用者的数据
                - 消息发送：发布到消息队列的内容
                - 文件导出：生成的下载文件
                
                **设备输出**:
                - 打印内容：发送到打印机的数据
                - 显示屏：推送到显示设备的信息
                - 控制指令：发送给执行设备的命令
                
                ### Exit特殊情况
                **确认消息**:
                - 操作成功确认 = 1 Exit
                - 错误消息 = 1 Exit
                - 简单状态码不单独计算
                
                **分页输出**:
                - 每页相同内容 = 1 Exit
                - 不同页面内容 = 多个Exit
                
                ## Read识别详解
                
                ### 典型Read场景
                **数据库查询**:
                - 单表查询：读取特定表的记录
                - 多表关联：读取关联的数据组
                - 配置查询：读取系统配置信息
                
                **文件读取**:
                - 配置文件：读取应用配置
                - 数据文件：读取业务数据
                - 模板文件：读取报表模板
                
                **缓存访问**:
                - 内存缓存：读取缓存的数据
                - 分布式缓存：读取共享缓存
                - 会话数据：读取用户会话信息
                
                ### Read计算规则
                **相同数据多次读取**:
                - 同一功能过程中 = 1 Read
                - 不同功能过程中 = 分别计算
                
                **不同目的读取**:
                - 验证用途：读取用于验证的数据
                - 显示用途：读取用于显示的数据
                - 处理用途：读取用于业务处理的数据
                
                ## Write识别详解
                
                ### 典型Write场景
                **数据库操作**:
                - 插入记录：创建新的数据记录
                - 更新记录：修改现有数据
                - 删除记录：删除数据记录
                
                **文件操作**:
                - 创建文件：生成新文件
                - 更新文件：修改文件内容
                - 日志写入：记录操作日志
                
                **缓存操作**:
                - 缓存更新：更新缓存数据
                - 会话保存：保存会话状态
                - 临时存储：保存临时数据
                
                ### Write特殊情况
                **批量操作**:
                - 批量插入相同结构 = 1 Write
                - 批量更新不同记录 = 1 Write
                - 批量操作不同数据组 = 多个Write
                
                **事务操作**:
                - 事务内多个Write = 分别计算
                - 原子操作作为整体 = 按数据组计算
                
                ## 边界验证检查表
                
                ### Entry验证
                - [ ] 数据来源是功能用户？
                - [ ] 数据穿越软件边界？
                - [ ] 数据组逻辑完整？
                - [ ] 避免重复计算？
                
                ### Exit验证
                - [ ] 数据目标是功能用户？
                - [ ] 数据穿越软件边界？
                - [ ] 输出内容有意义？
                - [ ] 确认消息已考虑？
                
                ### Read验证
                - [ ] 数据来源是持久存储？
                - [ ] 读取是功能必需的？
                - [ ] 避免重复读取计算？
                - [ ] 数据组粒度适当？
                
                ### Write验证
                - [ ] 数据目标是持久存储？
                - [ ] 写入是功能必需的？
                - [ ] 操作类型识别正确？
                - [ ] 批量操作处理恰当？
                """
            }
        ]
        
        return cosmic_documents
    
    async def create_comparison_documents(self) -> List[Dict[str, Any]]:
        """创建NESMA与COSMIC对比文档"""
        
        comparison_documents = [
            {
                "title": "NESMA vs COSMIC 对比分析",
                "content": """
                NESMA与COSMIC标准对比分析
                
                ## 基本理念对比
                
                ### NESMA理念
                **数据导向**: 以数据文件和事务为核心
                **复杂度分级**: 通过DET/RET计算复杂度等级
                **权重系统**: 不同复杂度对应不同权重
                **成熟度**: 发展历史悠久，应用广泛
                
                ### COSMIC理念
                **过程导向**: 以功能过程和数据移动为核心
                **等权重**: 所有数据移动等权重(1 CFP)
                **边界清晰**: 明确的软件边界和存储边界
                **现代化**: 适应现代软件架构
                
                ## 测量对象对比
                
                ### NESMA测量对象
                | 类型 | 说明 | 示例 |
                |------|------|------|
                | ILF | 内部逻辑文件 | 客户档案、产品目录 |
                | EIF | 外部接口文件 | 外部系统数据 |
                | EI | 外部输入 | 数据录入、更新操作 |
                | EO | 外部输出 | 计算报表、统计分析 |
                | EQ | 外部查询 | 简单查询、数据显示 |
                
                ### COSMIC测量对象
                | 类型 | 说明 | 示例 |
                |------|------|------|
                | Entry | 数据输入 | 用户输入、接口调用 |
                | Exit | 数据输出 | 结果显示、数据返回 |
                | Read | 数据读取 | 数据库查询、文件读取 |
                | Write | 数据写入 | 数据保存、状态更新 |
                
                ## 适用场景对比
                
                ### NESMA适用场景
                **优势领域**:
                - 传统企业应用
                - 数据处理密集型系统
                - 报表和查询为主的应用
                - 需要详细复杂度分析的项目
                
                **技术特征**:
                - 关系型数据库为主
                - 传统三层架构
                - 批处理系统
                - 企业资源规划(ERP)
                
                ### COSMIC适用场景
                **优势领域**:
                - 现代分布式系统
                - 面向服务架构(SOA)
                - 微服务架构
                - 实时处理系统
                
                **技术特征**:
                - API驱动的架构
                - 云原生应用
                - 移动应用
                - 物联网系统
                
                ## 测量精度对比
                
                ### NESMA精度特点
                **细粒度分析**:
                - DET/RET详细计算
                - 复杂度等级精确分类
                - 权重差异反映复杂度
                
                **潜在偏差**:
                - 复杂度评估主观性
                - 权重表固定可能不准确
                - DET/RET计算复杂易错
                
                ### COSMIC精度特点
                **一致性优势**:
                - 等权重减少主观判断
                - 数据移动客观可计数
                - 边界定义相对清晰
                
                **潜在限制**:
                - 忽略复杂度差异
                - 粒度选择影响结果
                - 边界识别仍需经验
                
                ## 工作量对比
                
                ### NESMA工作量
                **时间投入**: 相对较高
                **技能要求**: 需要NESMA专业知识
                **文档要求**: 详细的DET/RET分析
                **质量保证**: 多轮复杂度验证
                
                ### COSMIC工作量
                **时间投入**: 相对较低
                **技能要求**: 需要理解数据流
                **文档要求**: 功能过程和数据移动
                **质量保证**: 边界一致性检查
                
                ## 结果可比性
                
                ### 转换关系研究
                **经验比率**:
                - 一般情况: 1 NESMA UFP ≈ 1-1.2 COSMIC CFP
                - 简单系统: 比率可能更高
                - 复杂系统: 比率可能更低
                
                **影响因素**:
                - 系统复杂度分布
                - 数据处理密度
                - 界面复杂程度
                - 技术架构差异
                
                ### 对比分析建议
                **双标准项目**:
                1. 先确定边界一致性
                2. 分别独立测量
                3. 分析差异原因
                4. 建立项目特定比率
                
                **结果验证**:
                - 10%以内差异：正常范围
                - 10-25%差异：需分析原因
                - 25%以上差异：重新检查测量
                
                ## 选择建议
                
                ### 选择NESMA的情况
                - 传统企业应用开发
                - 团队熟悉NESMA方法
                - 需要详细复杂度分析
                - 与历史项目对比
                
                ### 选择COSMIC的情况
                - 现代软件架构项目
                - 敏捷开发环境
                - 分布式系统估算
                - 国际标准化要求
                
                ### 双标准的情况
                - 大型关键项目
                - 估算精度要求极高
                - 团队能力充足
                - 时间资源允许
                """
            }
        ]
        
        return comparison_documents
    
    async def enhance_knowledge_base(self) -> Dict[str, Any]:
        """增强知识库"""
        
        logger.info("🚀 开始增强知识库...")
        
        # 创建文档
        nesma_docs = await self.create_nesma_knowledge_documents()
        cosmic_docs = await self.create_cosmic_knowledge_documents()
        comparison_docs = await self.create_comparison_documents()
        
        all_documents = nesma_docs + cosmic_docs + comparison_docs
        
        # 保存文档到文件
        docs_dir = Path("knowledge_base/documents")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_docs = []
        
        for doc in all_documents:
            # 分割文档
            splits = self.text_splitter.split_text(doc["content"])
            
            for i, split in enumerate(splits):
                enhanced_doc = {
                    "title": f"{doc['title']} - 第{i+1}部分",
                    "content": split,
                    "source_type": self._determine_source_type(doc["title"]),
                    "chunk_index": i,
                    "total_chunks": len(splits),
                    "created_at": datetime.now().isoformat()
                }
                enhanced_docs.append(enhanced_doc)
        
        # 保存增强文档
        enhanced_file = docs_dir / "enhanced_knowledge.json"
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_docs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 保存了 {len(enhanced_docs)} 个文档块到 {enhanced_file}")
        
        # 创建向量存储（如果配置了）
        try:
            await self._create_vector_store(enhanced_docs)
        except Exception as e:
            logger.warning(f"向量存储创建失败: {e}")
        
        return {
            "total_documents": len(all_documents),
            "total_chunks": len(enhanced_docs),
            "nesma_documents": len(nesma_docs),
            "cosmic_documents": len(cosmic_docs),
            "comparison_documents": len(comparison_docs),
            "saved_to": str(enhanced_file)
        }
    
    def _determine_source_type(self, title: str) -> str:
        """确定文档来源类型"""
        if "NESMA" in title:
            return "NESMA"
        elif "COSMIC" in title:
            return "COSMIC"
        elif "对比" in title or "vs" in title:
            return "COMPARISON"
        else:
            return "COMMON"
    
    async def _create_vector_store(self, documents: List[Dict[str, Any]]) -> None:
        """创建向量存储"""
        
        logger.info("📊 创建向量存储...")
        
        # 准备文档内容
        texts = [doc["content"] for doc in documents]
        metadatas = [
            {
                "title": doc["title"],
                "source_type": doc["source_type"],
                "chunk_index": doc["chunk_index"],
                "created_at": doc["created_at"]
            }
            for doc in documents
        ]
        
        # 创建Chroma向量存储（开发环境）
        try:
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory="./chroma_db_enhanced"
            )
            logger.info("✅ Chroma向量存储创建成功")
        except Exception as e:
            logger.error(f"Chroma向量存储创建失败: {e}")


async def main():
    """主函数"""
    
    enhancer = KnowledgeBaseEnhancer()
    
    try:
        result = await enhancer.enhance_knowledge_base()
        
        print("🎉 知识库增强完成!")
        print(f"总文档数: {result['total_documents']}")
        print(f"总块数: {result['total_chunks']}")
        print(f"NESMA文档: {result['nesma_documents']}")
        print(f"COSMIC文档: {result['cosmic_documents']}")
        print(f"对比文档: {result['comparison_documents']}")
        print(f"保存位置: {result['saved_to']}")
        
    except Exception as e:
        logger.error(f"知识库增强失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 