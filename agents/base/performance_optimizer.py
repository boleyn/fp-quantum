"""
量子智能化功能点估算系统 - 性能优化器

提供缓存、批处理、并发等性能优化功能
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, TypeVar, Union
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """性能指标"""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cache_hit: bool = False
    batch_size: Optional[int] = None
    concurrent_tasks: Optional[int] = None
    memory_usage_mb: Optional[float] = None


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times: Dict[str, datetime] = {}
        
    def get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            "function": func_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }
        return hashlib.md5(json.dumps(key_data).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key not in self.cache:
            return None
            
        # 检查TTL
        if self._is_expired(key):
            self.delete(key)
            return None
            
        # 更新访问时间
        self.access_times[key] = datetime.now()
        return self.cache[key]["value"]
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存"""
        # 如果缓存已满，清理过期和最少使用的条目
        if len(self.cache) >= self.max_size:
            self._evict_cache()
            
        self.cache[key] = {
            "value": value,
            "created_at": datetime.now()
        }
        self.access_times[key] = datetime.now()
        
    def delete(self, key: str) -> None:
        """删除缓存"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
            
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        
    def _is_expired(self, key: str) -> bool:
        """检查是否过期"""
        if key not in self.cache:
            return True
            
        created_at = self.cache[key]["created_at"]
        return datetime.now() - created_at > timedelta(seconds=self.ttl_seconds)
    
    def _evict_cache(self) -> None:
        """清理缓存"""
        # 首先清理过期的
        expired_keys = [key for key in self.cache.keys() if self._is_expired(key)]
        for key in expired_keys:
            self.delete(key)
            
        # 如果还是太多，清理最少使用的
        if len(self.cache) >= self.max_size:
            # 按访问时间排序，删除最旧的
            sorted_keys = sorted(
                self.access_times.keys(),
                key=lambda k: self.access_times[k]
            )
            
            # 删除最旧的25%
            to_delete = sorted_keys[:max(1, len(sorted_keys) // 4)]
            for key in to_delete:
                self.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_ratio": getattr(self, "_hit_count", 0) / max(getattr(self, "_access_count", 1), 1),
            "expired_entries": len([k for k in self.cache.keys() if self._is_expired(k)])
        }


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_tasks: List[Dict[str, Any]] = []
        self.last_batch_time = time.time()
        
    async def add_task(
        self, 
        func: Callable, 
        args: tuple = (), 
        kwargs: dict = None
    ) -> Any:
        """添加任务到批次"""
        if kwargs is None:
            kwargs = {}
            
        task = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "future": asyncio.Future()
        }
        
        self.pending_tasks.append(task)
        
        # 检查是否需要执行批次
        if (len(self.pending_tasks) >= self.batch_size or 
            time.time() - self.last_batch_time > self.max_wait_time):
            await self._execute_batch()
            
        return await task["future"]
    
    async def _execute_batch(self) -> None:
        """执行批次"""
        if not self.pending_tasks:
            return
            
        batch = self.pending_tasks.copy()
        self.pending_tasks.clear()
        self.last_batch_time = time.time()
        
        # 按函数分组
        function_groups: Dict[str, List[Dict[str, Any]]] = {}
        for task in batch:
            func_name = task["func"].__name__
            if func_name not in function_groups:
                function_groups[func_name] = []
            function_groups[func_name].append(task)
        
        # 并行执行每个函数组
        for func_name, tasks in function_groups.items():
            asyncio.create_task(self._execute_function_batch(tasks))
    
    async def _execute_function_batch(self, tasks: List[Dict[str, Any]]) -> None:
        """执行同一函数的批次"""
        try:
            # 如果函数支持批处理，尝试批量执行
            first_task = tasks[0]
            func = first_task["func"]
            
            if hasattr(func, "_supports_batch"):
                # 批量执行
                all_args = [task["args"] for task in tasks]
                all_kwargs = [task["kwargs"] for task in tasks]
                
                results = await func(all_args, all_kwargs)
                
                # 分发结果
                for task, result in zip(tasks, results):
                    if not task["future"].done():
                        task["future"].set_result(result)
            else:
                # 单独执行
                for task in tasks:
                    try:
                        if asyncio.iscoroutinefunction(task["func"]):
                            result = await task["func"](*task["args"], **task["kwargs"])
                        else:
                            result = task["func"](*task["args"], **task["kwargs"])
                        
                        if not task["future"].done():
                            task["future"].set_result(result)
                    except Exception as e:
                        if not task["future"].done():
                            task["future"].set_exception(e)
                            
        except Exception as e:
            # 如果批处理失败，设置所有任务的异常
            for task in tasks:
                if not task["future"].done():
                    task["future"].set_exception(e)


class ConcurrencyManager:
    """并发管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
    async def execute_concurrent(
        self, 
        tasks: List[Callable], 
        task_args: List[tuple] = None,
        task_kwargs: List[dict] = None
    ) -> List[Any]:
        """并发执行任务列表"""
        
        if task_args is None:
            task_args = [() for _ in tasks]
        if task_kwargs is None:
            task_kwargs = [{} for _ in tasks]
            
        async def limited_task(func, args, kwargs):
            async with self.semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        # 创建任务
        concurrent_tasks = [
            asyncio.create_task(limited_task(func, args, kwargs))
            for func, args, kwargs in zip(tasks, task_args, task_kwargs)
        ]
        
        # 等待所有任务完成
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        return results
    
    async def execute_with_timeout(
        self, 
        func: Callable, 
        timeout: float,
        *args, 
        **kwargs
    ) -> Any:
        """执行带超时的任务"""
        
        async def task():
            async with self.semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        try:
            return await asyncio.wait_for(task(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"任务 {func.__name__} 执行超时 ({timeout}秒)")
            raise


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        batch_size: int = 10,
        max_concurrent: int = 10
    ):
        self.cache_manager = CacheManager(cache_size, cache_ttl)
        self.batch_processor = BatchProcessor(batch_size)
        self.concurrency_manager = ConcurrencyManager(max_concurrent)
        self.metrics: List[PerformanceMetrics] = []
        
    def cached(self, ttl: Optional[int] = None):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self.cache_manager.get_cache_key(
                    func.__name__, args, kwargs
                )
                
                # 尝试从缓存获取
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"缓存命中: {func.__name__}")
                    return cached_result
                
                # 执行函数
                start_time = datetime.now()
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 存储到缓存
                self.cache_manager.set(cache_key, result)
                
                # 记录指标
                end_time = datetime.now()
                metrics = PerformanceMetrics(
                    operation_name=func.__name__,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds(),
                    cache_hit=False
                )
                self.metrics.append(metrics)
                
                return result
            
            return wrapper
        return decorator
    
    def batched(self, batch_size: Optional[int] = None):
        """批处理装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                processor = BatchProcessor(
                    batch_size or self.batch_processor.batch_size
                )
                return await processor.add_task(func, args, kwargs)
            
            return wrapper
        return decorator
    
    def timed(self, func):
        """计时装饰器"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
                
            finally:
                end_time = datetime.now()
                metrics = PerformanceMetrics(
                    operation_name=func.__name__,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=(end_time - start_time).total_seconds()
                )
                self.metrics.append(metrics)
                logger.info(
                    f"函数 {func.__name__} 执行时间: "
                    f"{metrics.duration_seconds:.3f}秒"
                )
        
        return wrapper
    
    async def optimize_large_dataset_processing(
        self,
        data: List[Any],
        processor_func: Callable,
        chunk_size: int = 100
    ) -> List[Any]:
        """优化大数据集处理"""
        
        if len(data) <= chunk_size:
            # 小数据集直接处理
            return await processor_func(data)
        
        # 分块处理
        chunks = [
            data[i:i + chunk_size] 
            for i in range(0, len(data), chunk_size)
        ]
        
        # 并发处理块
        tasks = [processor_func(chunk) for chunk in chunks]
        chunk_results = await self.concurrency_manager.execute_concurrent(tasks)
        
        # 合并结果
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
            else:
                results.append(chunk_result)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics:
            return {"message": "暂无性能数据"}
        
        # 按操作分组
        operation_stats: Dict[str, List[float]] = {}
        for metric in self.metrics:
            if metric.operation_name not in operation_stats:
                operation_stats[metric.operation_name] = []
            operation_stats[metric.operation_name].append(metric.duration_seconds)
        
        # 计算统计信息
        report = {
            "总操作数": len(self.metrics),
            "操作统计": {},
            "缓存统计": self.cache_manager.get_stats(),
            "最慢操作": self._get_slowest_operations(),
            "性能建议": self._get_performance_recommendations()
        }
        
        for op_name, durations in operation_stats.items():
            report["操作统计"][op_name] = {
                "调用次数": len(durations),
                "平均耗时": sum(durations) / len(durations),
                "最大耗时": max(durations),
                "最小耗时": min(durations)
            }
        
        return report
    
    def _get_slowest_operations(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取最慢的操作"""
        sorted_metrics = sorted(
            self.metrics,
            key=lambda m: m.duration_seconds,
            reverse=True
        )
        
        return [
            {
                "操作名": metric.operation_name,
                "耗时": f"{metric.duration_seconds:.3f}秒",
                "时间": metric.start_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            for metric in sorted_metrics[:top_n]
        ]
    
    def _get_performance_recommendations(self) -> List[str]:
        """获取性能建议"""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        # 分析缓存命中率
        cache_stats = self.cache_manager.get_stats()
        if cache_stats["hit_ratio"] < 0.5:
            recommendations.append("考虑增加缓存大小或优化缓存策略")
        
        # 分析慢操作
        slow_operations = [
            m for m in self.metrics 
            if m.duration_seconds > 5.0
        ]
        if slow_operations:
            recommendations.append(
                f"发现 {len(slow_operations)} 个慢操作，建议优化或使用异步处理"
            )
        
        # 分析并发度
        concurrent_operations = len([
            m for m in self.metrics
            if m.concurrent_tasks and m.concurrent_tasks > 1
        ])
        if concurrent_operations < len(self.metrics) * 0.3:
            recommendations.append("考虑增加并发处理以提高性能")
        
        return recommendations or ["当前性能表现良好"]
    
    def clear_metrics(self) -> None:
        """清空性能指标"""
        self.metrics.clear()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache_manager.clear()


# 全局性能优化器实例
global_optimizer = PerformanceOptimizer()


def optimize_agent_performance(agent_class):
    """智能体性能优化装饰器"""
    
    class OptimizedAgent(agent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.optimizer = PerformanceOptimizer()
        
        async def execute_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """优化的任务执行"""
            
            # 使用缓存和计时
            @self.optimizer.cached()
            @self.optimizer.timed
            async def cached_execute():
                return await super(OptimizedAgent, self).execute_task(task_name, inputs)
            
            return await cached_execute()
        
        def get_performance_stats(self) -> Dict[str, Any]:
            """获取性能统计"""
            return self.optimizer.get_performance_report()
    
    return OptimizedAgent


if __name__ == "__main__":
    async def test_performance_optimizer():
        """测试性能优化器"""
        
        optimizer = PerformanceOptimizer()
        
        # 测试缓存
        @optimizer.cached()
        async def expensive_operation(x: int) -> int:
            await asyncio.sleep(1)  # 模拟耗时操作
            return x * x
        
        # 测试计时
        @optimizer.timed
        async def timed_operation(x: int) -> int:
            await asyncio.sleep(0.5)
            return x + 1
        
        print("🚀 测试性能优化器...")
        
        # 第一次调用（无缓存）
        result1 = await expensive_operation(5)
        print(f"第一次调用结果: {result1}")
        
        # 第二次调用（有缓存）
        result2 = await expensive_operation(5)
        print(f"第二次调用结果: {result2}")
        
        # 测试计时
        result3 = await timed_operation(10)
        print(f"计时操作结果: {result3}")
        
        # 打印性能报告
        report = optimizer.get_performance_report()
        print("\n📊 性能报告:")
        for key, value in report.items():
            print(f"{key}: {value}")
    
    asyncio.run(test_performance_optimizer()) 