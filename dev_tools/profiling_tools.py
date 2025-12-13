"""
Performance Profiling Tools for Qwen3-VL Model

This module provides comprehensive performance profiling tools to visualize 
bottlenecks and optimization effectiveness in the Qwen3-VL model.
"""

import os
import time
import json
import threading
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from functools import wraps
from contextlib import contextmanager
import cProfile
import pstats
from io import StringIO


@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceProfiler:
    """Main performance profiling class"""
    
    def __init__(self):
        self.metrics = []
        self.profile_data = {}
        self.system_monitor = SystemMonitor()
        self.benchmark_results = {}
        self.profiling_enabled = True
        self.lock = threading.Lock()
    
    def measure_time(self, func_name: str = None):
        """Decorator to measure execution time of functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)
                
                name = func_name or func.__name__
                start_time = time.time()
                
                # Monitor system resources during execution
                start_resources = self.system_monitor.get_current_resources()
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                duration = end_time - start_time
                
                end_resources = self.system_monitor.get_current_resources()
                
                # Record performance metric
                with self.lock:
                    self.metrics.append(PerformanceMetric(
                        name=f"{name}_time",
                        value=duration,
                        unit="seconds",
                        timestamp=start_time,
                        metadata={
                            'function': name,
                            'args_len': len(args),
                            'kwargs_len': len(kwargs),
                            'resources_before': start_resources,
                            'resources_after': end_resources
                        }
                    ))
                
                return result
            return wrapper
        return decorator
    
    def profile_function(self, func_name: str = None, record_stats: bool = True):
        """Decorator to profile function with cProfile"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)
                
                name = func_name or func.__name__
                
                profiler = cProfile.Profile()
                profiler.enable()
                
                result = func(*args, **kwargs)
                
                profiler.disable()
                
                if record_stats:
                    s = StringIO()
                    ps = pstats.Stats(profiler, stream=s)
                    ps.sort_stats('cumulative')
                    stats_str = s.getvalue()
                    
                    with self.lock:
                        self.profile_data[name] = {
                            'stats': stats_str,
                            'timestamp': time.time(),
                            'function': name
                        }
                
                return result
            return wrapper
        return decorator
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: List[str] = None, metadata: Dict = None):
        """Record a custom performance metric"""
        if not self.profiling_enabled:
            return
        
        with self.lock:
            self.metrics.append(PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=time.time(),
                tags=tags,
                metadata=metadata
            ))
    
    def benchmark_function(self, func: Callable, name: str, iterations: int = 100, warmup: int = 10):
        """Benchmark a function for multiple iterations"""
        if not self.profiling_enabled:
            return None
        
        # Warmup
        for _ in range(warmup):
            func()
        
        # Actual benchmarking
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        benchmark_result = {
            'name': name,
            'iterations': iterations,
            'warmup': warmup,
            'times': times.tolist(),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'p95_time': float(np.percentile(times, 95)),
            'throughput': float(iterations / np.sum(times))  # ops/sec
        }
        
        with self.lock:
            self.benchmark_results[name] = benchmark_result
        
        return benchmark_result
    
    def get_metrics_by_name(self, name: str) -> List[PerformanceMetric]:
        """Get all metrics with a specific name"""
        return [metric for metric in self.metrics if metric.name == name]
    
    def get_latest_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get the most recent metric with a specific name"""
        matching = [metric for metric in self.metrics if metric.name == name]
        return matching[-1] if matching else None
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self.lock:
            self.metrics.clear()
            self.profile_data.clear()
            self.benchmark_results.clear()
    
    def save_metrics(self, path: str):
        """Save collected metrics to file"""
        metrics_dict = [asdict(metric) for metric in self.metrics]
        
        data = {
            'metrics': metrics_dict,
            'profile_data': self.profile_data,
            'benchmark_results': self.benchmark_results,
            'system_info': self.system_monitor.get_system_info()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_metrics(self, path: str):
        """Load metrics from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.metrics = [PerformanceMetric(**metric) for metric in data['metrics']]
        self.profile_data = data['profile_data']
        self.benchmark_results = data['benchmark_results']
    
    def generate_report(self, output_path: str = None):
        """Generate a comprehensive performance report"""
        report = {
            'summary': self._generate_summary(),
            'metrics': self._generate_metrics_report(),
            'benchmarks': self.benchmark_results,
            'system_info': self.system_monitor.get_system_info()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_summary(self):
        """Generate a summary of performance metrics"""
        if not self.metrics:
            return {'message': 'No metrics collected'}
        
        # Calculate summary statistics
        total_time = sum(m.value for m in self.metrics if 'time' in m.unit.lower())
        avg_time = np.mean([m.value for m in self.metrics if 'time' in m.unit.lower()]) if any('time' in m.unit.lower() for m in self.metrics) else 0
        total_ops = len([m for m in self.metrics if 'ops' in m.name.lower()])
        
        return {
            'total_operations': len(self.metrics),
            'total_time_seconds': total_time,
            'average_time_per_op': avg_time,
            'operations_count': total_ops,
            'first_timestamp': min(m.timestamp for m in self.metrics),
            'last_timestamp': max(m.timestamp for m in self.metrics)
        }
    
    def _generate_metrics_report(self):
        """Generate a detailed metrics report"""
        report = {}
        
        # Group metrics by name
        grouped = defaultdict(list)
        for metric in self.metrics:
            grouped[metric.name].append(metric)
        
        for name, metrics in grouped.items():
            values = [m.value for m in metrics]
            report[name] = {
                'count': len(values),
                'min': float(min(values)),
                'max': float(max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
        
        return report
    
    def visualize_metrics(self, metric_names: List[str] = None, figsize: tuple = (12, 8)):
        """Visualize collected metrics"""
        if not self.metrics:
            print("No metrics to visualize")
            return
        
        if metric_names is None:
            # Get all unique metric names
            metric_names = list(set(m.name for m in self.metrics))
        
        # Filter metrics by name
        filtered_metrics = [m for m in self.metrics if m.name in metric_names]
        
        if not filtered_metrics:
            print(f"No metrics found for names: {metric_names}")
            return
        
        # Create plots
        fig, axes = plt.subplots(len(metric_names), 1, figsize=figsize)
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, name in enumerate(metric_names):
            name_metrics = [m for m in filtered_metrics if m.name == name]
            timestamps = [m.timestamp for m in name_metrics]
            values = [m.value for m in name_metrics]
            
            axes[i].plot(timestamps, values, marker='o', label=name)
            axes[i].set_title(f'Metric: {name}')
            axes[i].set_xlabel('Timestamp')
            axes[i].set_ylabel(f'Value ({name_metrics[0].unit})')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_benchmarks(self):
        """Visualize benchmark results"""
        if not self.benchmark_results:
            print("No benchmark results to visualize")
            return
        
        names = list(self.benchmark_results.keys())
        means = [self.benchmark_results[name]['mean_time'] for name in names]
        stds = [self.benchmark_results[name]['std_time'] for name in names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Function Benchmark Results (Mean Execution Time)')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{mean_val:.4f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.resource_history = []
        self.max_history = 1000  # Maximum number of samples to keep
    
    def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent
            
            # GPU memory if available
            gpu_memory = 0
            gpu_util = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            
            resources = {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'system_memory_percent': system_memory_percent,
                'gpu_memory_mb': gpu_memory,
                'gpu_util_percent': gpu_util,
                'timestamp': time.time()
            }
            
            return resources
        except Exception as e:
            print(f"Error getting system resources: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            cpu_count = psutil.cpu_count(logical=False)
            logical_cpu_count = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            system_info = {
                'cpu_physical_cores': cpu_count,
                'cpu_logical_cores': logical_cpu_count,
                'total_memory_gb': memory_gb,
                'platform': os.sys.platform,
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
            
            if torch.cuda.is_available():
                gpu_info = {
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0
                }
                system_info.update(gpu_info)
            
            return system_info
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {}
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring in a separate thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                resources = self.get_current_resources()
                self.resource_history.append(resources)
                
                # Keep history size reasonable
                if len(self.resource_history) > self.max_history:
                    self.resource_history.pop(0)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
    
    def get_resource_usage_timeline(self) -> Dict[str, List]:
        """Get timeline of resource usage"""
        if not self.resource_history:
            return {}
        
        timeline = {
            'timestamps': [r['timestamp'] for r in self.resource_history],
            'cpu_percent': [r['cpu_percent'] for r in self.resource_history],
            'memory_mb': [r['memory_mb'] for r in self.resource_history],
            'system_memory_percent': [r['system_memory_percent'] for r in self.resource_history],
            'gpu_memory_mb': [r['gpu_memory_mb'] for r in self.resource_history],
            'gpu_util_percent': [r['gpu_util_percent'] for r in self.resource_history]
        }
        
        return timeline
    
    def visualize_resource_usage(self, figsize: tuple = (14, 10)):
        """Visualize resource usage timeline"""
        timeline = self.get_resource_usage_timeline()
        if not timeline or not timeline['timestamps']:
            print("No resource usage data to visualize")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('System Resource Usage Timeline')
        
        # CPU usage
        axes[0, 0].plot(timeline['timestamps'], timeline['cpu_percent'], 'b-')
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True)
        
        # Memory usage
        axes[0, 1].plot(timeline['timestamps'], timeline['memory_mb'], 'g-')
        axes[0, 1].set_title('Process Memory Usage (MB)')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].grid(True)
        
        # System memory usage
        axes[1, 0].plot(timeline['timestamps'], timeline['system_memory_percent'], 'r-')
        axes[1, 0].set_title('System Memory Usage (%)')
        axes[1, 0].set_ylabel('System Memory %')
        axes[1, 0].grid(True)
        
        # GPU memory usage
        axes[1, 1].plot(timeline['timestamps'], timeline['gpu_memory_mb'], 'm-')
        axes[1, 1].set_title('GPU Memory Usage (MB)')
        axes[1, 1].set_ylabel('GPU Memory (MB)')
        axes[1, 1].grid(True)
        
        # GPU utilization
        axes[2, 0].plot(timeline['timestamps'], timeline['gpu_util_percent'], 'c-')
        axes[2, 0].set_title('GPU Utilization (%)')
        axes[2, 0].set_ylabel('GPU Util %')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].grid(True)
        
        # Memory vs CPU correlation
        axes[2, 1].scatter(timeline['memory_mb'], timeline['cpu_percent'], alpha=0.6)
        axes[2, 1].set_title('Memory vs CPU Correlation')
        axes[2, 1].set_xlabel('Memory (MB)')
        axes[2, 1].set_ylabel('CPU %')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


class BottleneckDetector:
    """Detect performance bottlenecks in model execution"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.bottlenecks = []
    
    def detect_bottlenecks(self) -> List[Dict]:
        """Detect potential bottlenecks based on collected metrics"""
        self.bottlenecks = []
        
        # Group metrics by name
        grouped_metrics = defaultdict(list)
        for metric in self.profiler.metrics:
            grouped_metrics[metric.name].append(metric)
        
        # Analyze each group for potential bottlenecks
        for name, metrics in grouped_metrics.items():
            values = [m.value for m in metrics]
            
            # Check for high average values (potential bottleneck)
            avg_value = np.mean(values)
            std_value = np.std(values)
            
            # Check for high variance (unstable performance)
            cv = std_value / avg_value if avg_value != 0 else 0
            
            # Check for outliers
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + 1.5 * iqr
            outliers = [v for v in values if v > upper_bound]
            
            bottleneck_score = 0
            issues = []
            
            if avg_value > np.mean([np.mean(g) for g in grouped_metrics.values()]) * 1.5:
                bottleneck_score += 1
                issues.append("High average value")
            
            if cv > 0.5:  # High coefficient of variation
                bottleneck_score += 1
                issues.append("High variance")
            
            if len(outliers) > len(values) * 0.1:  # More than 10% outliers
                bottleneck_score += 1
                issues.append("Many outliers")
            
            if bottleneck_score > 0:
                self.bottlenecks.append({
                    'metric_name': name,
                    'bottleneck_score': bottleneck_score,
                    'issues': issues,
                    'avg_value': float(avg_value),
                    'std_value': float(std_value),
                    'cv': float(cv),
                    'outlier_count': len(outliers),
                    'total_samples': len(values)
                })
        
        return self.bottlenecks
    
    def print_bottleneck_report(self):
        """Print a report of detected bottlenecks"""
        bottlenecks = self.detect_bottlenecks()
        
        if not bottlenecks:
            print("No bottlenecks detected.")
            return
        
        print("=== Performance Bottleneck Report ===")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"\n{i}. Metric: {bottleneck['metric_name']}")
            print(f"   Bottleneck Score: {bottleneck['bottleneck_score']}/3")
            print(f"   Average Value: {bottleneck['avg_value']:.4f}")
            print(f"   Standard Deviation: {bottleneck['std_value']:.4f}")
            print(f"   Coefficient of Variation: {bottleneck['cv']:.4f}")
            print(f"   Outliers: {bottleneck['outlier_count']}/{bottleneck['total_samples']}")
            print(f"   Issues: {', '.join(bottleneck['issues'])}")


@contextmanager
def profile_block(name: str, profiler: PerformanceProfiler = None):
    """Context manager to profile a block of code"""
    if profiler is None:
        profiler = global_profiler
    
    start_time = time.time()
    start_resources = profiler.system_monitor.get_current_resources()
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        end_resources = profiler.system_monitor.get_current_resources()
        
        profiler.record_metric(
            name=f"{name}_time",
            value=duration,
            unit="seconds",
            metadata={
                'block': name,
                'resources_before': start_resources,
                'resources_after': end_resources
            }
        )


# Global profiler instance
global_profiler = PerformanceProfiler()
bottleneck_detector = BottleneckDetector(global_profiler)


def enable_profiling():
    """Enable performance profiling"""
    global_profiler.profiling_enabled = True


def disable_profiling():
    """Disable performance profiling"""
    global_profiler.profiling_enabled = False


# Example usage functions
def example_profiling():
    """Example of profiling usage"""
    print("=== Performance Profiling Example ===")
    
    # Example function to profile
    @global_profiler.measure_time("matrix_multiplication")
    def matrix_mult(size=1000):
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        return torch.mm(a, b)
    
    # Profile the function
    for i in range(5):
        result = matrix_mult(500)
        global_profiler.record_metric(
            name="peak_memory_mb",
            value=torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            unit="MB",
            metadata={"iteration": i}
        )
    
    # Generate and show report
    report = global_profiler.generate_report()
    print(f"Profiled {len(global_profiler.metrics)} metrics")
    
    # Visualize metrics
    global_profiler.visualize_metrics(["matrix_multiplication_time"])
    
    # Detect bottlenecks
    bottleneck_detector.print_bottleneck_report()


def example_benchmarking():
    """Example of benchmarking usage"""
    print("\n=== Benchmarking Example ===")
    
    # Define functions to benchmark
    def simple_add():
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        return a + b
    
    def complex_operation():
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        c = torch.matmul(a, b)
        d = torch.relu(c)
        return torch.sum(d)
    
    # Benchmark functions
    global_profiler.benchmark_function(simple_add, "simple_addition", iterations=50, warmup=5)
    global_profiler.benchmark_function(complex_operation, "complex_matmul", iterations=50, warmup=5)
    
    # Show results
    print("Benchmark Results:")
    for name, result in global_profiler.benchmark_results.items():
        print(f"  {name}: {result['mean_time']:.6f}s ± {result['std_time']:.6f}s")


if __name__ == "__main__":
    enable_profiling()
    
    # Start system monitoring
    global_profiler.system_monitor.start_monitoring(interval=0.5)
    
    # Run examples
    example_profiling()
    example_benchmarking()
    
    # Stop monitoring
    global_profiler.system_monitor.stop_monitoring()
    
    # Visualize system resource usage
    global_profiler.system_monitor.visualize_resource_usage()