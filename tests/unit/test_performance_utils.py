"""
Utilities for performance testing
"""
import time
import torch
import psutil
from contextlib import contextmanager
from typing import Dict, Callable, Any


@contextmanager
def performance_monitor():
    """Context manager to monitor performance metrics."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        start_gpu_memory = 0
    
    yield
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory if available
    if torch.cuda.is_available():
        end_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        end_gpu_memory = 0
    
    metrics = {
        'execution_time': end_time - start_time,
        'memory_usage': end_memory - start_memory,
        'peak_gpu_memory': end_gpu_memory - start_gpu_memory if torch.cuda.is_available() else 0
    }
    
    print(f"Execution time: {metrics['execution_time']:.4f}s")
    print(f"Memory usage: {metrics['memory_usage']:.2f}MB")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {metrics['peak_gpu_memory']:.2f}MB")
    
    yield metrics


def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Benchmark a function and return performance metrics."""
    with performance_monitor() as metrics:
        result = func(*args, **kwargs)
    
    return {
        'result': result,
        'metrics': next(metrics)  # Get the metrics from the generator
    }


def measure_gpu_utilization():
    """Measure GPU utilization if available."""
    if torch.cuda.is_available():
        import pynvml
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return {'gpu_util': util.gpu, 'memory_util': util.memory}
        except:
            return {'gpu_util': 0, 'memory_util': 0}
    else:
        return {'gpu_util': 0, 'memory_util': 0}