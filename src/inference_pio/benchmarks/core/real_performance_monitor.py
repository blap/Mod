"""
Real Performance Monitoring Module

This module provides real performance measurement capabilities instead of simulated metrics.
It monitors actual system resources, model performance, and provides accurate benchmarks.
"""

import gc
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import GPUtil
import psutil
import torch


@dataclass
class PerformanceMetrics:
    """Data class to hold performance metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None


class RealPerformanceMonitor:
    """Real performance monitoring class that collects actual system metrics."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.metrics_queue = queue.Queue()

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system metrics."""
        timestamp = time.time()

        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()

        # GPU metrics if available
        gpu_percent = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
            gpu_percent = gpu.load * 100
            gpu_memory_used_mb = gpu.memoryUsed
            gpu_memory_total_mb = gpu.memoryTotal

        # PyTorch memory metrics if CUDA is available
        memory_allocated_mb = None
        memory_reserved_mb = None
        if torch.cuda.is_available():
            memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_info.percent,
            memory_used_mb=memory_info.used / 1024 / 1024,
            memory_available_mb=memory_info.available / 1024 / 1024,
            gpu_percent=gpu_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            memory_allocated_mb=memory_allocated_mb,
            memory_reserved_mb=memory_reserved_mb,
        )

    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.stop_event.clear()

        def monitor_loop():
            while self.monitoring_active and not self.stop_event.is_set():
                try:
                    metrics = self.get_current_metrics()
                    self.metrics_queue.put(metrics)
                    self.metrics_history.append(metrics)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
                    break

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.stop_event:
            self.stop_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

    def measure_inference_performance(
        self, model_callable, *args, **kwargs
    ) -> PerformanceMetrics:
        """Measure actual inference performance."""
        # Clear CUDA cache to get accurate measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Pre-measurement
        pre_metrics = self.get_current_metrics()

        # Synchronize before timing (for GPU)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Execute the model call
        result = model_callable(*args, **kwargs)

        # Synchronize after timing (for GPU)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Post-measurement
        post_metrics = self.get_current_metrics()

        inference_time_ms = (end_time - start_time) * 1000

        # Calculate tokens per second if we can determine the output length
        tokens_per_second = None
        if hasattr(result, "shape") and len(result.shape) > 0:
            output_length = (
                result.shape[-1] if len(result.shape) > 1 else result.shape[0]
            )
            tokens_per_second = (
                output_length / (end_time - start_time)
                if (end_time - start_time) > 0
                else 0
            )

        # Combine metrics with inference time
        combined_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=post_metrics.cpu_percent,
            memory_percent=post_metrics.memory_percent,
            memory_used_mb=post_metrics.memory_used_mb,
            memory_available_mb=post_metrics.memory_available_mb,
            gpu_percent=post_metrics.gpu_percent,
            gpu_memory_used_mb=post_metrics.gpu_memory_used_mb,
            gpu_memory_total_mb=post_metrics.gpu_memory_total_mb,
            inference_time_ms=inference_time_ms,
            tokens_per_second=tokens_per_second,
            memory_allocated_mb=post_metrics.memory_allocated_mb,
            memory_reserved_mb=post_metrics.memory_reserved_mb,
        )

        self.metrics_history.append(combined_metrics)
        return combined_metrics

    def get_average_metrics(self, last_n: int = 10) -> Optional[PerformanceMetrics]:
        """Get average metrics from the last N measurements."""
        if len(self.metrics_history) == 0:
            return None

        recent_metrics = self.metrics_history[-last_n:]

        avg_timestamp = sum(m.timestamp for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_mem = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_mem_used = sum(m.memory_used_mb for m in recent_metrics) / len(
            recent_metrics
        )
        avg_mem_avail = sum(m.memory_available_mb for m in recent_metrics) / len(
            recent_metrics
        )

        # Only average GPU metrics if they exist
        avg_gpu = None
        if all(m.gpu_percent is not None for m in recent_metrics):
            avg_gpu = sum(m.gpu_percent for m in recent_metrics) / len(recent_metrics)

        avg_gpu_mem_used = None
        if all(m.gpu_memory_used_mb is not None for m in recent_metrics):
            avg_gpu_mem_used = sum(m.gpu_memory_used_mb for m in recent_metrics) / len(
                recent_metrics
            )

        avg_inf_time = None
        if all(m.inference_time_ms is not None for m in recent_metrics):
            avg_inf_time = sum(m.inference_time_ms for m in recent_metrics) / len(
                recent_metrics
            )

        avg_tps = None
        if all(m.tokens_per_second is not None for m in recent_metrics):
            avg_tps = sum(m.tokens_per_second for m in recent_metrics) / len(
                recent_metrics
            )

        return PerformanceMetrics(
            timestamp=avg_timestamp,
            cpu_percent=avg_cpu,
            memory_percent=avg_mem,
            memory_used_mb=avg_mem_used,
            memory_available_mb=avg_mem_avail,
            gpu_percent=avg_gpu,
            gpu_memory_used_mb=avg_gpu_mem_used,
            inference_time_ms=avg_inf_time,
            tokens_per_second=avg_tps,
            memory_allocated_mb=None,  # Would need special averaging
            memory_reserved_mb=None,  # Would need special averaging
        )


# Global performance monitor instance
performance_monitor = RealPerformanceMonitor()


@contextmanager
def performance_monitor_context():
    """Context manager for easy performance monitoring."""
    performance_monitor.start_monitoring()
    try:
        yield performance_monitor
    finally:
        performance_monitor.stop_monitoring()


def benchmark_function_real(func, *args, iterations=10, **kwargs):
    """
    Benchmark a function with real performance measurements.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Dict with benchmark results
    """
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)

    # Clear caches
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Actual benchmark
    times = []
    metrics_list = []

    for i in range(iterations):
        metrics = performance_monitor.measure_inference_performance(
            func, *args, **kwargs
        )
        times.append(metrics.inference_time_ms)
        metrics_list.append(metrics)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate throughput if tokens_per_second is available
    tps_values = [
        m.tokens_per_second for m in metrics_list if m.tokens_per_second is not None
    ]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else None

    return {
        "iterations": iterations,
        "times_ms": times,
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "avg_tokens_per_second": avg_tps,
        "metrics_history": metrics_list,
        "final_system_metrics": performance_monitor.get_average_metrics(last_n=5),
    }


def get_real_system_metrics():
    """Get current real system metrics."""
    return performance_monitor.get_current_metrics()


def start_continuous_monitoring():
    """Start continuous performance monitoring."""
    performance_monitor.start_monitoring()


def stop_continuous_monitoring():
    """Stop continuous performance monitoring."""
    performance_monitor.stop_monitoring()
