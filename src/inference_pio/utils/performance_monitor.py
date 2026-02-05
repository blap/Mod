"""
Performance Monitor Module

This module provides a performance monitoring class that can be used across different
components of the Inference-PIO system to track and measure performance metrics.
"""

import time
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PerformanceMetrics:
    """Data class to hold performance metrics."""
    timestamp: float
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    cache_miss_rate: Optional[float] = None


class PerformanceMonitor:
    """
    Monitors performance metrics for various system components.
    """
    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.prefetch_hit_count = 0
        self.prefetch_miss_count = 0
        self.total_operations = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.metrics_history = []

    def record_hit(self, is_prefetch: bool = False):
        """Record a cache hit."""
        self.hit_count += 1
        if is_prefetch:
            self.prefetch_hit_count += 1
        self.total_operations += 1

    def record_miss(self, is_prefetch: bool = False):
        """Record a cache miss."""
        self.miss_count += 1
        if is_prefetch:
            self.prefetch_miss_count += 1
        self.total_operations += 1

    def record_eviction(self):
        """Record a cache eviction."""
        self.eviction_count += 1
        self.total_operations += 1

    def get_metrics(self) -> dict:
        """Get current performance metrics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        miss_rate = self.miss_count / total_requests if total_requests > 0 else 0
        
        prefetch_total = self.prefetch_hit_count + self.prefetch_miss_count
        prefetch_hit_rate = self.prefetch_hit_count / prefetch_total if prefetch_total > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "prefetch_hit_rate": prefetch_hit_rate,
            "total_operations": self.total_operations,
            "eviction_count": self.eviction_count,
            "uptime_seconds": time.time() - self.start_time
        }

    def reset(self):
        """Reset all counters."""
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.prefetch_hit_count = 0
        self.prefetch_miss_count = 0
        self.total_operations = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def record_completion(self, operation: str):
        """Record completion of an operation."""
        self.total_operations += 1
        # This can be extended to track different types of operations
        pass


class RealPerformanceMonitor(PerformanceMonitor):
    """
    Extended performance monitor that includes real system metrics.
    """
    def __init__(self, log_interval: int = 50):
        super().__init__(log_interval)
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False

    def get_current_system_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        timestamp = time.time()
        
        cpu_percent = None
        memory_percent = None
        memory_used_mb = None
        
        if self.psutil_available:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_used_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            memory_percent = self.process.memory_percent()
        
        # Calculate cache metrics
        total_requests = self.hit_count + self.miss_count
        cache_hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        cache_miss_rate = self.miss_count / total_requests if total_requests > 0 else 0
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            cache_hit_rate=cache_hit_rate,
            cache_miss_rate=cache_miss_rate
        )