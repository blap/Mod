"""
Metrics collection utilities for the Qwen3-VL system.
"""

from typing import Any, Dict
import time
import threading
from collections import defaultdict, deque


class MetricsCollector:
    """
    A centralized metrics collector for the Qwen3-VL system.
    """
    
    def __init__(self):
        self._metrics = defaultdict(deque)
        self._timings = defaultdict(deque)
        self._counters = defaultdict(float)
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, metric_type: str = "gauge", component: str = "unknown"):
        """
        Record a metric value.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            metric_type: Type of metric ('gauge', 'counter', 'histogram', 'timing')
            component: Component that generated the metric
        """
        with self._lock:
            metric_data = {
                'value': value,
                'type': metric_type,
                'component': component,
                'timestamp': time.time()
            }
            self._metrics[name].append(metric_data)
    
    def record_timing(self, name: str, value: float, component: str = "unknown"):
        """
        Record a timing metric.
        
        Args:
            name: Name of the timing metric
            value: Timing value in seconds
            component: Component that generated the metric
        """
        with self._lock:
            timing_data = {
                'value': value,
                'component': component,
                'timestamp': time.time()
            }
            self._timings[name].append(timing_data)
    
    def record_counter(self, name: str, value: float, component: str = "unknown"):
        """
        Record a counter metric.
        
        Args:
            name: Name of the counter metric
            value: Counter increment value
            component: Component that generated the metric
        """
        with self._lock:
            self._counters[name] += value
    
    def get_metric(self, name: str) -> Any:
        """
        Get a specific metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric data
        """
        with self._lock:
            if name in self._metrics:
                return list(self._metrics[name])
            elif name in self._timings:
                return list(self._timings[name])
            elif name in self._counters:
                return self._counters[name]
            else:
                return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            return {
                'metrics': dict(self._metrics),
                'timings': dict(self._timings),
                'counters': dict(self._counters)
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._timings.clear()
            self._counters.clear()


# Global metrics collector instance
_collector = MetricsCollector()


def record_metric(name: str, value: float, metric_type: str = "gauge", component: str = "unknown"):
    """
    Record a metric value using the global collector.
    
    Args:
        name: Name of the metric
        value: Value of the metric
        metric_type: Type of metric ('gauge', 'counter', 'histogram', 'timing')
        component: Component that generated the metric
    """
    _collector.record_metric(name, value, metric_type, component)


def record_timing(name: str, value: float, component: str = "unknown"):
    """
    Record a timing metric using the global collector.
    
    Args:
        name: Name of the timing metric
        value: Timing value in seconds
        component: Component that generated the metric
    """
    _collector.record_timing(name, value, component)


def record_counter(name: str, value: float, component: str = "unknown"):
    """
    Record a counter metric using the global collector.
    
    Args:
        name: Name of the counter metric
        value: Counter increment value
        component: Component that generated the metric
    """
    _collector.record_counter(name, value, component)


def get_metric(name: str) -> Any:
    """
    Get a specific metric from the global collector.
    
    Args:
        name: Name of the metric
        
    Returns:
        Metric data
    """
    return _collector.get_metric(name)


def get_all_metrics() -> Dict[str, Any]:
    """
    Get all metrics from the global collector.
    
    Returns:
        Dictionary with all metrics
    """
    return _collector.get_all_metrics()