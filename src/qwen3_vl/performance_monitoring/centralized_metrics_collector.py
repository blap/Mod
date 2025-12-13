"""
Centralized Metrics Collector System

This module provides a centralized system for collecting, aggregating, and reporting
performance metrics across the Qwen3-VL-2B-Instruct project. It supports various
metric types, provides real-time monitoring capabilities, and offers aggregation
and reporting features.

The system is designed to be thread-safe and efficient to minimize performance
overhead while providing comprehensive monitoring capabilities.
"""

import threading
import time
import json
import csv
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from datetime import datetime
import os
from enum import Enum


class MetricType(Enum):
    """
    Enum for different types of metrics.
    """
    TIME = "time"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COUNTER = "counter"
    RATIO = "ratio"
    PERCENTAGE = "percentage"
    CUSTOM = "custom"


@dataclass
class Metric:
    """
    Data class representing a single metric.
    
    Attributes:
        name (str): Name of the metric
        value (float): Value of the metric
        metric_type (MetricType): Type of the metric
        timestamp (float): Unix timestamp when metric was recorded
        source (str): Source component that generated the metric
        tags (Dict[str, str]): Additional tags for categorizing metrics
    """
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    source: str
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetricsAggregator:
    """
    Aggregates metrics over time windows and computes statistics.
    
    This class maintains rolling windows of metrics and computes various
    statistical measures to help understand performance trends.
    
    Attributes:
        window_size (int): Number of metrics to keep in the rolling window
        metrics (Dict[str, deque]): Rolling window of metrics for each name
        stats (Dict[str, Dict[str, float]]): Computed statistics for each metric
        _lock (threading.Lock): Lock for thread-safe operations
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize the metrics aggregator.
        
        Args:
            window_size: Number of metrics to keep in the rolling window
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def add_metric(self, name: str, value: float) -> None:
        """
        Add a metric value to the aggregator.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        with self._lock:
            self.metrics[name].append(value)
            self._update_stats(name)
    
    def _update_stats(self, name: str) -> None:
        """
        Update statistics for a metric.
        
        Args:
            name: Name of the metric to update stats for
        """
        values = list(self.metrics[name])
        if not values:
            return
        
        count = len(values)
        total = sum(values)
        avg = total / count
        min_val = min(values)
        max_val = max(values)
        
        # Calculate standard deviation
        variance = sum((x - avg) ** 2 for x in values) / count if count > 0 else 0
        std_dev = variance ** 0.5
        
        self.stats[name] = {
            'count': count,
            'total': total,
            'average': avg,
            'min': min_val,
            'max': max_val,
            'std_dev': std_dev,
            'last_value': values[-1] if values else 0
        }
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for one or all metrics.
        
        Args:
            name: Name of specific metric (None for all metrics)
            
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            if name:
                return self.stats.get(name, {})
            else:
                return dict(self.stats)
    
    def get_recent_values(self, name: str, count: int = 10) -> List[float]:
        """
        Get recent values for a metric.
        
        Args:
            name: Name of the metric
            count: Number of recent values to return
            
        Returns:
            List of recent metric values
        """
        with self._lock:
            values = list(self.metrics[name])
            return values[-count:] if len(values) >= count else values[:]
    
    def clear(self) -> None:
        """Clear all collected metrics and statistics."""
        with self._lock:
            self.metrics.clear()
            self.stats.clear()


class MetricsFilter:
    """
    Filters metrics based on various criteria.
    
    This class provides methods to filter metrics by name, type, time range,
    or custom criteria.
    """
    
    def __init__(self):
        self.filters: List[Callable[[Metric], bool]] = []
    
    def add_name_filter(self, name_pattern: str) -> 'MetricsFilter':
        """
        Add a filter for metric names using a pattern.
        
        Args:
            name_pattern: Pattern to match metric names (supports wildcards)
            
        Returns:
            Self for chaining
        """
        def name_filter(metric: Metric) -> bool:
            import fnmatch
            return fnmatch.fnmatch(metric.name, name_pattern)
        
        self.filters.append(name_filter)
        return self
    
    def add_type_filter(self, metric_type: MetricType) -> 'MetricsFilter':
        """
        Add a filter for metric types.
        
        Args:
            metric_type: Type of metric to include
            
        Returns:
            Self for chaining
        """
        def type_filter(metric: Metric) -> bool:
            return metric.metric_type == metric_type
        
        self.filters.append(type_filter)
        return self
    
    def add_source_filter(self, source_pattern: str) -> 'MetricsFilter':
        """
        Add a filter for metric sources.
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Self for chaining
        """
        def source_filter(metric: Metric) -> bool:
            import fnmatch
            return fnmatch.fnmatch(metric.source, source_pattern)
        
        self.filters.append(source_filter)
        return self
    
    def add_time_range_filter(self, start_time: float, end_time: float) -> 'MetricsFilter':
        """
        Add a filter for time range.
        
        Args:
            start_time: Start of time range (Unix timestamp)
            end_time: End of time range (Unix timestamp)
            
        Returns:
            Self for chaining
        """
        def time_filter(metric: Metric) -> bool:
            return start_time <= metric.timestamp <= end_time
        
        self.filters.append(time_filter)
        return self
    
    def apply(self, metrics: List[Metric]) -> List[Metric]:
        """
        Apply all filters to a list of metrics.
        
        Args:
            metrics: List of metrics to filter
            
        Returns:
            Filtered list of metrics
        """
        result = metrics
        for filter_func in self.filters:
            result = [m for m in result if filter_func(m)]
        return result


class CentralizedMetricsCollector:
    """
    Centralized system for collecting, aggregating, and reporting performance metrics.
    
    This singleton class provides a central point for collecting metrics from
    different parts of the system. It maintains thread-safe storage, provides
    aggregation capabilities, and offers various export methods.
    
    Attributes:
        _instance (Optional[CentralizedMetricsCollector]): Singleton instance
        _lock (threading.Lock): Lock for thread-safe singleton access
        metrics (List[Metric]): List of all collected metrics
        aggregator (MetricsAggregator): Aggregator for computing statistics
        enabled (bool): Whether metrics collection is enabled
        logger (logging.Logger): Logger instance
    """
    
    _instance: Optional['CentralizedMetricsCollector'] = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls):
        """
        Implement singleton pattern for the metrics collector.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the metrics collector.
        """
        if self._initialized:
            return

        self.metrics: List[Metric] = []
        self.aggregator = MetricsAggregator()
        self.enabled = True
        self.logger = logging.getLogger(__name__)

        # Thread safety for metrics list
        self._metrics_lock = threading.RLock()

        # Performance optimization: cache common metric names
        self._metric_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'CentralizedMetricsCollector':
        """
        Get the singleton instance of the metrics collector.
        
        Returns:
            CentralizedMetricsCollector instance
        """
        return cls()
    
    def enable(self) -> None:
        """Enable metrics collection."""
        self.enabled = True
        self.logger.info("Metrics collection enabled")
    
    def disable(self) -> None:
        """Disable metrics collection."""
        self.enabled = False
        self.logger.info("Metrics collection disabled")
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: Union[MetricType, str],
                     source: str = "unknown",
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            metric_type: Type of the metric (MetricType enum or string)
            source: Source component that generated the metric
            tags: Additional tags for categorizing metrics
        """
        if not self.enabled:
            return
        
        # Convert string to MetricType if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                self.logger.warning(f"Invalid metric type '{metric_type}', using CUSTOM")
                metric_type = MetricType.CUSTOM
        
        # Create metric object
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            source=source,
            tags=tags or {}
        )
        
        # Thread-safe addition to metrics list
        with self._metrics_lock:
            self.metrics.append(metric)
        
        # Add to aggregator for statistics
        self.aggregator.add_metric(name, value)
        
        # Cache for performance optimization
        self._metric_cache[name].append(value)
        
        self.logger.debug(f"Recorded metric: {name} = {value} ({metric_type.value}) from {source}")
    
    def record_timing(self, 
                     operation_name: str, 
                     execution_time: float,
                     source: str = "unknown") -> None:
        """
        Record a timing metric.
        
        Args:
            operation_name: Name of the operation being timed
            execution_time: Execution time in seconds
            source: Source component that generated the metric
        """
        self.record_metric(
            name=f"{operation_name}_time",
            value=execution_time,
            metric_type=MetricType.TIME,
            source=source,
            tags={"operation": operation_name}
        )
    
    def record_counter(self, 
                      name: str, 
                      value: float = 1.0,
                      source: str = "unknown") -> None:
        """
        Record a counter metric (incrementing counter).
        
        Args:
            name: Name of the counter
            value: Value to add to the counter (default 1.0)
            source: Source component that generated the metric
        """
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            source=source
        )
    
    def record_memory(self, 
                     name: str, 
                     value: float,
                     source: str = "unknown") -> None:
        """
        Record a memory metric.
        
        Args:
            name: Name of the memory metric
            value: Memory value in bytes
            source: Source component that generated the metric
        """
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.MEMORY,
            source=source
        )
    
    def record_throughput(self, 
                         name: str, 
                         value: float,
                         source: str = "unknown") -> None:
        """
        Record a throughput metric.
        
        Args:
            name: Name of the throughput metric
            value: Throughput value (e.g., operations per second)
            source: Source component that generated the metric
        """
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.THROUGHPUT,
            source=source
        )
    
    def get_metrics(self, 
                   filter_func: Optional[Callable[[Metric], bool]] = None,
                   limit: Optional[int] = None) -> List[Metric]:
        """
        Get collected metrics with optional filtering.
        
        Args:
            filter_func: Optional function to filter metrics
            limit: Optional limit on number of metrics returned
            
        Returns:
            List of metrics
        """
        with self._metrics_lock:
            metrics = self.metrics if filter_func is None else [m for m in self.metrics if filter_func(m)]
            return metrics[:limit] if limit else metrics[:]
    
    def get_latest_metrics(self, 
                          count: int = 10,
                          metric_type: Optional[MetricType] = None) -> List[Metric]:
        """
        Get the most recent metrics.
        
        Args:
            count: Number of recent metrics to return
            metric_type: Optional metric type to filter by
            
        Returns:
            List of recent metrics
        """
        with self._metrics_lock:
            if metric_type:
                filtered_metrics = [m for m in self.metrics if m.metric_type == metric_type]
            else:
                filtered_metrics = self.metrics
            
            return filtered_metrics[-count:] if len(filtered_metrics) >= count else filtered_metrics[:]
    
    def get_metric_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical information for metrics.
        
        Args:
            name: Name of specific metric (None for all metrics)
            
        Returns:
            Dictionary containing statistical information
        """
        return self.aggregator.get_stats(name)
    
    def get_metric_history(self, name: str, count: int = 100) -> List[Metric]:
        """
        Get historical values for a specific metric.
        
        Args:
            name: Name of the metric
            count: Number of historical values to return
            
        Returns:
            List of historical metric values
        """
        with self._metrics_lock:
            # Get metrics in reverse order to get the most recent ones
            all_metrics = [m for m in reversed(self.metrics) if m.name == name]
            return all_metrics[:count]
    
    def export_to_json(self, 
                      filename: str, 
                      filter_func: Optional[Callable[[Metric], bool]] = None) -> None:
        """
        Export metrics to a JSON file.
        
        Args:
            filename: Path to the output file
            filter_func: Optional function to filter metrics before export
        """
        metrics = self.get_metrics(filter_func)
        metrics_dict = [asdict(m) for m in metrics]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(metrics)} metrics to {filename}")
    
    def export_to_csv(self, 
                     filename: str, 
                     filter_func: Optional[Callable[[Metric], bool]] = None) -> None:
        """
        Export metrics to a CSV file.
        
        Args:
            filename: Path to the output file
            filter_func: Optional function to filter metrics before export
        """
        metrics = self.get_metrics(filter_func)
        
        if not metrics:
            self.logger.warning("No metrics to export")
            return
        
        fieldnames = ['name', 'value', 'metric_type', 'timestamp', 'source', 'tags']
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                # Convert metric to dictionary and handle tags
                row = asdict(metric)
                row['metric_type'] = row['metric_type'].value
                row['tags'] = json.dumps(row['tags'])  # Serialize tags as JSON string
                writer.writerow(row)
        
        self.logger.info(f"Exported {len(metrics)} metrics to {filename}")
    
    def export_to_prometheus(self, 
                            filename: str, 
                            filter_func: Optional[Callable[[Metric], bool]] = None) -> None:
        """
        Export metrics in Prometheus format.
        
        Args:
            filename: Path to the output file
            filter_func: Optional function to filter metrics before export
        """
        metrics = self.get_metrics(filter_func)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Metrics exported from Qwen3-VL-2B-Instruct\n")
            f.write(f"# Export time: {datetime.now().isoformat()}\n\n")
            
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            for name, group in metric_groups.items():
                # Write metric type comment
                metric_type = group[0].metric_type.value
                f.write(f"# TYPE {name} {metric_type}\n")
                
                # Write metric values
                for metric in group:
                    timestamp_ms = int(metric.timestamp * 1000)
                    f.write(f'{name}{{{self._format_tags(metric.tags)}}} {metric.value} {timestamp_ms}\n')
                
                f.write("\n")
        
        self.logger.info(f"Exported {len(metrics)} metrics to {filename} in Prometheus format")
    
    def _format_tags(self, tags: Dict[str, str]) -> str:
        """
        Format tags dictionary as a Prometheus label string.
        
        Args:
            tags: Dictionary of tags
            
        Returns:
            Formatted label string
        """
        if not tags:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in tags.items()]
        return ",".join(label_pairs)
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report.
        
        Returns:
            Dictionary containing metrics report
        """
        with self._metrics_lock:
            total_metrics = len(self.metrics)
        
        # Get aggregated statistics
        all_stats = self.aggregator.get_stats()
        
        # Get recent activity
        recent_metrics = self.get_latest_metrics(10)
        
        # Calculate time range
        if self.metrics:
            start_time = min(m.timestamp for m in self.metrics)
            end_time = max(m.timestamp for m in self.metrics)
            duration = end_time - start_time
        else:
            start_time = end_time = duration = 0
        
        report = {
            "summary": {
                "total_metrics": total_metrics,
                "time_range": {
                    "start": datetime.fromtimestamp(start_time).isoformat() if start_time else None,
                    "end": datetime.fromtimestamp(end_time).isoformat() if end_time else None,
                    "duration_seconds": duration
                },
                "active_metrics": list(all_stats.keys())
            },
            "statistics": all_stats,
            "recent_metrics": [asdict(m) for m in recent_metrics],
            "export_time": datetime.now().isoformat()
        }
        
        return report
    
    def print_report(self) -> None:
        """
        Print a formatted metrics report to the console.
        """
        report = self.get_report()
        
        print("\n" + "="*60)
        print("Qwen3-VL-2B-Instruct Metrics Report")
        print("="*60)
        
        print(f"Total Metrics Collected: {report['summary']['total_metrics']}")
        print(f"Time Range: {report['summary']['time_range']['start']} to {report['summary']['time_range']['end']}")
        print(f"Duration: {report['summary']['time_range']['duration_seconds']:.2f} seconds")
        print(f"Active Metrics: {len(report['summary']['active_metrics'])}")
        
        print("\nTop Metrics by Average Value:")
        sorted_stats = sorted(
            report['statistics'].items(),
            key=lambda x: x[1].get('average', 0),
            reverse=True
        )[:10]  # Top 10 metrics
        
        for name, stats in sorted_stats:
            print(f"  {name}: avg={stats.get('average', 0):.4f}, min={stats.get('min', 0):.4f}, max={stats.get('max', 0):.4f}")
        
        print("\nRecent Metrics:")
        for metric in report['recent_metrics']:
            print(f"  {metric['name']}: {metric['value']} ({metric['metric_type']}) - {datetime.fromtimestamp(metric['timestamp']).strftime('%H:%M:%S')}")
        
        print("="*60)
    
    def clear_metrics(self) -> None:
        """
        Clear all collected metrics.
        """
        with self._metrics_lock:
            self.metrics.clear()
        
        self.aggregator = MetricsAggregator()  # Reset aggregator
        self._metric_cache.clear()
        
        self.logger.info("Cleared all collected metrics")
    
    def start_monitoring_thread(self, 
                               interval: float = 5.0,
                               callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> threading.Thread:
        """
        Start a background thread to periodically collect system metrics.
        
        Args:
            interval: Interval between metric collections in seconds
            callback: Optional callback function to process metrics
            
        Returns:
            Monitoring thread
        """
        def monitoring_loop():
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                
                while getattr(monitoring_loop, 'running', True):
                    # Collect system metrics
                    timestamp = time.time()
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.record_metric(
                        name="system_cpu_percent",
                        value=cpu_percent,
                        metric_type=MetricType.PERCENTAGE,
                        source="system_monitor"
                    )
                    
                    # Memory usage
                    memory_info = process.memory_info()
                    self.record_metric(
                        name="system_memory_rss",
                        value=memory_info.rss,
                        metric_type=MetricType.MEMORY,
                        source="system_monitor"
                    )
                    self.record_metric(
                        name="system_memory_vms",
                        value=memory_info.vms,
                        metric_type=MetricType.MEMORY,
                        source="system_monitor"
                    )
                    
                    # Additional metrics if available
                    try:
                        memory_percent = process.memory_percent()
                        self.record_metric(
                            name="system_memory_percent",
                            value=memory_percent,
                            metric_type=MetricType.PERCENTAGE,
                            source="system_monitor"
                        )
                    except:
                        pass  # Skip if memory percent is not available
                    
                    # Thread count
                    thread_count = process.num_threads()
                    self.record_metric(
                        name="system_thread_count",
                        value=thread_count,
                        metric_type=MetricType.COUNTER,
                        source="system_monitor"
                    )
                    
                    # Call custom callback if provided
                    if callback:
                        report = self.get_report()
                        callback(report)
                    
                    time.sleep(interval)
                    
            except ImportError:
                self.logger.warning("psutil not available, skipping system monitoring")
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

        # Add a running attribute to the function to track its state
        monitoring_loop.running = True  # type: ignore[attr-defined]
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()

        self.logger.info(f"Started system monitoring thread with {interval}s interval")
        return thread
    
    def stop_monitoring_thread(self, thread: threading.Thread) -> None:
        """
        Stop a monitoring thread.
        
        Args:
            thread: Monitoring thread to stop
        """
        if hasattr(thread, 'running'):
            thread.running = False
        
        thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
        self.logger.info("Stopped system monitoring thread")


# Initialize logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


# Convenience functions for common operations
def get_metrics_collector() -> CentralizedMetricsCollector:
    """
    Get the centralized metrics collector instance.
    
    Returns:
        CentralizedMetricsCollector instance
    """
    return CentralizedMetricsCollector.get_instance()


def record_metric(name: str, 
                 value: float, 
                 metric_type: Union[MetricType, str] = MetricType.CUSTOM,
                 source: str = "unknown",
                 tags: Optional[Dict[str, str]] = None) -> None:
    """
    Convenience function to record a metric using the centralized collector.
    
    Args:
        name: Name of the metric
        value: Value of the metric
        metric_type: Type of the metric
        source: Source component that generated the metric
        tags: Additional tags for categorizing metrics
    """
    collector = get_metrics_collector()
    collector.record_metric(name, value, metric_type, source, tags)


def record_timing(operation_name: str, 
                 execution_time: float,
                 source: str = "unknown") -> None:
    """
    Convenience function to record a timing metric.
    
    Args:
        operation_name: Name of the operation being timed
        execution_time: Execution time in seconds
        source: Source component that generated the metric
    """
    collector = get_metrics_collector()
    collector.record_timing(operation_name, execution_time, source)


def record_counter(name: str, 
                  value: float = 1.0,
                  source: str = "unknown") -> None:
    """
    Convenience function to record a counter metric.
    
    Args:
        name: Name of the counter
        value: Value to add to the counter
        source: Source component that generated the metric
    """
    collector = get_metrics_collector()
    collector.record_counter(name, value, source)


def print_metrics_report() -> None:
    """
    Print a formatted metrics report to the console.
    """
    collector = get_metrics_collector()
    collector.print_report()


# Initialize the collector on import
get_metrics_collector()