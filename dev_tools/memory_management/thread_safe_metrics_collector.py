"""
Thread-Safe Centralized Metrics Collection System for Qwen3-VL Model

This module implements a comprehensive thread-safe metrics collection system that gathers performance,
memory, and operational metrics from all components of the Qwen3-VL model. The system
provides real-time metrics collection, aggregation, and export capabilities to support
monitoring, optimization, and debugging of the model across different hardware configurations.

Features:
- Thread-safe metrics collection from all components
- Standardized metric formats and naming conventions
- Real-time metrics collection and aggregation
- Export capabilities to JSON, CSV, and Prometheus formats
- Integration with existing performance monitoring functions
- Metrics validation and error handling
- Comprehensive documentation and examples
"""

import json
import csv
import time
import threading
import queue
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Deque
from collections import defaultdict, deque
import psutil
import torch
import gc
from enum import Enum
import logging
import os
from concurrent.futures import ThreadPoolExecutor


class MetricType(Enum):
    """Enumeration for different metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ThreadSafeMetricValue:
    """Thread-safe representation of a single metric value with metadata."""

    def __init__(self,
                 name: str,
                 value: Union[int, float, str, bool],
                 metric_type: MetricType,
                 labels: Optional[Dict[str, str]] = None,
                 description: str = "",
                 timestamp: Optional[float] = None):
        self.name = name
        self.value = value
        self.metric_type = metric_type
        self.labels = labels or {}
        self.description = description
        self.timestamp = timestamp or time.time()
        self._lock = threading.RLock()  # Use RLock for recursive locking

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        with self._lock:
            return {
                "name": self.name,
                "value": self.value,
                "type": self.metric_type.value,
                "labels": self.labels,
                "description": self.description,
                "timestamp": self.timestamp,
                "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
            }

    def __repr__(self) -> str:
        with self._lock:
            return f"ThreadSafeMetricValue(name='{self.name}', value={self.value}, type={self.metric_type.value})"


class ThreadSafeMetricsCollector:
    """
    Thread-safe centralized metrics collector for the Qwen3-VL model ecosystem.

    This class collects metrics from various components of the model including:
    - Performance metrics (latency, throughput, etc.)
    - Memory metrics (GPU/CPU usage, allocation patterns, etc.)
    - Hardware metrics (temperature, power, etc.)
    - Custom metrics from different optimization components
    """

    def __init__(self,
                 collection_interval: float = 1.0,
                 max_metrics_buffer: int = 10000,
                 enable_prometheus_export: bool = True,
                 max_history_per_metric: int = 1000,
                 validation_enabled: bool = True):
        """
        Initialize the thread-safe metrics collector.

        Args:
            collection_interval: Interval in seconds between metric collections
            max_metrics_buffer: Maximum number of metrics to keep in memory
            enable_prometheus_export: Whether to enable Prometheus export functionality
            max_history_per_metric: Maximum number of historical values to keep per metric
            validation_enabled: Whether to enable metric validation
        """
        self.collection_interval = collection_interval
        self.max_metrics_buffer = max_metrics_buffer
        self.enable_prometheus_export = enable_prometheus_export
        self.max_history_per_metric = max_history_per_metric
        self.validation_enabled = validation_enabled

        # Storage for metrics - all protected by locks
        self._metrics_buffer: Deque[ThreadSafeMetricValue] = deque(maxlen=max_metrics_buffer)
        self._current_metrics: Dict[str, ThreadSafeMetricValue] = {}
        self._metrics_history: Dict[str, Deque[ThreadSafeMetricValue]] = defaultdict(lambda: deque(maxlen=max_history_per_metric))

        # Threading and synchronization
        self._collection_thread: Optional[threading.Thread] = None
        self._is_collecting = False
        self._collection_lock = threading.RLock()  # Use RLock for recursive locking
        self._shutdown_event = threading.Event()
        self._metrics_queue = queue.Queue(maxsize=1000)  # Queue for metrics to be processed
        self._buffer_lock = threading.RLock()  # Separate lock for buffer operations
        self._current_metrics_lock = threading.RLock()  # Separate lock for current metrics
        self._history_lock = threading.RLock()  # Separate lock for history

        # Validation rules
        self._validation_rules: Dict[str, Callable[[Any], bool]] = {}

        # Export formats
        self._export_formats = {
            'json': self.export_to_json,
            'csv': self.export_to_csv,
            'prometheus': self.export_to_prometheus
        }

        # System metrics collectors
        self._system_collectors = {
            'cpu': self._collect_cpu_metrics,
            'memory': self._collect_memory_metrics,
            'gpu': self._collect_gpu_metrics,
            'process': self._collect_process_metrics
        }

        # Performance tracking
        self._performance_trackers: Dict[str, Dict[str, Any]] = {}
        self._perf_tracker_lock = threading.RLock()

        # Logger setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def start_collection(self):
        """Start the metrics collection thread."""
        with self._collection_lock:
            if self._is_collecting:
                self.logger.warning("Metrics collection is already running")
                return

            self._is_collecting = True
            self._shutdown_event.clear()
            self._collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
            self._collection_thread.start()
            self.logger.info("Started metrics collection")

    def stop_collection(self):
        """Stop the metrics collection thread."""
        with self._collection_lock:
            if not self._is_collecting:
                self.logger.warning("Metrics collection is not running")
                return

            self._is_collecting = False
            self._shutdown_event.set()

            if self._collection_thread:
                self._collection_thread.join(timeout=2.0)
                self._collection_thread = None

            self._shutdown_event.clear()
            self.logger.info("Stopped metrics collection")

    def _collection_worker(self):
        """Background worker for collecting metrics."""
        while self._is_collecting and not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect custom metrics from registered sources
                self._collect_custom_metrics()

                # Process any queued metrics
                self._process_queued_metrics()

                # Sleep for the collection interval
                time.sleep(self.collection_interval)

            except Exception as e:
                self.logger.error(f"Error in metrics collection worker: {e}")
                # Continue collection even if there's an error
                time.sleep(self.collection_interval)

    def _process_queued_metrics(self):
        """Process metrics from the queue."""
        while not self._metrics_queue.empty():
            try:
                metric_data = self._metrics_queue.get_nowait()
                self.add_metric(**metric_data)
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing queued metric: {e}")

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            self.add_metric(
                name="system_cpu_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="CPU utilization percentage"
            )

            self.add_metric(
                name="system_cpu_count",
                value=cpu_count,
                metric_type=MetricType.GAUGE,
                labels={"unit": "count"},
                description="Number of CPU cores"
            )

            if cpu_freq:
                self.add_metric(
                    name="system_cpu_frequency_current",
                    value=cpu_freq.current,
                    metric_type=MetricType.GAUGE,
                    labels={"unit": "mhz"},
                    description="Current CPU frequency"
                )

            # Memory metrics
            memory = psutil.virtual_memory()

            self.add_metric(
                name="system_memory_total",
                value=memory.total,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Total system memory"
            )

            self.add_metric(
                name="system_memory_available",
                value=memory.available,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Available system memory"
            )

            self.add_metric(
                name="system_memory_used",
                value=memory.used,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Used system memory"
            )

            self.add_metric(
                name="system_memory_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="System memory usage percentage"
            )

            # Swap metrics
            swap = psutil.swap_memory()
            self.add_metric(
                name="system_swap_total",
                value=swap.total,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Total swap memory"
            )

            self.add_metric(
                name="system_swap_percent",
                value=swap.percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="Swap memory usage percentage"
            )

            # GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_reserved = torch.cuda.memory_reserved()
                gpu_device_count = torch.cuda.device_count()

                self.add_metric(
                    name="gpu_memory_allocated",
                    value=gpu_memory_allocated,
                    metric_type=MetricType.GAUGE,
                    labels={"unit": "bytes", "device": f"cuda:{torch.cuda.current_device()}" if gpu_device_count > 0 else "none"},
                    description="GPU memory allocated by PyTorch"
                )

                self.add_metric(
                    name="gpu_memory_reserved",
                    value=gpu_memory_reserved,
                    metric_type=MetricType.GAUGE,
                    labels={"unit": "bytes", "device": f"cuda:{torch.cuda.current_device()}" if gpu_device_count > 0 else "none"},
                    description="GPU memory reserved by PyTorch"
                )

                self.add_metric(
                    name="gpu_device_count",
                    value=gpu_device_count,
                    metric_type=MetricType.GAUGE,
                    labels={"unit": "count"},
                    description="Number of available GPU devices"
                )

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def _collect_custom_metrics(self):
        """Collect custom metrics from registered sources."""
        # This is where we would collect metrics from various model components
        # For now, we'll add some placeholder metrics
        pass

    def _collect_cpu_metrics(self) -> List[ThreadSafeMetricValue]:
        """Collect CPU-related metrics."""
        metrics = []

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            metrics.append(ThreadSafeMetricValue(
                name="cpu_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="CPU utilization percentage"
            ))

            metrics.append(ThreadSafeMetricValue(
                name="cpu_count",
                value=cpu_count,
                metric_type=MetricType.GAUGE,
                labels={"unit": "count"},
                description="Number of logical CPU cores"
            ))

            if cpu_freq:
                metrics.append(ThreadSafeMetricValue(
                    name="cpu_frequency_current",
                    value=cpu_freq.current,
                    metric_type=MetricType.GAUGE,
                    labels={"unit": "mhz"},
                    description="Current CPU frequency"
                ))

                metrics.append(ThreadSafeMetricValue(
                    name="cpu_frequency_max",
                    value=cpu_freq.max,
                    metric_type=MetricType.GAUGE,
                    labels={"unit": "mhz"},
                    description="Maximum CPU frequency"
                ))

        except Exception as e:
            self.logger.error(f"Error collecting CPU metrics: {e}")

        return metrics

    def _collect_memory_metrics(self) -> List[ThreadSafeMetricValue]:
        """Collect memory-related metrics."""
        metrics = []

        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            metrics.append(ThreadSafeMetricValue(
                name="memory_total",
                value=memory.total,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Total system memory"
            ))

            metrics.append(ThreadSafeMetricValue(
                name="memory_available",
                value=memory.available,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Available system memory"
            ))

            metrics.append(ThreadSafeMetricValue(
                name="memory_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="System memory usage percentage"
            ))

            metrics.append(ThreadSafeMetricValue(
                name="swap_total",
                value=swap.total,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Total swap memory"
            ))

            metrics.append(ThreadSafeMetricValue(
                name="swap_percent",
                value=swap.percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="Swap memory usage percentage"
            ))

        except Exception as e:
            self.logger.error(f"Error collecting memory metrics: {e}")

        return metrics

    def _collect_gpu_metrics(self) -> List[ThreadSafeMetricValue]:
        """Collect GPU-related metrics."""
        metrics = []

        if torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()

                for i in range(device_count):
                    # Memory metrics
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory

                    metrics.append(ThreadSafeMetricValue(
                        name="gpu_memory_allocated",
                        value=memory_allocated,
                        metric_type=MetricType.GAUGE,
                        labels={"device": f"cuda:{i}", "unit": "bytes"},
                        description=f"GPU {i} memory allocated by PyTorch"
                    ))

                    metrics.append(ThreadSafeMetricValue(
                        name="gpu_memory_reserved",
                        value=memory_reserved,
                        metric_type=MetricType.GAUGE,
                        labels={"device": f"cuda:{i}", "unit": "bytes"},
                        description=f"GPU {i} memory reserved by PyTorch"
                    ))

                    metrics.append(ThreadSafeMetricValue(
                        name="gpu_memory_utilization_percent",
                        value=(memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                        metric_type=MetricType.GAUGE,
                        labels={"device": f"cuda:{i}", "unit": "percent"},
                        description=f"GPU {i} memory utilization percentage"
                    ))

                    # Additional metrics if nvidia-ml-py is available
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                        # Temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics.append(ThreadSafeMetricValue(
                            name="gpu_temperature",
                            value=temp,
                            metric_type=MetricType.GAUGE,
                            labels={"device": f"cuda:{i}", "unit": "celsius"},
                            description=f"GPU {i} temperature"
                        ))

                        # Power usage
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        metrics.append(ThreadSafeMetricValue(
                            name="gpu_power_usage",
                            value=power,
                            metric_type=MetricType.GAUGE,
                            labels={"device": f"cuda:{i}", "unit": "watts"},
                            description=f"GPU {i} power usage"
                        ))

                        # Utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics.append(ThreadSafeMetricValue(
                            name="gpu_gpu_utilization_percent",
                            value=util.gpu,
                            metric_type=MetricType.GAUGE,
                            labels={"device": f"cuda:{i}", "unit": "percent"},
                            description=f"GPU {i} utilization percentage"
                        ))

                        metrics.append(ThreadSafeMetricValue(
                            name="gpu_memory_utilization_percent",
                            value=util.memory,
                            metric_type=MetricType.GAUGE,
                            labels={"device": f"cuda:{i}", "unit": "percent"},
                            description=f"GPU {i} memory utilization percentage"
                        ))

                    except ImportError:
                        # If pynvml is not available, skip these metrics
                        self.logger.debug("pynvml not available, skipping extended GPU metrics")
                        pass
                    except Exception as e:
                        self.logger.error(f"Error collecting extended GPU metrics for device {i}: {e}")

            except Exception as e:
                self.logger.error(f"Error collecting GPU metrics: {e}")

        return metrics

    def _collect_process_metrics(self) -> List[ThreadSafeMetricValue]:
        """Collect process-related metrics."""
        metrics = []

        try:
            process = psutil.Process()

            # Memory info
            memory_info = process.memory_info()
            metrics.append(ThreadSafeMetricValue(
                name="process_memory_rss",
                value=memory_info.rss,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Process resident set size (RSS)"
            ))

            metrics.append(ThreadSafeMetricValue(
                name="process_memory_vms",
                value=memory_info.vms,
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description="Process virtual memory size (VMS)"
            ))

            # CPU info
            cpu_percent = process.cpu_percent()
            metrics.append(ThreadSafeMetricValue(
                name="process_cpu_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                labels={"unit": "percent"},
                description="Process CPU utilization percentage"
            ))

            # Number of threads
            num_threads = process.num_threads()
            metrics.append(ThreadSafeMetricValue(
                name="process_num_threads",
                value=num_threads,
                metric_type=MetricType.GAUGE,
                labels={"unit": "count"},
                description="Number of process threads"
            ))

            # Number of open files
            num_fds = process.num_fds()
            metrics.append(ThreadSafeMetricValue(
                name="process_num_fds",
                value=num_fds,
                metric_type=MetricType.GAUGE,
                labels={"unit": "count"},
                description="Number of open file descriptors"
            ))

        except Exception as e:
            self.logger.error(f"Error collecting process metrics: {e}")

        return metrics

    def add_metric(self,
                   name: str,
                   value: Union[int, float, str, bool],
                   metric_type: MetricType,
                   labels: Optional[Dict[str, str]] = None,
                   description: str = "",
                   timestamp: Optional[float] = None) -> bool:
        """
        Add a metric to the collector.

        Args:
            name: Name of the metric
            value: Value of the metric
            metric_type: Type of the metric (counter, gauge, histogram, summary)
            labels: Optional labels for the metric
            description: Optional description of the metric
            timestamp: Optional timestamp for the metric (defaults to current time)

        Returns:
            bool: True if metric was added successfully, False otherwise
        """
        try:
            # Validate metric name
            if self.validation_enabled and not self._validate_metric_name(name):
                self.logger.error(f"Invalid metric name: {name}")
                return False

            # Validate metric value
            if self.validation_enabled and not self._validate_metric_value(value):
                self.logger.error(f"Invalid metric value: {value}")
                return False

            # Create metric value object
            metric = ThreadSafeMetricValue(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels,
                description=description,
                timestamp=timestamp
            )

            # Add to buffer with proper locking
            with self._buffer_lock:
                self._metrics_buffer.append(metric)

            # Update current metrics with proper locking
            key = self._get_metric_key(name, labels)
            with self._current_metrics_lock:
                self._current_metrics[key] = metric

            # Update history with proper locking
            with self._history_lock:
                self._metrics_history[name].append(metric)

            return True
        except Exception as e:
            self.logger.error(f"Error adding metric {name}: {e}")
            return False

    def _validate_metric_name(self, name: str) -> bool:
        """Validate metric name according to naming conventions."""
        # Metric names should follow Prometheus naming conventions:
        # - Start with a letter
        # - Contain only letters, numbers, and underscores
        # - Not start with reserved prefixes like "prometheus_"
        pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, name)) and not name.startswith('prometheus_')

    def _validate_metric_value(self, value: Union[int, float, str, bool]) -> bool:
        """Validate metric value."""
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                # Check for NaN and infinity
                return not (value != value or value == float('inf') or value == float('-inf'))
            return True
        elif isinstance(value, str):
            # Limit string length
            return len(value) <= 1000
        elif isinstance(value, bool):
            return True
        return False

    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate a unique key for a metric."""
        if labels:
            label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return f"{name}{{}}"  # Return name with empty braces when no labels

    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[ThreadSafeMetricValue]:
        """Get the current value of a specific metric."""
        key = self._get_metric_key(name, labels)
        with self._current_metrics_lock:
            return self._current_metrics.get(key)

    def get_metrics_history(self, name: str, limit: Optional[int] = None) -> List[ThreadSafeMetricValue]:
        """Get historical values for a specific metric."""
        with self._history_lock:
            history = list(self._metrics_history.get(name, []))
            if limit:
                history = history[-limit:]
            return history

    def get_all_current_metrics(self) -> Dict[str, ThreadSafeMetricValue]:
        """Get all current metric values."""
        with self._current_metrics_lock:
            return self._current_metrics.copy()

    def get_all_metrics_buffer(self) -> List[ThreadSafeMetricValue]:
        """Get all metrics in the buffer."""
        with self._buffer_lock:
            return list(self._metrics_buffer)

    def register_validation_rule(self, name: str, validator: Callable[[Any], bool]):
        """Register a validation rule for a metric."""
        with self._collection_lock:
            self._validation_rules[name] = validator

    def validate_metric(self, name: str, value: Any) -> bool:
        """Validate a metric value against registered rules."""
        with self._collection_lock:
            if name in self._validation_rules:
                return self._validation_rules[name](value)
            return True

    def export_to_json(self,
                      output_path: Optional[str] = None,
                      include_history: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Export metrics to JSON format.

        Args:
            output_path: Optional path to write JSON file
            include_history: Whether to include historical data

        Returns:
            JSON string if output_path is None, otherwise writes to file
        """
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": [
                    metric.to_dict()
                    for metric in self.get_all_metrics_buffer()
                ]
            }

            if include_history:
                with self._history_lock:
                    data["history"] = {}
                    for name, history in self._metrics_history.items():
                        data["history"][name] = [metric.to_dict() for metric in history]

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return data
            else:
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            raise

    def export_to_csv(self,
                     output_path: str,
                     include_history: bool = False) -> None:
        """
        Export metrics to CSV format.

        Args:
            output_path: Path to write CSV file
            include_history: Whether to include historical data
        """
        try:
            metrics = self.get_all_metrics_buffer()

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['name', 'value', 'type', 'labels', 'description', 'timestamp', 'datetime']
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                for metric in metrics:
                    row = metric.to_dict()
                    # Convert labels to string representation
                    row['labels'] = str(row['labels'])
                    writer.writerow(row)
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus text format string
        """
        try:
            output = []

            # Add header
            output.append("# Metrics collected from Qwen3-VL model")
            output.append(f"# Collected at: {datetime.now().isoformat()}")
            output.append("")

            # Group metrics by name - need to get all metrics from buffer
            all_metrics = self.get_all_metrics_buffer()
            
            # Group metrics by name
            metrics_by_name = defaultdict(list)
            for metric in all_metrics:
                metrics_by_name[metric.name].append(metric)

            # Format each metric group
            for name, metrics in metrics_by_name.items():
                # Add type information
                metric_type = metrics[0].metric_type.value
                output.append(f"# TYPE {name} {metric_type}")

                # Add metric descriptions
                if metrics[0].description:
                    output.append(f"# HELP {name} {metrics[0].description}")

                # Add metric values
                for metric in metrics:
                    # Format labels
                    label_str = ""
                    if metric.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                        label_str = f"{{{','.join(label_pairs)}}}"

                    # Format value (convert boolean values to 1/0)
                    value = metric.value
                    if isinstance(value, bool):
                        value = 1 if value else 0

                    output.append(f"{name}{label_str} {value} {int(metric.timestamp * 1000)}")

                output.append("")

            return "\n".join(output)
        except Exception as e:
            self.logger.error(f"Error exporting to Prometheus: {e}")
            raise

    def export(self,
              format_type: str,
              output_path: Optional[str] = None,
              **kwargs) -> Any:
        """
        Export metrics in the specified format.

        Args:
            format_type: Format to export ('json', 'csv', 'prometheus')
            output_path: Optional path to write output
            **kwargs: Additional arguments for export functions

        Returns:
            Exported data in the specified format
        """
        if format_type not in self._export_formats:
            raise ValueError(f"Unsupported export format: {format_type}")

        try:
            if format_type == 'prometheus':
                # Prometheus export doesn't support output_path parameter
                return self._export_formats[format_type](**kwargs)
            else:
                return self._export_formats[format_type](output_path=output_path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error exporting in {format_type} format: {e}")
            raise

    def add_performance_tracker(self, name: str,
                              start_time: Optional[float] = None,
                              start_memory: Optional[int] = None) -> None:
        """Add a performance tracker for measuring execution time and memory."""
        with self._perf_tracker_lock:
            self._performance_trackers[name] = {
                'start_time': start_time or time.time(),
                'start_memory': start_memory or (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
                'end_time': None,
                'end_memory': None
            }

    def end_performance_tracker(self, name: str) -> Optional[Dict[str, Any]]:
        """End a performance tracker and return results."""
        with self._perf_tracker_lock:
            if name not in self._performance_trackers:
                return None

            tracker = self._performance_trackers[name]
            tracker['end_time'] = time.time()
            tracker['end_memory'] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            result = {
                'duration': tracker['end_time'] - tracker['start_time'],
                'memory_delta': tracker['end_memory'] - tracker['start_memory'],
                'start_time': tracker['start_time'],
                'end_time': tracker['end_time'],
                'start_memory': tracker['start_memory'],
                'end_memory': tracker['end_memory']
            }

            # Remove tracker before adding metrics to avoid race conditions
            del self._performance_trackers[name]

            # Add metrics
            success1 = self.add_metric(
                name=f"performance_tracker_{name}_duration",
                value=result['duration'],
                metric_type=MetricType.GAUGE,
                labels={"unit": "seconds"},
                description=f"Execution time for {name}"
            )

            success2 = self.add_metric(
                name=f"performance_tracker_{name}_memory_delta",
                value=result['memory_delta'],
                metric_type=MetricType.GAUGE,
                labels={"unit": "bytes"},
                description=f"Memory delta for {name}"
            )

            if not success1 or not success2:
                self.logger.error(f"Failed to add performance tracking metrics for {name}")

            return result

    def reset(self):
        """Reset all collected metrics."""
        with self._buffer_lock:
            self._metrics_buffer.clear()
        
        with self._current_metrics_lock:
            self._current_metrics.clear()
        
        with self._history_lock:
            self._metrics_history.clear()
        
        with self._perf_tracker_lock:
            self._performance_trackers.clear()
        
        # Clear the queue
        with self._collection_lock:
            while not self._metrics_queue.empty():
                try:
                    self._metrics_queue.get_nowait()
                except queue.Empty:
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the metrics collector."""
        return {
            "total_metrics_collected": len(self.get_all_metrics_buffer()),
            "current_metrics_count": len(self.get_all_current_metrics()),
            "metrics_history_count": {name: len(history) for name, history in self._metrics_history.items()},
            "collection_interval": self.collection_interval,
            "max_buffer_size": self.max_metrics_buffer,
            "max_history_per_metric": self.max_history_per_metric,
            "is_collecting": self._is_collecting,
            "enabled_export_formats": list(self._export_formats.keys()),
            "queue_size": self._metrics_queue.qsize(),
            "performance_trackers_count": len(self._performance_trackers)
        }

    def queue_metric(self,
                     name: str,
                     value: Union[int, float, str, bool],
                     metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None,
                     description: str = "",
                     timestamp: Optional[float] = None) -> bool:
        """
        Queue a metric for later processing. This is useful for high-frequency metrics
        to avoid blocking the main application thread.

        Args:
            name: Name of the metric
            value: Value of the metric
            metric_type: Type of the metric (counter, gauge, histogram, summary)
            labels: Optional labels for the metric
            description: Optional description of the metric
            timestamp: Optional timestamp for the metric

        Returns:
            bool: True if metric was queued successfully, False otherwise
        """
        try:
            metric_data = {
                'name': name,
                'value': value,
                'metric_type': metric_type,
                'labels': labels,
                'description': description,
                'timestamp': timestamp
            }
            self._metrics_queue.put_nowait(metric_data)
            return True
        except queue.Full:
            self.logger.warning(f"Metrics queue is full, dropping metric: {name}")
            return False
        except Exception as e:
            self.logger.error(f"Error queuing metric {name}: {e}")
            return False


# Global thread-safe metrics collector instance
global_thread_safe_metrics_collector = ThreadSafeMetricsCollector()


def get_thread_safe_metrics_collector() -> ThreadSafeMetricsCollector:
    """Get the global thread-safe metrics collector instance."""
    return global_thread_safe_metrics_collector


def collect_system_metrics():
    """Convenience function to collect system metrics."""
    collector = get_thread_safe_metrics_collector()

    # Add system metrics
    collector._collect_system_metrics()


def add_performance_metric(name: str, value: Union[int, float],
                          labels: Optional[Dict[str, str]] = None,
                          description: str = ""):
    """Convenience function to add a performance metric."""
    collector = get_thread_safe_metrics_collector()
    collector.add_metric(
        name=name,
        value=value,
        metric_type=MetricType.GAUGE,
        labels=labels,
        description=description
    )


def start_performance_tracking(name: str):
    """Start tracking performance for a specific operation."""
    collector = get_thread_safe_metrics_collector()
    collector.add_performance_tracker(name)


def end_performance_tracking(name: str) -> Optional[Dict[str, Any]]:
    """End performance tracking and return results."""
    collector = get_thread_safe_metrics_collector()
    return collector.end_performance_tracker(name)


def queue_performance_metric(name: str, value: Union[int, float],
                           labels: Optional[Dict[str, str]] = None,
                           description: str = ""):
    """Convenience function to queue a performance metric."""
    collector = get_thread_safe_metrics_collector()
    collector.queue_metric(
        name=name,
        value=value,
        metric_type=MetricType.GAUGE,
        labels=labels,
        description=description
    )


# Example usage and testing
if __name__ == "__main__":
    # Initialize the collector
    collector = get_thread_safe_metrics_collector()
    collector.start_collection()

    # Add some example metrics
    collector.add_metric(
        name="model_inference_latency",
        value=0.125,
        metric_type=MetricType.GAUGE,
        labels={"model": "qwen3-vl", "input_size": "large"},
        description="Model inference latency in seconds"
    )

    collector.add_metric(
        name="model_throughput",
        value=8.0,
        metric_type=MetricType.GAUGE,
        labels={"model": "qwen3-vl", "unit": "requests_per_second"},
        description="Model throughput in requests per second"
    )

    # Start and end performance tracking
    start_performance_tracking("example_operation")
    time.sleep(0.1)  # Simulate some work
    perf_result = end_performance_tracking("example_operation")
    print(f"Performance tracking result: {perf_result}")

    # Test queue functionality
    collector.queue_metric(
        name="queued_metric_example",
        value=42,
        metric_type=MetricType.COUNTER,
        labels={"test": "queued"},
        description="Example of a queued metric"
    )

    # Export to different formats
    print("\nJSON Export:")
    json_data = collector.export_to_json()
    print(json_data[:500] + "..." if len(json_data) > 500 else json_data)

    print("\nPrometheus Export:")
    prometheus_data = collector.export_to_prometheus()
    print(prometheus_data[:500] + "..." if len(prometheus_data) > 500 else prometheus_data)

    # Export to CSV
    collector.export_to_csv("thread_safe_example_metrics.csv")
    print("\nMetrics exported to CSV: thread_safe_example_metrics.csv")

    # Show statistics
    stats = collector.get_statistics()
    print(f"\nCollector Statistics: {stats}")

    # Stop collection
    collector.stop_collection()

    print("\nMetrics collection stopped.")
    print(f"Total metrics collected: {len(collector.get_all_metrics_buffer())}")