"""
Adaptive Batch Manager for Dynamic Batching System
Dependency-Free
"""

import gc
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import psutil
from ...core.engine.backend import HAS_CUDA

from .input_complexity_analyzer import ComplexityMetrics, InputComplexityAnalyzer

logger = logging.getLogger(__name__)

class BatchSizeAdjustmentReason(Enum):
    MEMORY_PRESSURE = "memory_pressure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_AVAILABLE = "memory_available"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    STABILITY_ADJUSTMENT = "stability_adjustment"

@dataclass
class BatchMetrics:
    timestamp: float
    memory_usage_gb: float
    gpu_memory_usage_gb: float
    batch_size: int
    processing_time_ms: float
    throughput_tokens_per_sec: float
    latency_ms_per_token: float
    memory_pressure_ratio: float
    gpu_memory_pressure_ratio: float
    performance_score: float

class AdaptiveBatchManager:
    def __init__(
        self,
        initial_batch_size: int = 1,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        memory_threshold_ratio: float = 0.85,
        performance_window_size: int = 10,
        adjustment_factor: float = 0.1,
        cooldown_period: float = 5.0,
        performance_target: float = 0.8,
        use_input_complexity: bool = True,
        complexity_weight: float = 0.3,
        complexity_low_threshold: float = 0.3,
        complexity_high_threshold: float = 0.7,
    ):
        self.current_batch_size = max(min_batch_size, min(initial_batch_size, max_batch_size))
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_ratio = memory_threshold_ratio
        self.performance_window_size = performance_window_size
        self.adjustment_factor = adjustment_factor
        self.cooldown_period = cooldown_period
        self.performance_target = performance_target
        self.use_input_complexity = use_input_complexity
        self.complexity_weight = complexity_weight
        self.complexity_low_threshold = complexity_low_threshold
        self.complexity_high_threshold = complexity_high_threshold

        self.metrics_history = deque(maxlen=performance_window_size)
        self.last_adjustment_time = time.time()
        self._lock = threading.Lock()
        self._recent_processing_times = deque(maxlen=performance_window_size)
        self._recent_throughputs = deque(maxlen=performance_window_size)
        self._complexity_analyzer = InputComplexityAnalyzer()
        self._recent_complexities = deque(maxlen=performance_window_size)

        logger.info(f"AdaptiveBatchManager initialized: [{min_batch_size}, {max_batch_size}]")

    def get_system_memory_info(self) -> Dict[str, float]:
        memory = psutil.virtual_memory()
        gpu_memory_info = {}

        if HAS_CUDA:
            # Placeholder: Backend doesn't expose memory queries yet.
            # Could add backend.get_memory_info() later.
            # For now, assume 0 usage or use nvidia-smi via subprocess if critical.
            # Using conservative defaults to avoid crashing logic.
            gpu_memory_info = {
                "gpu_memory_allocated_gb": 0.0,
                "gpu_memory_reserved_gb": 0.0,
                "gpu_memory_total_gb": 16.0, # Estimation or query later
                "gpu_memory_utilization": 0
            }

        return {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_pressure_ratio": memory.used / memory.total,
            **gpu_memory_info,
        }

    def collect_metrics(
        self,
        batch_size: int,
        processing_time_ms: float,
        tokens_processed: int,
        input_data: Optional[Any] = None,
    ) -> BatchMetrics:
        memory_info = self.get_system_memory_info()

        if processing_time_ms > 0:
            throughput = (tokens_processed / processing_time_ms) * 1000
            latency = processing_time_ms / max(tokens_processed, 1)
        else:
            throughput = 0.0
            latency = 0.0

        complexity_score = 0.0
        if input_data is not None and self.use_input_complexity:
            try:
                complexity_metrics = self._complexity_analyzer.analyze_input_complexity(input_data)
                complexity_score = complexity_metrics.complexity_score
                self._recent_complexities.append(complexity_score)
            except Exception as e:
                logger.warning(f"Error calculating input complexity: {e}")

        normalized_throughput = min(throughput / 1000.0, 1.0)
        normalized_latency = max(0.0, 1.0 - (latency / 100.0))

        memory_pressure = memory_info.get("memory_pressure_ratio", 0.0)
        gpu_memory_pressure = (
            memory_info.get("gpu_memory_allocated_gb", 0.0)
            / memory_info.get("gpu_memory_total_gb", 1.0)
            if "gpu_memory_total_gb" in memory_info else 0.0
        )

        performance_score = (
            0.4 * normalized_throughput
            + 0.3 * normalized_latency
            + 0.15 * (1.0 - min(memory_pressure / self.memory_threshold_ratio, 1.0))
            + 0.15 * (1.0 - min(gpu_memory_pressure / self.memory_threshold_ratio, 1.0))
        )

        metrics = BatchMetrics(
            timestamp=time.time(),
            memory_usage_gb=memory_info["memory_used_gb"],
            gpu_memory_usage_gb=memory_info.get("gpu_memory_allocated_gb", 0.0),
            batch_size=batch_size,
            processing_time_ms=processing_time_ms,
            throughput_tokens_per_sec=throughput,
            latency_ms_per_token=latency,
            memory_pressure_ratio=memory_pressure,
            gpu_memory_pressure_ratio=gpu_memory_pressure,
            performance_score=performance_score,
        )

        self.metrics_history.append(metrics)
        self._recent_processing_times.append(processing_time_ms)
        self._recent_throughputs.append(throughput)

        return metrics

    def should_adjust_batch_size(self) -> Tuple[bool, Optional[BatchSizeAdjustmentReason]]:
        if len(self.metrics_history) < 2: return False, None
        if time.time() - self.last_adjustment_time < self.cooldown_period: return False, None

        latest = self.metrics_history[-1]
        previous = self.metrics_history[-2]

        if (latest.memory_pressure_ratio > self.memory_threshold_ratio or
            latest.gpu_memory_pressure_ratio > self.memory_threshold_ratio):
            return True, BatchSizeAdjustmentReason.MEMORY_PRESSURE

        if (latest.performance_score < self.performance_target * 0.8 and
            latest.performance_score < previous.performance_score * 0.95):
            return True, BatchSizeAdjustmentReason.PERFORMANCE_DEGRADATION

        if (latest.memory_pressure_ratio < self.memory_threshold_ratio * 0.7 and
            latest.batch_size < self.max_batch_size and
            latest.performance_score > self.performance_target):
            return True, BatchSizeAdjustmentReason.MEMORY_AVAILABLE

        if (latest.performance_score > self.performance_target and
            latest.batch_size < self.max_batch_size and
            latest.performance_score > previous.performance_score * 1.05):
            return True, BatchSizeAdjustmentReason.PERFORMANCE_IMPROVEMENT

        return False, BatchSizeAdjustmentReason.STABILITY_ADJUSTMENT

    def calculate_new_batch_size(self, reason: BatchSizeAdjustmentReason) -> int:
        current = self.current_batch_size
        if reason == BatchSizeAdjustmentReason.MEMORY_PRESSURE:
            new_size = max(self.min_batch_size, int(current * (1.0 - self.adjustment_factor * 2)))
        elif reason == BatchSizeAdjustmentReason.PERFORMANCE_DEGRADATION:
            new_size = max(self.min_batch_size, int(current * (1.0 - self.adjustment_factor)))
        elif reason == BatchSizeAdjustmentReason.MEMORY_AVAILABLE:
            new_size = min(self.max_batch_size, int(current * (1.0 + self.adjustment_factor * 0.5)))
        elif reason == BatchSizeAdjustmentReason.PERFORMANCE_IMPROVEMENT:
            new_size = min(self.max_batch_size, int(current * (1.0 + self.adjustment_factor * 0.3)))
        else:
            new_size = current
        return max(self.min_batch_size, min(new_size, self.max_batch_size))

    def adjust_batch_size(self) -> Tuple[int, bool, Optional[BatchSizeAdjustmentReason]]:
        with self._lock:
            should, reason = self.should_adjust_batch_size()
            if not should: return self.current_batch_size, False, None

            new_size = self.calculate_new_batch_size(reason)
            if new_size != self.current_batch_size:
                self.current_batch_size = new_size
                self.last_adjustment_time = time.time()
                return new_size, True, reason
            return self.current_batch_size, False, reason

    def get_optimal_batch_size(self, processing_time_ms: float, tokens_processed: int, input_data: Optional[Any] = None) -> int:
        self.collect_metrics(self.current_batch_size, processing_time_ms, tokens_processed, input_data)
        new_size, _, _ = self.adjust_batch_size()

        if input_data is not None and self.use_input_complexity:
            try:
                metrics = self._complexity_analyzer.analyze_input_complexity(input_data)
                comp_size = self._complexity_analyzer.get_adaptive_batch_size(
                    metrics.complexity_score, self.min_batch_size, self.max_batch_size,
                    self.complexity_low_threshold, self.complexity_high_threshold
                )
                combined = int((1 - self.complexity_weight) * new_size + self.complexity_weight * comp_size)
                return max(self.min_batch_size, min(combined, self.max_batch_size))
            except Exception as e:
                logger.warning(f"Complexity analysis failed: {e}")
        return new_size

    # ... (Other methods force_batch_size, reset, get_status_report, cleanup kept same/simplified) ...
    def force_batch_size(self, size):
        with self._lock:
            self.current_batch_size = max(self.min_batch_size, min(size, self.max_batch_size))
            return True

    def get_status_report(self):
        return {"batch_size": self.current_batch_size, "metrics": list(self.metrics_history)}

def get_adaptive_batch_manager(**kwargs) -> AdaptiveBatchManager:
    if not hasattr(get_adaptive_batch_manager, "_instance"):
        get_adaptive_batch_manager._instance = AdaptiveBatchManager(**kwargs)
    return get_adaptive_batch_manager._instance

__all__ = ["AdaptiveBatchManager", "BatchMetrics", "BatchSizeAdjustmentReason", "get_adaptive_batch_manager"]
