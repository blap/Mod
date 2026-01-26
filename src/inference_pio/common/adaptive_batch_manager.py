"""
Adaptive Batch Manager for Dynamic Batching System

This module implements an adaptive batch manager that monitors memory usage and
adjusts batch sizes based on available memory, performance metrics, and input complexity.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import psutil
import torch
import gc
from collections import deque

from .input_complexity_analyzer import InputComplexityAnalyzer, ComplexityMetrics


logger = logging.getLogger(__name__)


class BatchSizeAdjustmentReason(Enum):
    """Reasons for batch size adjustment."""
    MEMORY_PRESSURE = "memory_pressure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_AVAILABLE = "memory_available"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    STABILITY_ADJUSTMENT = "stability_adjustment"


@dataclass
class BatchMetrics:
    """Metrics collected for batch size decision making."""
    timestamp: float
    memory_usage_gb: float
    gpu_memory_usage_gb: float
    batch_size: int
    processing_time_ms: float
    throughput_tokens_per_sec: float
    latency_ms_per_token: float
    memory_pressure_ratio: float
    gpu_memory_pressure_ratio: float
    performance_score: float  # Higher is better


class AdaptiveBatchManager:
    """
    Adaptive batch manager that dynamically adjusts batch sizes based on memory usage,
    performance metrics, and input complexity.
    """

    def __init__(self,
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
                 complexity_high_threshold: float = 0.7):
        """
        Initialize the adaptive batch manager.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold_ratio: Memory usage ratio that triggers adjustments (0.0 to 1.0)
            performance_window_size: Number of recent samples to consider for performance evaluation
            adjustment_factor: Factor controlling how aggressively to adjust batch size
            cooldown_period: Time in seconds to wait between adjustments
            performance_target: Target performance score (0.0 to 1.0)
            use_input_complexity: Whether to consider input complexity when adjusting batch size
            complexity_weight: Weight of input complexity in batch size calculation (0.0 to 1.0)
            complexity_low_threshold: Below this complexity, use max batch size
            complexity_high_threshold: Above this complexity, use min batch size
        """
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

        # Metrics tracking
        self.metrics_history = deque(maxlen=performance_window_size)
        self.last_adjustment_time = time.time()

        # Threading lock for thread safety
        self._lock = threading.Lock()

        # Performance tracking
        self._recent_processing_times = deque(maxlen=performance_window_size)
        self._recent_throughputs = deque(maxlen=performance_window_size)

        # Complexity tracking
        self._complexity_analyzer = InputComplexityAnalyzer()
        self._recent_complexities = deque(maxlen=performance_window_size)

        logger.info(f"AdaptiveBatchManager initialized with batch_size range [{min_batch_size}, {max_batch_size}], "
                   f"initial batch_size: {initial_batch_size}, memory_threshold: {memory_threshold_ratio}, "
                   f"use_input_complexity: {use_input_complexity}, complexity_weight: {complexity_weight}")

    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information."""
        memory = psutil.virtual_memory()
        gpu_memory_info = {}
        
        if torch.cuda.is_available():
            try:
                gpu_memory_info = {
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
                    'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                    'gpu_memory_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                }
            except:
                # Handle cases where GPU utilization cannot be retrieved
                gpu_memory_info = {
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
                    'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                    'gpu_memory_utilization': 0
                }
        else:
            gpu_memory_info = {}
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024 ** 3),
            'memory_total_gb': memory.total / (1024 ** 3),
            'memory_used_gb': memory.used / (1024 ** 3),
            'memory_pressure_ratio': memory.used / memory.total,
            **gpu_memory_info
        }

    def collect_metrics(self,
                       batch_size: int,
                       processing_time_ms: float,
                       tokens_processed: int,
                       input_data: Optional[Union[str, list, torch.Tensor]] = None) -> BatchMetrics:
        """Collect metrics for the current batch, including input complexity if provided."""
        memory_info = self.get_system_memory_info()

        # Calculate throughput and latency
        if processing_time_ms > 0:
            throughput = (tokens_processed / processing_time_ms) * 1000  # tokens per second
            latency = processing_time_ms / max(tokens_processed, 1)  # ms per token
        else:
            throughput = 0.0
            latency = 0.0

        # Calculate input complexity if input data is provided
        complexity_score = 0.0
        if input_data is not None and self.use_input_complexity:
            try:
                complexity_metrics = self._complexity_analyzer.analyze_input_complexity(input_data)
                complexity_score = complexity_metrics.complexity_score
                self._recent_complexities.append(complexity_score)
            except Exception as e:
                logger.warning(f"Error calculating input complexity: {e}")
                complexity_score = 0.0

        # Calculate performance score (higher is better)
        # Normalize values to 0-1 range and combine them
        normalized_throughput = min(throughput / 1000.0, 1.0)  # Assume 1000 tokens/sec is excellent
        normalized_latency = max(0.0, 1.0 - (latency / 100.0))  # Assume 100ms/token is poor

        # Memory pressure (lower is better for performance)
        memory_pressure = memory_info.get('memory_pressure_ratio', 0.0)
        gpu_memory_pressure = memory_info.get('gpu_memory_allocated_gb', 0.0) / memory_info.get('gpu_memory_total_gb', 1.0) \
                              if 'gpu_memory_total_gb' in memory_info else 0.0

        # Performance score: higher is better
        performance_score = (
            0.4 * normalized_throughput +
            0.3 * normalized_latency +
            0.15 * (1.0 - min(memory_pressure / self.memory_threshold_ratio, 1.0)) +
            0.15 * (1.0 - min(gpu_memory_pressure / self.memory_threshold_ratio, 1.0))
        )

        metrics = BatchMetrics(
            timestamp=time.time(),
            memory_usage_gb=memory_info['memory_used_gb'],
            gpu_memory_usage_gb=memory_info.get('gpu_memory_allocated_gb', 0.0),
            batch_size=batch_size,
            processing_time_ms=processing_time_ms,
            throughput_tokens_per_sec=throughput,
            latency_ms_per_token=latency,
            memory_pressure_ratio=memory_pressure,
            gpu_memory_pressure_ratio=gpu_memory_pressure,
            performance_score=performance_score
        )

        # Store metrics for history
        self.metrics_history.append(metrics)
        self._recent_processing_times.append(processing_time_ms)
        self._recent_throughputs.append(throughput)

        return metrics

    def should_adjust_batch_size(self) -> Tuple[bool, Optional[BatchSizeAdjustmentReason]]:
        """Determine if batch size should be adjusted and why."""
        if len(self.metrics_history) < 2:
            return False, None
        
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.cooldown_period:
            return False, None
        
        latest_metrics = self.metrics_history[-1]
        previous_metrics = self.metrics_history[-2]
        
        # Check for memory pressure
        if latest_metrics.memory_pressure_ratio > self.memory_threshold_ratio or \
           latest_metrics.gpu_memory_pressure_ratio > self.memory_threshold_ratio:
            return True, BatchSizeAdjustmentReason.MEMORY_PRESSURE
        
        # Check for performance degradation
        if latest_metrics.performance_score < self.performance_target * 0.8 and \
           latest_metrics.performance_score < previous_metrics.performance_score * 0.95:
            return True, BatchSizeAdjustmentReason.PERFORMANCE_DEGRADATION
        
        # Check if memory is available for larger batches
        if latest_metrics.memory_pressure_ratio < self.memory_threshold_ratio * 0.7 and \
           latest_metrics.batch_size < self.max_batch_size and \
           latest_metrics.performance_score > self.performance_target:
            return True, BatchSizeAdjustmentReason.MEMORY_AVAILABLE
        
        # Check for performance improvement opportunity
        if latest_metrics.performance_score > self.performance_target and \
           latest_metrics.batch_size < self.max_batch_size and \
           latest_metrics.performance_score > previous_metrics.performance_score * 1.05:
            return True, BatchSizeAdjustmentReason.PERFORMANCE_IMPROVEMENT
        
        return False, BatchSizeAdjustmentReason.STABILITY_ADJUSTMENT

    def calculate_new_batch_size(self, reason: BatchSizeAdjustmentReason) -> int:
        """Calculate the new batch size based on the adjustment reason."""
        current_batch_size = self.current_batch_size
        
        if reason == BatchSizeAdjustmentReason.MEMORY_PRESSURE:
            # Reduce batch size significantly when under memory pressure
            new_size = max(
                self.min_batch_size,
                int(current_batch_size * (1.0 - self.adjustment_factor * 2))
            )
        elif reason == BatchSizeAdjustmentReason.PERFORMANCE_DEGRADATION:
            # Reduce batch size moderately when performance degrades
            new_size = max(
                self.min_batch_size,
                int(current_batch_size * (1.0 - self.adjustment_factor))
            )
        elif reason == BatchSizeAdjustmentReason.MEMORY_AVAILABLE:
            # Increase batch size when memory is available
            new_size = min(
                self.max_batch_size,
                int(current_batch_size * (1.0 + self.adjustment_factor * 0.5))
            )
        elif reason == BatchSizeAdjustmentReason.PERFORMANCE_IMPROVEMENT:
            # Gradually increase batch size when performance improves
            new_size = min(
                self.max_batch_size,
                int(current_batch_size * (1.0 + self.adjustment_factor * 0.3))
            )
        else:  # STABILITY_ADJUSTMENT
            # Make minor adjustments for stability
            if len(self._recent_throughputs) >= 2:
                recent_avg = sum(list(self._recent_throughputs)[-3:]) / min(3, len(self._recent_throughputs))
                prev_avg = sum(list(self._recent_throughputs)[-6:-3]) / min(3, max(1, len(self._recent_throughputs)-3))
                
                if recent_avg > prev_avg * 1.05:  # Performance improved
                    new_size = min(self.max_batch_size, current_batch_size + 1)
                elif recent_avg < prev_avg * 0.95:  # Performance degraded
                    new_size = max(self.min_batch_size, current_batch_size - 1)
                else:
                    new_size = current_batch_size
            else:
                new_size = current_batch_size
        
        # Ensure the new size is within bounds
        new_size = max(self.min_batch_size, min(new_size, self.max_batch_size))
        
        return new_size

    def adjust_batch_size(self) -> Tuple[int, bool, Optional[BatchSizeAdjustmentReason]]:
        """
        Adjust the batch size based on current metrics and return the new size.
        
        Returns:
            Tuple of (new_batch_size, was_adjusted, reason_for_adjustment)
        """
        with self._lock:
            should_adjust, reason = self.should_adjust_batch_size()
            
            if not should_adjust:
                return self.current_batch_size, False, None
            
            new_batch_size = self.calculate_new_batch_size(reason)
            
            if new_batch_size != self.current_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = new_batch_size
                self.last_adjustment_time = time.time()
                
                logger.info(f"Batch size adjusted from {old_size} to {new_batch_size} "
                           f"due to {reason.value}. Memory pressure: "
                           f"sys={self.metrics_history[-1].memory_pressure_ratio:.2f}, "
                           f"gpu={self.metrics_history[-1].gpu_memory_pressure_ratio:.2f}")
                
                return new_batch_size, True, reason
            else:
                return self.current_batch_size, False, reason

    def get_optimal_batch_size(self,
                               processing_time_ms: float,
                               tokens_processed: int,
                               input_data: Optional[Union[str, list, torch.Tensor]] = None) -> int:
        """
        Get the optimal batch size for the next batch based on performance metrics and input complexity.

        Args:
            processing_time_ms: Processing time for the current batch in milliseconds
            tokens_processed: Number of tokens processed in the current batch
            input_data: Input data to analyze for complexity-based batch sizing

        Returns:
            Recommended batch size for the next batch
        """
        # Collect metrics for the current batch
        self.collect_metrics(self.current_batch_size, processing_time_ms, tokens_processed, input_data)

        # Adjust batch size if needed
        new_size, was_adjusted, reason = self.adjust_batch_size()

        # If using input complexity, adjust the batch size based on complexity
        if input_data is not None and self.use_input_complexity:
            try:
                complexity_metrics = self._complexity_analyzer.analyze_input_complexity(input_data)
                complexity_based_size = self._complexity_analyzer.get_adaptive_batch_size(
                    complexity_score=complexity_metrics.complexity_score,
                    base_min_batch=self.min_batch_size,
                    base_max_batch=self.max_batch_size,
                    complexity_threshold_low=self.complexity_low_threshold,
                    complexity_threshold_high=self.complexity_high_threshold
                )

                # Combine memory/performance-based size with complexity-based size
                # Use weighted average based on complexity weight
                combined_size = int(
                    (1 - self.complexity_weight) * new_size +
                    self.complexity_weight * complexity_based_size
                )

                # Ensure the result is within bounds
                combined_size = max(self.min_batch_size, min(combined_size, self.max_batch_size))

                # Log the complexity-based adjustment
                logger.debug(f"Complexity-based batch size adjustment: "
                           f"memory/performance={new_size}, complexity-based={complexity_based_size}, "
                           f"final={combined_size}, complexity_score={complexity_metrics.complexity_score}")

                return combined_size
            except Exception as e:
                logger.warning(f"Error applying complexity-based batch sizing: {e}")
                # Fall back to memory/performance-based sizing
                return new_size
        else:
            return new_size

    def force_batch_size(self, batch_size: int) -> bool:
        """
        Force a specific batch size (useful for manual overrides or testing).
        
        Args:
            batch_size: The batch size to force
            
        Returns:
            True if the batch size was set successfully, False otherwise
        """
        with self._lock:
            if self.min_batch_size <= batch_size <= self.max_batch_size:
                if batch_size != self.current_batch_size:
                    old_size = self.current_batch_size
                    self.current_batch_size = batch_size
                    logger.info(f"Batch size forced from {old_size} to {batch_size}")
                return True
            else:
                logger.warning(f"Attempted to set batch size to {batch_size}, "
                              f"but it's outside the allowed range [{self.min_batch_size}, {self.max_batch_size}]")
                return False

    def reset_performance_tracking(self):
        """Reset performance tracking metrics."""
        with self._lock:
            self.metrics_history.clear()
            self._recent_processing_times.clear()
            self._recent_throughputs.clear()
            logger.info("Performance tracking metrics reset")

    def get_status_report(self) -> Dict[str, Any]:
        """Get a status report of the current adaptive batching state."""
        with self._lock:
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                avg_processing_time = sum(self._recent_processing_times) / len(self._recent_processing_times) \
                                      if self._recent_processing_times else 0
                avg_throughput = sum(self._recent_throughputs) / len(self._recent_throughputs) \
                                 if self._recent_throughputs else 0
            else:
                latest_metrics = None
                avg_processing_time = 0
                avg_throughput = 0

            if self._recent_complexities:
                avg_complexity = sum(self._recent_complexities) / len(self._recent_complexities)
            else:
                avg_complexity = 0

            memory_info = self.get_system_memory_info()

            return {
                'current_batch_size': self.current_batch_size,
                'min_batch_size': self.min_batch_size,
                'max_batch_size': self.max_batch_size,
                'memory_threshold_ratio': self.memory_threshold_ratio,
                'last_adjustment_time': self.last_adjustment_time,
                'cooldown_remaining': max(0, self.cooldown_period - (time.time() - self.last_adjustment_time)),
                'metrics_history_length': len(self.metrics_history),
                'latest_metrics': latest_metrics.__dict__ if latest_metrics else {},
                'average_processing_time_ms': avg_processing_time,
                'average_throughput_tokens_per_sec': avg_throughput,
                'average_input_complexity': avg_complexity,
                'recent_complexities_count': len(self._recent_complexities),
                'use_input_complexity': self.use_input_complexity,
                'complexity_weight': self.complexity_weight,
                'system_memory_info': memory_info
            }

    def cleanup(self):
        """Clean up resources used by the batch manager."""
        with self._lock:
            self.metrics_history.clear()
            self._recent_processing_times.clear()
            self._recent_throughputs.clear()
            logger.info("AdaptiveBatchManager cleaned up")


def get_adaptive_batch_manager(
    initial_batch_size: int = 1,
    min_batch_size: int = 1,
    max_batch_size: int = 16,
    memory_threshold_ratio: float = 0.85,
    performance_window_size: int = 10,
    adjustment_factor: float = 0.1,
    cooldown_period: float = 5.0,
    performance_target: float = 0.8,
    use_input_complexity: bool = True,
    complexity_weight: float = 0.3,
    complexity_low_threshold: float = 0.3,
    complexity_high_threshold: float = 0.7
) -> AdaptiveBatchManager:
    """
    Get a global instance of the adaptive batch manager.

    Args:
        initial_batch_size: Starting batch size
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
        memory_threshold_ratio: Memory usage ratio that triggers adjustments (0.0 to 1.0)
        performance_window_size: Number of recent samples to consider for performance evaluation
        adjustment_factor: Factor controlling how aggressively to adjust batch size
        cooldown_period: Time in seconds to wait between adjustments
        performance_target: Target performance score (0.0 to 1.0)
        use_input_complexity: Whether to consider input complexity when adjusting batch size
        complexity_weight: Weight of input complexity in batch size calculation (0.0 to 1.0)
        complexity_low_threshold: Below this complexity, use max batch size
        complexity_high_threshold: Above this complexity, use min batch size

    Returns:
        AdaptiveBatchManager instance
    """
    if not hasattr(get_adaptive_batch_manager, '_instance'):
        get_adaptive_batch_manager._instance = AdaptiveBatchManager(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            memory_threshold_ratio=memory_threshold_ratio,
            performance_window_size=performance_window_size,
            adjustment_factor=adjustment_factor,
            cooldown_period=cooldown_period,
            performance_target=performance_target,
            use_input_complexity=use_input_complexity,
            complexity_weight=complexity_weight,
            complexity_low_threshold=complexity_low_threshold,
            complexity_high_threshold=complexity_high_threshold
        )
    return get_adaptive_batch_manager._instance


__all__ = [
    "AdaptiveBatchManager",
    "BatchMetrics",
    "BatchSizeAdjustmentReason",
    "get_adaptive_batch_manager"
]