"""
Dynamic Text Batching System for Unimodal Models

This module implements a sophisticated dynamic batching system specifically optimized for textual inputs
in unimodal models (GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b). The system adjusts batch sizes
based on input complexity, memory constraints, and performance metrics to optimize throughput and latency.
"""

import logging
import time
import threading
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import torch
import gc
from collections import deque
import re
import numpy as np

from .input_complexity_analyzer import InputComplexityAnalyzer, ComplexityMetrics
from .adaptive_batch_manager import AdaptiveBatchManager, BatchMetrics, BatchSizeAdjustmentReason


logger = logging.getLogger(__name__)


class TextBatchType(Enum):
    """Types of text batches based on content characteristics."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    TECHNICAL = "technical"
    CODE = "code"
    LONG_CONTEXT = "long_context"


@dataclass
class TextBatchInfo:
    """Information about a text batch for dynamic sizing."""
    batch_type: TextBatchType
    complexity_score: float
    avg_sequence_length: int
    vocabulary_richness: float
    semantic_density: float
    token_count: int
    estimated_memory_usage: float  # in GB
    recommended_batch_size: int


class DynamicTextBatchManager:
    """
    Advanced dynamic batching manager specifically optimized for textual inputs.
    
    This manager considers multiple factors specific to text processing:
    - Input complexity and type
    - Sequence length variations
    - Memory usage patterns for text models
    - Performance characteristics of different text types
    """

    def __init__(self,
                 initial_batch_size: int = 4,
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 memory_threshold_ratio: float = 0.80,
                 performance_window_size: int = 10,
                 adjustment_factor: float = 0.15,
                 cooldown_period: float = 3.0,
                 performance_target: float = 0.85,
                 complexity_weight: float = 0.4,
                 sequence_length_weight: float = 0.3,
                 memory_safety_margin: float = 0.15):
        """
        Initialize the dynamic text batch manager.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold_ratio: Memory usage ratio that triggers adjustments (0.0 to 1.0)
            performance_window_size: Number of recent samples to consider for performance evaluation
            adjustment_factor: Factor controlling how aggressively to adjust batch size
            cooldown_period: Time in seconds to wait between adjustments
            performance_target: Target performance score (0.0 to 1.0)
            complexity_weight: Weight of input complexity in batch size calculation (0.0 to 1.0)
            sequence_length_weight: Weight of sequence length in batch size calculation (0.0 to 1.0)
            memory_safety_margin: Safety margin for memory calculations (0.0 to 1.0)
        """
        self.current_batch_size = max(min_batch_size, min(initial_batch_size, max_batch_size))
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_ratio = memory_threshold_ratio
        self.performance_window_size = performance_window_size
        self.adjustment_factor = adjustment_factor
        self.cooldown_period = cooldown_period
        self.performance_target = performance_target
        self.complexity_weight = complexity_weight
        self.sequence_length_weight = sequence_length_weight
        self.memory_safety_margin = memory_safety_margin

        # Metrics tracking
        self.metrics_history = deque(maxlen=performance_window_size)
        self.text_batch_history = deque(maxlen=performance_window_size)
        self.last_adjustment_time = time.time()

        # Threading lock for thread safety
        self._lock = threading.Lock()

        # Performance tracking
        self._recent_processing_times = deque(maxlen=performance_window_size)
        self._recent_throughputs = deque(maxlen=performance_window_size)

        # Complexity tracking
        self._complexity_analyzer = InputComplexityAnalyzer()
        
        # Text-specific metrics
        self._text_type_distribution = {
            TextBatchType.SIMPLE: 0,
            TextBatchType.MODERATE: 0,
            TextBatchType.COMPLEX: 0,
            TextBatchType.TECHNICAL: 0,
            TextBatchType.CODE: 0,
            TextBatchType.LONG_CONTEXT: 0
        }

        logger.info(f"DynamicTextBatchManager initialized with batch_size range [{min_batch_size}, {max_batch_size}], "
                   f"initial batch_size: {initial_batch_size}, memory_threshold: {memory_threshold_ratio}, "
                   f"complexity_weight: {complexity_weight}, sequence_length_weight: {sequence_length_weight}")

    def analyze_text_batch_type(self, texts: Union[str, List[str]]) -> TextBatchType:
        """
        Analyze the type of text batch based on content characteristics.

        Args:
            texts: Input text or list of texts

        Returns:
            TextBatchType indicating the type of text content
        """
        if isinstance(texts, str):
            texts = [texts]

        # Join all texts for analysis
        full_text = " ".join(texts)

        # Analyze various characteristics
        avg_length = np.mean([len(t) for t in texts])
        word_count = len(full_text.split())

        # Check for technical terms
        technical_indicators = [
            r'\b(?:function|class|def|var|let|const|public|private|protected|static|void|int|float|double|char|bool|string)\b',
            r'\b(?:if|else|for|while|switch|case|break|continue|return|try|catch|finally|throw)\b',
            r'\b(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|CREATE|TABLE|DATABASE)\b',
            r'\b(?:import|from|as|with|async|await|yield|lambda|map|filter|reduce)\b',
            r'\b(?:function|method|class|object|instance|attribute|property|constructor)\b'
        ]

        technical_matches = sum(len(re.findall(pattern, full_text, re.IGNORECASE))
                               for pattern in technical_indicators)

        # Check for code-like patterns
        code_patterns = [
            r'[{}()\[\];,]',
            r'(?:def |class |function |var |let |const |public |private )',
            r'(?:import |from |as |with |async |await )',
            r'(?:SELECT |INSERT |UPDATE |DELETE |FROM |WHERE |JOIN )'
        ]

        code_matches = sum(len(re.findall(pattern, full_text)) for pattern in code_patterns)

        # Check for complex vocabulary
        words = re.findall(r'\b\w+\b', full_text.lower())
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0

        # Determine batch type based on characteristics
        if avg_length > 2000:  # Long context
            return TextBatchType.LONG_CONTEXT
        elif code_matches > len(texts) * 2:  # More than 2 code indicators per text
            return TextBatchType.CODE
        elif technical_matches > len(texts) * 1:  # More than 1 technical term per text
            return TextBatchType.TECHNICAL
        elif vocabulary_richness > 0.7:  # High vocabulary richness
            return TextBatchType.COMPLEX
        elif vocabulary_richness > 0.4:  # Moderate vocabulary richness
            return TextBatchType.MODERATE
        else:
            return TextBatchType.SIMPLE

    def estimate_memory_usage(self, texts: Union[str, List[str]], batch_size: int) -> float:
        """
        Estimate memory usage for processing a batch of texts.
        
        Args:
            texts: Input text or list of texts
            batch_size: Expected batch size
            
        Returns:
            Estimated memory usage in GB
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Calculate average sequence length
        avg_length = np.mean([len(t) for t in texts])
        
        # Estimate memory usage based on sequence length and batch size
        # This is a simplified model - in practice, this would be calibrated based on actual measurements
        memory_per_token_gb = 2e-9  # ~2 bytes per token (simplified estimate)
        max_seq_len = max(len(t) for t in texts) if texts else 1
        
        # Calculate estimated memory usage
        estimated_memory_gb = batch_size * max_seq_len * memory_per_token_gb * 1.5  # 1.5x safety factor
        
        return estimated_memory_gb

    def collect_text_batch_metrics(self,
                                  texts: Union[str, List[str]],
                                  processing_time_ms: float,
                                  tokens_processed: int,
                                  batch_size: int) -> TextBatchInfo:
        """
        Collect metrics specific to text batch processing.
        
        Args:
            texts: Input text or list of texts
            processing_time_ms: Processing time in milliseconds
            tokens_processed: Number of tokens processed
            batch_size: Current batch size
            
        Returns:
            TextBatchInfo with collected metrics
        """
        # Analyze text complexity
        complexity_metrics = self._complexity_analyzer.analyze_input_complexity(texts)
        
        # Determine text batch type
        batch_type = self.analyze_text_batch_type(texts)
        
        # Update text type distribution
        self._text_type_distribution[batch_type] += 1
        
        # Calculate average sequence length
        if isinstance(texts, str):
            avg_sequence_length = len(texts)
        else:
            avg_sequence_length = np.mean([len(t) for t in texts]) if texts else 0
        
        # Estimate memory usage
        estimated_memory_usage = self.estimate_memory_usage(texts, batch_size)
        
        # Calculate recommended batch size based on complexity
        complexity_based_size = self._complexity_analyzer.get_adaptive_batch_size(
            complexity_score=complexity_metrics.complexity_score,
            base_min_batch=self.min_batch_size,
            base_max_batch=self.max_batch_size
        )
        
        # Adjust based on sequence length
        length_factor = min(1.0, 512.0 / max(avg_sequence_length, 1))
        length_adjusted_size = int(complexity_based_size * length_factor)
        
        # Ensure within bounds
        recommended_batch_size = max(
            self.min_batch_size,
            min(length_adjusted_size, self.max_batch_size)
        )
        
        text_batch_info = TextBatchInfo(
            batch_type=batch_type,
            complexity_score=complexity_metrics.complexity_score,
            avg_sequence_length=avg_sequence_length,
            vocabulary_richness=complexity_metrics.vocabulary_richness,
            semantic_density=complexity_metrics.semantic_density,
            token_count=tokens_processed,
            estimated_memory_usage=estimated_memory_usage,
            recommended_batch_size=recommended_batch_size
        )
        
        # Store in history
        self.text_batch_history.append(text_batch_info)
        
        return text_batch_info

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
                       input_data: Optional[Union[str, List[str], torch.Tensor]] = None) -> BatchMetrics:
        """Collect metrics for the current batch, including text-specific metrics if applicable."""
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
        if input_data is not None:
            try:
                complexity_metrics = self._complexity_analyzer.analyze_input_complexity(input_data)
                complexity_score = complexity_metrics.complexity_score
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

    def calculate_new_batch_size(self, 
                                reason: BatchSizeAdjustmentReason, 
                                text_batch_info: Optional[TextBatchInfo] = None) -> int:
        """Calculate the new batch size based on the adjustment reason and text characteristics."""
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

        # Apply text-specific adjustments if available
        if text_batch_info is not None:
            # Adjust based on text type
            if text_batch_info.batch_type == TextBatchType.LONG_CONTEXT:
                # Long context requires more memory, reduce batch size
                new_size = max(self.min_batch_size, int(new_size * 0.7))
            elif text_batch_info.batch_type == TextBatchType.CODE:
                # Code typically requires more memory due to syntax complexity
                new_size = max(self.min_batch_size, int(new_size * 0.8))
            elif text_batch_info.batch_type == TextBatchType.TECHNICAL:
                # Technical text may require more memory
                new_size = max(self.min_batch_size, int(new_size * 0.85))
            elif text_batch_info.batch_type == TextBatchType.SIMPLE:
                # Simple text can handle larger batches
                new_size = min(self.max_batch_size, int(new_size * 1.2))

            # Adjust based on complexity score
            if text_batch_info.complexity_score > 0.7:
                # High complexity reduces batch size
                new_size = max(self.min_batch_size, int(new_size * (1.0 - text_batch_info.complexity_score * 0.3)))
            elif text_batch_info.complexity_score < 0.3:
                # Low complexity allows larger batches
                new_size = min(self.max_batch_size, int(new_size * (1.0 + (0.3 - text_batch_info.complexity_score) * 0.5)))

            # Adjust based on sequence length
            if text_batch_info.avg_sequence_length > 1024:
                # Long sequences reduce batch size
                length_factor = min(1.0, 1024.0 / text_batch_info.avg_sequence_length)
                new_size = max(self.min_batch_size, int(new_size * length_factor))

        # Ensure the new size is within bounds
        new_size = max(self.min_batch_size, min(new_size, self.max_batch_size))

        return new_size

    def adjust_batch_size(self, text_batch_info: Optional[TextBatchInfo] = None) -> Tuple[int, bool, Optional[BatchSizeAdjustmentReason]]:
        """
        Adjust the batch size based on current metrics and text characteristics.

        Returns:
            Tuple of (new_batch_size, was_adjusted, reason_for_adjustment)
        """
        with self._lock:
            should_adjust, reason = self.should_adjust_batch_size()

            if not should_adjust:
                return self.current_batch_size, False, None

            new_batch_size = self.calculate_new_batch_size(reason, text_batch_info)

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
                              input_data: Optional[Union[str, List[str], torch.Tensor]] = None) -> int:
        """
        Get the optimal batch size for the next batch based on performance metrics and text characteristics.

        Args:
            processing_time_ms: Processing time for the current batch in milliseconds
            tokens_processed: Number of tokens processed in the current batch
            input_data: Input data to analyze for complexity-based batch sizing

        Returns:
            Recommended batch size for the next batch
        """
        # Collect general metrics for the current batch
        self.collect_metrics(self.current_batch_size, processing_time_ms, tokens_processed, input_data)

        # Collect text-specific metrics if input is text
        text_batch_info = None
        if input_data is not None and isinstance(input_data, (str, list)) and all(isinstance(x, str) for x in ([input_data] if isinstance(input_data, str) else input_data)):
            text_batch_info = self.collect_text_batch_metrics(
                input_data, processing_time_ms, tokens_processed, self.current_batch_size
            )

        # Adjust batch size if needed
        new_size, was_adjusted, reason = self.adjust_batch_size(text_batch_info)

        # If input data is provided, apply additional text-specific adjustments
        if input_data is not None and text_batch_info is not None:
            # Apply final adjustments based on text characteristics
            final_size = text_batch_info.recommended_batch_size
            
            # Combine with performance-based size using weighted average
            combined_size = int(
                (1 - self.complexity_weight) * new_size +
                self.complexity_weight * final_size
            )
            
            # Ensure within bounds
            combined_size = max(self.min_batch_size, min(combined_size, self.max_batch_size))
            
            return combined_size
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
            self.text_batch_history.clear()
            self._recent_processing_times.clear()
            self._recent_throughputs.clear()
            logger.info("Performance tracking metrics reset")

    def get_status_report(self) -> Dict[str, Any]:
        """Get a status report of the current dynamic text batching state."""
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

            memory_info = self.get_system_memory_info()

            # Calculate text type distribution percentages
            total_batches = sum(self._text_type_distribution.values())
            text_type_percentages = {}
            if total_batches > 0:
                for text_type, count in self._text_type_distribution.items():
                    text_type_percentages[text_type.value] = count / total_batches

            return {
                'current_batch_size': self.current_batch_size,
                'min_batch_size': self.min_batch_size,
                'max_batch_size': self.max_batch_size,
                'memory_threshold_ratio': self.memory_threshold_ratio,
                'last_adjustment_time': self.last_adjustment_time,
                'cooldown_remaining': max(0, self.cooldown_period - (time.time() - self.last_adjustment_time)),
                'metrics_history_length': len(self.metrics_history),
                'text_batch_history_length': len(self.text_batch_history),
                'latest_metrics': latest_metrics.__dict__ if latest_metrics else {},
                'average_processing_time_ms': avg_processing_time,
                'average_throughput_tokens_per_sec': avg_throughput,
                'system_memory_info': memory_info,
                'text_type_distribution': text_type_percentages,
                'complexity_weight': self.complexity_weight,
                'sequence_length_weight': self.sequence_length_weight
            }

    def cleanup(self):
        """Clean up resources used by the batch manager."""
        with self._lock:
            self.metrics_history.clear()
            self.text_batch_history.clear()
            self._recent_processing_times.clear()
            self._recent_throughputs.clear()
            logger.info("DynamicTextBatchManager cleaned up")


def get_dynamic_text_batch_manager(
    initial_batch_size: int = 4,
    min_batch_size: int = 1,
    max_batch_size: int = 32,
    memory_threshold_ratio: float = 0.80,
    performance_window_size: int = 10,
    adjustment_factor: float = 0.15,
    cooldown_period: float = 3.0,
    performance_target: float = 0.85,
    complexity_weight: float = 0.4,
    sequence_length_weight: float = 0.3,
    memory_safety_margin: float = 0.15
) -> DynamicTextBatchManager:
    """
    Get a global instance of the dynamic text batch manager.

    Args:
        initial_batch_size: Starting batch size
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
        memory_threshold_ratio: Memory usage ratio that triggers adjustments (0.0 to 1.0)
        performance_window_size: Number of recent samples to consider for performance evaluation
        adjustment_factor: Factor controlling how aggressively to adjust batch size
        cooldown_period: Time in seconds to wait between adjustments
        performance_target: Target performance score (0.0 to 1.0)
        complexity_weight: Weight of input complexity in batch size calculation (0.0 to 1.0)
        sequence_length_weight: Weight of sequence length in batch size calculation (0.0 to 1.0)
        memory_safety_margin: Safety margin for memory calculations (0.0 to 1.0)

    Returns:
        DynamicTextBatchManager instance
    """
    if not hasattr(get_dynamic_text_batch_manager, '_instance'):
        get_dynamic_text_batch_manager._instance = DynamicTextBatchManager(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            memory_threshold_ratio=memory_threshold_ratio,
            performance_window_size=performance_window_size,
            adjustment_factor=adjustment_factor,
            cooldown_period=cooldown_period,
            performance_target=performance_target,
            complexity_weight=complexity_weight,
            sequence_length_weight=sequence_length_weight,
            memory_safety_margin=memory_safety_margin
        )
    return get_dynamic_text_batch_manager._instance


__all__ = [
    "DynamicTextBatchManager",
    "TextBatchType",
    "TextBatchInfo",
    "get_dynamic_text_batch_manager"
]