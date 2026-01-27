"""
Dynamic Multimodal Batching System for Vision-Language Models

This module implements a dynamic batching system specifically optimized for multimodal inputs
(text and image combinations) in vision-language models like Qwen3-VL-2b.
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from PIL import Image
from collections import defaultdict, deque

from ...common.adaptive_batch_manager import AdaptiveBatchManager, BatchMetrics
from ...common.input_complexity_analyzer import InputComplexityAnalyzer, ComplexityMetrics


logger = logging.getLogger(__name__)


class MultimodalBatchType(Enum):
    """Types of multimodal batches."""
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    TEXT_IMAGE = "text_image"
    MIXED = "mixed"


@dataclass
class MultimodalBatchInfo:
    """Information about a multimodal batch."""
    batch_type: MultimodalBatchType
    text_complexity: float
    image_complexity: float
    combined_complexity: float
    batch_size: int
    processing_time_ms: float
    memory_usage_gb: float
    throughput_tokens_per_sec: float


class DynamicMultimodalBatchManager:
    """
    Dynamic batch manager specifically for multimodal inputs (text and images).
    Adjusts batch sizes based on the complexity of both text and image inputs.
    """

    def __init__(self,
                 initial_batch_size: int = 1,
                 min_batch_size: int = 1,
                 max_batch_size: int = 8,
                 memory_threshold_ratio: float = 0.85,
                 performance_window_size: int = 10,
                 adjustment_factor: float = 0.1,
                 cooldown_period: float = 5.0,
                 performance_target: float = 0.8,
                 text_weight: float = 0.4,
                 image_weight: float = 0.6,
                 complexity_threshold_low: float = 0.3,
                 complexity_threshold_high: float = 0.7):
        """
        Initialize the dynamic multimodal batch manager.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold_ratio: Memory usage ratio that triggers adjustments (0.0 to 1.0)
            performance_window_size: Number of recent samples to consider for performance evaluation
            adjustment_factor: Factor controlling how aggressively to adjust batch size
            cooldown_period: Time in seconds to wait between adjustments
            performance_target: Target performance score (0.0 to 1.0)
            text_weight: Weight of text complexity in combined complexity calculation
            image_weight: Weight of image complexity in combined complexity calculation
            complexity_threshold_low: Below this complexity, use max batch size
            complexity_threshold_high: Above this complexity, use min batch size
        """
        self.current_batch_size = max(min_batch_size, min(initial_batch_size, max_batch_size))
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_ratio = memory_threshold_ratio
        self.performance_window_size = performance_window_size
        self.adjustment_factor = adjustment_factor
        self.cooldown_period = cooldown_period
        self.performance_target = performance_target
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.complexity_threshold_low = complexity_threshold_low
        self.complexity_threshold_high = complexity_threshold_high

        # Metrics tracking
        self.metrics_history = deque(maxlen=performance_window_size)
        
        # Complexity analyzers
        self.text_complexity_analyzer = InputComplexityAnalyzer()
        self.image_complexity_analyzer = ImageComplexityAnalyzer()
        
        # Recent complexities tracking
        self._recent_text_complexities = deque(maxlen=performance_window_size)
        self._recent_image_complexities = deque(maxlen=performance_window_size)
        self._recent_combined_complexities = deque(maxlen=performance_window_size)

        logger.info(f"DynamicMultimodalBatchManager initialized with batch_size range [{min_batch_size}, {max_batch_size}], "
                   f"initial batch_size: {initial_batch_size}, memory_threshold: {memory_threshold_ratio}, "
                   f"text_weight: {text_weight}, image_weight: {image_weight}")

    def analyze_multimodal_complexity(self, inputs: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """
        Analyze the complexity of multimodal inputs (text and images).

        Args:
            inputs: List of input dictionaries containing 'text' and/or 'image' keys

        Returns:
            Tuple of (average_text_complexity, average_image_complexity, combined_complexity)
        """
        text_complexities = []
        image_complexities = []

        for input_item in inputs:
            # Analyze text complexity if present
            if 'text' in input_item and input_item['text']:
                text_complexity = self.text_complexity_analyzer.analyze_input_complexity(input_item['text']).complexity_score
                text_complexities.append(text_complexity)
            
            # Analyze image complexity if present
            if 'image' in input_item and input_item['image']:
                image_complexity = self.image_complexity_analyzer.analyze_image_complexity(input_item['image'])
                image_complexities.append(image_complexity)

        # Calculate averages
        avg_text_complexity = np.mean(text_complexities) if text_complexities else 0.0
        avg_image_complexity = np.mean(image_complexities) if image_complexities else 0.0
        
        # Calculate combined complexity using weighted average
        combined_complexity = (
            self.text_weight * avg_text_complexity + 
            self.image_weight * avg_image_complexity
        )

        return avg_text_complexity, avg_image_complexity, combined_complexity

    def determine_batch_type(self, inputs: List[Dict[str, Any]]) -> MultimodalBatchType:
        """
        Determine the type of multimodal batch based on input composition.

        Args:
            inputs: List of input dictionaries

        Returns:
            MultimodalBatchType indicating the batch type
        """
        has_text = any('text' in inp and inp['text'] for inp in inputs)
        has_image = any('image' in inp and inp['image'] for inp in inputs)

        if has_text and has_image:
            return MultimodalBatchType.TEXT_IMAGE
        elif has_text and not has_image:
            return MultimodalBatchType.TEXT_ONLY
        elif not has_text and has_image:
            return MultimodalBatchType.IMAGE_ONLY
        else:
            return MultimodalBatchType.MIXED  # This shouldn't happen if inputs are valid

    def collect_multimodal_metrics(self,
                                  batch_size: int,
                                  processing_time_ms: float,
                                  tokens_processed: int,
                                  inputs: List[Dict[str, Any]]) -> MultimodalBatchInfo:
        """
        Collect metrics for the current multimodal batch.

        Args:
            batch_size: Size of the current batch
            processing_time_ms: Processing time in milliseconds
            tokens_processed: Number of tokens processed
            inputs: List of input dictionaries

        Returns:
            MultimodalBatchInfo with collected metrics
        """
        # Analyze multimodal complexity
        avg_text_complexity, avg_image_complexity, combined_complexity = self.analyze_multimodal_complexity(inputs)
        
        # Determine batch type
        batch_type = self.determine_batch_type(inputs)

        # Calculate throughput and latency
        if processing_time_ms > 0:
            throughput = (tokens_processed / processing_time_ms) * 1000  # tokens per second
            latency = processing_time_ms / max(tokens_processed, 1)  # ms per token
        else:
            throughput = 0.0
            latency = 0.0

        # Get memory usage
        memory_info = self._get_system_memory_info()
        memory_usage_gb = memory_info['memory_used_gb']

        # Calculate performance score
        normalized_throughput = min(throughput / 1000.0, 1.0)  # Assume 1000 tokens/sec is excellent
        normalized_latency = max(0.0, 1.0 - (latency / 100.0))  # Assume 100ms/token is poor
        memory_pressure = memory_info.get('memory_pressure_ratio', 0.0)

        performance_score = (
            0.4 * normalized_throughput +
            0.3 * normalized_latency +
            0.3 * (1.0 - min(memory_pressure / self.memory_threshold_ratio, 1.0))
        )

        # Create metrics entry
        metrics_entry = BatchMetrics(
            timestamp=time.time(),
            memory_usage_gb=memory_usage_gb,
            gpu_memory_usage_gb=memory_info.get('gpu_memory_allocated_gb', 0.0),
            batch_size=batch_size,
            processing_time_ms=processing_time_ms,
            throughput_tokens_per_sec=throughput,
            latency_ms_per_token=latency,
            memory_pressure_ratio=memory_pressure,
            gpu_memory_pressure_ratio=memory_info.get('gpu_memory_allocated_gb', 0.0) / memory_info.get('gpu_memory_total_gb', 1.0),
            performance_score=performance_score
        )

        # Store metrics
        self.metrics_history.append(metrics_entry)
        self._recent_text_complexities.append(avg_text_complexity)
        self._recent_image_complexities.append(avg_image_complexity)
        self._recent_combined_complexities.append(combined_complexity)

        return MultimodalBatchInfo(
            batch_type=batch_type,
            text_complexity=avg_text_complexity,
            image_complexity=avg_image_complexity,
            combined_complexity=combined_complexity,
            batch_size=batch_size,
            processing_time_ms=processing_time_ms,
            memory_usage_gb=memory_usage_gb,
            throughput_tokens_per_sec=throughput
        )

    def get_optimal_multimodal_batch_size(self,
                                        processing_time_ms: float,
                                        tokens_processed: int,
                                        inputs: List[Dict[str, Any]]) -> int:
        """
        Get the optimal batch size for the next multimodal batch based on performance metrics and input complexity.

        Args:
            processing_time_ms: Processing time for the current batch in milliseconds
            tokens_processed: Number of tokens processed in the current batch
            inputs: List of input dictionaries for complexity analysis

        Returns:
            Recommended batch size for the next batch
        """
        # Collect metrics for the current batch
        batch_info = self.collect_multimodal_metrics(
            self.current_batch_size, 
            processing_time_ms, 
            tokens_processed, 
            inputs
        )

        # Determine complexity-based batch size
        _, _, combined_complexity = self.analyze_multimodal_complexity(inputs)
        complexity_based_size = self._get_complexity_based_batch_size(combined_complexity)

        # Determine performance-based batch size using the parent class logic
        performance_based_size = self._get_performance_based_batch_size(processing_time_ms, tokens_processed)

        # Combine both approaches - prioritize complexity-based sizing for multimodal inputs
        # since they have different resource requirements than pure text
        combined_size = int(
            0.7 * complexity_based_size + 0.3 * performance_based_size
        )

        # Ensure the result is within bounds
        final_size = max(self.min_batch_size, min(combined_size, self.max_batch_size))

        logger.debug(f"Multimodal batch sizing: complexity_based={complexity_based_size}, "
                    f"performance_based={performance_based_size}, final={final_size}, "
                    f"combined_complexity={combined_complexity:.3f}")

        return final_size

    def _get_complexity_based_batch_size(self, combined_complexity: float) -> int:
        """
        Get batch size based on the combined complexity of inputs.

        Args:
            combined_complexity: Combined complexity score (0.0 to 1.0)

        Returns:
            Recommended batch size based on complexity
        """
        if combined_complexity <= self.complexity_threshold_low:
            # Simple inputs - use larger batch size
            return self.max_batch_size
        elif combined_complexity >= self.complexity_threshold_high:
            # Complex inputs - use smaller batch size
            return self.min_batch_size
        else:
            # Moderate complexity - interpolate between min and max
            complexity_range = self.complexity_threshold_high - self.complexity_threshold_low
            normalized_complexity = (combined_complexity - self.complexity_threshold_low) / complexity_range
            batch_range = self.max_batch_size - self.min_batch_size

            # Calculate batch size with inverse relationship to complexity
            calculated_batch_size = int(self.max_batch_size - (normalized_complexity * batch_range))

            # Ensure it stays within bounds
            return max(self.min_batch_size, min(calculated_batch_size, self.max_batch_size))

    def _get_performance_based_batch_size(self, processing_time_ms: float, tokens_processed: int) -> int:
        """
        Get batch size based on performance metrics (using parent AdaptiveBatchManager logic).

        Args:
            processing_time_ms: Processing time for the current batch
            tokens_processed: Number of tokens processed

        Returns:
            Recommended batch size based on performance
        """
        # Create a temporary AdaptiveBatchManager to leverage its logic
        temp_manager = AdaptiveBatchManager(
            initial_batch_size=self.current_batch_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            memory_threshold_ratio=self.memory_threshold_ratio,
            performance_window_size=self.performance_window_size,
            adjustment_factor=self.adjustment_factor,
            cooldown_period=self.cooldown_period,
            performance_target=self.performance_target
        )

        # Use the temporary manager to get a performance-based recommendation
        return temp_manager.get_optimal_batch_size(processing_time_ms, tokens_processed, None)

    def _get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information."""
        import psutil

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


class ImageComplexityAnalyzer:
    """
    Analyzes the complexity of images for multimodal batching decisions.
    """

    def analyze_image_complexity(self, image: Union[Image.Image, torch.Tensor]) -> float:
        """
        Analyze the complexity of an image.

        Args:
            image: PIL Image or torch Tensor representing an image

        Returns:
            Complexity score between 0.0 (simple) and 1.0 (complex)
        """
        if isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            img_array = np.array(image)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            img_array = image.cpu().numpy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Calculate various complexity metrics
        height, width = img_array.shape[:2]
        
        # Resolution-based complexity (larger images are more complex)
        resolution_complexity = min(1.0, (height * width) / (224 * 224))  # Normalize against 224x224

        # Color diversity (more colors = more complex)
        if len(img_array.shape) == 3:  # RGB image
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            max_possible_colors = min(256**3, height * width)  # Cap at image size
            color_diversity = unique_colors / max_possible_colors
        else:  # Grayscale
            unique_colors = len(np.unique(img_array))
            max_possible_colors = 256
            color_diversity = unique_colors / max_possible_colors

        # Texture complexity using local binary patterns (simplified)
        texture_complexity = self._calculate_texture_complexity(img_array)

        # Edge complexity (more edges = more complex)
        edge_complexity = self._calculate_edge_complexity(img_array)

        # Weighted combination of all factors
        complexity_score = (
            0.3 * resolution_complexity +
            0.25 * color_diversity +
            0.25 * texture_complexity +
            0.2 * edge_complexity
        )

        return min(1.0, complexity_score)  # Clamp to 0-1 range

    def _calculate_texture_complexity(self, img_array: np.ndarray) -> float:
        """
        Calculate a simplified texture complexity score.
        """
        if len(img_array.shape) == 3:
            # Convert to grayscale for simplicity
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Calculate local variance as a measure of texture
        from scipy.ndimage import generic_filter
        local_std = generic_filter(gray, np.std, size=5)
        avg_std = np.mean(local_std)
        
        # Normalize against maximum possible standard deviation for uint8
        return min(1.0, avg_std / 128.0)

    def _calculate_edge_complexity(self, img_array: np.ndarray) -> float:
        """
        Calculate edge complexity using Sobel operator.
        """
        if len(img_array.shape) == 3:
            # Convert to grayscale for simplicity
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Calculate gradients using Sobel operator
        from scipy.ndimage import sobel
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize by maximum possible gradient magnitude
        max_possible_gradient = 255 * np.sqrt(2)
        return min(1.0, np.mean(magnitude) / max_possible_gradient)


def get_dynamic_multimodal_batch_manager(
    initial_batch_size: int = 1,
    min_batch_size: int = 1,
    max_batch_size: int = 8,
    memory_threshold_ratio: float = 0.85,
    performance_window_size: int = 10,
    adjustment_factor: float = 0.1,
    cooldown_period: float = 5.0,
    performance_target: float = 0.8,
    text_weight: float = 0.4,
    image_weight: float = 0.6,
    complexity_threshold_low: float = 0.3,
    complexity_threshold_high: float = 0.7
) -> DynamicMultimodalBatchManager:
    """
    Get a global instance of the dynamic multimodal batch manager.

    Args:
        initial_batch_size: Starting batch size
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size
        memory_threshold_ratio: Memory usage ratio that triggers adjustments (0.0 to 1.0)
        performance_window_size: Number of recent samples to consider for performance evaluation
        adjustment_factor: Factor controlling how aggressively to adjust batch size
        cooldown_period: Time in seconds to wait between adjustments
        performance_target: Target performance score (0.0 to 1.0)
        text_weight: Weight of text complexity in combined complexity calculation
        image_weight: Weight of image complexity in combined complexity calculation
        complexity_threshold_low: Below this complexity, use max batch size
        complexity_threshold_high: Above this complexity, use min batch size

    Returns:
        DynamicMultimodalBatchManager instance
    """
    if not hasattr(get_dynamic_multimodal_batch_manager, '_instance'):
        get_dynamic_multimodal_batch_manager._instance = DynamicMultimodalBatchManager(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            memory_threshold_ratio=memory_threshold_ratio,
            performance_window_size=performance_window_size,
            adjustment_factor=adjustment_factor,
            cooldown_period=cooldown_period,
            performance_target=performance_target,
            text_weight=text_weight,
            image_weight=image_weight,
            complexity_threshold_low=complexity_threshold_low,
            complexity_threshold_high=complexity_threshold_high
        )
    return get_dynamic_multimodal_batch_manager._instance


__all__ = [
    "DynamicMultimodalBatchManager",
    "MultimodalBatchType",
    "MultimodalBatchInfo",
    "ImageComplexityAnalyzer",
    "get_dynamic_multimodal_batch_manager"
]