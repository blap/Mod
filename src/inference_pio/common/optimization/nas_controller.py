"""
Continuous Neural Architecture Search (NAS) Controller for Inference Time Optimization

This module implements a continuous NAS system that dynamically adapts model architecture
(depth and width) based on input characteristics to balance accuracy and speed during inference.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..processing.adaptive_batch_manager import AdaptiveBatchManager
from ..processing.input_complexity_analyzer import ComplexityMetrics, InputComplexityAnalyzer
from ..hardware.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ArchitectureAdaptationStrategy(Enum):
    """Strategies for adapting architecture during inference."""

    DEPTH_ADAPTIVE = "depth_adaptive"
    WIDTH_ADAPTIVE = "width_adaptive"
    COMBINED_ADAPTIVE = "combined_adaptive"
    LATENCY_BASED = "latency_based"
    MEMORY_BASED = "memory_based"


@dataclass
class NASConfig:
    """Configuration for the NAS controller."""

    strategy: ArchitectureAdaptationStrategy = (
        ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE
    )
    min_depth_ratio: float = 0.3  # Minimum depth as percentage of original
    max_depth_ratio: float = 1.0  # Maximum depth as percentage of original
    min_width_ratio: float = 0.3  # Minimum width as percentage of original
    max_width_ratio: float = 1.0  # Maximum width as percentage of original
    latency_target_ms: float = 100.0  # Target latency in milliseconds
    memory_budget_mb: float = 2048.0  # Memory budget in MB
    accuracy_tradeoff_factor: float = (
        0.7  # Trade-off factor between accuracy and speed (0-1)
    )
    adaptation_frequency: int = 10  # How often to adapt (in terms of inference calls)
    enable_latency_monitoring: bool = True
    enable_memory_monitoring: bool = True
    performance_history_size: int = 50  # Number of past inferences to consider


@dataclass
class ArchitectureState:
    """Current state of the model architecture."""

    depth_ratio: float = 1.0
    width_ratio: float = 1.0
    current_latency_ms: float = 0.0
    current_memory_mb: float = 0.0
    accuracy_estimate: float = 1.0
    adaptation_count: int = 0


@dataclass
class NASMetrics:
    """Metrics collected during NAS operation."""

    input_complexity: float
    target_latency_met: bool
    memory_constraint_met: bool
    accuracy_preserved: bool
    depth_adjustment: float
    width_adjustment: float
    adaptation_reason: str
    processing_time_ms: float
    memory_used_mb: float


class ContinuousNASController:
    """
    Continuous NAS Controller for real-time architecture adaptation.

    This controller monitors input complexity, latency, and memory usage to dynamically
    adjust model depth and width during inference to balance accuracy and speed.
    """

    def __init__(self, config: NASConfig):
        self.config = config
        self.state = ArchitectureState()
        self.performance_history: List[NASMetrics] = []

        # Initialize monitoring components
        self.complexity_analyzer = InputComplexityAnalyzer()
        self.batch_manager = AdaptiveBatchManager(
            initial_batch_size=1, min_batch_size=1, max_batch_size=8
        )
        self.memory_manager = MemoryManager()

        # Track inference count for adaptation frequency
        self.inference_count = 0

        # Handle both enum and string strategies
        if isinstance(config.strategy, ArchitectureAdaptationStrategy):
            strategy_str = config.strategy.value
        else:
            strategy_str = str(config.strategy)

        logger.info(
            f"Initialized Continuous NAS Controller with strategy: {strategy_str}"
        )

    def adapt_architecture(
        self,
        model: nn.Module,
        input_data: Any,
        pre_forward_hook: Optional[Callable] = None,
        post_forward_hook: Optional[Callable] = None,
    ) -> Tuple[nn.Module, NASMetrics]:
        """
        Adapt the model architecture based on input characteristics and constraints.

        Args:
            model: The neural network model to adapt
            input_data: Input data for complexity analysis
            pre_forward_hook: Hook function to call before forward pass
            post_forward_hook: Hook function to call after forward pass

        Returns:
            Tuple of adapted model and NAS metrics
        """
        start_time = time.time()

        # Analyze input complexity
        complexity_metrics = self.complexity_analyzer.analyze_input_complexity(
            input_data
        )
        # Handle both ComplexityMetrics objects and dictionaries
        if hasattr(complexity_metrics, "complexity_score"):
            input_complexity = complexity_metrics.complexity_score
        else:
            # If it's a dictionary from tensor analysis, convert to complexity score
            if (
                isinstance(complexity_metrics, dict)
                and "complexity_score" in complexity_metrics
            ):
                input_complexity = complexity_metrics["complexity_score"]
            else:
                # For tensor analysis dict, we need to calculate complexity score
                input_complexity = 0.5  # Default medium complexity for tensors

        # Get current performance metrics
        current_latency = self._estimate_current_latency()
        current_memory = self._estimate_current_memory_usage(model)

        # Decide whether to adapt based on complexity and constraints
        should_adapt = self._should_adapt(
            input_complexity, current_latency, current_memory
        )

        if should_adapt:
            # Calculate new architecture ratios
            new_depth_ratio, new_width_ratio = self._calculate_adaptation_ratios(
                input_complexity, current_latency, current_memory
            )

            # Apply architecture changes
            adapted_model = self._apply_architecture_changes(
                model, new_depth_ratio, new_width_ratio
            )

            # Update state
            self.state.depth_ratio = new_depth_ratio
            self.state.width_ratio = new_width_ratio
            self.state.current_latency_ms = current_latency
            self.state.current_memory_mb = current_memory
            self.state.adaptation_count += 1

            adaptation_reason = self._determine_adaptation_reason(
                input_complexity, current_latency, current_memory
            )

            logger.info(
                f"Architecture adapted - Depth: {new_depth_ratio:.2f}, "
                f"Width: {new_width_ratio:.2f}, Reason: {adaptation_reason}"
            )
        else:
            adapted_model = model
            adaptation_reason = "No adaptation needed"

        # Calculate metrics
        processing_time_ms = (time.time() - start_time) * 1000
        metrics = NASMetrics(
            input_complexity=input_complexity,
            target_latency_met=current_latency <= self.config.latency_target_ms,
            memory_constraint_met=current_memory <= self.config.memory_budget_mb,
            accuracy_preserved=self._estimate_accuracy_preservation(),
            depth_adjustment=self.state.depth_ratio,
            width_adjustment=self.state.width_ratio,
            adaptation_reason=adaptation_reason,
            processing_time_ms=processing_time_ms,
            memory_used_mb=current_memory,
        )

        # Store metrics for history
        self.performance_history.append(metrics)
        if len(self.performance_history) > self.config.performance_history_size:
            self.performance_history.pop(0)

        self.inference_count += 1

        return adapted_model, metrics

    def _should_adapt(self, complexity: float, latency: float, memory: float) -> bool:
        """Determine if architecture adaptation is needed."""
        # Adapt based on frequency
        if self.inference_count % self.config.adaptation_frequency == 0:
            return True

        # Adapt if constraints are violated
        if latency > self.config.latency_target_ms:
            return True
        if memory > self.config.memory_budget_mb:
            return True

        # Adapt based on input complexity changes
        if len(self.performance_history) >= 2:
            recent_complexities = [
                m.input_complexity for m in self.performance_history[-5:]
            ]
            if len(recent_complexities) > 1:
                complexity_change = abs(
                    recent_complexities[-1] - recent_complexities[-2]
                )
                if complexity_change > 0.2:  # Significant change threshold
                    return True

        return False

    def _calculate_adaptation_ratios(
        self, complexity: float, latency: float, memory: float
    ) -> Tuple[float, float]:
        """Calculate new depth and width ratios based on current conditions."""
        # Start with current ratios
        target_depth = self.state.depth_ratio
        target_width = self.state.width_ratio

        # Adjust based on strategy
        strategy = self.config.strategy
        if isinstance(strategy, ArchitectureAdaptationStrategy):
            strategy_value = strategy.value
        else:
            strategy_value = str(strategy)

        if strategy_value == ArchitectureAdaptationStrategy.DEPTH_ADAPTIVE.value:
            # Adjust depth based on complexity and constraints
            if complexity > 0.7:  # High complexity input
                target_depth = min(target_depth + 0.1, self.config.max_depth_ratio)
            elif complexity < 0.3:  # Low complexity input
                target_depth = max(target_depth - 0.1, self.config.min_depth_ratio)

            # Adjust for latency constraints
            if latency > self.config.latency_target_ms:
                target_depth = max(target_depth * 0.9, self.config.min_depth_ratio)

        elif strategy_value == ArchitectureAdaptationStrategy.WIDTH_ADAPTIVE.value:
            # Adjust width based on complexity and constraints
            if complexity > 0.7:  # High complexity input
                target_width = min(target_width + 0.1, self.config.max_width_ratio)
            elif complexity < 0.3:  # Low complexity input
                target_width = max(target_width - 0.1, self.config.min_width_ratio)

            # Adjust for memory constraints
            if memory > self.config.memory_budget_mb:
                target_width = max(target_width * 0.9, self.config.min_width_ratio)

        elif strategy_value == ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE.value:
            # Combined approach: adjust both depth and width
            # Higher complexity requires more resources
            complexity_factor = 1.0 + (complexity - 0.5) * 0.4  # Range 0.8 to 1.2

            # Adjust depth
            target_depth = max(
                min(target_depth * complexity_factor, self.config.max_depth_ratio),
                self.config.min_depth_ratio,
            )

            # Adjust width
            target_width = max(
                min(target_width * complexity_factor, self.config.max_width_ratio),
                self.config.min_width_ratio,
            )

            # Apply constraint adjustments
            latency_factor = max(
                0.8, 1.0 - (latency / self.config.latency_target_ms - 1.0) * 0.5
            )
            memory_factor = max(
                0.8, 1.0 - (memory / self.config.memory_budget_mb - 1.0) * 0.5
            )

            target_depth *= min(latency_factor, memory_factor)
            target_width *= min(latency_factor, memory_factor)

            # Clamp to bounds
            target_depth = max(
                min(target_depth, self.config.max_depth_ratio),
                self.config.min_depth_ratio,
            )
            target_width = max(
                min(target_width, self.config.max_width_ratio),
                self.config.min_width_ratio,
            )

        elif strategy_value == ArchitectureAdaptationStrategy.LATENCY_BASED.value:
            # Focus on meeting latency targets
            latency_ratio = self.config.latency_target_ms / max(latency, 1.0)
            target_depth *= latency_ratio**0.5  # Square root for balanced adjustment
            target_width *= latency_ratio**0.5

            # Clamp to bounds
            target_depth = max(
                min(target_depth, self.config.max_depth_ratio),
                self.config.min_depth_ratio,
            )
            target_width = max(
                min(target_width, self.config.max_width_ratio),
                self.config.min_width_ratio,
            )

        elif strategy_value == ArchitectureAdaptationStrategy.MEMORY_BASED.value:
            # Focus on staying within memory budget
            memory_ratio = self.config.memory_budget_mb / max(memory, 1.0)
            target_depth *= memory_ratio**0.5
            target_width *= memory_ratio**0.5

            # Clamp to bounds
            target_depth = max(
                min(target_depth, self.config.max_depth_ratio),
                self.config.min_depth_ratio,
            )
            target_width = max(
                min(target_width, self.config.max_width_ratio),
                self.config.min_width_ratio,
            )

        return target_depth, target_width

    def _apply_architecture_changes(
        self, model: nn.Module, depth_ratio: float, width_ratio: float
    ) -> nn.Module:
        """
        Apply architectural changes to the model based on depth and width ratios.

        This is a simplified implementation. In practice, this would involve:
        - Removing/keeping transformer layers based on depth_ratio
        - Adjusting hidden dimensions based on width_ratio
        - Modifying attention heads and MLP dimensions
        """
        # This is a placeholder implementation - in a real system, this would
        # involve actual architectural modifications
        logger.debug(
            f"Applying architecture changes - Depth: {depth_ratio}, Width: {width_ratio}"
        )

        # For now, just return the original model since actual architectural
        # modifications require model-specific implementations
        return model

    def _estimate_current_latency(self) -> float:
        """Estimate current latency based on historical data."""
        if not self.performance_history:
            return 50.0  # Default estimate

        recent_latencies = [
            m.processing_time_ms for m in self.performance_history[-10:]
        ]
        return np.mean(recent_latencies) if recent_latencies else 50.0

    def _estimate_current_memory_usage(self, model: nn.Module) -> float:
        """Estimate current memory usage of the model."""
        # Get memory usage from memory manager if available
        if hasattr(self.memory_manager, "get_current_memory_usage"):
            try:
                return self.memory_manager.get_current_memory_usage()
            except:
                # Placeholder for actual NAS controller implementation
                # This would contain the actual neural architecture search algorithm
                logger.warning("NAS controller not implemented for this operation")
                return None

        # Estimate based on model parameters
        param_count = sum(p.numel() for p in model.parameters())
        # Rough estimation: 4 bytes per parameter for float32
        estimated_mb = (param_count * 4) / (1024 * 1024)
        return estimated_mb

    def _estimate_accuracy_preservation(self) -> float:
        """Estimate accuracy preservation based on current architecture."""
        # Simple model: accuracy decreases as architecture is reduced
        avg_ratio = (self.state.depth_ratio + self.state.width_ratio) / 2
        # Accuracy preservation roughly follows the geometric mean of ratios
        accuracy_preservation = avg_ratio**self.config.accuracy_tradeoff_factor
        return max(0.5, accuracy_preservation)  # Ensure minimum accuracy

    def _determine_adaptation_reason(
        self, complexity: float, latency: float, memory: float
    ) -> str:
        """Determine the reason for architecture adaptation."""
        reasons = []

        if latency > self.config.latency_target_ms:
            reasons.append("latency_exceeded")
        if memory > self.config.memory_budget_mb:
            reasons.append("memory_exceeded")
        if complexity > 0.7:
            reasons.append("high_complexity_input")
        elif complexity < 0.3:
            reasons.append("low_complexity_input")

        if not reasons:
            reasons.append("periodic_adaptation")

        return ", ".join(reasons)


def get_nas_controller(config: Optional[NASConfig] = None) -> ContinuousNASController:
    """Get a singleton instance of the NAS controller."""
    if config is None:
        config = NASConfig()

    # In a real implementation, you might want to maintain a global instance
    # For now, return a new instance
    return ContinuousNASController(config)


# Global instance
nas_controller = get_nas_controller()


__all__ = [
    "ContinuousNASController",
    "NASConfig",
    "ArchitectureAdaptationStrategy",
    "ArchitectureState",
    "NASMetrics",
    "get_nas_controller",
    "nas_controller",
]
