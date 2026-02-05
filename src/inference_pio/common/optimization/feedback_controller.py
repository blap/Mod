"""
Feedback Controller for Continuous Optimization Adjustments

This module implements a centralized feedback system that monitors model performance
metrics and adjusts optimization strategies based on observed accuracy and latency.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


class FeedbackEventType(Enum):
    """Types of feedback events that can be processed."""

    ACCURACY_UPDATE = "accuracy_update"
    LATENCY_UPDATE = "latency_update"
    PERFORMANCE_ADJUSTMENT = "performance_adjustment"
    OPTIMIZATION_CHANGE = "optimization_change"


@dataclass
class FeedbackEvent:
    """Represents a feedback event with timestamp and metadata."""

    event_type: FeedbackEventType
    model_id: str
    timestamp: float
    metrics: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Container for performance metrics collected from models."""

    accuracy: float = 0.0
    latency: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: float = 0.0

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.timestamp = time.time()


@dataclass
class OptimizationAdjustment:
    """Describes an adjustment to optimization parameters."""

    strategy: str
    parameters: Dict[str, Any]
    reason: str
    confidence: float = 1.0


class FeedbackController:
    """
    Centralized feedback controller that monitors model performance and adjusts
    optimization strategies based on observed metrics.

    This system continuously collects performance metrics from models and makes
    adjustments to optimization parameters to maintain optimal performance.
    """

    def __init__(self, window_size: int = 100, adjustment_threshold: float = 0.05):
        """
        Initialize the feedback controller.

        Args:
            window_size: Number of recent metrics to consider for adjustments
            adjustment_threshold: Threshold for triggering optimization adjustments
        """
        self.window_size = window_size
        self.adjustment_threshold = adjustment_threshold

        # Storage for performance metrics per model
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.current_metrics: Dict[str, PerformanceMetrics] = {}

        # Callbacks for optimization adjustments
        self.adjustment_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Event queue for feedback processing
        self.event_queue: deque = deque()
        self.event_lock = threading.Lock()

        # Performance targets
        self.performance_targets: Dict[str, Dict[str, float]] = {
            "accuracy": {"target": 0.95, "weight": 0.7},
            "latency": {"target": 0.1, "weight": 0.3},  # Target latency in seconds
        }

        # Logger setup
        self.logger = logging.getLogger(__name__)

        # Background thread for continuous monitoring
        self.monitoring_thread = None
        self.is_monitoring = False

    def register_model(self, model_id: str):
        """Register a model with the feedback controller."""
        if model_id not in self.metrics_history:
            self.metrics_history[model_id] = deque(maxlen=self.window_size)
            self.current_metrics[model_id] = PerformanceMetrics()

    def unregister_model(self, model_id: str):
        """Unregister a model from the feedback controller."""
        if model_id in self.metrics_history:
            del self.metrics_history[model_id]
            del self.current_metrics[model_id]

    def record_metrics(self, model_id: str, metrics: PerformanceMetrics):
        """Record performance metrics for a specific model."""
        self.register_model(model_id)

        # Store in history
        self.metrics_history[model_id].append(metrics)

        # Update current metrics
        self.current_metrics[model_id] = metrics

        # Add to event queue
        event = FeedbackEvent(
            event_type=(
                FeedbackEventType.ACCURACY_UPDATE
                if metrics.accuracy > 0
                else FeedbackEventType.LATENCY_UPDATE
            ),
            model_id=model_id,
            timestamp=time.time(),
            metrics={
                "accuracy": metrics.accuracy,
                "latency": metrics.latency,
                "throughput": metrics.throughput,
                "memory_usage": metrics.memory_usage,
                "gpu_utilization": metrics.gpu_utilization,
            },
        )

        with self.event_lock:
            self.event_queue.append(event)

        # Check if adjustment is needed
        adjustment = self._evaluate_performance_and_adjust(model_id)
        if adjustment:
            self._apply_adjustment(model_id, adjustment)

    def add_adjustment_callback(
        self, model_id: str, callback: Callable[[OptimizationAdjustment], None]
    ):
        """Add a callback to handle optimization adjustments for a specific model."""
        self.adjustment_callbacks[model_id].append(callback)

    def remove_adjustment_callback(
        self, model_id: str, callback: Callable[[OptimizationAdjustment], None]
    ):
        """Remove a callback for a specific model."""
        if model_id in self.adjustment_callbacks:
            try:
                self.adjustment_callbacks[model_id].remove(callback)
            except ValueError:
                logger.warning(f"Callback not found for event: {event_type}")

    def set_performance_target(
        self, metric_name: str, target_value: float, weight: float = 1.0
    ):
        """Set performance targets for specific metrics."""
        self.performance_targets[metric_name] = {
            "target": target_value,
            "weight": weight,
        }

    def get_current_metrics(self, model_id: str) -> Optional[PerformanceMetrics]:
        """Get the current performance metrics for a model."""
        return self.current_metrics.get(model_id)

    def get_historical_metrics(
        self, model_id: str, count: int = 10
    ) -> List[PerformanceMetrics]:
        """Get historical performance metrics for a model."""
        history = self.metrics_history.get(model_id, deque())
        return list(history)[-count:]

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

    def _evaluate_performance_and_adjust(
        self, model_id: str
    ) -> Optional[OptimizationAdjustment]:
        """Evaluate performance and determine if adjustments are needed."""
        if model_id not in self.current_metrics:
            return None

        current = self.current_metrics[model_id]
        history = list(self.metrics_history[model_id])

        if len(history) < 2:
            return None

        # Calculate trends
        avg_accuracy = (
            np.mean([m.accuracy for m in history[-10:]])
            if len(history) >= 10
            else current.accuracy
        )
        avg_latency = (
            np.mean([m.latency for m in history[-10:]])
            if len(history) >= 10
            else current.latency
        )

        # Check if we need to adjust based on targets
        accuracy_deviation = abs(
            avg_accuracy - self.performance_targets["accuracy"]["target"]
        )
        latency_deviation = abs(
            avg_latency - self.performance_targets["latency"]["target"]
        )

        # Determine if adjustment is needed
        needs_adjustment = (
            accuracy_deviation > self.adjustment_threshold
            or latency_deviation > self.adjustment_threshold
        )

        if not needs_adjustment:
            return None

        # Determine adjustment strategy based on which metric is off-target
        if accuracy_deviation > self.adjustment_threshold:
            if avg_accuracy < self.performance_targets["accuracy"]["target"]:
                # Accuracy too low - increase precision or reduce aggressive optimizations
                strategy = "increase_precision"
                params = {
                    "precision": "float32" if current.accuracy < 0.9 else "float16",
                    "reduce_compression": True,
                    "increase_attention_heads": True,
                }
                reason = f"Accuracy below target: {avg_accuracy:.3f} < {self.performance_targets['accuracy']['target']:.3f}"
            else:
                # Accuracy too high - could potentially optimize for speed
                strategy = "optimize_for_speed"
                params = {
                    "precision": "float16",
                    "enable_compression": True,
                    "reduce_attention_heads": False,
                }
                reason = f"Accuracy above target: {avg_accuracy:.3f} > {self.performance_targets['accuracy']['target']:.3f}"

        elif latency_deviation > self.adjustment_threshold:
            if avg_latency > self.performance_targets["latency"]["target"]:
                # Latency too high - optimize for speed
                strategy = "optimize_for_speed"
                params = {
                    "precision": "float16",
                    "enable_compression": True,
                    "reduce_attention_heads": True,
                    "batch_size_reduction": True,
                }
                reason = f"Latency above target: {avg_latency:.3f}s > {self.performance_targets['latency']['target']:.3f}s"
            else:
                # Latency too low - could potentially improve accuracy
                strategy = "increase_accuracy"
                params = {
                    "precision": "float32",
                    "reduce_compression": True,
                    "increase_attention_heads": True,
                }
                reason = f"Latency below target: {avg_latency:.3f}s < {self.performance_targets['latency']['target']:.3f}s"
        else:
            return None

        # Calculate confidence based on trend stability
        accuracy_trend_stable = self._calculate_trend_stability(
            [m.accuracy for m in history]
        )
        latency_trend_stable = self._calculate_trend_stability(
            [m.latency for m in history]
        )
        confidence = min(accuracy_trend_stable, latency_trend_stable)

        return OptimizationAdjustment(
            strategy=strategy, parameters=params, reason=reason, confidence=confidence
        )

    def _calculate_trend_stability(self, values: List[float]) -> float:
        """Calculate the stability of a trend based on variance."""
        if len(values) < 2:
            return 0.0

        # Calculate coefficient of variation (lower is more stable)
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 0.0

        cv = std_val / abs(mean_val)
        # Convert to confidence score (0-1, where 1 is most stable)
        return max(0.0, 1.0 - cv)

    def _apply_adjustment(self, model_id: str, adjustment: OptimizationAdjustment):
        """Apply an optimization adjustment to a model."""
        self.logger.info(
            f"Applying adjustment to {model_id}: {adjustment.strategy} - {adjustment.reason}"
        )

        # Trigger callbacks for this model
        for callback in self.adjustment_callbacks[model_id]:
            try:
                callback(adjustment)
            except Exception as e:
                self.logger.error(
                    f"Error in adjustment callback for {model_id}: {str(e)}"
                )

        # Add to event queue
        event = FeedbackEvent(
            event_type=FeedbackEventType.OPTIMIZATION_CHANGE,
            model_id=model_id,
            timestamp=time.time(),
            metrics={
                "strategy": adjustment.strategy,
                "parameters": adjustment.parameters,
            },
            context={"reason": adjustment.reason, "confidence": adjustment.confidence},
        )

        with self.event_lock:
            self.event_queue.append(event)

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                # Process events in queue
                with self.event_lock:
                    while self.event_queue:
                        event = self.event_queue.popleft()
                        self._process_event(event)

                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error

    def _process_event(self, event: FeedbackEvent):
        """Process a feedback event."""
        # Log the event
        self.logger.debug(
            f"Processing event: {event.event_type.value} for {event.model_id}"
        )

        # Additional event processing can be added here


# Global instance for singleton pattern
feedback_controller = FeedbackController()


def get_feedback_controller() -> FeedbackController:
    """Get the global feedback controller instance."""
    return feedback_controller
