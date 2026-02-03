"""
Unified Machine Learning Optimization System for Inference-PIO

This module provides a unified system that combines ML-based optimization selection
with hyperparameter tuning for all model types.
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .hyperparameter_optimizer import (
    PerformanceHyperparameterOptimizer,
    get_performance_optimizer,
)
from ..processing.input_complexity_analyzer import get_complexity_analyzer
from .ml_optimization_selector import (
    AutoOptimizationSelector,
    PerformanceMetrics,
    get_auto_selector,
)
from .optimization_manager import OptimizationConfig, get_optimization_manager

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""

    GLM_4_7_FLASH = "glm_4_7_flash"
    QWEN3_4B_INSTRUCT_2507 = "qwen3_4b_instruct_2507"
    QWEN3_CODER_30B = "qwen3_coder_30b"
    QWEN3_VL_2B = "qwen3_vl_2b"


@dataclass
class MLBasedOptimizationConfig:
    """Configuration for ML-based optimization."""

    model_type: ModelType
    enable_ml_selection: bool = True
    enable_hyperparameter_tuning: bool = True
    target_metric: str = "latency"  # "latency", "memory", "throughput", "energy"
    max_optimizations: int = 5
    optimization_frequency: int = 10  # Re-optimize every N calls
    use_input_adaptation: bool = True  # Adapt based on input characteristics


class UnifiedMLOptimizationSystem:
    """Unified system for ML-based optimization across all models."""

    def __init__(self):
        self.auto_selector = get_auto_selector()
        self.performance_optimizer = get_performance_optimizer()
        self.configs: Dict[ModelType, MLBasedOptimizationConfig] = {}
        self.call_counts: Dict[str, int] = {}  # Track calls per model instance
        self.model_configs: Dict[str, Dict[str, Any]] = (
            {}
        )  # Store optimal configs per model
        self.hardware_info: Dict[str, Any] = self._get_hardware_info()

    def register_model_type(
        self, model_type: ModelType, config: MLBasedOptimizationConfig
    ):
        """Register configuration for a specific model type."""
        self.configs[model_type] = config

        # Set model family for the selector based on model type
        from .optimization_config import ModelFamily

        if model_type == ModelType.GLM_4_7_FLASH:
            self.auto_selector.set_model_family(ModelFamily.GLM)
        elif model_type in [
            ModelType.QWEN3_4B_INSTRUCT_2507,
            ModelType.QWEN3_CODER_30B,
            ModelType.QWEN3_VL_2B,
        ]:
            self.auto_selector.set_model_family(ModelFamily.QWEN)

    def optimize_model_for_input(
        self, model: nn.Module, input_data: Any, model_type: ModelType
    ) -> nn.Module:
        """Apply ML-based optimizations to a model based on input characteristics."""
        model_id = id(model)

        # Get configuration for this model type
        if model_type not in self.configs:
            logger.warning(
                f"No configuration found for model type {model_type}, using defaults"
            )
            config = MLBasedOptimizationConfig(model_type=model_type)
            self.register_model_type(model_type, config)
        else:
            config = self.configs[model_type]

        # Increment call count
        self.call_counts[model_id] = self.call_counts.get(model_id, 0) + 1
        should_reoptimize = (
            self.call_counts[model_id] % config.optimization_frequency
        ) == 0

        # Apply ML-based optimization selection
        if config.enable_ml_selection or should_reoptimize:
            optimized_model = self._apply_ml_optimizations(
                model, input_data, model_type, config
            )
        else:
            # Use previously stored configuration
            if str(model_id) in self.model_configs:
                optimized_model = self._apply_stored_config(
                    model, self.model_configs[str(model_id)]
                )
            else:
                optimized_model = model  # No optimizations applied

        # Apply hyperparameter tuning if enabled
        if config.enable_hyperparameter_tuning:
            optimized_model = self._apply_hyperparameter_tuning(
                optimized_model, input_data, config.target_metric
            )

        return optimized_model

    def _apply_ml_optimizations(
        self,
        model: nn.Module,
        input_data: Any,
        model_type: ModelType,
        config: MLBasedOptimizationConfig,
    ) -> nn.Module:
        """Apply ML-based optimization selection."""
        try:
            # Select optimizations using ML
            selected_optimizations = self.auto_selector.select_optimizations(
                model=model,
                input_data=input_data,
                hardware_info=self.hardware_info,
                target_metric=(
                    config.target_metric + "_ms"
                    if config.target_metric == "latency"
                    else config.target_metric
                ),
                max_optimizations=config.max_optimizations,
            )

            logger.info(
                f"Selected optimizations for {model_type.value}: {selected_optimizations}"
            )

            # Apply selected optimizations
            opt_manager = get_optimization_manager()

            # Configure optimizations
            for opt_name in selected_optimizations:
                if opt_name in opt_manager.get_available_optimizations():
                    # Create a basic config, in practice you'd customize based on input
                    opt_config = OptimizationConfig(
                        name=opt_name,
                        enabled=True,
                        parameters={},  # Will be tuned later
                    )
                    opt_manager.configure_optimization(opt_name, opt_config)

            # Apply optimizations
            optimized_model = opt_manager.apply_optimizations(
                model, selected_optimizations
            )

            # Store configuration for this model
            model_id = str(id(model))
            self.model_configs[model_id] = {
                "optimizations": selected_optimizations,
                "timestamp": time.time(),
            }

            # Update ML model with performance feedback (simulated here)
            # In a real implementation, you'd measure actual performance
            perf_metrics = PerformanceMetrics(
                latency_ms=100.0,  # Placeholder
                memory_usage_mb=500.0,  # Placeholder
                throughput_tokens_per_sec=10.0,  # Placeholder
                energy_consumption=1.0,  # Placeholder
                accuracy_drop=0.0,  # Placeholder
            )

            self.auto_selector.update_with_performance_feedback(
                model=model,
                input_data=input_data,
                applied_optimizations=selected_optimizations,
                performance_metrics=perf_metrics,
                hardware_info=self.hardware_info,
            )

            return optimized_model

        except Exception as e:
            logger.error(f"Error applying ML optimizations: {e}")
            return model  # Return original model if optimization fails

    def _apply_hyperparameter_tuning(
        self, model: nn.Module, input_data: Any, target_metric: str
    ) -> nn.Module:
        """Apply hyperparameter tuning to the model."""
        try:
            # Perform hyperparameter optimization
            optimization_result = self.performance_optimizer.optimize_for_model(
                model=model,
                input_data=input_data,
                target_metric=target_metric,
                n_calls=20,  # Limit calls for performance
            )

            logger.info(
                f"Hyperparameter optimization completed. Best params: {optimization_result.best_params}"
            )

            # Apply the best parameters to the model
            self._apply_params_to_model(model, optimization_result.best_params)

            return model

        except Exception as e:
            logger.error(f"Error applying hyperparameter tuning: {e}")
            return model  # Return model unchanged if tuning fails

    def _apply_params_to_model(self, model: nn.Module, params: Dict[str, Any]):
        """Apply hyperparameters to the model."""
        # This is a simplified implementation - in practice, you'd need to
        # actually modify the model's configuration based on the parameters
        for param_name, param_value in params.items():
            # Only set attributes that exist and are modifiable
            if hasattr(model, param_name) and not param_name.startswith("_"):
                try:
                    setattr(model, param_name, param_value)
                except AttributeError:
                    # Some attributes might be read-only
                    continue

    def _apply_stored_config(
        self, model: nn.Module, config: Dict[str, Any]
    ) -> nn.Module:
        """Apply previously stored optimization configuration."""
        try:
            opt_manager = get_optimization_manager()
            return opt_manager.apply_optimizations(model, config["optimizations"])
        except Exception as e:
            logger.error(f"Error applying stored config: {e}")
            return model

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information."""
        return {
            "gpu_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.is_available()
                else 0
            ),
            "cpu_cores": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 4
            ),
            "cuda_cores": 0,  # Would need to query GPU specifics
            "compute_capability": (
                torch.cuda.get_device_capability()[0]
                + torch.cuda.get_device_capability()[1] / 10
                if torch.cuda.is_available()
                else 0.0
            ),
            "is_gpu_available": torch.cuda.is_available(),
        }

    def save_state(self, filepath: str):
        """Save the ML optimization system state."""
        import os
        import pickle

        state = {
            "call_counts": self.call_counts,
            "model_configs": self.model_configs,
            "configs": self.configs,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        # Also save the underlying ML models
        selector_path = filepath.replace(".pkl", "_selector.json")
        self.auto_selector.save_state(selector_path)

        logger.info(f"Unified ML Optimization System state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load the ML optimization system state."""
        import os
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.call_counts = state.get("call_counts", {})
        self.model_configs = state.get("model_configs", {})
        self.configs = state.get("configs", {})

        # Also load the underlying ML models
        selector_path = filepath.replace(".pkl", "_selector.json")
        if os.path.exists(selector_path):
            self.auto_selector.load_state(selector_path)

        logger.info(f"Unified ML Optimization System state loaded from {filepath}")


# Initialize the system with configurations for all 4 models
ml_optimization_system = UnifiedMLOptimizationSystem()

# Register configurations for all model types
ml_optimization_system.register_model_type(
    ModelType.GLM_4_7_FLASH,
    MLBasedOptimizationConfig(
        model_type=ModelType.GLM_4_7_FLASH,
        enable_ml_selection=True,
        enable_hyperparameter_tuning=True,
        target_metric="latency",
        max_optimizations=5,
        optimization_frequency=10,
        use_input_adaptation=True,
    ),
)

ml_optimization_system.register_model_type(
    ModelType.QWEN3_4B_INSTRUCT_2507,
    MLBasedOptimizationConfig(
        model_type=ModelType.QWEN3_4B_INSTRUCT_2507,
        enable_ml_selection=True,
        enable_hyperparameter_tuning=True,
        target_metric="throughput",
        max_optimizations=5,
        optimization_frequency=10,
        use_input_adaptation=True,
    ),
)

ml_optimization_system.register_model_type(
    ModelType.QWEN3_CODER_30B,
    MLBasedOptimizationConfig(
        model_type=ModelType.QWEN3_CODER_30B,
        enable_ml_selection=True,
        enable_hyperparameter_tuning=True,
        target_metric="memory",
        max_optimizations=5,
        optimization_frequency=10,
        use_input_adaptation=True,
    ),
)

ml_optimization_system.register_model_type(
    ModelType.QWEN3_VL_2B,
    MLBasedOptimizationConfig(
        model_type=ModelType.QWEN3_VL_2B,
        enable_ml_selection=True,
        enable_hyperparameter_tuning=True,
        target_metric="latency",
        max_optimizations=5,
        optimization_frequency=10,
        use_input_adaptation=True,
    ),
)


def get_ml_optimization_system() -> UnifiedMLOptimizationSystem:
    """Get the global ML optimization system instance."""
    return ml_optimization_system


logger.info("Unified ML Optimization System initialized successfully")


__all__ = [
    "ModelType",
    "MLBasedOptimizationConfig",
    "UnifiedMLOptimizationSystem",
    "get_ml_optimization_system",
    "ml_optimization_system",
]
