"""
Hyperparameter Optimization System for Inference-PIO

This module implements automatic hyperparameter optimization for performance tuning
across different models and input types.
"""

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution, minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

warnings.filterwarnings("ignore", category=UserWarning)


logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for a hyperparameter."""

    name: str
    type: str  # 'int', 'float', 'bool', 'categorical'
    bounds: Tuple[float, float]  # (min, max) for numeric params
    choices: Optional[List[Any]] = None  # For categorical params
    default_value: Any = None
    step: Optional[float] = None  # Step size for discrete parameters


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: Dict[str, Any]
    best_score: float
    optimization_trace: List[Tuple[Dict[str, Any], float]]  # History of evaluations
    num_evaluations: int


class HyperparameterOptimizer:
    """System for optimizing hyperparameters for performance."""

    def __init__(self):
        self.hyperparameters: List[HyperparameterConfig] = []
        self.gp_model = None
        self.evaluated_points = []
        self.evaluated_scores = []

    def add_hyperparameter(self, config: HyperparameterConfig):
        """Add a hyperparameter to optimize."""
        self.hyperparameters.append(config)

    def add_common_parameters(self):
        """Add common hyperparameters for performance optimization."""
        common_params = [
            HyperparameterConfig(
                name="batch_size", type="int", bounds=(1, 64), default_value=16, step=1
            ),
            HyperparameterConfig(
                name="attention_sparsity_ratio",
                type="float",
                bounds=(0.1, 0.9),
                default_value=0.25,
                step=0.05,
            ),
            HyperparameterConfig(
                name="compression_ratio",
                type="float",
                bounds=(0.1, 0.9),
                default_value=0.5,
                step=0.05,
            ),
            HyperparameterConfig(
                name="pruning_ratio",
                type="float",
                bounds=(0.05, 0.5),
                default_value=0.2,
                step=0.05,
            ),
            HyperparameterConfig(
                name="max_memory_ratio",
                type="float",
                bounds=(0.5, 0.95),
                default_value=0.8,
                step=0.05,
            ),
            HyperparameterConfig(
                name="kernel_fusion_enabled",
                type="bool",
                bounds=(0, 1),  # Used for bounds in optimization
                default_value=True,
            ),
            HyperparameterConfig(
                name="use_flash_attention",
                type="bool",
                bounds=(0, 1),
                default_value=True,
            ),
            HyperparameterConfig(
                name="use_sparse_attention",
                type="bool",
                bounds=(0, 1),
                default_value=True,
            ),
        ]

        for param in common_params:
            self.add_hyperparameter(param)

    def _param_dict_to_array(self, param_dict: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to array for optimization."""
        values = []
        for hp in self.hyperparameters:
            if hp.name in param_dict:
                val = param_dict[hp.name]
                if hp.type == "bool":
                    values.append(1.0 if val else 0.0)
                else:
                    values.append(float(val))
            else:
                # Use default value if not provided
                if hp.type == "bool":
                    values.append(1.0 if hp.default_value else 0.0)
                else:
                    values.append(float(hp.default_value))
        return np.array(values)

    def _array_to_param_dict(self, param_array: np.ndarray) -> Dict[str, Any]:
        """Convert parameter array back to dictionary."""
        param_dict = {}
        for i, hp in enumerate(self.hyperparameters):
            val = param_array[i]
            if hp.type == "int":
                param_dict[hp.name] = int(round(val))
            elif hp.type == "float":
                param_dict[hp.name] = float(val)
            elif hp.type == "bool":
                param_dict[hp.name] = bool(round(val))
            elif hp.type == "categorical":
                # For categorical, we'd map the continuous value to the closest choice
                # This is a simplified approach
                idx = min(int(round(val)), len(hp.choices) - 1) if hp.choices else 0
                param_dict[hp.name] = (
                    hp.choices[idx] if hp.choices else hp.default_value
                )
        return param_dict

    def _validate_params(self, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp parameters to their bounds."""
        validated = {}
        for hp in self.hyperparameters:
            if hp.name in param_dict:
                val = param_dict[hp.name]

                if hp.type in ["int", "float"]:
                    # Clamp to bounds
                    min_val, max_val = hp.bounds
                    val = max(min_val, min(max_val, val))

                    if hp.type == "int":
                        val = int(round(val))

                validated[hp.name] = val
            else:
                validated[hp.name] = hp.default_value

        return validated

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        n_calls: int = 50,
        n_random_starts: int = 10,
        minimize: bool = True,
    ) -> OptimizationResult:
        """Optimize hyperparameters using Bayesian optimization approach."""

        def objective_wrapper(param_array):
            param_dict = self._array_to_param_dict(param_array)
            score = objective_function(param_dict)

            # Store evaluated point
            self.evaluated_points.append(param_array.copy())
            self.evaluated_scores.append(score)

            # Return negative score if maximizing
            return -score if not minimize else score

        # Define bounds for optimization
        bounds = []
        for hp in self.hyperparameters:
            bounds.append(hp.bounds)

        bounds = np.array(bounds)

        # Use differential evolution for global optimization
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=n_calls,
            popsize=max(10, n_random_starts // len(bounds)),
            seed=42,
            disp=False,
        )

        best_params = self._array_to_param_dict(result.x)
        best_score = result.fun if minimize else -result.fun

        # Create optimization trace
        optimization_trace = []
        for point, score in zip(self.evaluated_points, self.evaluated_scores):
            param_dict = self._array_to_param_dict(point)
            optimization_trace.append((param_dict, score))

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_trace=optimization_trace,
            num_evaluations=len(self.evaluated_points),
        )

    def get_current_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found so far."""
        if not self.evaluated_scores:
            # Return defaults if no evaluations done
            return {hp.name: hp.default_value for hp in self.hyperparameters}

        best_idx = np.argmin(self.evaluated_scores) if self.evaluated_scores else 0
        best_point = self.evaluated_points[best_idx]
        return self._array_to_param_dict(best_point)


class PerformanceHyperparameterOptimizer:
    """High-level optimizer for performance hyperparameters."""

    def __init__(self):
        self.optimizer = HyperparameterOptimizer()
        self.optimizer.add_common_parameters()
        self.current_model = None
        self.current_input = None

    def optimize_for_model(
        self,
        model: nn.Module,
        input_data: Any,
        target_metric: str = "latency",  # "latency", "memory", "throughput"
        n_calls: int = 30,
    ) -> OptimizationResult:
        """Optimize hyperparameters for a specific model and input."""
        self.current_model = model
        self.current_input = input_data

        def objective_function(params: Dict[str, Any]) -> float:
            """Objective function that evaluates performance with given parameters."""
            return self._evaluate_performance(model, input_data, params, target_metric)

        return self.optimizer.optimize(
            objective_function=objective_function,
            n_calls=n_calls,
            minimize=(
                target_metric in ["latency", "memory"]
            ),  # Minimize for latency/memory
            n_random_starts=min(10, n_calls // 3),
        )

    def _evaluate_performance(
        self,
        model: nn.Module,
        input_data: Any,
        params: Dict[str, Any],
        target_metric: str,
    ) -> float:
        """Evaluate model performance with given hyperparameters."""
        # Apply parameters to model temporarily
        original_params = self._apply_params_to_model(model, params)

        # Benchmark performance
        start_time = time.time()
        memory_before = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        try:
            with torch.no_grad():
                if isinstance(input_data, torch.Tensor):
                    output = model(input_data)
                elif isinstance(input_data, dict):
                    output = model(**input_data)
                else:
                    output = model(input_data)
        except Exception as e:
            logger.warning(f"Error during performance evaluation: {e}")
            # Return worst possible score
            if target_metric in ["latency", "memory"]:
                return float("inf")  # Worst case for minimization
            else:
                return 0.0  # Worst case for maximization

        # Calculate metrics
        latency = time.time() - start_time
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (memory_after - memory_before) / (1024**2)  # MB

        if isinstance(output, torch.Tensor):
            tokens_processed = output.numel()
        else:
            tokens_processed = 1

        throughput = tokens_processed / max(latency, 1e-6)  # Tokens per second

        # Restore original parameters
        self._restore_params_from_model(model, original_params)

        # Return the target metric
        if target_metric == "latency":
            return latency * 1000  # Convert to milliseconds
        elif target_metric == "memory":
            return memory_used
        elif target_metric == "throughput":
            return throughput
        else:
            # Default to latency
            return latency * 1000

    def _apply_params_to_model(
        self, model: nn.Module, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply hyperparameters to the model and return original values."""
        original_values = {}

        # This is a simplified implementation - in practice, you'd need to
        # actually modify the model's configuration based on the parameters
        for param_name, param_value in params.items():
            # Store original value if it exists as an attribute
            if hasattr(model, param_name):
                original_values[param_name] = getattr(model, param_name)
                setattr(model, param_name, param_value)

        return original_values

    def _restore_params_from_model(
        self, model: nn.Module, original_values: Dict[str, Any]
    ):
        """Restore original hyperparameter values to the model."""
        for param_name, original_value in original_values.items():
            if hasattr(model, param_name):
                setattr(model, param_name, original_value)

    def get_optimal_params_for_model(self, model: nn.Module) -> Dict[str, Any]:
        """Get the optimal parameters found for a specific model."""
        # In a real implementation, we'd store and retrieve model-specific parameters
        return self.optimizer.get_current_best_params()


# Global instance
performance_optimizer = PerformanceHyperparameterOptimizer()


def get_performance_optimizer() -> PerformanceHyperparameterOptimizer:
    """Get the global performance hyperparameter optimizer instance."""
    return performance_optimizer


logger.info("Hyperparameter Optimization System loaded successfully")


__all__ = [
    "HyperparameterConfig",
    "OptimizationResult",
    "HyperparameterOptimizer",
    "PerformanceHyperparameterOptimizer",
    "get_performance_optimizer",
    "performance_optimizer",
]
