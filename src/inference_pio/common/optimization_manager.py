"""
Modular Optimization Manager for Inference-PIO System

This module provides a centralized system for managing activation/deactivation
of various optimizations across all models. The system allows for flexible
combinations of optimizations and facilitates maintenance and updates.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Enumeration of different optimization types."""
    ATTENTION = "attention"
    MEMORY = "memory"
    COMPUTE = "compute"
    ACTIVATION = "activation"
    MODEL_STRUCTURE = "model_structure"
    QUANTIZATION = "quantization"
    DISTRIBUTED = "distributed"


@dataclass
class OptimizationConfig:
    """Configuration for a specific optimization."""
    name: str
    enabled: bool = False
    optimization_type: OptimizationType = OptimizationType.COMPUTE
    priority: int = 0  # Lower numbers are applied first
    dependencies: List[str] = None  # Names of optimizations this depends on
    parameters: Dict[str, Any] = None  # Specific parameters for the optimization

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}


class OptimizationInterface(ABC):
    """Abstract base class for all optimizations."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.enabled = config.enabled

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """Apply the optimization to the model."""
        pass

    @abstractmethod
    def remove(self, model: nn.Module) -> nn.Module:
        """Remove the optimization from the model."""
        pass

    def is_applicable(self, model: nn.Module) -> bool:
        """Check if this optimization is applicable to the given model."""
        return True

    def get_requirements(self) -> List[str]:
        """Get any requirements for this optimization."""
        return []


class OptimizationRegistry:
    """Registry for managing optimization classes and instances."""

    def __init__(self):
        self._optimizations: Dict[str, Type[OptimizationInterface]] = {}
        self._instances: Dict[str, OptimizationInterface] = {}

    def register(self, name: str, optimization_class: Type[OptimizationInterface]):
        """Register an optimization class."""
        self._optimizations[name] = optimization_class

    def create_instance(self, name: str, config: OptimizationConfig) -> Optional[OptimizationInterface]:
        """Create an instance of an optimization."""
        if name not in self._optimizations:
            logger.error(f"Optimization '{name}' not registered")
            return None

        try:
            instance = self._optimizations[name](config)
            self._instances[name] = instance
            return instance
        except Exception as e:
            logger.error(f"Failed to create instance of optimization '{name}': {e}")
            return None

    def get_instance(self, name: str) -> Optional[OptimizationInterface]:
        """Get an existing instance of an optimization."""
        return self._instances.get(name)

    def get_registered_names(self) -> List[str]:
        """Get list of registered optimization names."""
        return list(self._optimizations.keys())


# Global registry instance
optimization_registry = OptimizationRegistry()


class ModularOptimizationManager:
    """
    Centralized manager for applying and controlling optimizations across models.
    """

    def __init__(self):
        self.registry = optimization_registry
        self.active_optimizations: Dict[str, OptimizationInterface] = {}
        self.applied_to_models: Dict[str, List[str]] = {}  # model_id -> list of applied opt names
        self.optimization_configs: Dict[str, OptimizationConfig] = {}

    def register_optimization(self, name: str, optimization_class: Type[OptimizationInterface]):
        """Register a new optimization type."""
        self.registry.register(name, optimization_class)
        logger.info(f"Registered optimization: {name}")

    def configure_optimization(self, name: str, config: OptimizationConfig):
        """Configure an optimization without applying it."""
        self.optimization_configs[name] = config
        logger.debug(f"Configured optimization: {name}, enabled: {config.enabled}")

    def get_available_optimizations(self) -> List[str]:
        """Get list of all registered optimizations."""
        return self.registry.get_registered_names()

    def apply_optimizations(self, model: nn.Module, optimization_names: List[str] = None) -> nn.Module:
        """
        Apply specified optimizations to the model.

        Args:
            model: Model to apply optimizations to
            optimization_names: List of optimization names to apply (if None, applies all enabled)

        Returns:
            Optimized model
        """
        if optimization_names is None:
            # Apply all enabled optimizations
            optimization_names = [name for name, config in self.optimization_configs.items() if config.enabled]

        # Sort optimizations by priority
        sorted_names = self._sort_optimizations_by_priority(optimization_names)

        # Check dependencies
        if not self._validate_dependencies(sorted_names):
            logger.error("Dependency validation failed, aborting optimization application")
            return model

        model_id = id(model)

        for opt_name in sorted_names:
            if opt_name not in self.optimization_configs:
                logger.warning(f"Optimization '{opt_name}' not configured, skipping")
                continue

            config = self.optimization_configs[opt_name]
            if not config.enabled:
                logger.debug(f"Optimization '{opt_name}' not enabled, skipping")
                continue

            # Create and apply optimization
            optimization = self.registry.create_instance(opt_name, config)
            if optimization is None:
                logger.error(f"Failed to create optimization instance for '{opt_name}', skipping")
                continue

            if not optimization.is_applicable(model):
                logger.info(f"Optimization '{opt_name}' not applicable to this model, skipping")
                continue

            try:
                model = optimization.apply(model)
                self.active_optimizations[opt_name] = optimization
                
                # Track which optimizations were applied to this model
                if model_id not in self.applied_to_models:
                    self.applied_to_models[model_id] = []
                if opt_name not in self.applied_to_models[model_id]:
                    self.applied_to_models[model_id].append(opt_name)
                
                logger.info(f"Applied optimization: {opt_name}")
            except Exception as e:
                logger.error(f"Failed to apply optimization '{opt_name}': {e}")

        return model

    def remove_optimizations(self, model: nn.Module, optimization_names: List[str] = None) -> nn.Module:
        """
        Remove specified optimizations from the model.

        Args:
            model: Model to remove optimizations from
            optimization_names: List of optimization names to remove (if None, removes all applied)

        Returns:
            Original model without optimizations
        """
        model_id = id(model)
        
        if optimization_names is None:
            # Remove all optimizations applied to this model
            optimization_names = self.applied_to_models.get(model_id, [])
        
        # Reverse the order of application for removal
        applied_to_model = self.applied_to_models.get(model_id, [])
        optimization_names = [name for name in reversed(applied_to_model) if name in optimization_names]

        for opt_name in optimization_names:
            if opt_name in self.active_optimizations:
                optimization = self.active_optimizations[opt_name]
                try:
                    model = optimization.remove(model)
                    logger.info(f"Removed optimization: {opt_name}")
                    
                    # Remove from tracking
                    if opt_name in self.active_optimizations:
                        del self.active_optimizations[opt_name]
                    if model_id in self.applied_to_models and opt_name in self.applied_to_models[model_id]:
                        self.applied_to_models[model_id].remove(opt_name)
                except Exception as e:
                    logger.error(f"Failed to remove optimization '{opt_name}': {e}")

        return model

    def _sort_optimizations_by_priority(self, optimization_names: List[str]) -> List[str]:
        """Sort optimizations by their priority."""
        optimizations_with_priority = []
        for name in optimization_names:
            config = self.optimization_configs.get(name)
            if config:
                optimizations_with_priority.append((name, config.priority))
            else:
                optimizations_with_priority.append((name, 0))  # Default priority
        
        # Sort by priority (lower numbers first)
        sorted_opts = sorted(optimizations_with_priority, key=lambda x: x[1])
        return [name for name, priority in sorted_opts]

    def _validate_dependencies(self, optimization_names: List[str]) -> bool:
        """Validate that all dependencies are satisfied."""
        for name in optimization_names:
            config = self.optimization_configs.get(name)
            if config:
                for dep in config.dependencies:
                    if dep not in optimization_names:
                        logger.error(f"Optimization '{name}' depends on '{dep}' which is not in the list")
                        return False
        return True

    def get_model_optimizations(self, model: nn.Module) -> List[str]:
        """Get list of optimizations applied to a specific model."""
        model_id = id(model)
        return self.applied_to_models.get(model_id, [])

    def get_optimization_status(self, name: str) -> Dict[str, Any]:
        """Get status information for a specific optimization."""
        config = self.optimization_configs.get(name)
        is_active = name in self.active_optimizations
        
        return {
            "name": name,
            "configured": config is not None,
            "enabled": config.enabled if config else False,
            "active": is_active,
            "priority": config.priority if config else 0,
            "dependencies": config.dependencies if config else [],
            "parameters": config.parameters if config else {}
        }

    def get_all_optimization_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all optimizations."""
        statuses = {}
        for name in self.get_available_optimizations():
            statuses[name] = self.get_optimization_status(name)
        return statuses

    def update_optimization_config(self, name: str, **kwargs):
        """Update configuration for an optimization."""
        if name in self.optimization_configs:
            config = self.optimization_configs[name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logger.info(f"Updated configuration for optimization: {name}")
        else:
            logger.warning(f"Optimization '{name}' not found in configurations")


# Global optimization manager instance
optimization_manager = ModularOptimizationManager()


def get_optimization_manager() -> ModularOptimizationManager:
    """
    Get the global optimization manager instance.

    Returns:
        ModularOptimizationManager instance
    """
    return optimization_manager


# Optimization implementations
class FlashAttentionOptimization(OptimizationInterface):
    """Optimization for FlashAttention 2.0."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .flash_attention_2 import create_flash_attention_2
            # Apply FlashAttention 2.0 to the model
            # This is a simplified implementation - in practice, you'd replace attention layers
            logger.info("Applying FlashAttention 2.0 optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("FlashAttention 2.0 not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing FlashAttention 2.0 optimization")
        return model


class SparseAttentionOptimization(OptimizationInterface):
    """Optimization for Sparse Attention."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .sparse_attention import create_sparse_attention
            logger.info("Applying Sparse Attention optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Sparse Attention not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Sparse Attention optimization")
        return model


class AdaptiveSparseAttentionOptimization(OptimizationInterface):
    """Optimization for Adaptive Sparse Attention."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .adaptive_sparse_attention import create_adaptive_sparse_attention
            logger.info("Applying Adaptive Sparse Attention optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Adaptive Sparse Attention not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Adaptive Sparse Attention optimization")
        return model


class DiskOffloadingOptimization(OptimizationInterface):
    """Optimization for Disk Offloading."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .disk_offloading import create_disk_offloader
            logger.info("Applying Disk Offloading optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Disk Offloading not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Disk Offloading optimization")
        return model


class ActivationOffloadingOptimization(OptimizationInterface):
    """Optimization for Activation Offloading."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .activation_offloading import create_activation_offloader
            logger.info("Applying Activation Offloading optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Activation Offloading not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Activation Offloading optimization")
        return model


class TensorCompressionOptimization(OptimizationInterface):
    """Optimization for Tensor Compression."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .tensor_compression import get_tensor_compressor
            logger.info("Applying Tensor Compression optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Tensor Compression not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Tensor Compression optimization")
        return model


class StructuredPruningOptimization(OptimizationInterface):
    """Optimization for Structured Pruning."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .structured_pruning import apply_structured_pruning
            logger.info("Applying Structured Pruning optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Structured Pruning not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Structured Pruning optimization")
        return model


class TensorDecompositionOptimization(OptimizationInterface):
    """Optimization for Tensor Decomposition."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .tensor_decomposition import decompose_model_weights
            logger.info("Applying Tensor Decomposition optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Tensor Decomposition not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Tensor Decomposition optimization")
        return model


class SNNOptimization(OptimizationInterface):
    """Optimization for Spiking Neural Networks."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .snn import convert_dense_to_snn
            logger.info("Applying SNN optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("SNN optimization not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing SNN optimization")
        return model


class KernelFusionOptimization(OptimizationInterface):
    """Optimization for Kernel Fusion."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .kernel_fusion import get_kernel_fusion_manager
            logger.info("Applying Kernel Fusion optimization")
            fusion_manager = get_kernel_fusion_manager()
            return fusion_manager.optimize_model(model)
        except ImportError:
            logger.warning("Kernel Fusion not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Kernel Fusion optimization")
        return model  # Cannot easily revert kernel fusion


class DistributedSimulationOptimization(OptimizationInterface):
    """Optimization for Distributed Simulation."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .distributed_simulation import DistributedSimulationManager
            logger.info("Applying Distributed Simulation optimization")
            return model  # Placeholder - actual implementation would modify the model
        except ImportError:
            logger.warning("Distributed Simulation not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Distributed Simulation optimization")
        return model


class AdaptiveBatchingOptimization(OptimizationInterface):
    """Optimization for Adaptive Batching."""

    def apply(self, model: nn.Module) -> nn.Module:
        try:
            from .adaptive_batch_manager import get_adaptive_batch_manager
            logger.info("Applying Adaptive Batching optimization")
            return model  # This optimization typically affects inference logic, not model structure
        except ImportError:
            logger.warning("Adaptive Batching not available")
            return model

    def remove(self, model: nn.Module) -> nn.Module:
        logger.info("Removing Adaptive Batching optimization")
        return model


# Register all optimizations
def register_default_optimizations():
    """Register all default optimizations with the manager."""
    manager = get_optimization_manager()
    
    optimizations = [
        ("flash_attention", FlashAttentionOptimization),
        ("sparse_attention", SparseAttentionOptimization),
        ("adaptive_sparse_attention", AdaptiveSparseAttentionOptimization),
        ("disk_offloading", DiskOffloadingOptimization),
        ("activation_offloading", ActivationOffloadingOptimization),
        ("tensor_compression", TensorCompressionOptimization),
        ("structured_pruning", StructuredPruningOptimization),
        ("tensor_decomposition", TensorDecompositionOptimization),
        ("snn", SNNOptimization),
        ("kernel_fusion", KernelFusionOptimization),
        ("distributed_simulation", DistributedSimulationOptimization),
        ("adaptive_batching", AdaptiveBatchingOptimization),
    ]
    
    for name, cls in optimizations:
        manager.register_optimization(name, cls)


# Register optimizations on module import
register_default_optimizations()


__all__ = [
    "OptimizationType",
    "OptimizationConfig",
    "OptimizationInterface",
    "OptimizationRegistry",
    "ModularOptimizationManager",
    "get_optimization_manager",
    "optimization_manager",
    "register_default_optimizations",
    # Optimization classes
    "FlashAttentionOptimization",
    "SparseAttentionOptimization",
    "AdaptiveSparseAttentionOptimization",
    "DiskOffloadingOptimization",
    "ActivationOffloadingOptimization",
    "TensorCompressionOptimization",
    "StructuredPruningOptimization",
    "TensorDecompositionOptimization",
    "SNNOptimization",
    "KernelFusionOptimization",
    "DistributedSimulationOptimization",
    "AdaptiveBatchingOptimization",
]