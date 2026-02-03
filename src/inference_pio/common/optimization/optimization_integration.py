"""
Optimization Integration Utilities for Inference-PIO

This module provides utilities to integrate the modular optimization system
with existing model implementations.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .optimization_config import (
    ModelFamily,
    ModelOptimizationConfig,
    get_config_manager,
)
from .optimization_manager import OptimizationConfig, get_optimization_manager

logger = logging.getLogger(__name__)


def apply_model_family_optimizations(
    model: nn.Module, model_family: ModelFamily, profile_name: Optional[str] = None
) -> nn.Module:
    """
    Apply optimizations specific to a model family using the configuration system.

    Args:
        model: Model to apply optimizations to
        model_family: Family of the model
        profile_name: Name of the global profile to use (optional)

    Returns:
        Optimized model with family-specific optimizations applied
    """
    config_manager = get_config_manager()
    opt_manager = get_optimization_manager()

    # Set the active profile if specified
    if profile_name:
        if not config_manager.set_active_profile(profile_name):
            logger.warning(
                f"Profile '{profile_name}' not found, using default settings"
            )

    # Get model-specific configuration
    model_config = config_manager.apply_profile_to_model_config(model_family)
    if not model_config:
        logger.warning(f"No configuration found for model family: {model_family}")
        return model

    # Configure optimizations based on model config
    for opt_config in model_config.optimizations:
        opt_manager.configure_optimization(opt_config.name, opt_config)

    # Apply optimizations in priority order
    optimization_names = [opt.name for opt in model_config.optimizations if opt.enabled]

    if model_config.priority_order:
        # Sort by priority order specified in config
        optimization_names = [
            name for name in model_config.priority_order if name in optimization_names
        ]

    logger.info(
        f"Applying optimizations to {model_family.value} model: {optimization_names}"
    )
    return opt_manager.apply_optimizations(model, optimization_names)


def apply_optimizations_by_config(
    model: nn.Module, config: ModelOptimizationConfig
) -> nn.Module:
    """
    Apply optimizations to a model based on a configuration object.

    Args:
        model: Model to apply optimizations to
        config: Configuration specifying which optimizations to apply

    Returns:
        Optimized model with specified optimizations applied
    """
    opt_manager = get_optimization_manager()

    # Configure optimizations based on config
    for opt_config in config.optimizations:
        opt_manager.configure_optimization(opt_config.name, opt_config)

    # Apply optimizations in priority order
    optimization_names = [opt.name for opt in config.optimizations if opt.enabled]

    if config.priority_order:
        # Sort by priority order specified in config
        optimization_names = [
            name for name in config.priority_order if name in optimization_names
        ]

    logger.info(f"Applying optimizations: {optimization_names}")
    return opt_manager.apply_optimizations(model, optimization_names)


def get_model_optimization_status(model: nn.Module) -> Dict[str, Any]:
    """
    Get the optimization status for a specific model.

    Args:
        model: Model to get status for

    Returns:
        Dictionary with detailed optimization status information for the model
    """
    opt_manager = get_optimization_manager()

    applied_opts = opt_manager.get_model_optimizations(model)
    all_statuses = opt_manager.get_all_optimization_statuses()

    return {
        "model_id": id(model),
        "applied_optimizations": applied_opts,
        "optimization_details": {
            name: status
            for name, status in all_statuses.items()
            if name in applied_opts
        },
        "total_optimizations_available": len(all_statuses),
    }


def update_model_optimization(
    model: nn.Module,
    optimization_name: str,
    enabled: bool,
    parameters: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Update a specific optimization on a model.

    Args:
        model: Model to update optimization for
        optimization_name: Name of the optimization to update
        enabled: Whether to enable or disable the optimization
        parameters: New parameters for the optimization (optional)

    Returns:
        Updated model with the specified optimization modified
    """
    opt_manager = get_optimization_manager()

    # Get current config and update it
    if optimization_name in opt_manager.optimization_configs:
        config = opt_manager.optimization_configs[optimization_name]
        config.enabled = enabled
        if parameters:
            config.parameters.update(parameters)
    else:
        # Create new config if it doesn't exist
        config = OptimizationConfig(
            name=optimization_name, enabled=enabled, parameters=parameters or {}
        )
        opt_manager.configure_optimization(optimization_name, config)

    # Apply or remove the optimization as needed
    if enabled:
        return opt_manager.apply_optimizations(model, [optimization_name])
    else:
        return opt_manager.remove_optimizations(model, [optimization_name])


def create_optimization_pipeline(
    model_family: ModelFamily,
    profile_name: str = "balanced",
    custom_configs: Optional[List[OptimizationConfig]] = None,
) -> callable:
    """
    Create an optimization pipeline function for a specific model family.

    Args:
        model_family: Family of the model
        profile_name: Name of the profile to use
        custom_configs: Custom optimization configurations to add/override

    Returns:
        Function that takes a model and returns an optimized model for the specified family
    """

    def optimization_pipeline(model: nn.Module) -> nn.Module:
        # Apply model family optimizations
        optimized_model = apply_model_family_optimizations(
            model, model_family, profile_name
        )

        # Apply any custom configurations
        if custom_configs:
            opt_manager = get_optimization_manager()

            for config in custom_configs:
                opt_manager.configure_optimization(config.name, config)

            custom_opt_names = [cfg.name for cfg in custom_configs if cfg.enabled]
            optimized_model = opt_manager.apply_optimizations(
                optimized_model, custom_opt_names
            )

        return optimized_model

    return optimization_pipeline


def get_supported_optimizations() -> List[str]:
    """
    Get list of all supported optimizations.

    Returns:
        List of supported optimization names available in the system
    """
    opt_manager = get_optimization_manager()
    return opt_manager.get_available_optimizations()


def reset_model_optimizations(model: nn.Module) -> nn.Module:
    """
    Remove all optimizations from a model.

    Args:
        model: Model to remove optimizations from

    Returns:
        Model with all optimizations removed, returning to original state
    """
    opt_manager = get_optimization_manager()
    applied_opts = opt_manager.get_model_optimizations(model)

    if applied_opts:
        logger.info(f"Removing optimizations from model: {applied_opts}")
        return opt_manager.remove_optimizations(model, applied_opts)

    return model


# Convenience functions for specific model families
def apply_glm_optimizations(
    model: nn.Module, profile_name: str = "balanced"
) -> nn.Module:
    """Apply GLM-specific optimizations to a model."""
    return apply_model_family_optimizations(model, ModelFamily.GLM, profile_name)


def apply_qwen_optimizations(
    model: nn.Module, profile_name: str = "balanced"
) -> nn.Module:
    """Apply Qwen-specific optimizations to a model."""
    return apply_model_family_optimizations(model, ModelFamily.QWEN, profile_name)


# Backward compatibility functions for existing code
def legacy_apply_flash_attention(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply FlashAttention, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="flash_attention",
        enabled=True,
        optimization_type=OptimizationType.ATTENTION,
        parameters={
            "use_triton": getattr(config, "use_flash_attention_2", True),
            "attention_dropout": getattr(config, "attention_dropout", 0.0),
        },
    )

    opt_manager.configure_optimization("flash_attention", opt_config)
    return opt_manager.apply_optimizations(model, ["flash_attention"])


def legacy_apply_sparse_attention(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply SparseAttention, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="sparse_attention",
        enabled=getattr(config, "use_sparse_attention", False),
        optimization_type=OptimizationType.ATTENTION,
        parameters={
            "sparsity_ratio": getattr(config, "sparse_attention_sparsity_ratio", 0.25),
            "block_size": getattr(config, "sparse_attention_block_size", 64),
        },
    )

    opt_manager.configure_optimization("sparse_attention", opt_config)
    return opt_manager.apply_optimizations(model, ["sparse_attention"])


def legacy_apply_disk_offloading(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply DiskOffloading, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="disk_offloading",
        enabled=getattr(config, "enable_disk_offloading", False),
        optimization_type=OptimizationType.MEMORY,
        parameters={
            "max_memory_ratio": getattr(config, "max_memory_ratio", 0.8),
            "offload_directory": getattr(config, "offload_directory", None),
            "page_size_mb": getattr(config, "page_size_mb", 16),
        },
    )

    opt_manager.configure_optimization("disk_offloading", opt_config)
    return opt_manager.apply_optimizations(model, ["disk_offloading"])


def legacy_apply_activation_offloading(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply ActivationOffloading, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="activation_offloading",
        enabled=getattr(config, "enable_activation_offloading", False),
        optimization_type=OptimizationType.ACTIVATION,
        parameters={
            "max_memory_ratio": getattr(config, "activation_max_memory_ratio", 0.7),
            "offload_directory": getattr(config, "activation_offload_directory", None),
            "page_size_mb": getattr(config, "activation_page_size_mb", 8),
        },
    )

    opt_manager.configure_optimization("activation_offloading", opt_config)
    return opt_manager.apply_optimizations(model, ["activation_offloading"])


def legacy_apply_tensor_compression(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply TensorCompression, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="tensor_compression",
        enabled=getattr(config, "enable_tensor_compression", False),
        optimization_type=OptimizationType.COMPUTE,
        parameters={
            "compression_method": getattr(
                config, "tensor_compression_method", "incremental_pca"
            ),
            "compression_ratio": getattr(config, "tensor_compression_ratio", 0.5),
            "max_components": getattr(config, "tensor_compression_max_components", 256),
        },
    )

    opt_manager.configure_optimization("tensor_compression", opt_config)
    return opt_manager.apply_optimizations(model, ["tensor_compression"])


def legacy_apply_structured_pruning(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply StructuredPruning, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="structured_pruning",
        enabled=getattr(config, "use_structured_pruning", False),
        optimization_type=OptimizationType.MODEL_STRUCTURE,
        parameters={
            "pruning_ratio": getattr(config, "pruning_ratio", 0.2),
            "method": getattr(config, "pruning_method", "layer_removal"),
            "block_size": getattr(config, "pruning_block_size", 1),
        },
    )

    opt_manager.configure_optimization("structured_pruning", opt_config)
    return opt_manager.apply_optimizations(model, ["structured_pruning"])


def legacy_apply_kernel_fusion(model: nn.Module, config: Any) -> nn.Module:
    """Legacy function to apply KernelFusion, now using the new system."""
    from .optimization_manager import (
        OptimizationConfig,
        OptimizationType,
        get_optimization_manager,
    )

    opt_manager = get_optimization_manager()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="kernel_fusion",
        enabled=getattr(config, "enable_kernel_fusion", False),
        optimization_type=OptimizationType.COMPUTE,
        parameters={
            "fusion_patterns": getattr(config, "kernel_fusion_patterns", []),
            "use_custom_cuda_kernels": getattr(config, "use_custom_cuda_kernels", True),
        },
    )

    opt_manager.configure_optimization("kernel_fusion", opt_config)
    return opt_manager.apply_optimizations(model, ["kernel_fusion"])


logger.info("Optimization integration utilities loaded successfully")


__all__ = [
    "apply_model_family_optimizations",
    "apply_optimizations_by_config",
    "get_model_optimization_status",
    "update_model_optimization",
    "create_optimization_pipeline",
    "get_supported_optimizations",
    "reset_model_optimizations",
    # Model family specific functions
    "apply_glm_optimizations",
    "apply_qwen_optimizations",
    # Legacy compatibility functions
    "legacy_apply_flash_attention",
    "legacy_apply_sparse_attention",
    "legacy_apply_disk_offloading",
    "legacy_apply_activation_offloading",
    "legacy_apply_tensor_compression",
    "legacy_apply_structured_pruning",
    "legacy_apply_kernel_fusion",
]
