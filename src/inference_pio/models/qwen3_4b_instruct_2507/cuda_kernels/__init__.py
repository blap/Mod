"""
Qwen3-4B-Instruct-2507 CUDA Kernels Optimization

This module provides CUDA kernel optimizations for the Qwen3-4B-Instruct-2507 model.
"""

from typing import Optional
import torch
import torch.nn as nn

from ..config import Qwen34BInstruct2507Config
from .qwen3_4b_specific_kernels import (
    apply_qwen3_4b_specific_cuda_optimizations_to_model,
    get_qwen3_4b_cuda_optimization_report,
    Qwen34BAttentionKernel,
    Qwen34BMLPKernel,
    Qwen34BRMSNormKernel,
    Qwen34BHardwareOptimizer,
    create_qwen3_4b_cuda_kernels
)


def apply_qwen3_4b_optimizations_to_model(model: nn.Module, config: Qwen34BInstruct2507Config) -> nn.Module:
    """
    Apply Qwen3-4B-Instruct-2507 specific CUDA optimizations to the model.

    Args:
        model: The model to optimize
        config: Model configuration

    Returns:
        Optimized model
    """
    # Apply Qwen3-4B specific optimizations using the detailed implementation
    optimized_model = apply_qwen3_4b_specific_cuda_optimizations_to_model(
        model=model,
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        config=config
    )

    return optimized_model


def get_qwen3_4b_optimization_report(model: nn.Module, config: Qwen34BInstruct2507Config) -> dict:
    """
    Get a report of Qwen3-4B optimizations applied to the model.

    Args:
        model: The model to analyze
        config: Model configuration

    Returns:
        Optimization report
    """
    return get_qwen3_4b_cuda_optimization_report(
        model=model,
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        config=config
    )


__all__ = [
    "apply_qwen3_4b_optimizations_to_model",
    "get_qwen3_4b_optimization_report",
    "Qwen34BAttentionKernel",
    "Qwen34BMLPKernel",
    "Qwen34BRMSNormKernel",
    "Qwen34BHardwareOptimizer",
    "create_qwen3_4b_cuda_kernels"
]