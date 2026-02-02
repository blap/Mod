"""
Qwen3-Coder-30B Specific Optimizations Package

This module provides the initialization for Qwen3-Coder-30B specific optimizations
in the Inference-PIO system.
"""

from .qwen3_coder_specific_optimizations import (
    Qwen3CoderOptimizationConfig,
    apply_qwen3_coder_specific_optimizations,
    get_qwen3_coder_optimization_report,
)

__all__ = [
    "apply_qwen3_coder_specific_optimizations",
    "get_qwen3_coder_optimization_report",
    "Qwen3CoderOptimizationConfig",
]
