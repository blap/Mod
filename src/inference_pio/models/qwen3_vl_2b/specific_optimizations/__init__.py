"""
Qwen3-VL-2B Specific Optimizations Package

This module provides the initialization for Qwen3-VL-2B specific optimizations
in the Inference-PIO system.
"""

from .qwen3_vl_specific_optimizations import (
    apply_qwen3_vl_specific_optimizations,
    get_qwen3_vl_optimization_report,
    Qwen3VLOptimizationConfig
)


__all__ = [
    "apply_qwen3_vl_specific_optimizations",
    "get_qwen3_vl_optimization_report",
    "Qwen3VLOptimizationConfig"
]