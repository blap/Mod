"""
GLM-4.7 Specific Optimizations Package

This module provides the initialization for GLM-4.7 specific optimizations
in the Inference-PIO system.
"""

from .glm47_specific_optimizations import (
    apply_glm47_specific_optimizations,
    get_glm47_optimization_report,
    GLM47OptimizationConfig
)


__all__ = [
    "apply_glm47_specific_optimizations",
    "get_glm47_optimization_report",
    "GLM47OptimizationConfig"
]