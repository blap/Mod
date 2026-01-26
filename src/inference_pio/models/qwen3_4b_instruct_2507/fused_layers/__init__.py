"""
Qwen3-4B-Instruct-2507 Fused Layers Module

This module provides fused layer implementations for the Qwen3-4B-Instruct-2507 model.
"""

from .fused_layer_norm import replace_layer_norm_in_model

__all__ = [
    "replace_layer_norm_in_model"
]