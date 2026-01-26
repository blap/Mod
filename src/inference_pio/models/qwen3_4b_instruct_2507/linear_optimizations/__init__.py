"""
Qwen3-4B-Instruct-2507 Linear Optimizations Module

This module provides linear layer optimizations for the Qwen3-4B-Instruct-2507 model.
"""

from .bias_removal import apply_bias_removal_to_model

__all__ = [
    "apply_bias_removal_to_model"
]