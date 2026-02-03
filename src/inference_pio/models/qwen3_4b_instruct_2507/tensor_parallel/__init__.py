"""
Qwen3-4B-Instruct-2507 Tensor Parallelism Module

This module provides tensor parallelism functionality for the Qwen3-4B-Instruct-2507 model.
"""

from .tensor_parallel_layers import (
    TensorParallelConfig,
    safe_convert_to_tensor_parallel,
)

__all__ = ["TensorParallelConfig", "safe_convert_to_tensor_parallel"]
