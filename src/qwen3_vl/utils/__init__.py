"""
Utilities package for Qwen3-VL model.

This package provides commonly used utility functions for tensor operations,
general utilities, and other helper functions across the codebase.
"""

from .tensor_utils import (
    repeat_kv, rotate_half, apply_rotary_pos_emb, apply_rotary_pos_emb_with_position_ids, 
    compute_attention_scores, mask_attention_scores, softmax_with_dtype, 
    apply_attention_weights, reshape_for_output
)

from .general_utils import (
    get_logger, validate_tensor_shape, safe_tensor_operation, get_available_memory, 
    get_tensor_memory_size, create_tensor_from_config, get_nested_attribute, 
    set_nested_attribute, merge_dicts, get_config_path, load_config, save_config, 
    time_it, temporary_seed, get_device_count, get_device_name
)

from .cuda_error_handler import (
    CUDAErrorHandler, MemoryPoolManager, apply_hardware_specific_optimizations,
    get_hardware_optimization_recommendations
)

__all__ = [
    # Tensor utilities
    'repeat_kv',
    'rotate_half',
    'apply_rotary_pos_emb',
    'apply_rotary_pos_emb_with_position_ids',
    'compute_attention_scores',
    'mask_attention_scores',
    'softmax_with_dtype',
    'apply_attention_weights',
    'reshape_for_output',
    
    # General utilities
    'get_logger',
    'validate_tensor_shape',
    'safe_tensor_operation',
    'get_available_memory',
    'get_tensor_memory_size',
    'create_tensor_from_config',
    'get_nested_attribute',
    'set_nested_attribute',
    'merge_dicts',
    'get_config_path',
    'load_config',
    'save_config',
    'time_it',
    'temporary_seed',
    'get_device_count',
    'get_device_name',
    
    # CUDA error handling utilities
    'CUDAErrorHandler',
    'MemoryPoolManager',
    'apply_hardware_specific_optimizations',
    'get_hardware_optimization_recommendations'
]