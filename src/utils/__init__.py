"""
Unified Utility Module for Inference-PIO System

This module consolidates all utility functions to prevent code duplication
and provide a centralized location for common utilities.
"""

from .cuda_kernels import *
from .error_handling import *
from .rotary_embeddings import *

# Import all utilities to make them available at the package level
from .snn_utils import *
from .tensor_utils import *

# Only import consolidated_test_utilities if it exists
try:
    from ..common.consolidated_test_utilities import *
except ImportError:
    pass

__all__ = [
    # SNN Utils exports
    "estimate_energy_savings",
    "calculate_spike_rate",
    "convert_to_spike_tensor",
    "temporal_encode_spike",
    "rate_encode_spike",
    "reset_neuron_state",
    "count_spikes_in_model",
    "optimize_snn_for_energy",
    # Tensor Utils exports
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_with_position_ids",
    "rotate_half",
    "repeat_kv",
    "apply_chunking_to_forward",
    "gelu_new",
    "silu",
    "swish",
    "softmax_with_temperature",
    "masked_fill_with_broadcast",
    "normalize_with_l2",
    "pad_sequence_to_length",
    "truncate_sequence_to_length",
    "safe_tensor_operation",
    "validate_tensor_shape",
    "calculate_tensor_memory",
    "move_tensor_to_device",
    "concatenate_tensors",
    "stack_tensors",
    # Rotary Embeddings exports
    "GenericRotaryEmbedding",
    "Qwen3RotaryEmbedding",
    "Qwen34BRotaryEmbedding",
    "Qwen3CoderRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "create_generic_rotary_embedding",
    "create_qwen3_rotary_embedding",
    "apply_rotary_embeddings_to_model",
    # Error Handling exports
    "ValidationError",
    "BenchmarkError",
    "TestError",
    "ModelError",
    "validate_input",
    "validate_positive_number",
    "validate_range",
    "safe_execute",
    "error_handler",
    "retry_on_failure",
    "validate_model_plugin",
    "validate_benchmark_result",
    "validate_tensor_dimensions",
    "validate_tensor_shape",
    "handle_plugin_initialization",
    "handle_model_loading",
    "validate_and_clean_text",
    "validate_device",
    "validate_config",
    "validate_list_items",
    "validate_dict_keys",
    "check_memory_usage",
    "validate_tensor_values",
    "safe_model_operation",
    "example_safe_benchmark_operation",
    "example_validated_test_helper",
    # CUDA Kernels exports
    "AttentionKernel",
    "MLPKernel",
    "LayerNormKernel",
    "RMSNormKernel",
    "HardwareOptimizer",
    "create_cuda_kernels",
    "apply_cuda_optimizations_to_model",
    "get_cuda_optimization_report",
    "fused_add_norm",
    "apply_rotary_embeddings_to_qkv",
]
