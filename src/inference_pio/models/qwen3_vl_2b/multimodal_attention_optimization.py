"""
Multimodal Attention Optimization System - Compatibility Module

This module provides compatibility for older imports by aliasing the generic implementation
to the Qwen3-VL-2B specific names. New implementations should use the generic version
and extend it as needed.
"""

from ...common.generic_multimodal_attention_optimization import (
    GenericMultimodalAttentionOptimizer as Qwen3VL2BMultimodalAttentionOptimizer,
    GenericMultimodalAttentionManager as Qwen3VL2BAttentionManager,
    create_generic_multimodal_attention_optimizer as create_qwen3_vl_multimodal_attention_optimizer,
    apply_generic_multimodal_attention_optimizations_to_model as apply_multimodal_attention_optimizations_to_model,
    get_generic_multimodal_attention_optimization_report as get_multimodal_attention_optimization_report
)

__all__ = [
    "Qwen3VL2BMultimodalAttentionOptimizer",
    "Qwen3VL2BAttentionManager",
    "create_qwen3_vl_multimodal_attention_optimizer",
    "apply_multimodal_attention_optimizations_to_model",
    "get_multimodal_attention_optimization_report"
]
