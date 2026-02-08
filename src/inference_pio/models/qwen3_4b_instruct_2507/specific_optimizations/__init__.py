"""
Qwen3-4B-Instruct-2507 Specific Optimizations

This module contains model-specific optimizations for the Qwen3-4B-Instruct-2507 model.
"""

from src.inference_pio.common.interfaces.model_adapter import Qwen34BInstruct2507ModelAdapter
from .qwen3_attention_optimizations import (
    apply_qwen3_attention_optimizations,
    apply_qwen3_gqa_optimizations,
    apply_qwen3_rope_optimizations,
)
from .qwen3_instruction_optimizations import (
    apply_qwen3_generation_optimizations,
    apply_qwen3_instruction_tuning_optimizations,
)
from .qwen3_kv_cache_optimizations import (
    apply_qwen3_compressed_kv_cache,
    apply_qwen3_kv_cache_optimizations,
)

__all__ = [
    "apply_qwen3_attention_optimizations",
    "apply_qwen3_gqa_optimizations",
    "apply_qwen3_rope_optimizations",
    "apply_qwen3_kv_cache_optimizations",
    "apply_qwen3_compressed_kv_cache",
    "apply_qwen3_instruction_tuning_optimizations",
    "apply_qwen3_generation_optimizations",
    "Qwen34BInstruct2507ModelAdapter",
]
