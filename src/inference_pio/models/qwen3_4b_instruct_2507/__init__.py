"""
Qwen3-4B-Instruct-2507 Package - Self-Contained Version

This module provides the initialization for the Qwen3-4B-Instruct-2507 model package
in the self-contained plugin architecture for the Inference-PIO system.
"""

from .plugin import Qwen3_4B_Instruct_2507_Plugin, create_qwen3_4b_instruct_2507_plugin
from .model import Qwen34BInstruct2507Model
from .config import Qwen34BInstruct2507Config
from .specific_optimizations.qwen3_attention_optimizations import (
    apply_qwen3_attention_optimizations,
    apply_qwen3_gqa_optimizations,
    apply_qwen3_rope_optimizations
)
from .specific_optimizations.qwen3_kv_cache_optimizations import (
    apply_qwen3_kv_cache_optimizations,
    apply_qwen3_compressed_kv_cache
)
from .specific_optimizations.qwen3_instruction_optimizations import (
    apply_qwen3_instruction_tuning_optimizations,
    apply_qwen3_generation_optimizations,
    enhance_qwen3_instruction_following_capability
)


__all__ = [
    "Qwen3_4B_Instruct_2507_Plugin",
    "create_qwen3_4b_instruct_2507_plugin",
    "Qwen34BInstruct2507Model",
    "Qwen34BInstruct2507Config",
    # Qwen3-specific optimization functions
    "apply_qwen3_attention_optimizations",
    "apply_qwen3_gqa_optimizations",
    "apply_qwen3_rope_optimizations",
    "apply_qwen3_kv_cache_optimizations",
    "apply_qwen3_compressed_kv_cache",
    "apply_qwen3_instruction_tuning_optimizations",
    "apply_qwen3_generation_optimizations",
    "enhance_qwen3_instruction_following_capability"
]