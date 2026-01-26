"""
Qwen3-4B-Instruct-2507 KV Cache Module

This module provides KV cache implementations for the Qwen3-4B-Instruct-2507 model.
"""

from .compression_techniques import apply_compressed_kv_cache_to_model

__all__ = [
    "apply_compressed_kv_cache_to_model"
]