"""
Qwen3-4B-Instruct-2507 Attention Modules

This module provides the attention implementations for the Qwen3-4B-Instruct-2507 model.
"""

from .flash_attention import create_qwen3_4b_flash_attention_2
from .multi_query_attention import create_mqa_gqa_attention
from .paged_attention import create_qwen3_4b_paged_attention
from .sliding_window_attention import create_qwen3_4b_sliding_window_attention
from .sparse_attention import create_qwen3_4b_sparse_attention

__all__ = [
    "create_qwen3_4b_flash_attention_2",
    "create_qwen3_4b_sparse_attention",
    "create_qwen3_4b_paged_attention",
    "create_qwen3_4b_sliding_window_attention",
    "create_mqa_gqa_attention",
]
