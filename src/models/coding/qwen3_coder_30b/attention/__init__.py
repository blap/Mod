"""
Attention module for Qwen3-Coder-30B model.
Exports the specialized Multi-Query Attention implementation.
"""

from .multi_query_attention import MultiQueryAttention, MultiQueryAttentionConfig, create_mqa_layer

__all__ = [
    "MultiQueryAttention",
    "MultiQueryAttentionConfig", 
    "create_mqa_layer"
]