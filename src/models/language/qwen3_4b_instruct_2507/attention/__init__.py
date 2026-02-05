"""
Attention module for Qwen3-4B-Instruct-2507 model.
Exports the specialized Grouped Query Attention implementation.
"""

from .grouped_query_attention import GroupedQueryAttention, GroupedQueryAttentionConfig, create_gqa_layer

__all__ = [
    "GroupedQueryAttention",
    "GroupedQueryAttentionConfig", 
    "create_gqa_layer"
]