"""
Attention module for Qwen3-0.6B model.
Exports the specialized Sparse Attention implementation.
"""

from .sparse_attention import SparseAttention, SparseAttentionConfig, create_sparse_attention_layer

__all__ = [
    "SparseAttention",
    "SparseAttentionConfig", 
    "create_sparse_attention_layer"
]