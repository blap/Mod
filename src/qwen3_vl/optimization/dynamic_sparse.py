"""
Dynamic sparse attention components for Qwen3-VL.
This module now imports the canonical implementation from src.qwen3_vl.attention.dynamic_sparse_attention
to avoid code duplication.
"""
from src.qwen3_vl.attention.dynamic_sparse_attention import DynamicSparseAttention, VisionDynamicSparseAttention

__all__ = ['DynamicSparseAttention', 'VisionDynamicSparseAttention']
