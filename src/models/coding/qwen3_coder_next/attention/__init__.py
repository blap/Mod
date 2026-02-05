"""
Attention module for Qwen3-Coder-Next model.
Exports the specialized Sliding Window Attention implementation.
"""

from .sliding_window_attention import SlidingWindowAttention, SlidingWindowAttentionConfig, create_sliding_window_attention_layer

__all__ = [
    "SlidingWindowAttention",
    "SlidingWindowAttentionConfig", 
    "create_sliding_window_attention_layer"
]