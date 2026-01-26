"""
Qwen3-VL-2B Sliding Window Attention Implementation

This module implements sliding window attention for the Qwen3-VL-2B model using the common implementation.
"""

from typing import Optional

import torch
import torch.nn as nn

from ....common.sliding_window_attention import SlidingWindowAttention as BaseSlidingWindowAttention, create_sliding_window_attention
from ..config import Qwen3VL2BConfig


class Qwen3VLSlidingWindowAttention(BaseSlidingWindowAttention):
    """
    Qwen3-VL-2B specific sliding window attention implementation using the common base implementation.

    This implementation provides optimized attention computation with limited context window
    to reduce memory usage and improve performance compared to standard attention mechanisms.
    It inherits from the common SlidingWindowAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine sliding window parameters from config
        num_attention_heads = getattr(config, 'num_attention_heads', 16)
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)
        is_causal = getattr(config, 'is_causal', True)
        sliding_window_size = getattr(config, 'sliding_window_size', 4096)
        use_flash_attention = getattr(config, 'use_flash_attention_2', True)

        # Initialize using the common SlidingWindowAttention implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
            sliding_window_size=sliding_window_size,
            use_flash_attention=use_flash_attention
        )
        self.config = config
        self.layer_idx = layer_idx


def create_qwen3_vl_sliding_window_attention(config: Qwen3VL2BConfig, layer_idx: Optional[int] = None):
    """
    Factory function to create Qwen3-VL-2B sliding window attention implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VLSlidingWindowAttention: The Qwen3-VL-2B sliding window attention implementation
    """
    return Qwen3VLSlidingWindowAttention(config, layer_idx)


__all__ = [
    "Qwen3VLSlidingWindowAttention",
    "create_qwen3_vl_sliding_window_attention"
]