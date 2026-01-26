"""
Qwen3-VL-2B Paged Attention Implementation

This module implements paged attention for the Qwen3-VL-2B model using the common implementation.
"""

from typing import Optional

import torch
import torch.nn as nn

from ....common.paged_attention import PagedAttention, create_paged_attention
from ..config import Qwen3VL2BConfig


class Qwen3VL2BPagedAttention(PagedAttention):
    """
    Qwen3-VL-2B specific paged attention implementation using the common base implementation.

    This implementation provides optimized attention computation with memory-efficient paging
    to reduce memory usage and improve performance compared to standard attention mechanisms.
    It inherits from the common PagedAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
        page_size: int = 256,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096
    ):
        # Determine paged attention parameters from config
        num_attention_heads = getattr(config, 'num_attention_heads', 16)
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)
        is_causal = getattr(config, 'is_causal', True)

        # Initialize using the common PagedAttention implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
            page_size=page_size,
            use_sliding_window=use_sliding_window,
            sliding_window_size=sliding_window_size
        )
        self.config = config
        self.layer_idx = layer_idx


def create_qwen3_vl_paged_attention(
    config: Qwen3VL2BConfig, 
    layer_idx: Optional[int] = None,
    page_size: int = 256,
    use_sliding_window: bool = False,
    sliding_window_size: int = 4096
):
    """
    Factory function to create Qwen3-VL-2B paged attention implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)
        page_size: Size of each page in tokens
        use_sliding_window: Whether to use sliding window attention with paging
        sliding_window_size: Size of the sliding window if using sliding window attention

    Returns:
        Qwen3VL2BPagedAttention: The Qwen3-VL-2B paged attention implementation
    """
    return Qwen3VL2BPagedAttention(
        config, 
        layer_idx, 
        page_size, 
        use_sliding_window, 
        sliding_window_size
    )


__all__ = [
    "Qwen3VL2BPagedAttention",
    "create_qwen3_vl_paged_attention"
]