"""
Qwen3-VL-2B Multi-Query and Grouped-Query Attention Implementation

This module implements Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) for the Qwen3-VL-2B model using the common implementation.
"""

from typing import Optional

import torch
import torch.nn as nn

from ....common.multi_query_attention import MultiQueryAttention as BaseMultiQueryAttention, GroupedQueryAttention as BaseGroupedQueryAttention
from ..config import Qwen3VL2BConfig


class Qwen3VLMultiQueryAttention(BaseMultiQueryAttention):
    """
    Qwen3-VL-2B specific Multi-Query Attention implementation using the common base implementation.

    This implementation provides optimized attention computation with reduced memory usage
    and improved performance compared to standard attention mechanisms. It inherits from
    the common MultiQueryAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine MQA parameters from config
        num_attention_heads = getattr(config, 'num_attention_heads', 16)
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)
        is_causal = getattr(config, 'is_causal', True)
        use_sliding_window = getattr(config, 'use_sliding_window_attention', False)
        sliding_window_size = getattr(config, 'sliding_window_size', 4096)

        # Initialize using the common MultiQueryAttention implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
            use_sliding_window=use_sliding_window,
            sliding_window_size=sliding_window_size
        )
        self.config = config
        self.layer_idx = layer_idx


class Qwen3VLGroupedQueryAttention(BaseGroupedQueryAttention):
    """
    Qwen3-VL-2B specific Grouped-Query Attention implementation using the common base implementation.

    This implementation provides optimized attention computation with reduced memory usage
    and improved performance compared to standard attention mechanisms. It inherits from
    the common GroupedQueryAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine GQA parameters from config
        num_attention_heads = getattr(config, 'num_attention_heads', 16)
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)
        is_causal = getattr(config, 'is_causal', True)
        use_sliding_window = getattr(config, 'use_sliding_window_attention', False)
        sliding_window_size = getattr(config, 'sliding_window_size', 4096)

        # Initialize using the common GroupedQueryAttention implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
            use_sliding_window=use_sliding_window,
            sliding_window_size=sliding_window_size
        )
        self.config = config
        self.layer_idx = layer_idx


def create_mqa_gqa_attention(config: Qwen3VL2BConfig, layer_idx: Optional[int] = None):
    """
    Factory function to create Qwen3-VL-2B MQA/GQA attention implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VLMultiQueryAttention or Qwen3VLGroupedQueryAttention: The Qwen3-VL-2B MQA/GQA attention implementation
    """
    # Determine if we should use GQA or MQA based on config
    use_grouped_query_attention = getattr(config, 'use_grouped_query_attention', True)
    
    if use_grouped_query_attention:
        return Qwen3VLGroupedQueryAttention(config, layer_idx)
    else:
        return Qwen3VLMultiQueryAttention(config, layer_idx)


__all__ = [
    "Qwen3VLMultiQueryAttention",
    "Qwen3VLGroupedQueryAttention", 
    "create_mqa_gqa_attention"
]