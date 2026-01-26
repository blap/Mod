"""
Qwen3-VL-2B Flash Attention Implementation

This module implements FlashAttention 2.0 for the Qwen3-VL-2B model using the common implementation.
"""

from typing import Optional

import torch
import torch.nn as nn

from ....common.flash_attention_2 import FlashAttention2, create_flash_attention_2
from ..config import Qwen3VL2BConfig


class Qwen3VL2BFlashAttention2(FlashAttention2):
    """
    Qwen3-VL-2B specific FlashAttention 2.0 implementation using the common base implementation.

    This implementation provides optimized attention computation with reduced memory usage
    and improved performance compared to standard attention mechanisms. It inherits from
    the common FlashAttention2 implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine FlashAttention parameters from config
        num_attention_heads = getattr(config, 'num_attention_heads', 16)
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)
        is_causal = getattr(config, 'is_causal', True)

        # Initialize using the common FlashAttention2 implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal
        )
        self.config = config
        self.layer_idx = layer_idx


def create_qwen3_vl_flash_attention_2(config: Qwen3VL2BConfig, layer_idx: Optional[int] = None):
    """
    Factory function to create Qwen3-VL-2B FlashAttention 2.0 implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VL2BFlashAttention2: The Qwen3-VL-2B FlashAttention 2.0 implementation
    """
    return Qwen3VL2BFlashAttention2(config, layer_idx)


__all__ = [
    "Qwen3VL2BFlashAttention2",
    "create_qwen3_vl_flash_attention_2"
]