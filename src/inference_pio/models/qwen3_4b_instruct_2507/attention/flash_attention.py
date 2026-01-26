"""
Qwen3-4B-Instruct-2507 Flash Attention Implementation

This module implements FlashAttention 2.0 for the Qwen3-4B-Instruct-2507 model using the common implementation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.flash_attention_2 import FlashAttention2, create_flash_attention_2
from ..config import Qwen34BInstruct2507Config


class Qwen34BFlashAttention2(FlashAttention2):
    """
    Qwen3-4B-Instruct-2507 specific FlashAttention 2.0 implementation using the common base implementation.

    This implementation provides optimized attention computation with reduced memory usage
    and improved performance compared to standard attention mechanisms. It inherits from
    the common FlashAttention2 implementation to ensure consistency across models.
    """

    def __init__(self, config: Qwen34BInstruct2507Config, layer_idx: Optional[int] = None):
        # Initialize using the common FlashAttention2 implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            attention_dropout=getattr(config, 'attention_dropout_prob', 0.0),
            bias=False,  # Qwen3-4B typically uses bias=False
            is_causal=True  # Qwen3-4B is typically autoregressive
        )
        self.config = config
        self.layer_idx = layer_idx


def create_qwen3_4b_flash_attention_2(config: Qwen34BInstruct2507Config, layer_idx: Optional[int] = None):
    """
    Factory function to create Qwen3-4B-Instruct-2507 FlashAttention 2.0 implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        Qwen34BFlashAttention2: The Qwen3-4B-Instruct-2507 FlashAttention 2.0 implementation
    """
    return Qwen34BFlashAttention2(config, layer_idx)


__all__ = [
    "Qwen34BFlashAttention2",
    "create_qwen3_4b_flash_attention_2"
]