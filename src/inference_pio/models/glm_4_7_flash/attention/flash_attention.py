"""
GLM-4.7 Flash Attention Implementation

This module implements FlashAttention 2.0 for the GLM-4.7 model using the common implementation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.flash_attention_2 import FlashAttention2, create_flash_attention_2
from ..config import GLM47Config


class GLM47FlashAttention2(FlashAttention2):
    """
    GLM-4.7 specific FlashAttention 2.0 implementation using the common base implementation.

    This implementation provides optimized attention computation with reduced memory usage
    and improved performance compared to standard attention mechanisms. It inherits from
    the common FlashAttention2 implementation to ensure consistency across models.
    """

    def __init__(self, config: GLM47Config, layer_idx: Optional[int] = None):
        # Initialize using the common FlashAttention2 implementation
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            attention_dropout=getattr(config, 'attention_dropout_prob', 0.0),
            bias=False,  # GLM-4.7 typically uses bias=False
            is_causal=True  # GLM-4.7 is typically autoregressive
        )
        self.config = config
        self.layer_idx = layer_idx


def create_glm47_flash_attention_2(config: GLM47Config, layer_idx: Optional[int] = None):
    """
    Factory function to create GLM-4.7 FlashAttention 2.0 implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        GLM47FlashAttention2: The GLM-4.7 FlashAttention 2.0 implementation
    """
    return GLM47FlashAttention2(config, layer_idx)


__all__ = [
    "GLM47FlashAttention2",
    "create_glm47_flash_attention_2"
]