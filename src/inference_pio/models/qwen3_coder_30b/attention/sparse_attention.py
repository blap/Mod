"""
Qwen3-Coder-30B Sparse Attention Implementation

This module implements sparse attention mechanisms for the Qwen3-Coder-30B model using the common implementation.
"""

from typing import Optional

import torch.nn as nn

from ....common.sparse_attention import SparseAttention, create_sparse_attention
from ..config import Qwen3Coder30BConfig


class Qwen3CoderSparseAttention(SparseAttention):
    """
    Qwen3-Coder-30B specific sparse attention implementation using the common base implementation.

    This implementation provides optimized attention computation with reduced memory usage
    and improved performance compared to standard attention mechanisms. It inherits from
    the common SparseAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3Coder30BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine sparse attention parameters from config
        sparse_pattern = getattr(config, 'sparse_attention_pattern', 'longformer')
        sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.25)
        block_size = getattr(config, 'sparse_attention_block_size', 64)
        local_window_size = getattr(config, 'sparse_attention_local_window_size', 128)
        use_global_attention = getattr(config, 'use_global_attention', True)
        global_attention_indices = getattr(config, 'global_attention_indices', [0])
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)

        # Initialize using the common SparseAttention implementation
        super().__init__(
            config,
            layer_idx=layer_idx,
            sparse_pattern=sparse_pattern,
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
            local_window_size=local_window_size,
            use_global_attention=use_global_attention,
            global_attention_indices=global_attention_indices,
            attention_dropout=attention_dropout,
            bias=bias,
        )


def create_qwen3_coder_sparse_attention(config: Qwen3Coder30BConfig, layer_idx: Optional[int] = None):
    """
    Factory function to create Qwen3-Coder-30B sparse attention implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3CoderSparseAttention: The Qwen3-Coder-30B sparse attention implementation
    """
    return Qwen3CoderSparseAttention(config, layer_idx)


__all__ = [
    "Qwen3CoderSparseAttention",
    "create_qwen3_coder_sparse_attention"
]