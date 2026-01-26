"""
Qwen3-4B-Instruct-2507 Sliding Window Attention Implementation

This module implements sliding window attention for the Qwen3-4B-Instruct-2507 model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseCausalAttention
from ..config import Qwen34BInstruct2507Config


class Qwen34BSlidingWindowAttention(BaseCausalAttention):
    """
    Qwen3-4B-Instruct-2507 specific sliding window attention implementation.

    This implementation provides efficient attention computation using a sliding window
    approach, limiting the attention span to improve performance for long sequences.
    """

    def __init__(
        self,
        config: Qwen34BInstruct2507Config,
        layer_idx: Optional[int] = None,
        window_size: int = 4096
    ):
        """
        Initialize sliding window attention for Qwen3-4B-Instruct-2507.

        Args:
            config: Model configuration
            layer_idx: Index of the transformer layer
            window_size: Size of the sliding window
        """
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=getattr(config, 'attention_dropout_prob', 0.0),
            bias=not getattr(config, 'remove_bias_in_attention', False)
        )
        
        self.config = config
        self.layer_idx = layer_idx
        self.window_size = window_size
        
        # Additional parameters specific to sliding window attention
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.num_key_value_groups = getattr(config, 'num_key_value_groups', 
                                          config.num_attention_heads // self.num_key_value_heads)
        
        # If using grouped-query attention, adjust projections
        if self.num_key_value_heads != config.num_attention_heads:
            self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with sliding window attention mechanism.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Attention mask
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        # Apply projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        
        # Handle GQA/MQA: reshape K and V appropriately
        if self.num_key_value_heads == self.num_heads:
            # Standard MHA
            k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        else:
            # GQA/MQA: repeat K and V for grouped attention
            k = k.view(bsz, src_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, src_len, head_dim)
            v = v.view(bsz, src_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, src_len, head_dim)
            
            # Repeat K and V to match number of query heads
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scale query
        q = q * self.scaling

        # Handle past key-value states for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            src_len = k.size(2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (bsz, num_heads, tgt_len, src_len)

        # Apply sliding window mask
        sliding_window_mask = self._create_sliding_window_mask(tgt_len, src_len, self.window_size)
        sliding_window_mask = sliding_window_mask.to(attn_weights.device).to(attn_weights.dtype)
        attn_weights = attn_weights + sliding_window_mask

        # Apply causal mask to prevent attending to future tokens
        causal_mask = torch.tril(
            torch.ones((tgt_len, src_len), dtype=torch.bool, device=attn_weights.device)
        ).view(1, 1, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Apply dropout if configured
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (bsz, num_heads, tgt_len, head_dim)

        # Reshape to combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Prepare past key-value states for caching
        past_key_value = (k, v) if use_cache else None

        return attn_output, attn_weights if output_attentions else None, past_key_value

    def _create_sliding_window_mask(self, tgt_len: int, src_len: int, window_size: int) -> torch.Tensor:
        """
        Create a sliding window mask for attention.

        Args:
            tgt_len: Target sequence length
            src_len: Source sequence length
            window_size: Size of the sliding window

        Returns:
            Sliding window mask
        """
        # Create distance matrix
        row_idx = torch.arange(tgt_len, dtype=torch.long).unsqueeze(1)  # (tgt_len, 1)
        col_idx = torch.arange(src_len, dtype=torch.long).unsqueeze(0)  # (1, src_len)
        distances = row_idx - col_idx  # (tgt_len, src_len)

        # Create mask: positions outside window get -inf, others get 0
        window_mask = torch.where(
            distances >= -window_size,
            torch.zeros_like(distances, dtype=torch.float),
            torch.full_like(distances, float("-inf"))
        ).unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

        return window_mask


def create_qwen3_4b_sliding_window_attention(
    config: Qwen34BInstruct2507Config, 
    layer_idx: Optional[int] = None
):
    """
    Factory function to create Qwen3-4B-Instruct-2507 sliding window attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        Qwen34BSlidingWindowAttention: The Qwen3-4B-Instruct-2507 sliding window attention implementation
    """
    window_size = getattr(config, 'sliding_window_size', 4096)
    return Qwen34BSlidingWindowAttention(config, layer_idx, window_size)


__all__ = [
    "Qwen34BSlidingWindowAttention",
    "create_qwen3_4b_sliding_window_attention"
]