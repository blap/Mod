"""
Qwen3-4B-Instruct-2507 Multi-Query and Grouped-Query Attention Implementation

This module implements Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) for the Qwen3-4B-Instruct-2507 model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseCausalAttention
from ..config import Qwen34BInstruct2507Config


class Qwen34BMultiQueryAttention(BaseCausalAttention):
    """
    Qwen3-4B-Instruct-2507 specific Multi-Query Attention implementation.

    This implementation provides efficient attention computation where each query
    attends to all keys and values, but with optimized memory usage.
    """

    def __init__(
        self,
        config: Qwen34BInstruct2507Config,
        layer_idx: Optional[int] = None,
        use_gqa: bool = False,
        num_key_value_groups: Optional[int] = None
    ):
        """
        Initialize MQA/GQA for Qwen3-4B-Instruct-2507.

        Args:
            config: Model configuration
            layer_idx: Index of the transformer layer
            use_gqa: Whether to use Grouped-Query Attention instead of MQA
            num_key_value_groups: Number of query heads per KV head for GQA
        """
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=getattr(config, 'attention_dropout_prob', 0.0),
            bias=not getattr(config, 'remove_bias_in_attention', False)
        )
        
        self.config = config
        self.layer_idx = layer_idx
        self.use_gqa = use_gqa
        
        # Determine number of KV heads
        if use_gqa and num_key_value_groups is not None:
            self.num_key_value_heads = config.num_attention_heads // num_key_value_groups
        else:
            # For MQA, we typically use 1 KV head per group, often just 1 total
            self.num_key_value_heads = 1  # Standard MQA uses single KV head
        
        self.num_key_value_groups = getattr(config, 'num_key_value_groups', 
                                          config.num_attention_heads // self.num_key_value_heads)
        
        # Initialize projections - MQA/GQA uses separate projections for queries and key-values
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

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
        Forward pass with MQA/GQA mechanism.

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
        
        # For MQA/GQA: reshape K and V with fewer heads
        k = k.view(bsz, src_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, src_len, head_dim)
        v = v.view(bsz, src_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, src_len, head_dim)
        
        # Expand K and V to match number of query heads for GQA, or repeat for MQA
        if self.use_gqa:
            # For GQA: repeat K and V for grouped attention
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        else:
            # For MQA: repeat K and V for all query heads
            k = k.expand(-1, self.num_heads, -1, -1)
            v = v.expand(-1, self.num_heads, -1, -1)

        # Scale query
        q = q * self.scaling

        # Handle past key-value states for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            src_len = k.size(2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (bsz, num_heads, tgt_len, src_len)

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


def create_mqa_gqa_attention(config: Qwen34BInstruct2507Config, layer_idx: Optional[int] = None):
    """
    Factory function to create Qwen3-4B-Instruct-2507 MQA/GQA attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        Qwen34BMultiQueryAttention: The Qwen3-4B-Instruct-2507 MQA/GQA attention implementation
    """
    use_gqa = getattr(config, 'attention_type', 'mha') == 'gqa'
    num_key_value_groups = getattr(config, 'num_key_value_groups', 
                                 config.num_attention_heads // config.num_key_value_heads)
    
    return Qwen34BMultiQueryAttention(
        config, 
        layer_idx, 
        use_gqa=use_gqa, 
        num_key_value_groups=num_key_value_groups
    )


__all__ = [
    "Qwen34BMultiQueryAttention",
    "create_mqa_gqa_attention"
]