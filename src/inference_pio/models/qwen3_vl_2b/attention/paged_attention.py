"""
Qwen3-VL-2B Paged Attention Implementation

This module implements paged attention for the Qwen3-VL-2B model
using a real Paged KV Cache backend.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from ....common.paged_attention import PagedAttention
from ..config import Qwen3VL2BConfig
from ..kv_cache.paged_kv_cache import PagedKVCache


class Qwen3VL2BPagedAttention(PagedAttention):
    """
    Qwen3-VL-2B specific paged attention implementation.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
        page_size: int = 16,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096,
        kv_cache: Optional[PagedKVCache] = None
    ):
        num_attention_heads = getattr(config, 'num_attention_heads', 16)
        attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        bias = not getattr(config, 'remove_bias_in_attention', False)
        is_causal = getattr(config, 'is_causal', True)

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
        self.kv_cache = kv_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_tables: Optional[List[List[int]]] = None,
        seq_lens: Optional[List[int]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Paged Attention.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Real Paged KV Cache Logic
        if self.kv_cache is not None and block_tables is not None and seq_lens is not None:
            # Append new tokens
            k_to_cache = key_states.transpose(1, 2)
            v_to_cache = value_states.transpose(1, 2)
            self.kv_cache.append(k_to_cache, v_to_cache, self.layer_idx, block_tables, seq_lens)

            # Retrieve full context
            k_full, v_full = self.kv_cache.get_kv_cache(self.layer_idx, block_tables, seq_lens)
            key_states = k_full.transpose(1, 2)
            value_states = v_full.transpose(1, 2)

        elif past_key_value is not None:
             # Fallback to standard concatenation if no paged cache provided
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if self.use_sliding_window:
            key_states = self._apply_sliding_window(key_states)
            value_states = self._apply_sliding_window(value_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if self.is_causal:
            # Note: Causal mask shape needs to match the retrieved key_states length
            target_len = q_len
            source_len = key_states.size(2)
            causal_mask = torch.triu(
                torch.ones(target_len, source_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=source_len - target_len + 1
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if use_cache:
            if self.kv_cache is not None:
                past_key_value = None # Managed internally
            else:
                past_key_value = (key_states, value_states)

        return attn_output, attn_weights if output_attentions else None, past_key_value


def create_qwen3_vl_paged_attention(
    config: Qwen3VL2BConfig, 
    layer_idx: Optional[int] = None,
    page_size: int = 16,
    use_sliding_window: bool = False,
    sliding_window_size: int = 4096,
    kv_cache: Optional[PagedKVCache] = None
):
    return Qwen3VL2BPagedAttention(
        config, 
        layer_idx, 
        page_size, 
        use_sliding_window, 
        sliding_window_size,
        kv_cache
    )


__all__ = [
    "Qwen3VL2BPagedAttention",
    "create_qwen3_vl_paged_attention"
]
