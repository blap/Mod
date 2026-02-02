"""
Qwen3-Coder-30B Paged Attention Implementation

This module implements paged attention for the Qwen3-Coder-30B model
using a real Paged KV Cache backend.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseCausalAttention
from ..config import Qwen3Coder30BConfig
from ..kv_cache.paged_kv_cache import PagedKVCache


class Qwen3CoderPagedAttention(BaseCausalAttention):
    """
    Qwen3-Coder-30B specific paged attention implementation.
    """

    def __init__(
        self,
        config: Qwen3Coder30BConfig,
        layer_idx: Optional[int] = None,
        page_size: int = 16,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096,
        kv_cache: Optional[PagedKVCache] = None,
    ):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=getattr(config, "attention_dropout_prob", 0.0),
            bias=not getattr(config, "remove_bias_in_attention", False),
        )

        self.config = config
        self.layer_idx = layer_idx
        self.kv_cache = kv_cache

        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.num_key_value_groups = getattr(
            config,
            "num_key_value_groups",
            config.num_attention_heads // self.num_key_value_heads,
        )

        if self.num_key_value_heads != config.num_attention_heads:
            self.k_proj = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.v_proj = nn.Linear(
                config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_tables: Optional[List[List[int]]] = None,
        seq_lens: Optional[List[int]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:

        bsz, q_len, embed_dim = query.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.num_key_value_heads == self.num_heads:
            k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
                1, 2
            )

        q = q * self.scaling

        if (
            self.kv_cache is not None
            and block_tables is not None
            and seq_lens is not None
        ):
            k_to_cache = k.transpose(1, 2)
            v_to_cache = v.transpose(1, 2)

            self.kv_cache.append(
                k_to_cache, v_to_cache, self.layer_idx, block_tables, seq_lens
            )
            k_full, v_full = self.kv_cache.get_kv_cache(
                self.layer_idx, block_tables, seq_lens
            )

            k = k_full.transpose(1, 2)
            v = v_full.transpose(1, 2)

            if self.num_key_value_heads != self.num_heads:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        else:
            if self.num_key_value_heads != self.num_heads:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_module.p if self.dropout_module else 0.0,
        )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, q_len, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, None, None


def create_qwen3_coder_paged_attention(
    config: Qwen3Coder30BConfig,
    layer_idx: Optional[int] = None,
    page_size: int = 16,
    use_sliding_window: bool = False,
    sliding_window_size: int = 4096,
    kv_cache: Optional[PagedKVCache] = None,
):
    return Qwen3CoderPagedAttention(
        config, layer_idx, page_size, use_sliding_window, sliding_window_size, kv_cache
    )


__all__ = ["Qwen3CoderPagedAttention", "create_qwen3_coder_paged_attention"]
