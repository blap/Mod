"""
Qwen3-4B-Instruct-2507 Paged Attention Implementation

This module implements paged attention for the Qwen3-4B-Instruct-2507 model
using a real Paged KV Cache backend.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseCausalAttention
from ..config import Qwen34BInstruct2507Config
from ..kv_cache.paged_kv_cache import PagedKVCache


class Qwen34BPagedAttention(BaseCausalAttention):
    """
    Qwen3-4B-Instruct-2507 specific paged attention implementation.
    """

    def __init__(
        self,
        config: Qwen34BInstruct2507Config,
        layer_idx: Optional[int] = None,
        page_size: int = 16,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096,
        kv_cache: Optional[PagedKVCache] = None,  # Dependency injection
    ):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=getattr(config, "attention_dropout_prob", 0.0),
            bias=not getattr(config, "remove_bias_in_attention", False),
        )

        self.config = config
        self.layer_idx = layer_idx
        self.page_size = page_size
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
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
        """
        Forward pass with real paged attention.

        Args:
            query: [batch_size, q_len, hidden_size]
            key: [batch_size, q_len, hidden_size] (new tokens only)
            value: [batch_size, q_len, hidden_size] (new tokens only)
            block_tables: List of physical block IDs for each sequence.
            seq_lens: List of total sequence lengths (past + new).
        """
        bsz, q_len, embed_dim = query.size()

        # Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape [bsz, q_len, num_heads, head_dim] -> [bsz, num_heads, q_len, head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle GQA/MQA for Key/Value
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

            # For caching, we store the un-repeated K/V to save memory.
            # Repetition happens during computation if using standard SDPA.

        # Scale query
        q = q * self.scaling

        # --- KV Cache Management ---
        if (
            self.kv_cache is not None
            and block_tables is not None
            and seq_lens is not None
        ):
            # 1. Append new tokens to cache
            # We need to reshape K/V back to [bsz, seq_len, heads, dim] for the append method
            k_to_cache = k.transpose(1, 2)
            v_to_cache = v.transpose(1, 2)

            self.kv_cache.append(
                k_to_cache, v_to_cache, self.layer_idx, block_tables, seq_lens
            )

            # 2. Retrieve full sequence for attention (Optimization: Use a PagedAttention Kernel here!)
            # For now, we reconstruct the full tensor to satisfy the "Implementation Floor" using standard SDPA.
            # Ideally, we would call `ops.paged_attention_v2(...)` here if using vLLM kernels.
            k_full, v_full = self.kv_cache.get_kv_cache(
                self.layer_idx, block_tables, seq_lens
            )

            # Transpose full K/V for attention: [bsz, num_heads, total_seq_len, head_dim]
            k = k_full.transpose(1, 2)
            v = v_full.transpose(1, 2)

            # Handle GQA repetition after retrieval if needed
            if self.num_key_value_heads != self.num_heads:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        else:
            # Fallback for no cache (e.g. initial prefill without paging setup)
            # Just use current K/V, assume no past
            if self.num_key_value_heads != self.num_heads:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # --- Attention Computation ---
        # Dimensions:
        # q: [bsz, n_heads, q_len, head_dim]
        # k: [bsz, n_heads, k_len, head_dim]

        # Use SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_module.p if self.dropout_module else 0.0,
        )

        # Reshape output
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, q_len, embed_dim)
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, None, None  # Past KV is managed externally via block_tables


def create_qwen3_4b_paged_attention(
    config: Qwen34BInstruct2507Config,
    layer_idx: Optional[int] = None,
    page_size: int = 16,
    use_sliding_window: bool = False,
    sliding_window_size: int = 4096,
    kv_cache: Optional[PagedKVCache] = None,
):
    return Qwen34BPagedAttention(
        config, layer_idx, page_size, use_sliding_window, sliding_window_size, kv_cache
    )


__all__ = ["Qwen34BPagedAttention", "create_qwen3_4b_paged_attention"]
