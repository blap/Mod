"""
Optimized Attention Wrapper for PaddleOCR-VL-1.5

This module wraps standard attention with Paged KV Cache support and
optional Flash Attention integration.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from ..kv_cache.paged_kv_cache import PagedKVCache

class OptimizedAttentionWrapper:
    def __init__(self, config):
        self.config = config
        self.use_flash_attn = config.enable_flash_attention
        self.head_dim = config.hidden_size // getattr(config, 'num_attention_heads', 16) # Fallback

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        kv_cache: Optional[PagedKVCache] = None,
        seq_id: Optional[int] = None,
        token_position: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass for attention.

        Args:
            query: [batch_size, num_heads, q_len, head_dim]
            key: [batch_size, num_heads, kv_len, head_dim]
            value: [batch_size, num_heads, kv_len, head_dim]
        """
        batch_size, num_heads, q_len, head_dim = query.shape

        # 1. Update KV Cache if present and we are in generation phase (q_len == 1 typically)
        # Note: If q_len > 1 (prefill), we might append all at once, but PagedKV typically handles token-by-token or block-by-block.
        # For simplicity in this wrapper, we assume standard usage:

        if kv_cache is not None and seq_id is not None:
            # If generating (q_len == 1)
            if q_len == 1 and token_position is not None:
                # Squeeze batch dim for single token append
                # key: [1, num_heads, 1, head_dim] -> [num_heads, head_dim]
                k_token = key.squeeze(0).squeeze(1)
                v_token = value.squeeze(0).squeeze(1)
                kv_cache.append_token_kv(seq_id, layer_idx, k_token, v_token, token_position)

                # Retrieve full history for attention
                # Ideally, we use a custom kernel that reads directly from block tables.
                # Since we are sticking to python/torch for now (unless we have custom kernels compiled),
                # we reconstruct the continuous KV from the cache for the current sequence.
                # This is "Slow Paged Attention" but functional for the floor requirement without CUDA kernels.

                key, value = self._gather_kv_from_cache(kv_cache, seq_id, layer_idx, num_heads, head_dim)

            # If prefill (q_len > 1), we just calculate attention normally but ALSO store in cache?
            # Usually prefill writes to cache but uses the input K/V directly for self-attention.
            elif q_len > 1:
                # Logic to write block of tokens to cache would go here
                pass

        # 2. Compute Attention
        if self.use_flash_attn:
            # F.scaled_dot_product_attention supports FlashAttention automatically if inputs are correct
            # SDPA expects [batch, heads, seq, dim]

            # Ensure proper scaling (defaults to 1/sqrt(dim) inside sdpa if scale is None)
            # scale=None implies 1/sqrt(dim)

            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False # Handled by mask usually, or set True if pure causal
            )
        else:
            # Manual implementation if needed (fallback)
            scale = scaling or (head_dim ** -0.5)
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def _gather_kv_from_cache(self, kv_cache: PagedKVCache, seq_id: int, layer_idx: int, num_heads: int, head_dim: int):
        """
        Reconstruct continuous K/V tensors from Paged Cache for standard SDPA usage.
        (Performance bottleneck: In production this is replaced by PagedAttention Kernel)
        """
        block_table = kv_cache.sequence_block_tables[seq_id]
        blocks = block_table.blocks

        k_blocks = kv_cache.k_cache[layer_idx][blocks] # [num_blocks, num_heads, block_size, head_dim]
        v_blocks = kv_cache.v_cache[layer_idx][blocks]

        # Flatten blocks -> [1, num_heads, total_seq_len, head_dim]
        # Reshape: [num_blocks, num_heads, block_size, head_dim] -> [num_heads, num_blocks, block_size, head_dim]
        k_cont = k_blocks.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
        v_cont = v_blocks.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)

        # We might have padding at the end of the last block, but SDPA handles it if we assume masked
        # or we slice it if we track exact length.
        # For this implementation, we assume the caller handles length/masking.

        return k_cont, v_cont
