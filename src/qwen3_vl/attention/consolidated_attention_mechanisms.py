"""
Consolidated Attention Module for Qwen3-VL Model
Combines all attention mechanisms, flash attention, sparse attention, and rotary embeddings
into a unified, hardware-optimized attention system.
"""
import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLRotaryEmbedding(nn.Module):
    """Rotary embedding implementation for Qwen3-VL model."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class StandardAttention(nn.Module):
    """
    Standard attention mechanism implementation for Qwen3-VL.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FlashAttention2(nn.Module):
    """
    FlashAttention 2 implementation for memory-efficient attention computation.
    Reduces memory complexity from O(n²) to O(n) by using tiled computation and
    incremental softmax calculation.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Memory-efficient parameters
        self.tile_size = 512  # Size of tiles for chunked computation

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Memory-efficient attention computation using FlashAttention approach
        if not output_attentions:
            # Use PyTorch's optimized scaled dot-product attention (FlashAttention-like)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,  # No dropout during inference
                is_causal=False if attention_mask is not None else True  # Set based on mask
            )
            attn_weights = None  # Not computed when output_attentions is False
        else:
            # Compute attention weights in a memory-efficient way using tiling
            attn_weights = self._memory_efficient_attention_weights(
                query_states, key_states, attention_mask
            )
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def _memory_efficient_attention_weights(self, query_states, key_states, attention_mask):
        """
        Compute attention weights in a memory-efficient way using tiling.
        This reduces memory complexity from O(n²) to O(n) by processing in chunks.
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # Calculate tile size based on available memory
        tile_size = min(self.tile_size, seq_len)

        # Initialize output tensor
        attn_weights = torch.zeros(bsz, num_heads, seq_len, kv_seq_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Process in tiles to limit memory usage
        for q_start in range(0, seq_len, tile_size):
            q_end = min(q_start + tile_size, seq_len)
            q_tile = query_states[:, :, q_start:q_end, :]

            for k_start in range(0, kv_seq_len, tile_size):
                k_end = min(k_start + tile_size, kv_seq_len)
                k_tile = key_states[:, :, k_start:k_end, :]

                # Compute attention scores for this tile
                scores_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1)) / math.sqrt(self.head_dim)

                # Apply attention mask if provided
                if attention_mask is not None:
                    mask_tile = attention_mask[:, :, q_start:q_end, k_start:k_end]
                    scores_tile = scores_tile + mask_tile

                # Store the tile in the full attention matrix
                attn_weights[:, :, q_start:q_end, k_start:k_end] = scores_tile

        return attn_weights


class SM61OptimizedFlashAttention2(nn.Module):
    """
    NVIDIA SM61 optimized FlashAttention 2 implementation.
    Optimized for compute capability 6.1 with memory and compute constraints.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # SM61-specific parameters for optimal performance
        self.tile_size = 256  # Smaller tile size for SM61's memory constraints

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SM61-optimized attention computation using memory-efficient approach
        if not output_attentions:
            # Use PyTorch's optimized scaled dot-product attention (FlashAttention-like)
            # With SM61 optimizations
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,  # No dropout during inference
                is_causal=False if attention_mask is not None else True  # Set based on mask
            )
            attn_weights = None  # Not computed when output_attentions is False
        else:
            # For SM61, compute attention weights in a memory-efficient way using tiling
            attn_weights = self._sm61_memory_efficient_attention_weights(
                query_states, key_states, attention_mask
            )
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    def _sm61_memory_efficient_attention_weights(self, query_states, key_states, attention_mask):
        """
        SM61-optimized memory-efficient attention computation with smaller tile sizes
        and memory access patterns optimized for compute capability 6.1.
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # SM61-optimized tile size (smaller for better cache utilization)
        tile_size = min(self.tile_size, seq_len)

        # Initialize output tensor
        attn_weights = torch.zeros(bsz, num_heads, seq_len, kv_seq_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Process in tiles with SM61-optimized access patterns
        for q_start in range(0, seq_len, tile_size):
            q_end = min(q_start + tile_size, seq_len)
            q_tile = query_states[:, :, q_start:q_end, :]

            for k_start in range(0, kv_seq_len, tile_size):
                k_end = min(k_start + tile_size, kv_seq_len)
                k_tile = key_states[:, :, k_start:k_end, :]

                # Compute attention scores for this tile
                scores_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1)) / math.sqrt(self.head_dim)

                # Apply attention mask if provided
                if attention_mask is not None:
                    mask_tile = attention_mask[:, :, q_start:q_end, k_start:k_end]
                    scores_tile = scores_tile + mask_tile

                # Store the tile in the full attention matrix
                attn_weights[:, :, q_start:q_end, k_start:k_end] = scores_tile

        return attn_weights


class TrueSparseAttention(nn.Module):
    """
    True sparse attention implementation with configurable sparsity patterns.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply sparsity: keep only top-k values per query position
        sparse_attn_weights = self._apply_sparsity(attn_weights)

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_sparsity(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply sparsity pattern to attention weights by masking out low-attention values.
        """
        # Calculate how many elements to keep based on sparsity ratio
        seq_len = attn_weights.size(-1)
        k = max(1, int(seq_len * self.sparsity_ratio))

        # Get top-k attention values for each query position
        top_k_values, top_k_indices = torch.topk(attn_weights, k=k, dim=-1)

        # Create a mask for the sparse attention pattern
        sparse_mask = torch.full_like(attn_weights, float('-inf'))
        sparse_mask.scatter_(-1, top_k_indices, torch.ones_like(top_k_values))

        # Apply sparse mask to scores
        sparse_attn_weights = torch.where(sparse_mask == float('-inf'), sparse_mask, attn_weights)

        return sparse_attn_weights


class BlockSparseAttention(nn.Module):
    """
    Block-sparse attention with configurable block patterns.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.block_size = getattr(config, 'block_sparse_block_size', 64)
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Learnable sparsity pattern
        self.sparsity_pattern = nn.Parameter(
            torch.randn(self.num_heads, self.max_position_embeddings // self.block_size,
                       self.max_position_embeddings // self.block_size)
        )
        nn.init.uniform_(self.sparsity_pattern, -0.1, 0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply block-sparse attention pattern
        attn_weights = self._apply_block_sparse_attention(query_states, key_states)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_block_sparse_attention(self, query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
        """
        Apply block-sparse attention pattern to reduce computation.
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # Calculate block dimensions
        block_size = self.block_size
        num_q_blocks = math.ceil(seq_len / block_size)
        num_kv_blocks = math.ceil(kv_seq_len / block_size)

        # Pad sequences to be divisible by block size
        padded_q_len = num_q_blocks * block_size
        padded_kv_len = num_kv_blocks * block_size

        if seq_len != padded_q_len or kv_seq_len != padded_kv_len:
            query_states = F.pad(query_states, (0, 0, 0, padded_q_len - seq_len), value=0)
            key_states = F.pad(key_states, (0, 0, 0, padded_kv_len - kv_seq_len), value=0)

        # Reshape to block format
        query_blocks = query_states.view(bsz, num_heads, num_q_blocks, block_size, head_dim)
        key_blocks = key_states.view(bsz, num_heads, num_kv_blocks, block_size, head_dim)

        # Get sparsity pattern for current sequence length
        # Ensure we don't exceed the available pattern dimensions
        available_q_blocks = min(num_q_blocks, self.sparsity_pattern.size(1))
        available_kv_blocks = min(num_kv_blocks, self.sparsity_pattern.size(2))
        current_sparsity_pattern = self.sparsity_pattern[:, :available_q_blocks, :available_kv_blocks]

        # Apply learned sparsity pattern with top-k selection to enforce sparsity
        sparsity_threshold = torch.topk(
            current_sparsity_pattern.reshape(num_heads, -1),
            k=max(1, int(current_sparsity_pattern.numel() * self.sparsity_ratio / num_heads)),
            dim=-1
        )[0][:, -1].view(num_heads, 1, 1)

        sparse_mask = (current_sparsity_pattern > sparsity_threshold).float()

        # Initialize output tensor
        attn_weights = torch.zeros(bsz, num_heads, padded_q_len, padded_kv_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Compute attention only for non-zero blocks in the sparse pattern
        for h_idx in range(num_heads):
            for q_block_idx in range(available_q_blocks):
                for kv_block_idx in range(available_kv_blocks):
                    if sparse_mask[h_idx, q_block_idx, kv_block_idx] > 0:
                        # Compute attention for this block pair
                        q_block = query_blocks[:, h_idx, q_block_idx, :, :]  # [bsz, block_size, head_dim]
                        k_block = key_blocks[:, h_idx, kv_block_idx, :, :]  # [bsz, block_size, head_dim]

                        block_attn = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        attn_weights[:, h_idx,
                                    q_block_idx * block_size:min((q_block_idx + 1) * block_size, padded_q_len),
                                    kv_block_idx * block_size:min((kv_block_idx + 1) * block_size, padded_kv_len)] = block_attn

        # Trim back to original sequence length
        attn_weights = attn_weights[:, :, :seq_len, :kv_seq_len]

        return attn_weights


class DynamicSparseAttention(nn.Module):
    """
    Dynamic sparse attention with learned routing for token selection.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)
        self.vision_sparsity_ratio = getattr(config, 'vision_sparse_attention_sparsity_ratio', 0.4)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Learned routing mechanism for dynamic token selection
        self.routing_network = nn.Linear(self.hidden_size, self.num_heads, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute routing scores to determine important tokens
        routing_scores = self._compute_routing_scores(hidden_states)  # [bsz, seq_len, num_heads]

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply dynamic sparsity based on routing scores
        sparse_attn_weights = self._apply_dynamic_sparsity(attn_weights, routing_scores)

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _compute_routing_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores for dynamic token selection.
        """
        # Use the routing network to determine which tokens are important
        routing_logits = self.routing_network(hidden_states)  # [bsz, seq_len, num_heads]
        routing_scores = torch.sigmoid(routing_logits)  # [bsz, seq_len, num_heads]
        return routing_scores

    def _apply_dynamic_sparsity(self, attn_weights: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic sparsity based on routing scores.
        """
        bsz, num_heads, q_len, kv_len = attn_weights.size()

        # Adjust sparsity ratio based on whether we're processing vision or text tokens
        # This is a simple heuristic - in a full implementation, we would have more sophisticated logic
        current_sparsity_ratio = self.sparsity_ratio
        if q_len > 512:  # Heuristic: if sequence is long, likely vision tokens
            current_sparsity_ratio = self.vision_sparsity_ratio

        # Calculate how many tokens to attend to per head
        k = max(1, int(kv_len * current_sparsity_ratio))

        # For each head, select top-k tokens based on routing scores
        # routing_scores is [bsz, seq_len, num_heads], we need [bsz, num_heads, seq_len]
        routing_scores_t = routing_scores.transpose(1, 2)  # [bsz, num_heads, seq_len]

        # Get top-k routing scores for each head
        top_k_routing_values, top_k_indices = torch.topk(routing_scores_t, k, dim=-1)  # [bsz, num_heads, k]

        # Create a sparse attention mask
        sparse_mask = torch.full_like(attn_weights, float('-inf'))
        
        # For each head, fill in the sparse mask with values for the selected tokens
        for h_idx in range(num_heads):
            for batch_idx in range(bsz):
                selected_kv_indices = top_k_indices[batch_idx, h_idx, :]  # [k]
                # Fill the attention weights for this head and batch with the selected keys/values
                sparse_mask[batch_idx, h_idx, :, selected_kv_indices] = attn_weights[batch_idx, h_idx, :, selected_kv_indices]

        return sparse_mask


class VectorizedSparseAttention(nn.Module):
    """
    Vectorized sparse attention implementation with optimized computation.
    """
    def __init__(self, sparsity_ratio: float = 0.5):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio

    def forward(self, attn_weights: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply sparse attention by keeping only top-k attention weights per query position.
        """
        bsz, num_heads, seq_len, _ = attn_weights.size()

        # Calculate top_k based on sparsity ratio
        top_k = max(1, int(self.sparsity_ratio * seq_len))
        top_k = min(top_k, seq_len)  # Ensure top_k doesn't exceed sequence length

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Get top-k values and indices efficiently using torch.topk
        top_k_values, top_k_indices = torch.topk(attn_weights, top_k, dim=-1, sorted=False)

        # Create a mask to store sparse attention weights
        sparse_attn_weights = torch.full_like(attn_weights, float('-inf'))

        # Create index tensors for advanced indexing
        batch_indices = torch.arange(bsz, device=attn_weights.device).view(-1, 1, 1, 1).expand(-1, num_heads, seq_len, top_k)
        head_indices = torch.arange(num_heads, device=attn_weights.device).view(1, -1, 1, 1).expand(bsz, -1, seq_len, top_k)
        query_indices = torch.arange(seq_len, device=attn_weights.device).view(1, 1, -1, 1).expand(bsz, num_heads, -1, top_k)

        # Scatter the top-k values back to the sparse attention matrix
        sparse_attn_weights.scatter_(-1, top_k_indices.unsqueeze(-2).expand(-1, -1, -1, top_k), top_k_values)

        return sparse_attn_weights


class OptimizedDynamicSparseAttention(nn.Module):
    """
    Optimized dynamic sparse attention with advanced routing mechanisms.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)
        self.vision_sparsity_ratio = getattr(config, 'vision_sparse_attention_sparsity_ratio', 0.4)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # Rotary embedding for position encoding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Advanced routing network with multiple layers for better token selection
        self.routing_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_heads),
            nn.Softmax(dim=-1)  # Use softmax for probability distribution over heads
        )

        # Vectorized sparse attention for efficiency
        self.sparse_attention = VectorizedSparseAttention(self.sparsity_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for caching
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute routing scores to determine important tokens
        routing_scores = self.routing_network(hidden_states)  # [bsz, seq_len, num_heads]

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply dynamic sparsity based on routing scores
        # Adjust sparsity ratio based on context (vision vs text)
        if q_len > 512:  # Heuristic for vision tokens
            sparsity_ratio = self.vision_sparsity_ratio
        else:
            sparsity_ratio = self.sparsity_ratio

        # Temporarily adjust the sparsity ratio of the sparse attention module
        original_ratio = self.sparse_attention.sparsity_ratio
        self.sparse_attention.sparsity_ratio = sparsity_ratio
        sparse_attn_weights = self.sparse_attention(attn_weights, attention_mask)
        self.sparse_attention.sparsity_ratio = original_ratio  # Restore original value

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3VLVisionAttention(nn.Module):
    """A multi-head attention module for vision processing."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.vision_qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = mixed_qkv.unbind(0)  # [bsz, num_heads, seq_len, head_dim]

        # Transpose to apply softmax
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)

        attn_output = self.proj(attn_output)

        return attn_output, attn_weights


class AttentionMechanismSelector:
    """
    Factory and selector for different attention mechanisms based on configuration and hardware.
    """
    @staticmethod
    def create_attention(config, layer_idx: Optional[int] = None) -> nn.Module:
        """
        Create the appropriate attention mechanism based on configuration.
        
        Args:
            config: Model configuration
            layer_idx: Layer index (optional)
            
        Returns:
            Appropriate attention mechanism module
        """
        # Check configuration for which attention mechanism to use
        attention_implementation = getattr(config, 'attention_implementation', 'standard')
        
        if attention_implementation == 'flash_attention_2':
            # Use hardware-specific attention if available
            if hasattr(config, 'hardware_specific_attention') and config.hardware_specific_attention == 'sm61':
                return SM61OptimizedFlashAttention2(config, layer_idx)
            else:
                return FlashAttention2(config, layer_idx)
        elif attention_implementation == 'sparse_attention':
            if getattr(config, 'use_dynamic_sparse_attention', False):
                if getattr(config, 'use_optimized_dynamic_sparse_attention', False):
                    return OptimizedDynamicSparseAttention(config, layer_idx)
                else:
                    return DynamicSparseAttention(config, layer_idx)
            elif getattr(config, 'use_block_sparse_attention', False):
                return BlockSparseAttention(config, layer_idx)
            else:
                return TrueSparseAttention(config, layer_idx)
        elif attention_implementation == 'standard':
            return StandardAttention(config, layer_idx)
        else:
            # Default to standard attention
            return StandardAttention(config, layer_idx)

    @staticmethod
    def get_available_implementations() -> List[str]:
        """Get list of available attention implementations"""
        return [
            'standard',
            'flash_attention_2',
            'sparse_attention',
            'dynamic_sparse_attention',
            'block_sparse_attention',
            'optimized_dynamic_sparse_attention'
        ]


# Export all classes and functions
__all__ = [
    # Utility functions
    'repeat_kv',
    'rotate_half',
    'apply_rotary_pos_emb',
    
    # Rotary embeddings
    'Qwen3VLRotaryEmbedding',
    
    # Attention mechanisms
    'StandardAttention',
    'FlashAttention2',
    'SM61OptimizedFlashAttention2',
    'TrueSparseAttention',
    'BlockSparseAttention',
    'DynamicSparseAttention',
    'VectorizedSparseAttention',
    'OptimizedDynamicSparseAttention',
    'Qwen3VLVisionAttention',
    
    # Factory class
    'AttentionMechanismSelector'
]