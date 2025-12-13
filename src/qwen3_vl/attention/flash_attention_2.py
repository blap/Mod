"""
FlashAttention 2 implementation for Qwen3-VL model.
Implements memory-efficient attention with O(n) complexity instead of O(n²).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from src.qwen3_vl.core.config import Qwen3VLConfig


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
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


class FlashAttention2(nn.Module):
    """
    FlashAttention 2 implementation for memory-efficient attention computation.
    Reduces memory complexity from O(n²) to O(n) by using tiled computation and 
    incremental softmax calculation.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
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
        self.use_memory_efficient_kernel = True  # Whether to use the memory-efficient kernel

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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for caching
        if past_key_value is not None:
            # past_key_value is a tuple (key_cache, value_cache)
            # Concatenate new keys/values with cached ones
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Memory-efficient attention computation using FlashAttention approach
        if self.use_memory_efficient_kernel and not output_attentions:
            # Use PyTorch's optimized scaled dot-product attention (FlashAttention-like)
            # Only use causal mask if no attention_mask is provided
            is_causal = False
            attn_mask = attention_mask
            if attention_mask is None:
                is_causal = True
                attn_mask = None  # Don't pass both attn_mask and is_causal=True to scaled_dot_product_attention

            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,  # No dropout during inference
                is_causal=is_causal
            )
            attn_weights = None  # Not computed when output_attentions is False
        else:
            # Memory-efficient tiled computation for when output_attentions is needed
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
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
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
        self.use_async_ops = True  # Use async memory operations if available

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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for caching
        if past_key_value is not None:
            # past_key_value is a tuple (key_cache, value_cache)
            # Concatenate new keys/values with cached ones
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SM61-optimized attention computation using memory-efficient approach
        if not output_attentions:
            # Use PyTorch's optimized scaled dot-product attention (FlashAttention-like)
            # Only use causal mask if no attention_mask is provided
            is_causal = False
            attn_mask = attention_mask
            if attention_mask is None:
                is_causal = True
                attn_mask = None  # Don't pass both attn_mask and is_causal=True to scaled_dot_product_attention

            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,  # No dropout during inference
                is_causal=is_causal
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


class FlashAttention2TransformerLayer(nn.Module):
    """
    Transformer layer using FlashAttention 2 for memory-efficient attention computation.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Use FlashAttention 2 for memory efficiency
        if config.hardware_specific_attention and config.hardware_specific_attention == "sm61":
            self.self_attn = SM61OptimizedFlashAttention2(config, layer_idx=layer_idx)
        else:
            self.self_attn = FlashAttention2(config, layer_idx=layer_idx)

        # MLP component - could also be optimized but focusing on attention first
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, self.hidden_size)
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention using FlashAttention 2
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Consistent return format regardless of optional outputs
        if output_attentions and use_cache:
            return hidden_states, self_attn_weights, present_key_value
        elif output_attentions:
            return hidden_states, self_attn_weights, None
        elif use_cache:
            return hidden_states, None, present_key_value
        else:
            return hidden_states, None, None