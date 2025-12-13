"""
Consolidated Sparse Attention Module for Qwen3-VL Model
Combines dynamic_sparse_attention.py, dynamic_sparse_attention_optimized.py, 
block_sparse_attention.py, linear_attention.py, and memory_efficient_patterns.py
"""
import math
import warnings
from typing import Optional, Tuple, Union, List
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
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
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
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        # Handle case where num_key_value_heads is None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
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

        # Initialize rotary embeddings if needed
        self.rotary_emb = None
        if hasattr(config, 'rope_theta'):
            try:
                from optimization.rotary_embeddings import Qwen3VLRotaryEmbedding
                self.rotary_emb = Qwen3VLRotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
                    base=getattr(config, 'rope_theta', 1000000)
                )
            except ImportError:
                # Fallback implementation if optimization.rotary_embeddings module is not available
                try:
                    from attention.rotary_embeddings import Qwen3VLRotaryEmbedding
                    self.rotary_emb = Qwen3VLRotaryEmbedding(
                        dim=self.head_dim,
                        max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
                        base=getattr(config, 'rope_theta', 1000000)
                    )
                except ImportError:
                    # If no rotary embedding module is available, skip initialization
                    pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None
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

        # Apply rotary embeddings if available
        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Update KV cache if provided
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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

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
        sparse_mask.scatter_(-1, top_k_indices, top_k_values)

        return sparse_mask


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
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
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
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

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
        cache_position: Optional[torch.LongTensor] = None
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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _apply_block_sparse_attention(self, query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
        """
        Apply block-sparse attention pattern to reduce computation.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        _, _, kv_len, _ = key_states.shape

        # Calculate block dimensions
        block_size = self.block_size
        num_q_blocks = math.ceil(q_len / block_size)
        num_kv_blocks = math.ceil(kv_len / block_size)

        # Pad sequences to be divisible by block size
        padded_q_len = num_q_blocks * block_size
        padded_kv_len = num_kv_blocks * block_size

        if q_len != padded_q_len or kv_len != padded_kv_len:
            query_states = F.pad(query_states, (0, 0, 0, padded_q_len - q_len), value=0)
            key_states = F.pad(key_states, (0, 0, 0, padded_kv_len - kv_len), value=0)

        # Reshape to block format
        query_blocks = query_states.view(bsz, num_heads, num_q_blocks, block_size, head_dim)
        key_blocks = key_states.view(bsz, num_heads, num_kv_blocks, block_size, head_dim)

        # Get sparsity pattern for current sequence length
        # Ensure we don't exceed the available pattern dimensions
        available_q_blocks = min(num_q_blocks, self.sparsity_pattern.size(1))
        available_kv_blocks = min(num_kv_blocks, self.sparsity_pattern.size(2))
        current_sparsity_pattern = self.sparsity_pattern[:, :available_q_blocks, :available_kv_blocks]

        # Apply learned sparsity pattern with top-k selection to enforce sparsity
        k = max(1, int(current_sparsity_pattern.numel() * self.sparsity_ratio / num_heads))
        if k > 0 and current_sparsity_pattern.numel() > 0:
            flat_pattern = current_sparsity_pattern.view(num_heads, -1)
            top_k_values, top_k_indices = torch.topk(flat_pattern, k=min(k, flat_pattern.size(1)), dim=-1)
            sparsity_threshold = top_k_values[:, -1].view(num_heads, 1, 1)
            sparse_mask = (current_sparsity_pattern > sparsity_threshold).float()
        else:
            # If no sparsity can be applied, use full attention
            sparse_mask = torch.ones_like(current_sparsity_pattern)

        # Initialize attention weights tensor
        attn_weights = torch.zeros(bsz, num_heads, padded_q_len, padded_kv_len,
                                   dtype=query_states.dtype, device=query_states.device)

        # Compute attention only for non-zero blocks in the sparse pattern
        for h_idx in range(num_heads):
            for q_block_idx in range(available_q_blocks):
                for kv_block_idx in range(available_kv_blocks):
                    if sparse_mask[h_idx, q_block_idx, kv_block_idx] > 0:
                        # Compute attention for this block pair
                        q_block = query_blocks[:, h_idx, q_block_idx, :, :]  # [bsz, block_size, head_dim]
                        k_block = key_blocks[:, h_idx, kv_block_idx, :, :]   # [bsz, block_size, head_dim]

                        block_attn = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        attn_weights[:, h_idx,
                                    q_block_idx * block_size:min((q_block_idx + 1) * block_size, padded_q_len),
                                    kv_block_idx * block_size:min((kv_block_idx + 1) * block_size, padded_kv_len)] = block_attn

        # Trim back to original sequence length
        attn_weights = attn_weights[:, :, :q_len, :kv_len]

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
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
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
        cache_position: Optional[torch.LongTensor] = None
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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

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
        # This assumes that vision tokens might be handled differently based on sequence characteristics
        current_sparsity_ratio = self.sparsity_ratio
        if q_len > 512:  # Heuristic for vision tokens
            current_sparsity_ratio = self.vision_sparsity_ratio

        # Calculate how many tokens to attend to per head
        k = max(1, int(kv_len * current_sparsity_ratio))

        # For each head, select top-k tokens based on routing scores
        # routing_scores is [bsz, seq_len, num_heads], we need [bsz, num_heads, seq_len]
        routing_scores_t = routing_scores.transpose(1, 2)  # [bsz, num_heads, seq_len]

        # Get top-k routing scores for each head
        top_k_routing_values, top_k_indices = torch.topk(routing_scores_t, k=min(k, routing_scores_t.size(-1)), dim=-1)  # [bsz, num_heads, k]

        # Create a sparse attention mask
        sparse_attn_weights = torch.full_like(attn_weights, float('-inf'))

        # For each head, fill the sparse attention matrix with values for the selected tokens
        for h_idx in range(num_heads):
            for batch_idx in range(bsz):
                selected_kv_indices = top_k_indices[batch_idx, h_idx, :]  # [k]
                # Fill the attention weights for this head and batch with the selected keys/values
                sparse_attn_weights[batch_idx, h_idx, :, selected_kv_indices] = attn_weights[batch_idx, h_idx, :, selected_kv_indices]

        return sparse_attn_weights


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
        k = max(1, int(self.sparsity_ratio * seq_len))
        k = min(k, seq_len)  # Ensure k doesn't exceed sequence length

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Get top-k values and indices efficiently using torch.topk
        top_k_values, top_k_indices = torch.topk(attn_weights, k=k, dim=-1, sorted=False)

        # Create a mask to store sparse attention weights
        sparse_attn_weights = torch.full_like(attn_weights, float('-inf'))

        # Create index tensors for advanced indexing
        batch_indices = torch.arange(bsz, device=attn_weights.device).view(-1, 1, 1, 1).expand(-1, num_heads, seq_len, k)
        head_indices = torch.arange(num_heads, device=attn_weights.device).view(1, -1, 1, 1).expand(bsz, -1, seq_len, k)
        query_indices = torch.arange(seq_len, device=attn_weights.device).view(1, 1, -1, 1).expand(bsz, num_heads, -1, k)

        # Scatter the top-k values back to the sparse attention matrix
        sparse_attn_weights.scatter_(-1, top_k_indices.unsqueeze(-2).expand(-1, -1, -1, k), top_k_values)

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
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None
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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

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
        routing_scores = routing_logits  # [bsz, seq_len, num_heads] - already softmax applied
        return routing_scores

    def _apply_dynamic_sparsity(self, attn_weights: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic sparsity based on routing scores.
        """
        bsz, num_heads, q_len, kv_len = attn_weights.size()

        # Adjust sparsity ratio based on whether we're processing vision or text tokens
        current_sparsity_ratio = self.sparsity_ratio
        if q_len > 1000:  # Heuristic for vision tokens
            current_sparsity_ratio = self.vision_sparsity_ratio

        # Calculate how many tokens to attend to per head
        k = max(1, int(kv_len * current_sparsity_ratio))

        # For each head, select top-k tokens based on routing scores
        # routing_scores is [bsz, seq_len, num_heads], we need [bsz, num_heads, seq_len]
        routing_scores_t = routing_scores.transpose(1, 2)  # [bsz, num_heads, seq_len]

        # Get top-k routing scores for each head
        top_k_values, top_k_indices = torch.topk(routing_scores_t, k=min(k, routing_scores_t.size(-1)), dim=-1)  # [bsz, num_heads, k]

        # Create a sparse attention mask
        sparse_attn_weights = torch.full_like(attn_weights, float('-inf'))

        # For each head, fill the sparse attention matrix with values for the selected tokens
        for h_idx in range(num_heads):
            for batch_idx in range(bsz):
                selected_kv_indices = top_k_indices[batch_idx, h_idx, :]  # [k]
                # Fill the attention weights for this head and batch with the selected keys/values
                sparse_attn_weights[batch_idx, h_idx, :, selected_kv_indices] = attn_weights[batch_idx, h_idx, :, selected_kv_indices]

        return sparse_attn_weights


class VisionDynamicSparseAttention(nn.Module):
    """
    Dynamic sparse attention specifically optimized for vision processing.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.sparsity_ratio = getattr(config, 'vision_sparse_attention_sparsity_ratio', 0.4)

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)

        # Vision-specific routing network
        self.routing_network = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.num_heads),
            nn.Sigmoid()  # Sigmoid for vision token selection probabilities
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Project Q, K, V
        qkv = self.qkv_proj(hidden_states).reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv.unbind(0)  # [batch, num_heads, seq_len, head_dim]

        # Compute routing scores to determine important vision tokens
        routing_scores = self._compute_routing_scores(hidden_states)  # [bsz, seq_len, num_heads]

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply dynamic sparsity based on routing scores
        sparse_attn_weights = self._apply_dynamic_sparsity(attn_weights, routing_scores)

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _compute_routing_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores for vision token selection.
        """
        # Use the routing network to determine which vision tokens are important
        routing_logits = self.routing_network(hidden_states)  # [bsz, seq_len, num_heads]
        routing_scores = torch.sigmoid(routing_logits)  # [bsz, seq_len, num_heads]
        return routing_scores

    def _apply_dynamic_sparsity(self, attn_weights: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic sparsity based on routing scores for vision tokens.
        """
        bsz, num_heads, q_len, kv_len = attn_weights.size()

        # Calculate how many tokens to attend to per head
        k = max(1, int(kv_len * self.sparsity_ratio))

        # For each head, select top-k tokens based on routing scores
        # routing_scores is [bsz, seq_len, num_heads], we need [bsz, num_heads, seq_len]
        routing_scores_t = routing_scores.transpose(1, 2)  # [bsz, num_heads, seq_len]

        # Get top-k routing scores for each head
        top_k_values, top_k_indices = torch.topk(routing_scores_t, k=min(k, routing_scores_t.size(-1)), dim=-1)  # [bsz, num_heads, k]

        # Create a sparse attention mask
        sparse_attn_weights = torch.full_like(attn_weights, float('-inf'))

        # For each head, fill the sparse attention matrix with values for the selected tokens
        for h_idx in range(num_heads):
            for batch_idx in range(bsz):
                selected_kv_indices = top_k_indices[batch_idx, h_idx, :]  # [k]
                # Fill the attention weights for this head and batch with the selected keys/values
                sparse_attn_weights[batch_idx, h_idx, :, selected_kv_indices] = attn_weights[batch_idx, h_idx, :, selected_kv_indices]

        return sparse_attn_weights


class OptimizedVisionDynamicSparseAttention(nn.Module):
    """
    Optimized version of vision dynamic sparse attention with hardware-specific optimizations.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.sparsity_ratio = getattr(config, 'vision_sparse_attention_sparsity_ratio', 0.4)

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections with fused QKV for efficiency
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)

        # Optimized vision-specific routing network
        self.routing_network = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 8),  # Even smaller for efficiency
            nn.ReLU(),
            nn.Linear(self.embed_dim // 8, self.num_heads),
            nn.Softmax(dim=-2)  # Softmax over sequence dimension for vision
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states).reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv.unbind(0)  # [batch, num_heads, seq_len, head_dim]

        # Compute routing scores for vision tokens
        routing_scores = self._compute_routing_scores(hidden_states)  # [bsz, seq_len, num_heads]

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply optimized vision-specific dynamic sparsity
        sparse_attn_weights = self._apply_dynamic_sparsity(attn_weights, routing_scores)

        # Apply softmax
        attn_weights = nn.functional.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _compute_routing_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores for optimized vision token selection.
        """
        # Use the optimized routing network to determine which vision tokens are important
        routing_logits = self.routing_network(hidden_states)  # [bsz, seq_len, num_heads]
        routing_scores = routing_logits  # [bsz, seq_len, num_heads] - already softmax applied
        return routing_scores

    def _apply_dynamic_sparsity(self, attn_weights: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic sparsity based on routing scores for optimized vision processing.
        """
        bsz, num_heads, q_len, kv_len = attn_weights.size()

        # Calculate how many tokens to attend to per head
        k = max(1, int(kv_len * self.sparsity_ratio))

        # For each head, select top-k tokens based on routing scores
        # routing_scores is [bsz, seq_len, num_heads], we need [bsz, num_heads, seq_len]
        routing_scores_t = routing_scores.transpose(1, 2)  # [bsz, num_heads, seq_len]

        # Get top-k routing scores for each head
        top_k_values, top_k_indices = torch.topk(routing_scores_t, k=min(k, routing_scores_t.size(-1)), dim=-1)  # [bsz, num_heads, k]

        # Create a sparse attention mask
        sparse_attn_weights = torch.full_like(attn_weights, float('-inf'))

        # For each head, fill the sparse attention matrix with values for the selected tokens
        for h_idx in range(num_heads):
            for batch_idx in range(bsz):
                selected_kv_indices = top_k_indices[batch_idx, h_idx, :]  # [k]
                # Fill the attention weights for this head and batch with the selected keys/values
                sparse_attn_weights[batch_idx, h_idx, :, selected_kv_indices] = attn_weights[batch_idx, h_idx, :, selected_kv_indices]

        return sparse_attn_weights


class LinearAttention(nn.Module):
    """
    Linear attention mechanism that computes attention in linear time.
    Uses feature map approximation for efficiency.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

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

        # Feature map for linear attention (using elu + 1 for positive features)
        self.feature_map = lambda x: F.elu(x) + 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None
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

        # Apply rotary position embeddings if provided
        if position_ids is not None:
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

        # Apply feature map to queries and keys for linear attention
        query_states = self.feature_map(query_states)
        key_states = self.feature_map(key_states)

        # Compute linear attention: (QK^T)V -> (Q(VK^T)) where we compute KV^T first
        # This changes complexity from O(n^2) to O(n) in sequence length
        key_value = torch.einsum("bhld,bhlv->bhvd", key_states, value_states)
        attn_output = torch.einsum("bhqd,bhvd->bhqv", query_states, key_value)

        # Normalize by sum of attention weights
        normalizer = torch.einsum("bhld,bhl->bhd", key_states, torch.ones_like(key_states[..., 0]))
        normalizer = normalizer.unsqueeze(-2)  # Add sequence dimension
        attn_output = attn_output / (normalizer + 1e-10)  # Normalize

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MemoryEfficientPatterns(nn.Module):
    """
    Memory-efficient attention patterns with chunked processing.
    Implements flash_attention_chunked_forward and related patterns.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.chunk_size = getattr(config, 'attention_chunk_size', 512)

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
        cache_position: Optional[torch.LongTensor] = None
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

        # Compute attention using memory-efficient chunked processing
        attn_output = self._chunked_attention_forward(
            query_states, key_states, value_states, attention_mask
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _chunked_attention_forward(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage from O(n²) to O(n).
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, _, kv_seq_len, _ = key_states.shape

        # Process in chunks to limit memory usage
        chunk_size = min(self.chunk_size, seq_len)
        attn_output = torch.zeros_like(query_states)

        # Initialize normalization terms for incremental softmax
        max_scores = torch.full((bsz, num_heads, seq_len), float('-inf'), device=query_states.device, dtype=torch.float32)
        sums = torch.zeros((bsz, num_heads, seq_len), device=query_states.device, dtype=torch.float32)

        for q_start in range(0, seq_len, chunk_size):
            q_end = min(q_start + chunk_size, seq_len)
            q_chunk = query_states[:, :, q_start:q_end, :]  # [bsz, num_heads, chunk_size, head_dim]

            # Process key-value pairs in chunks for this query chunk
            for kv_start in range(0, kv_seq_len, chunk_size):
                kv_end = min(kv_start + chunk_size, kv_seq_len)
                k_chunk = key_states[:, :, kv_start:kv_end, :]  # [bsz, num_heads, chunk_size, head_dim]
                v_chunk = value_states[:, :, kv_start:kv_end, :]  # [bsz, num_heads, chunk_size, head_dim]

                # Compute attention scores for this chunk pair
                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(self.head_dim)

                if attention_mask is not None:
                    mask_chunk = attention_mask[:, :, q_start:q_end, kv_start:kv_end]
                    attn_scores = attn_scores + mask_chunk

                # Incremental softmax computation
                # Get new max for numerical stability
                new_max_scores = torch.maximum(max_scores[:, :, q_start:q_end], torch.max(attn_scores, dim=-1, keepdim=True)[0].to(torch.float32))
                
                # Compute scaling factors
                exp_old = torch.exp(max_scores[:, :, q_start:q_end] - new_max_scores)
                exp_new = torch.exp(attn_scores - new_max_scores)

                # Update sums
                sums[:, :, q_start:q_end] = sums[:, :, q_start:q_end] * exp_old + torch.sum(exp_new, dim=-1)

                # Update max scores
                max_scores[:, :, q_start:q_end] = new_max_scores

                # Compute weighted values
                weighted_values = torch.matmul(exp_new, v_chunk)

                # Accumulate attention output
                attn_output[:, :, q_start:q_end, :] += weighted_values

        # Final normalization
        attn_output = attn_output / (sums.unsqueeze(-1) + 1e-10)

        return attn_output


def flash_attention_chunked_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Chunked forward pass for flash attention implementation.
    This function implements memory-efficient attention computation using chunked processing.
    """
    # This is a simplified version - in practice, FlashAttention uses more complex algorithms
    # like the ones described in the paper "FlashAttention: Fast and Memory-Efficient Exact Attention"
    if softmax_scale is None:
        softmax_scale = query.size(-1) ** -0.5

    batch_size, num_heads, q_seq_len, head_dim = query.size()
    _, _, kv_seq_len, _ = key.size()

    # For simplicity, we'll implement a basic chunked attention
    # In practice, FlashAttention uses more sophisticated tiling and incremental softmax
    chunk_size = min(512, q_seq_len)  # Use 512 as default chunk size
    output = torch.zeros_like(query)

    for q_start in range(0, q_seq_len, chunk_size):
        q_end = min(q_start + chunk_size, q_seq_len)
        q_chunk = query[:, :, q_start:q_end, :]  # [batch, num_heads, chunk_size, head_dim]

        # Compute attention scores for this query chunk with all keys
        attn_scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * softmax_scale  # [batch, num_heads, chunk_size, kv_seq_len]

        if causal:
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(q_end - q_start, kv_seq_len, dtype=torch.bool, device=query.device),
                diagonal=q_start + 1
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size, kv_seq_len]
            attn_scores.masked_fill_(causal_mask, float('-inf'))

        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, q_start:q_end, :]  # [batch, 1, chunk_size, kv_seq_len]
            attn_scores = attn_scores + mask_chunk

        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout_p > 0.0:
            attn_probs = F.dropout(attn_probs, p=dropout_p)

        # Apply attention to values
        output_chunk = torch.matmul(attn_probs, value)  # [batch, num_heads, chunk_size, head_dim]

        # Store output chunk
        output[:, :, q_start:q_end, :] = output_chunk

    return output


class BlockSparseAttentionFactory:
    """
    Factory for creating different types of block sparse attention mechanisms.
    """
    @staticmethod
    def create_attention(config, layer_idx: Optional[int] = None, attention_type: str = "standard"):
        """
        Create an attention mechanism based on the specified type.
        """
        if attention_type == "true_sparse":
            return TrueSparseAttention(config, layer_idx)
        elif attention_type == "block_sparse":
            return BlockSparseAttention(config, layer_idx)
        elif attention_type == "dynamic_sparse":
            return DynamicSparseAttention(config, layer_idx)
        elif attention_type == "vectorized_sparse":
            return OptimizedDynamicSparseAttention(config, layer_idx)
        elif attention_type == "vision_sparse":
            return VisionDynamicSparseAttention(config, layer_idx)
        elif attention_type == "optimized_vision_sparse":
            return OptimizedVisionDynamicSparseAttention(config, layer_idx)
        elif attention_type == "linear":
            return LinearAttention(config, layer_idx)
        elif attention_type == "memory_efficient":
            return MemoryEfficientPatterns(config, layer_idx)
        else:
            # Default to standard attention
            from attention.standard_attention import StandardAttention
            return StandardAttention(config, layer_idx)