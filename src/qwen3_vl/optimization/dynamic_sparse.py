"""
Dynamic sparse attention components for Qwen3-VL.
Implements sparse attention mechanisms that dynamically select important tokens
to reduce computational complexity while maintaining performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

class DynamicSparseAttention(nn.Module):
    """
    Implementation of Dynamic Sparse Attention that keeps only the top-k most relevant
    tokens based on attention scores, plus a local window.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Sparsity parameters
        self.sparsity_factor = getattr(config, 'sparsity_factor', 4) # Retain 1/4 tokens by default
        self.window_size = getattr(config, 'local_window_size', 32)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate initial attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, q_len):
                 attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights + attention_mask

        # Dynamic Sparsity Selection
        # We want to keep top-k values + local window
        if q_len > self.window_size * 2:
            # 1. Local Window Mask
            # Create a band mask for local window
            indices = torch.arange(q_len, device=hidden_states.device)
            dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
            local_mask = dist <= self.window_size
            local_mask = local_mask.unsqueeze(0).unsqueeze(0) # [1, 1, q_len, q_len]

            # 2. Top-K Selection (Global tokens)
            k = max(1, q_len // self.sparsity_factor)

            # Find top-k values for each query
            # We don't want to select from masked positions (large negative values)
            # but topk handles them correctly (they will be at the bottom)
            topk_values, topk_indices = torch.topk(attn_weights, k, dim=-1)

            # Create a sparse mask from top-k
            # This is memory intensive for large sequences, but fine for "implementation" phase
            sparse_mask = torch.zeros_like(attn_weights, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_indices, True)

            # Combine masks: keep if (Local OR Top-K)
            final_mask = local_mask | sparse_mask

            # Apply sparsity: mask out non-selected elements with -inf
            # We use a very small number instead of -inf to avoid NaNs in gradient
            min_dtype = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.where(final_mask, attn_weights, torch.tensor(min_dtype, dtype=attn_weights.dtype))

        # Softmax and output
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class VisionDynamicSparseAttention(nn.Module):
    """
    Dynamic Sparse Attention optimized for 2D vision tokens.
    It prioritizes spatial locality and high-activation features.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config.attention_bias)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Vision sparsity: focus on "foreground" or high-activation areas
        self.keep_ratio = getattr(config, 'vision_sparsity_ratio', 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention score calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Vision-specific sparsity
        # Calculate "importance" of each token based on its max attention score to others
        # Tokens that are attended to strongly by many others are "important" (e.g. objects)
        token_importance = attn.sum(dim=-2) # [bsz, num_heads, seq_len]

        # Select top-k important tokens to keep in the key/value set for the next layers?
        # Or here, we just mask out low-importance interactions

        # For this implementation, we simply sparsify the attention matrix
        k_tokens = int(seq_len * self.keep_ratio)
        if k_tokens < seq_len:
            top_val, _ = torch.topk(attn, k_tokens, dim=-1)
            # Threshold is the k-th largest value
            threshold = top_val[..., -1].unsqueeze(-1)

            # Mask values below threshold
            mask = attn < threshold
            attn = attn.masked_fill(mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        x = self.proj(x)
        return x
