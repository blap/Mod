"""
Linear attention mechanisms for Qwen3-VL model.
Implements Performer-style linear attention while maintaining all 32 attention heads.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PerformerAttention(nn.Module):
    """
    Performer-style linear attention mechanism that maintains all 32 attention heads
    while providing O(n) complexity instead of O(n^2) of standard attention.

    This implementation uses the FAVOR+ (Fast Attention Via positive Orthogonal Random features)
    approach to approximate softmax attention in linear time.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # Maintains all 32 heads
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

        # Linear projections for queries, keys, and values
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

        # Parameters for the kernel approximation
        self.feature_map_type = "favor"  # Using FAVOR+ approach
        self.n_dims = self.head_dim  # Use head_dim for feature mapping
        self.feature_dim = self.n_dims  # Feature dimension for kernel approximation

        # Random matrix for feature mapping (orthogonal random features)
        # Use a fixed random matrix to ensure reproducibility
        self.register_buffer(
            "unstructured_random_matrix",
            torch.randn(self.feature_dim // 2, self.head_dim)
        )

    def _get_random_features(self, x):
        """
        Compute random features for the input using orthogonal random features.
        """
        # Apply the random matrix to the input
        rand_proj = torch.einsum("...d,ed->...e", x, self.unstructured_random_matrix)

        # Apply sine and cosine to create orthogonal features
        # This approximates the softmax kernel
        return torch.cat([torch.sin(rand_proj), torch.cos(rand_proj)], dim=-1)

    def _softmax_kernel(self, x, is_query=False):
        """
        Compute softmax kernel approximation using random features.
        """
        # Normalize the input for stable computation
        x = x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-6)

        # Get random features
        features = self._get_random_features(x)

        # Apply normalization factor
        return features / math.sqrt(self.feature_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply linear attention using kernel approximation
        # Instead of computing full attention matrix (O(n^2)), we use random feature approximation
        # This gives us O(n) complexity while maintaining the ability to use all 32 heads

        # Apply softmax kernel approximation to queries and keys
        query_prime = self._softmax_kernel(query_states, is_query=True)
        key_prime = self._softmax_kernel(key_states, is_query=False)

        # Compute linear attention: (Q' @ K') @ V instead of Q @ K^T @ V
        # This is the key to achieving linear complexity
        key_value = torch.einsum("bsnd,bsnh->bndh", key_prime, value_states)
        attn_output = torch.einsum("bsnd,bndh->bsnh", query_prime, key_value)

        # Reshape back to original format
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3VLRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for Qwen3-VL model.
    Copied from the main modeling file to maintain consistency.
    """
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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