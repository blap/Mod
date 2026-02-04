"""
Rotary Positional Embeddings for Qwen3-Coder-Next
"""

import torch
import torch.nn as nn

class Qwen3CoderNextRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=262144, base=1000000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `forward` faster
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from standard RoPE, we might use specific interleaving or complex construction
        # Standard implementation:
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Ensure RoPE is applied to a portion of the head dimension if necessary
    # Or expand cos/sin to match head dim if the intention was full rotation

    # In Qwen3-Coder-Next, rope_dim might be < head_dim (e.g. 64 vs 256)
    # If so, we only rotate the first `rope_dim` features

    rope_dim = cos.shape[-1]

    q_rot = q[..., :rope_dim]
    q_pass = q[..., rope_dim:]
    k_rot = k[..., :rope_dim]
    k_pass = k[..., rope_dim:]

    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    # Ensure cos/sin match the rotated part dimension
    cos = cos[..., :rope_dim]
    sin = sin[..., :rope_dim]

    # Check if implicit broadcasting is causing dimension mismatch
    # If q_rot is [B, H, S, D] and cos is [B, 1, S, D], it should work.

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)
