"""
Rotary embeddings for Qwen3-VL model
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Replicate keys and values n_rep times along the head dimension.

    Args:
        hidden_states: [batch_size, num_key_value_heads, seq_len, head_dim]
        n_rep: Number of times to replicate

    Returns:
        Replicated hidden states: [batch_size, num_attention_heads, seq_len, head_dim]
    """
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies rotary position embeddings to query and key tensors."""
    # Index cos and sin using position_ids to get the embeddings for the specific positions
    # cos, sin shape after squeeze: [max_pos+1, dim] - embeddings for all possible positions up to max position
    # position_ids shape: [bs, seq_len]
    # cos[position_ids] shape: [bs, seq_len, dim]
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim] -> actually [max_pos+1, dim] where max_pos is max position id
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim] -> actually [max_pos+1, dim] where max_pos is max position id
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]

        # Get the maximum position id to create embeddings for all positions up to that point
        max_pos = torch.max(position_ids).item() + 1
        max_pos = min(max_pos, self.max_position_embeddings)  # Cap at max_position_embeddings

        # Create embeddings for all positions from 0 to max_pos
        t = torch.arange(max_pos, dtype=torch.float32, device=position_ids.device)
        freqs = torch.outer(t, self.inv_freq)  # [max_pos, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, dim]
        cos = emb.cos()  # [max_pos, dim]
        sin = emb.sin()  # [max_pos, dim]

        # Return embeddings for the requested positions
        # Expand cos and sin to match expected dimensions [bs, 1, max_pos, dim]
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, max_pos, dim]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, max_pos, dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)