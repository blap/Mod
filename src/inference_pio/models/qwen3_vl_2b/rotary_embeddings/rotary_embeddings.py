"""
Rotary Embeddings for Inference-PIO System

This module provides the RotaryEmbedding class for applying rotary positional embeddings to attention mechanisms.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Embedding module that precomputes and caches sinusoidal embeddings.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[str] = None,
    ):
        """
        Initialize Rotary Embedding.

        Args:
            dim: Dimension of the embeddings
            max_position_embeddings: Maximum sequence length
            base: Base value for computing frequencies
            device: Device to store embeddings on
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: str, dtype: torch.dtype):
        """Precompute and cache cosine and sine values."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get cached cosine and sine values.

        Args:
            x: Input tensor (used to determine device and dtype)
            seq_len: Sequence length (if None, uses max_position_embeddings)

        Returns:
            Tuple of (cos, sin) tensors
        """
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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine tensor
        sin: Sine tensor
        position_ids: Position IDs (optional, used for indexing if provided)

    Returns:
        Tuple of (q_embed, k_embed)
    """
    # Note: Qwen models might have different rotary application (e.g. interleaved vs half-rotated)
    # The standard implementation usually follows:
    # (x * cos) + (rotate_half(x) * sin)

    # Check if cos/sin need to be indexed by position_ids
    if position_ids is not None:
        # Cos/Sin: (seq_len, dim) -> (batch, seq_len, dim) via indexing
        cos = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Assuming cos/sin are already expanded or broadcastable
        # Standard: (seq_len, dim) -> (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb", "rotate_half"]
