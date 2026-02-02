"""
GLM-4.7 Optimized Rotary Embeddings Implementation

This module implements optimized rotary embeddings for the GLM-4.7 model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import GLM47FlashConfig


class GLM47RotaryEmbedding(nn.Module):
    """
    Rotary Embedding implementation optimized for GLM-4.7 model.

    This implementation provides efficient computation of rotary embeddings
    with caching for commonly used sequence lengths to improve performance.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 1000000.0,
        precision: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.precision = precision
        self.device = device

        # Precompute frequencies
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

        # Initialize cache for cos and sin values
        self._setup_cache()

    def _setup_cache(self):
        """Set up the cache for cos and sin values."""
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cos_sin_cache(
        self, x: torch.Tensor, seq_len: int, position_ids: torch.Tensor
    ):
        """
        Update the cos and sin caches based on the current sequence length.
        """
        # Reconstruct actual sequence length from position_ids if provided
        if position_ids is not None and position_ids.numel() > 0:
            max_pos = int(position_ids.max().item()) + 1
            seq_len = max(max_pos, seq_len)

        # If the sequence length hasn't changed, return cached values
        if seq_len <= self._seq_len_cached:
            return self._cos_cached[:seq_len].to(x.dtype), self._sin_cached[
                :seq_len
            ].to(x.dtype)

        # Calculate new cos and sin values
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Update cached values
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)

        self._cos_cached = cos
        self._sin_cached = sin

        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Sequence length (optional, will be inferred from tensors if not provided)
            position_ids: Position IDs for each token in the sequence

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Determine sequence length
        if seq_len is None:
            seq_len = q.size(2)

        # Update cos and sin caches
        cos, sin = self._update_cos_sin_cache(q, seq_len, position_ids)

        # Apply rotary embeddings
        if position_ids is not None:
            # Use specific position IDs for each token
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        else:
            # Use sequential positions
            cos = cos[None, None, :, :]  # [1, 1, seq_len, dim]
            sin = sin[None, None, :, :]  # [1, 1, seq_len, dim]

        # Rotate query and key
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions of the input.

        Args:
            x: Input tensor of shape (..., head_dim)

        Returns:
            Rotated tensor of the same shape
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine values
        sin: Sine values
        position_ids: Position IDs (optional)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    if position_ids is not None:
        # Use specific position IDs
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def rotate_half(x):
    """
    Rotate half the hidden dimensions of the input.

    Args:
        x: Input tensor

    Returns:
        Rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


__all__ = ["GLM47RotaryEmbedding", "apply_rotary_pos_emb"]
