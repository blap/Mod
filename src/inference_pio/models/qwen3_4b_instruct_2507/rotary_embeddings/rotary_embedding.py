"""
Qwen3-4B-Instruct-2507 Rotary Embedding Implementation

This module implements optimized rotary embeddings for the Qwen3-4B-Instruct-2507 model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Qwen34BRotaryEmbedding(nn.Module):
    """
    Qwen3-4B-Instruct-2507 specific Rotary Embedding implementation.

    This implementation is optimized for the Qwen3-4B-Instruct-2507 model's architecture
    and provides efficient rotary position embeddings for attention mechanisms.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,  # Qwen3-4B specific
        base: float = 1000000.0,  # Qwen3-4B specific base
        precision: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        """
        Initialize Qwen3-4B-Instruct-2507 Rotary Embedding.

        Args:
            dim: Dimension of the embeddings
            max_position_embeddings: Maximum sequence length (Qwen3-4B specific)
            base: Base value for computing frequencies (Qwen3-4B specific)
            precision: Precision for computations
            device: Device to store embeddings on
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.precision = precision

        # Calculate inverse frequencies
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin embeddings
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=device or "cpu", dtype=precision
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
        if seq_len is None:
            seq_len = x.shape[1]  # Get sequence length from input

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
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

    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine embeddings
            sin: Sine embeddings

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        cos = cos.unsqueeze(1)  # Add head dimension
        sin = sin.unsqueeze(1)  # Add head dimension

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
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


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine embeddings
        sin: Sine embeddings

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    cos = cos.unsqueeze(1)  # Add head dimension
    sin = sin.unsqueeze(1)  # Add head dimension

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


__all__ = ["Qwen34BRotaryEmbedding", "rotate_half", "apply_rotary_pos_emb"]
