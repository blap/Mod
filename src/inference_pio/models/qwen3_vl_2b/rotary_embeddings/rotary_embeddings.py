"""
Qwen3-VL-2B Rotary Embedding Implementation - Self-Contained Version

This module implements optimized rotary embeddings specifically for the Qwen3-VL-2B model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Qwen3VL2BRotaryEmbedding(nn.Module):
    """
    Qwen3-VL-2B specific Rotary Embedding implementation with optimizations.

    This implementation is optimized for the Qwen3-VL-2B model's architecture and parameters,
    including extended context length support and efficient computation patterns.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,  # Extended for Qwen3-VL-2B
        base: float = 1000000.0,  # Qwen3 specific base
        device: Optional[torch.device] = None,
        precision: torch.dtype = torch.float16
    ):
        """
        Initialize Qwen3-VL-2B Rotary Embedding.

        Args:
            dim: Dimension of the rotary embedding
            max_position_embeddings: Maximum sequence length (Qwen3-VL-2B specific)
            base: Base value for computing frequencies (Qwen3 specific)
            device: Device to place the embedding on
            precision: Precision for the embedding computation
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.precision = precision

        # Calculate inverse frequencies with Qwen3-specific parameters
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute position IDs and cos/sin embeddings up to max_position_embeddings
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=precision)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Precompute cos and sin embeddings for the given sequence length.
        """
        self.max_seq_len_cached = seq_len
        # Create position IDs tensor
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # Calculate frequencies
        freqs = torch.outer(t, self.inv_freq)

        # Pad freqs to match dimension if needed
        if freqs.size(1) < self.dim // 2:
            pad_size = self.dim // 2 - freqs.size(1)
            freqs = torch.cat([freqs, torch.zeros(freqs.size(0), pad_size, device=device, dtype=freqs.dtype)], dim=1)

        # Calculate embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get cached cosine and sine values.

        Args:
            x: Input tensor (used to determine device and dtype)
            seq_len: Current sequence length (optional, computed from x if not provided)

        Returns:
            Tuple of (cos, sin) embeddings for the current sequence length
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Re-create cache if needed for longer sequence
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=self.precision)

        # Return cos and sin embeddings for the current sequence length
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
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

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine embeddings
            sin: Sine embeddings

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Ensure dimensions match
        cos = cos[None, :q.shape[1], None, :]  # [1, seq_len, 1, dim]
        sin = sin[None, :q.shape[1], None, :]  # [1, seq_len, 1, dim]

        # Apply rotary embeddings
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


def create_qwen3_vl_rotary_embedding(
    dim: int,
    max_position_embeddings: int = 32768,
    base: float = 1000000.0,
    device: Optional[torch.device] = None,
    precision: torch.dtype = torch.float16
) -> Qwen3VL2BRotaryEmbedding:
    """
    Factory function to create Qwen3-VL-2B rotary embedding.

    Args:
        dim: Dimension of the rotary embedding
        max_position_embeddings: Maximum sequence length
        base: Base value for computing frequencies
        device: Device to place the embedding on
        precision: Precision for the embedding computation

    Returns:
        Qwen3VL2BRotaryEmbedding: The Qwen3-VL-2B rotary embedding implementation
    """
    return Qwen3VL2BRotaryEmbedding(dim, max_position_embeddings, base, device, precision)


__all__ = [
    "Qwen3VL2BRotaryEmbedding",
    "create_qwen3_vl_rotary_embedding"
]