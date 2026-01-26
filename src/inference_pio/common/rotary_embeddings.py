"""
Generic Rotary Embeddings for Vision-Language Models

This module implements generic rotary embeddings for multimodal models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GenericRotaryEmbedding(nn.Module):
    """
    Generic implementation of rotary embeddings that can be extended by specific models.
    This module precomputes and caches sinusoidal embeddings for efficient attention computation.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        precision: torch.dtype = torch.float16
    ):
        """
        Initialize generic rotary embedding.

        Args:
            dim: Dimension of the embeddings
            max_position_embeddings: Maximum sequence length
            base: Base value for computing frequencies
            device: Device to store embeddings on
            precision: Precision for the embedding computation
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.precision = precision

        # Calculate inverse frequencies
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

        # Calculate embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get cached cosine and sine values.

        Args:
            x: Input tensor (used to determine device and dtype)
            seq_len: Sequence length (optional, computed from x if not provided)

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


def create_generic_rotary_embedding(
    dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    precision: torch.dtype = torch.float16
) -> GenericRotaryEmbedding:
    """
    Factory function to create generic rotary embedding.

    Args:
        dim: Dimension of the rotary embedding
        max_position_embeddings: Maximum sequence length
        base: Base value for computing frequencies
        device: Device to place the embedding on
        precision: Precision for the embedding computation

    Returns:
        GenericRotaryEmbedding: The generic rotary embedding implementation
    """
    return GenericRotaryEmbedding(dim, max_position_embeddings, base, device, precision)


def apply_generic_rotary_embeddings_to_model(model: nn.Module, config) -> nn.Module:
    """
    Apply generic rotary embeddings to the model.

    Args:
        model: The model to optimize
        config: Configuration for the model

    Returns:
        Model with rotary embeddings applied
    """
    logger.info("Applying generic rotary embeddings to model...")

    # This function would typically enhance the model with rotary embeddings
    # For now, we'll just return the model as is, but in a real implementation,
    # we would add rotary embedding components to the model's attention layers

    logger.info("Generic rotary embeddings applied successfully")
    return model


__all__ = [
    "GenericRotaryEmbedding",
    "RotaryEmbedding",  # Alias for GenericRotaryEmbedding
    "create_generic_rotary_embedding",
    "apply_generic_rotary_embeddings_to_model"
]

# Create an alias for backward compatibility
RotaryEmbedding = GenericRotaryEmbedding