"""
Generic Rotary Embeddings Module for Inference-PIO System
Dependency-Free Version
"""

import logging
from typing import Optional, Tuple
from ...core.engine.backend import Tensor, Module, precompute_freqs_cis

logger = logging.getLogger(__name__)

class GenericRotaryEmbedding(Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.cos_cached, self.sin_cached = precompute_freqs_cis(dim, max_position_embeddings, base, device)
        self.register_buffer("cos_cached", self.cos_cached)
        self.register_buffer("sin_cached", self.sin_cached)

    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_position_embeddings:
            # Recompute if needed (simple expansion support)
            self.max_position_embeddings = seq_len
            self.cos_cached, self.sin_cached = precompute_freqs_cis(self.dim, seq_len, self.base, x.device)

        start = [0, 0]
        shape = [seq_len, self.cos_cached.shape[1]]
        return self.cos_cached.slice(start, shape), self.sin_cached.slice(start, shape)

class Qwen3RotaryEmbedding(GenericRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
        device: str = "cpu",
    ):
        super().__init__(dim, max_position_embeddings, base, device)

# Aliases
Qwen34BRotaryEmbedding = Qwen3RotaryEmbedding
Qwen3CoderRotaryEmbedding = Qwen3RotaryEmbedding

def create_generic_rotary_embedding(dim: int, max_pos: int = 2048, base: float = 10000.0, device="cpu") -> GenericRotaryEmbedding:
    return GenericRotaryEmbedding(dim, max_pos, base, device)

def create_qwen3_rotary_embedding(dim: int, max_pos: int = 32768, base: float = 1000000.0, device="cpu") -> Qwen3RotaryEmbedding:
    return Qwen3RotaryEmbedding(dim, max_pos, base, device)

__all__ = [
    "GenericRotaryEmbedding",
    "Qwen3RotaryEmbedding",
    "Qwen34BRotaryEmbedding",
    "Qwen3CoderRotaryEmbedding",
    "create_generic_rotary_embedding",
    "create_qwen3_rotary_embedding"
]
