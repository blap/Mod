"""
Cross modal compression components for Qwen3-VL.
Reduces memory footprint by compressing multimodal representations.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class CrossModalMemoryCompressor(nn.Module):
    """
    Compresses visual and textual tokens into a unified, smaller representation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.compression_rate = getattr(config, 'memory_compression_rate', 2)
        hidden_size = getattr(config, 'hidden_size', 1024)

        # Simple compression: Linear projection to reduce dimensionality or token count
        # Here we implement a token pooling mechanism
        self.merger = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states: torch.Tensor, modality_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compresses the input sequence.
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        """
        batch, seq_len, dim = hidden_states.shape

        # If sequence is too short, don't compress
        if seq_len < 2:
            return hidden_states

        # Basic implementation: Pairwise token merging
        # Pad if odd length
        if seq_len % 2 != 0:
            hidden_states = torch.cat([hidden_states, hidden_states[:, -1:, :]], dim=1)
            seq_len += 1

        # Reshape to [batch, seq_len/2, 2, dim]
        reshaped = hidden_states.view(batch, seq_len // 2, 2, dim)

        # Concatenate pairs: [batch, seq_len/2, 2*dim]
        concatenated = torch.cat([reshaped[:, :, 0, :], reshaped[:, :, 1, :]], dim=-1)

        # Merge: [batch, seq_len/2, dim]
        compressed = self.merger(concatenated)

        return compressed
