"""
Adapter layers for Qwen3-VL.

This module contains adapter layers for parameter-efficient fine-tuning.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AdapterConfig:
    """
    Configuration for adapter layers.
    """
    adapter_dim: int = 64  # Bottleneck dimension for adapters
    adapter_scale: float = 1.0  # Scaling factor for adapter output
    adapter_dropout: float = 0.1  # Dropout probability for adapters
    use_adapter: bool = True  # Whether to use adapters
    adapter_layers: Tuple[int, ...] = (6, 12, 18, 24)  # Layers to place adapters in

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.adapter_dim <= 0:
            raise ValueError(f"adapter_dim must be positive, got {self.adapter_dim}")


class AdapterLayer(nn.Module):
    """
    A simple adapter layer implementation.
    """
    def __init__(self, hidden_size: int, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.down_proj = nn.Linear(hidden_size, config.adapter_dim, bias=False)
        self.up_proj = nn.Linear(config.adapter_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(config.adapter_dropout)
        self.scale = config.adapter_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter layer.
        """
        residual = x
        down = self.down_proj(x)
        activated = torch.nn.functional.relu(down)
        up = self.up_proj(activated)
        up = self.dropout(up)
        return residual + up * self.scale