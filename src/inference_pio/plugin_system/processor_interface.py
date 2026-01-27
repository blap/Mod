"""
Processor Plugin Interface

This module defines the interface for processor-specific plugins that handle
low-level operations (like matrix multiplication, attention, activation functions)
optimized for specific hardware architectures (Intel AVX, Apple NEON, Generic CPU, etc).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
from torch import Tensor

class ProcessorPluginInterface(ABC):
    """
    Abstract base class for processor plugins.
    Plugins implement specific optimizations for mathematical operations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the plugin."""
        pass

    @property
    @abstractmethod
    def supported_architectures(self) -> list[str]:
        """List of supported architectures (e.g., 'x86_64', 'arm64')."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.
        """
        pass

    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Perform matrix multiplication optimized for the processor.
        """
        pass

    @abstractmethod
    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False
    ) -> Tensor:
        """
        Perform scaled dot product attention.
        """
        pass

    @abstractmethod
    def apply_activation(self, x: Tensor, activation_type: str) -> Tensor:
        """
        Apply an activation function (e.g., 'silu', 'gelu', 'relu').
        """
        pass

    @abstractmethod
    def manage_threads(self, num_threads: int):
        """
        Set the number of threads for parallel execution.
        """
        pass
