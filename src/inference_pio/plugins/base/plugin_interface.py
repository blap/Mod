"""
Hardware-Specific Processor Plugin Interface

This module defines the standardized interface for processor-specific plugins that handle
low-level operations (like matrix multiplication, attention, activation functions)
optimized for specific hardware architectures (Intel AVX, Apple NEON, Generic CPU, etc).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


class HardwareProcessorPluginInterface(ABC):
    """
    Abstract base class for hardware-specific processor plugins.

    This interface defines the contract for plugins that implement
    hardware-optimized operations for mathematical computations,
    including matrix multiplication, attention mechanisms, and activation functions.
    """

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Unique name of the hardware processor plugin."""
        raise NotImplementedError("Method not implemented")

    @property
    @abstractmethod
    def supported_hardware_architectures(self) -> list[str]:
        """List of supported hardware architectures (e.g., 'x86_64', 'arm64')."""
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the hardware processor plugin with configuration.

        Args:
            config: Configuration dictionary for plugin initialization

        Returns:
            True if initialization was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def optimized_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Perform matrix multiplication optimized for the specific hardware processor.

        Args:
            a: First input tensor
            b: Second input tensor

        Returns:
            Result of matrix multiplication
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def optimized_scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Perform scaled dot product attention optimized for the hardware.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Attention mask (optional)
            dropout_p: Dropout probability (default 0.0)
            is_causal: Whether to apply causal masking (default False)

        Returns:
            Attention output tensor
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def optimized_apply_activation(self, x: Tensor, activation_type: str) -> Tensor:
        """
        Apply an activation function optimized for the hardware (e.g., 'silu', 'gelu', 'relu').

        Args:
            x: Input tensor
            activation_type: Type of activation function to apply

        Returns:
            Tensor with activation applied
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def configure_thread_management(self, num_threads: int):
        """
        Configure the number of threads for parallel execution on the hardware.

        Args:
            num_threads: Number of threads to use for parallel execution
        """
        raise NotImplementedError("Method not implemented")


__all__ = ["HardwareProcessorPluginInterface"]
