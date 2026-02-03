"""
Interface for tensor compression functionality in the Mod project.

This module defines a clear interface for tensor compression operations
that can be implemented by different compression strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TensorCompressionManagerInterface(ABC):
    """
    Interface for tensor compression operations.
    """

    @abstractmethod
    def setup_tensor_compression(self, **kwargs) -> bool:
        """
        Set up tensor compression system for model weights and activations.

        Args:
            **kwargs: Tensor compression configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        pass

    @abstractmethod
    def compress_model_weights(self, compression_ratio: float = 0.5, **kwargs) -> bool:
        """
        Compress model weights using tensor compression techniques.

        Args:
            compression_ratio: Target compression ratio (0.0 to 1.0)
            **kwargs: Additional compression parameters

        Returns:
            True if compression was successful, False otherwise
        """
        pass

    @abstractmethod
    def decompress_model_weights(self) -> bool:
        """
        Decompress model weights back to original form.

        Returns:
            True if decompression was successful, False otherwise
        """
        pass

    @abstractmethod
    def compress_activations(self, **kwargs) -> bool:
        """
        Compress model activations during inference.

        Args:
            **kwargs: Activation compression parameters

        Returns:
            True if activation compression was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get tensor compression statistics.

        Returns:
            Dictionary containing compression statistics
        """
        pass

    @abstractmethod
    def enable_adaptive_compression(self, **kwargs) -> bool:
        """
        Enable adaptive compression that adjusts based on available memory.

        Args:
            **kwargs: Adaptive compression configuration parameters

        Returns:
            True if adaptive compression was enabled successfully, False otherwise
        """
        pass