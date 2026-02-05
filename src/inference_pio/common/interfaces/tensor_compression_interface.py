"""
Interface for tensor compression functionality in the Mod project.

This module defines a clear interface for tensor compression operations
that can be implemented by different compression strategies.
"""

import logging
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
        raise NotImplementedError("Method not implemented")

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
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def decompress_model_weights(self) -> bool:
        """
        Decompress model weights back to original form.

        Returns:
            True if decompression was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def compress_activations(self, **kwargs) -> bool:
        """
        Compress model activations during inference.

        Args:
            **kwargs: Activation compression parameters

        Returns:
            True if activation compression was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get tensor compression statistics.

        Returns:
            Dictionary containing compression statistics
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def enable_adaptive_compression(self, **kwargs) -> bool:
        """
        Enable adaptive compression that adjusts based on available memory.

        Args:
            **kwargs: Adaptive compression configuration parameters

        Returns:
            True if adaptive compression was enabled successfully, False otherwise
        """
        raise NotImplementedError("Method not implemented")


# Import the concrete implementation to provide a default
try:
    from ..optimization.tensor_compression import TensorCompressionManager
    DefaultTensorCompressionManager = TensorCompressionManager
except ImportError:
    # If the concrete implementation is not available, use a basic implementation
    class DefaultTensorCompressionManager(TensorCompressionManagerInterface):
        """
        Default implementation of tensor compression manager for fallback purposes.
        """

        def __init__(self):
            self.compression_stats = {}
            self.compression_enabled = False

        def setup_tensor_compression(self, **kwargs) -> bool:
            self.compression_enabled = kwargs.get('compression_enabled', True)
            logger = logging.getLogger(__name__)
            logger.warning("Using default tensor compression implementation - no actual compression performed")
            return True

        def compress_model_weights(self, compression_ratio: float = 0.5, **kwargs) -> bool:
            # In a real implementation, this would compress the weights
            return True

        def decompress_model_weights(self) -> bool:
            # In a real implementation, this would decompress the weights
            return True

        def compress_activations(self, **kwargs) -> bool:
            # In a real implementation, this would compress activations
            return True

        def get_compression_stats(self) -> Dict[str, Any]:
            return self.compression_stats

        def enable_adaptive_compression(self, **kwargs) -> bool:
            # In a real implementation, this would enable adaptive compression
            return True