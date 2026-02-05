"""
Interface for sharding functionality in the Mod project.

This module defines a clear interface for sharding operations
that can be implemented by different sharding strategies.
"""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ShardingManagerInterface(ABC):
    """
    Interface for sharding operations.
    """

    @abstractmethod
    def enable_sharding(
        self, num_shards: int = 500, storage_path: str = "./shards", **kwargs
    ) -> bool:
        """
        Enable extreme sharding for the model.

        Args:
            num_shards: Number of shards to create (default 500 for extreme sharding)
            storage_path: Path to store shard files
            **kwargs: Additional sharding configuration parameters

        Returns:
            True if sharding was enabled successfully, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def disable_sharding(self) -> bool:
        """
        Disable sharding for the model.

        Returns:
            True if sharding was disabled successfully, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def shard_model(self, model: torch.nn.Module, num_shards: int = 500) -> bool:
        """
        Shard the model into hundreds of tiny fragments.

        Args:
            model: Model to shard
            num_shards: Number of shards to create

        Returns:
            True if sharding was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def prepare_inference_context(
        self, context_id: str, input_shape: Tuple, inference_type: str = "forward"
    ) -> List[str]:
        """
        Prepare an inference context by determining and loading required shards.

        Args:
            context_id: Unique identifier for this inference context
            input_shape: Shape of the input tensor
            inference_type: Type of inference ("forward", "generate", etc.)

        Returns:
            List of shard IDs loaded for this context
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def execute_with_shards(
        self, context_id: str, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute inference in a prepared context using only required shards.

        Args:
            context_id: Context ID from prepare_inference_context
            input_tensor: Input tensor for inference

        Returns:
            Output tensor from the computation
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def cleanup_inference_context(self, context_id: str, force_unload: bool = True):
        """
        Clean up an inference context and optionally unload shards.

        Args:
            context_id: Context ID to clean up
            force_unload: Whether to force unload all shards for this context
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_sharding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sharding system.

        Returns:
            Dictionary containing sharding statistics
        """
        pass