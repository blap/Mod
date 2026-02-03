"""
Model Sharding Component for Inference-PIO System

This module contains the sharding-related functionality extracted from the base plugin interface
to reduce the size of the main interface file and improve modularity.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelShardingMixin:
    """
    Mixin class that provides model sharding functionality to plugin interfaces.
    """

    def __init__(self):
        # Sharding attributes
        self._sharder = None
        self._streaming_loader = None
        self._sharding_enabled = False
        self._current_inference_context = None

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
        try:
            from .model_sharder import create_extreme_sharding_system

            self._sharder, self._streaming_loader = create_extreme_sharding_system(
                storage_path=storage_path, num_shards=num_shards
            )
            self._sharding_enabled = True
            logger.info(f"Extreme sharding enabled with {num_shards} shards")
            return True
        except Exception as e:
            logger.error(f"Failed to enable sharding: {e}")
            return False

    def disable_sharding(self) -> bool:
        """
        Disable sharding for the model.

        Returns:
            True if sharding was disabled successfully, False otherwise
        """
        try:
            if self._sharder:
                self._sharder.cleanup()
                self._sharder = None
            if self._streaming_loader:
                self._streaming_loader.cleanup_all_contexts()
                self._streaming_loader = None
            self._sharding_enabled = False
            logger.info("Sharding disabled")
            return True
        except Exception as e:
            logger.error(f"Failed to disable sharding: {e}")
            return False

    def shard_model(self, model: nn.Module, num_shards: int = 500) -> bool:
        """
        Shard the model into hundreds of tiny fragments.

        Args:
            model: Model to shard
            num_shards: Number of shards to create

        Returns:
            True if sharding was successful, False otherwise
        """
        try:
            if not self._sharder:
                logger.error("Sharding not enabled, call enable_sharding first")
                return False

            self._sharder.shard_model(model, num_shards)
            logger.info(f"Model successfully sharded into {num_shards} fragments")
            return True
        except Exception as e:
            logger.error(f"Failed to shard model: {e}")
            return False

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
        try:
            if not self._streaming_loader:
                logger.error("Streaming loader not initialized")
                return []

            return self._streaming_loader.prepare_inference_context(
                context_id, input_shape, inference_type
            )
        except Exception as e:
            logger.error(f"Failed to prepare inference context: {e}")
            return []

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
        try:
            if not self._streaming_loader:
                logger.error("Streaming loader not initialized")
                return input_tensor

            return self._streaming_loader.execute_in_context(context_id, input_tensor)
        except Exception as e:
            logger.error(f"Failed to execute with shards: {e}")
            # Fallback to regular inference if sharding fails
            if hasattr(self, "infer"):
                return self.infer(input_tensor)
            else:
                return input_tensor

    def cleanup_inference_context(self, context_id: str, force_unload: bool = True):
        """
        Clean up an inference context and optionally unload shards.

        Args:
            context_id: Context ID to clean up
            force_unload: Whether to force unload all shards for this context
        """
        try:
            if self._streaming_loader:
                self._streaming_loader.cleanup_context(context_id, force_unload)
        except Exception as e:
            logger.error(f"Failed to cleanup inference context: {e}")

    def get_sharding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sharding system.

        Returns:
            Dictionary containing sharding statistics
        """
        try:
            if self._sharder:
                return self._sharder.get_memory_usage()
            else:
                return {
                    "sharding_enabled": False,
                    "total_shards": 0,
                    "loaded_shards": 0,
                    "total_size_bytes": 0,
                    "loaded_size_bytes": 0,
                    "memory_utilization_ratio": 0.0,
                }
        except Exception as e:
            logger.error(f"Failed to get sharding stats: {e}")
            return {}


__all__ = ["ModelShardingMixin"]
