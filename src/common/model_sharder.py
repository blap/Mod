"""
Extreme Model Sharding System for Inference-PIO

This module implements an extreme sharding system that splits models into hundreds of tiny fragments
and implements a streaming system that loads only necessary parts for each inference step,
immediately unloading them after use.
"""

import hashlib
import json
import logging
import os
import pickle
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ShardStatus(Enum):
    """Status of a model shard."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ShardInfo:
    """Information about a model shard."""

    id: str
    layer_indices: Tuple[int, int]  # (start_layer, end_layer)
    size_bytes: int
    dependencies: List[str]  # List of shard IDs this shard depends on
    compute_requirements: Dict[str, Any]  # GPU memory, compute capability, etc.
    status: ShardStatus = ShardStatus.UNLOADED
    device: Optional[str] = None
    last_access_time: float = 0.0
    access_count: int = 0
    priority: int = 0  # Higher priority means more important


class ModelSharder:
    """System for splitting models into hundreds of tiny fragments."""

    def __init__(self, storage_path: str = "./shards"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Track all shards
        self.shards: Dict[str, ShardInfo] = {}
        self.shard_locks: Dict[str, threading.Lock] = {}

        # Track loaded shards
        self.loaded_shards: Dict[str, nn.Module] = {}
        self.shard_load_order: List[str] = []  # LRU order

        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loading_queue = queue.Queue()
        self.unloading_queue = queue.Queue()

        # Statistics
        self.stats = {
            "total_shards_created": 0,
            "total_loads": 0,
            "total_unloads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_load_time": 0.0,
            "avg_unload_time": 0.0,
        }

        # Background thread for managing shard lifecycle
        self.background_thread = threading.Thread(
            target=self._background_manager, daemon=True
        )
        self.background_thread.start()

    def shard_model(self, model: nn.Module, num_shards: int = 500) -> List[ShardInfo]:
        """
        Split a model into hundreds of tiny fragments.

        Args:
            model: The model to shard
            num_shards: Number of shards to create (default 500 for extreme sharding)

        Returns:
            List of ShardInfo objects describing the created shards
        """
        logger.info(f"Starting extreme sharding of model into {num_shards} shards...")

        # Get all named modules in the model
        all_modules = list(model.named_modules())

        # Filter out the root module itself
        child_modules = [(name, mod) for name, mod in all_modules if name != ""]

        if len(child_modules) < num_shards:
            # If we have fewer modules than requested shards, we'll need to split individual modules
            # For now, we'll distribute modules as evenly as possible
            modules_per_shard = max(1, len(child_modules) // num_shards)
        else:
            modules_per_shard = len(child_modules) // num_shards

        shards = []
        start_idx = 0

        for i in range(num_shards):
            end_idx = min(start_idx + modules_per_shard, len(child_modules))

            # If we're at the last shard and haven't used all modules, extend it
            if i == num_shards - 1:
                end_idx = len(child_modules)

            if start_idx >= len(child_modules):
                break

            # Create a shard from the selected modules
            shard_modules = child_modules[start_idx:end_idx]

            # Calculate shard size
            shard_size = 0
            for _, module in shard_modules:
                for param in module.parameters(recurse=True):
                    shard_size += param.numel() * param.element_size()

            # Create shard ID
            shard_id = f"shard_{i:04d}"

            # Create shard info
            shard_info = ShardInfo(
                id=shard_id,
                layer_indices=(start_idx, end_idx - 1),
                size_bytes=shard_size,
                dependencies=[],  # Will be calculated later based on model structure
                compute_requirements={
                    "min_gpu_memory_mb": max(
                        10, shard_size // (1024 * 1024)
                    ),  # Rough estimate
                    "precision": (
                        "float16"
                        if next(model.parameters()).dtype == torch.float16
                        else "float32"
                    ),
                },
            )

            # Save shard to disk
            self._save_shard(shard_info, [mod for _, mod in shard_modules])

            # Add to our tracking
            self.shards[shard_id] = shard_info
            self.shard_locks[shard_id] = threading.Lock()

            shards.append(shard_info)
            start_idx = end_idx

        self.stats["total_shards_created"] = len(shards)
        logger.info(f"Created {len(shards)} shards from model")

        return shards

    def _save_shard(self, shard_info: ShardInfo, modules: List[nn.Module]):
        """Save a shard to disk."""
        shard_path = self.storage_path / f"{shard_info.id}.pkl"

        # Create a container for the shard
        shard_data = {
            "id": shard_info.id,
            "modules": modules,
            "info": {
                "layer_indices": shard_info.layer_indices,
                "size_bytes": shard_info.size_bytes,
                "compute_requirements": shard_info.compute_requirements,
            },
        }

        with open(shard_path, "wb") as f:
            pickle.dump(shard_data, f)

    def _load_shard_from_disk(self, shard_id: str) -> nn.Module:
        """Load a shard from disk."""
        shard_path = self.storage_path / f"{shard_id}.pkl"

        with open(shard_path, "rb") as f:
            shard_data = pickle.load(f)

        # Return the modules as a sequential container
        if len(shard_data["modules"]) == 1:
            return shard_data["modules"][0]
        else:
            return nn.Sequential(*shard_data["modules"])

    def load_shard(self, shard_id: str, device: str = "cpu") -> nn.Module:
        """
        Load a specific shard into memory.

        Args:
            shard_id: ID of the shard to load
            device: Device to load the shard onto

        Returns:
            Loaded shard module
        """
        if shard_id not in self.shards:
            raise ValueError(f"Shard {shard_id} does not exist")

        shard_info = self.shards[shard_id]

        with self.shard_locks[shard_id]:
            if shard_id in self.loaded_shards:
                # Already loaded, just return it
                self.stats["cache_hits"] += 1
                shard_info.last_access_time = time.time()
                shard_info.access_count += 1
                return self.loaded_shards[shard_id]

            # Mark as loading
            shard_info.status = ShardStatus.LOADING
            start_time = time.time()

            try:
                # Load from disk
                shard_module = self._load_shard_from_disk(shard_id)

                # Move to specified device
                shard_module = shard_module.to(device)

                # Update shard info
                shard_info.status = ShardStatus.LOADED
                shard_info.device = device
                shard_info.last_access_time = time.time()
                shard_info.access_count += 1

                # Add to loaded shards
                self.loaded_shards[shard_id] = shard_module
                self.shard_load_order.append(shard_id)

                # Update stats
                load_time = time.time() - start_time
                self.stats["total_loads"] += 1
                self.stats["cache_misses"] += 1
                self.stats["avg_load_time"] = (
                    self.stats["avg_load_time"] * (self.stats["total_loads"] - 1)
                    + load_time
                ) / self.stats["total_loads"]

                logger.debug(f"Loaded shard {shard_id} to {device} in {load_time:.3f}s")

                return shard_module
            except Exception as e:
                shard_info.status = ShardStatus.ERROR
                logger.error(f"Failed to load shard {shard_id}: {e}")
                raise

    def unload_shard(self, shard_id: str) -> bool:
        """
        Unload a specific shard from memory.

        Args:
            shard_id: ID of the shard to unload

        Returns:
            True if successful, False otherwise
        """
        if shard_id not in self.shards:
            return False

        shard_info = self.shards[shard_id]

        with self.shard_locks[shard_id]:
            if shard_id not in self.loaded_shards:
                # Already unloaded
                return True

            start_time = time.time()

            try:
                # Remove from loaded shards
                del self.loaded_shards[shard_id]

                # Remove from load order
                if shard_id in self.shard_load_order:
                    self.shard_load_order.remove(shard_id)

                # Update shard info
                shard_info.status = ShardStatus.UNLOADED
                shard_info.device = None

                # Update stats
                unload_time = time.time() - start_time
                self.stats["total_unloads"] += 1
                self.stats["avg_unload_time"] = (
                    self.stats["avg_unload_time"] * (self.stats["total_unloads"] - 1)
                    + unload_time
                ) / self.stats["total_unloads"]

                logger.debug(f"Unloaded shard {shard_id} in {unload_time:.3f}s")

                return True
            except Exception as e:
                logger.error(f"Failed to unload shard {shard_id}: {e}")
                return False

    def get_required_shards(
        self, input_shape: Tuple, inference_step: str = "forward"
    ) -> List[str]:
        """
        Determine which shards are needed for a specific inference step.

        Args:
            input_shape: Shape of the input tensor
            inference_step: Type of inference step ("forward", "backward", etc.)

        Returns:
            List of shard IDs required for this step
        """
        # This is a simplified implementation - in practice, this would be more sophisticated
        # based on the model architecture and the specific computation needed

        if inference_step == "forward":
            # For forward pass, we typically need shards in sequence
            # For extreme sharding, we'll return a subset based on input position
            # This is a placeholder algorithm - real implementation would be model-specific
            input_seq_len = input_shape[1] if len(input_shape) > 1 else 1
            shard_subset_size = max(
                1, len(self.shards) // 10
            )  # Load 10% of shards at a time

            # Calculate which shards to load based on input position
            start_shard_idx = (input_seq_len * 7) % len(
                self.shards
            )  # Pseudo-random based on input
            end_shard_idx = min(start_shard_idx + shard_subset_size, len(self.shards))

            required_shards = []
            for i, shard_id in enumerate(self.shards.keys()):
                if start_shard_idx <= i < end_shard_idx:
                    required_shards.append(shard_id)
        else:
            # For other steps, return all shards (simplified)
            required_shards = list(self.shards.keys())

        return required_shards

    def execute_with_shards(
        self,
        input_tensor: torch.Tensor,
        required_shards: List[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Execute inference using only the required shards.

        Args:
            input_tensor: Input tensor for inference
            required_shards: List of shard IDs to use
            device: Device to execute on

        Returns:
            Output tensor from the computation
        """
        # Load required shards
        loaded_modules = []
        for shard_id in required_shards:
            if shard_id not in self.loaded_shards:
                self.load_shard(shard_id, device)
            loaded_modules.append(self.loaded_shards[shard_id])

        # Execute the computation using the loaded shards
        # This is a simplified implementation - real implementation would depend on model architecture
        output = input_tensor
        for module in loaded_modules:
            output = module(output)

        # Immediately unload the shards after use (unless they're frequently accessed)
        for shard_id in required_shards:
            # Only unload if access count is low (meaning it's not frequently used)
            if self.shards[shard_id].access_count < 5:  # Threshold configurable
                self.unload_shard(shard_id)

        return output

    def _background_manager(self):
        """Background thread to manage shard lifecycle."""
        while True:
            try:
                # Process loading/unloading queues if needed
                # For now, just sleep - in a real implementation this would handle async operations
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in background manager: {e}")
                time.sleep(1)  # Wait before retrying

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for the sharding system."""
        total_size = sum(shard.size_bytes for shard in self.shards.values())
        loaded_size = (
            sum(
                self.loaded_shards[shard_id].numel()
                * next(self.loaded_shards[shard_id].parameters()).element_size()
                for shard_id in self.loaded_shards
            )
            if self.loaded_shards
            else 0
        )

        return {
            "total_shards": len(self.shards),
            "loaded_shards": len(self.loaded_shards),
            "total_size_bytes": total_size,
            "loaded_size_bytes": loaded_size,
            "memory_utilization_ratio": (
                loaded_size / total_size if total_size > 0 else 0
            ),
            "stats": self.stats.copy(),
        }

    def cleanup(self):
        """Clean up all loaded shards and resources."""
        for shard_id in list(self.loaded_shards.keys()):
            self.unload_shard(shard_id)

        self.executor.shutdown(wait=True)


class StreamingModelLoader:
    """Streaming loader that manages loading/unloading of model fragments."""

    def __init__(self, model_sharder: ModelSharder, max_loaded_shards: int = 10):
        self.sharder = model_sharder
        self.max_loaded_shards = max_loaded_shards
        self.active_inference_contexts = {}  # Track active inference contexts
        self.context_lock = threading.Lock()

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
        required_shards = self.sharder.get_required_shards(input_shape, inference_type)

        # Load required shards
        loaded_shards = []
        for shard_id in required_shards[
            : self.max_loaded_shards
        ]:  # Limit to max loaded shards
            try:
                self.sharder.load_shard(shard_id)
                loaded_shards.append(shard_id)
            except Exception as e:
                logger.error(
                    f"Failed to load shard {shard_id} for context {context_id}: {e}"
                )

        # Track this context
        with self.context_lock:
            self.active_inference_contexts[context_id] = {
                "required_shards": required_shards,
                "loaded_shards": loaded_shards,
                "created_at": time.time(),
                "last_access": time.time(),
            }

        return loaded_shards

    def execute_in_context(
        self, context_id: str, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute inference in a prepared context.

        Args:
            context_id: Context ID from prepare_inference_context
            input_tensor: Input tensor for inference

        Returns:
            Output tensor from the computation
        """
        with self.context_lock:
            if context_id not in self.active_inference_contexts:
                raise ValueError(f"Context {context_id} not found")

            context = self.active_inference_contexts[context_id]
            context["last_access"] = time.time()
            required_shards = context["loaded_shards"]

        # Execute using the loaded shards
        return self.sharder.execute_with_shards(input_tensor, required_shards)

    def cleanup_context(self, context_id: str, force_unload: bool = True):
        """
        Clean up an inference context and optionally unload shards.

        Args:
            context_id: Context ID to clean up
            force_unload: Whether to force unload all shards for this context
        """
        with self.context_lock:
            if context_id not in self.active_inference_contexts:
                return

            context = self.active_inference_contexts[context_id]
            del self.active_inference_contexts[context_id]

        if force_unload:
            for shard_id in context["loaded_shards"]:
                # Only unload if the shard isn't shared with other contexts
                if self._is_shard_exclusive_to_context(shard_id, context_id):
                    self.sharder.unload_shard(shard_id)

    def _is_shard_exclusive_to_context(self, shard_id: str, context_id: str) -> bool:
        """Check if a shard is exclusively used by a specific context."""
        with self.context_lock:
            for ctx_id, ctx_info in self.active_inference_contexts.items():
                if ctx_id != context_id and shard_id in ctx_info["loaded_shards"]:
                    return False
        return True

    def get_active_contexts(self) -> Dict[str, Any]:
        """Get information about all active inference contexts."""
        with self.context_lock:
            return self.active_inference_contexts.copy()

    def cleanup_all_contexts(self):
        """Clean up all active inference contexts."""
        with self.context_lock:
            context_ids = list(self.active_inference_contexts.keys())

        for context_id in context_ids:
            self.cleanup_context(context_id, force_unload=True)


def create_extreme_sharding_system(
    storage_path: str = "./shards", num_shards: int = 500
) -> Tuple[ModelSharder, StreamingModelLoader]:
    """
    Create an extreme sharding system with streaming loader.

    Args:
        storage_path: Path to store shard files
        num_shards: Number of shards to create (default 500 for extreme sharding)

    Returns:
        Tuple of (ModelSharder, StreamingModelLoader)
    """
    sharder = ModelSharder(storage_path)
    loader = StreamingModelLoader(
        sharder, max_loaded_shards=10
    )  # Limit to 10 shards at a time

    return sharder, loader


__all__ = [
    "ModelSharder",
    "StreamingModelLoader",
    "ShardStatus",
    "ShardInfo",
    "create_extreme_sharding_system",
]
