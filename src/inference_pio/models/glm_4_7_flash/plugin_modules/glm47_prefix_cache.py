"""
GLM-4.7 Prefix Caching Implementation

This module implements prefix caching for the GLM-4.7 model.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass
import threading
import time

from ..config import GLM47FlashConfig


class EvictionPolicy(Enum):
    """
    Enum for different cache eviction policies.
    """
    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"


@dataclass
class PrefixCacheConfig:
    """
    Configuration for prefix caching.
    """
    max_cache_size: int = 1024 * 1024 * 256  # 256MB
    cache_precision: torch.dtype = torch.float16
    compression_enabled: bool = True
    compression_method: str = "fp16"  # Options: "fp16", "int8", "sparse"
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_prefetching: bool = True
    prefetch_distance: int = 1
    max_prefix_length: int = 2048
    min_prefix_length: int = 8
    cache_warmup_threshold: int = 3


class PrefixCacheManager:
    """
    Prefix cache manager for GLM-4.7 model.

    This class manages the caching of common prefixes to avoid recomputation.
    """
    def __init__(self, config: PrefixCacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.access_counts = {}  # For LFU policy
        self.insertion_times = {}  # For FIFO policy
        self.cache_size = 0
        self.max_cache_size = config.max_cache_size
        self.precision = config.cache_precision
        self.compression_enabled = config.compression_enabled
        self.compression_method = config.compression_method
        self.eviction_policy = config.eviction_policy
        self.enable_prefetching = config.enable_prefetching
        self.prefetch_distance = config.prefetch_distance
        self.max_prefix_length = config.max_prefix_length
        self.min_prefix_length = config.min_prefix_length
        self.warmup_threshold = config.cache_warmup_threshold

        # Lock for thread safety
        self.lock = threading.Lock()

    def _estimate_tensor_size(self, tensor: torch.Tensor) -> int:
        """
        Estimate the size of a tensor in bytes.

        Args:
            tensor: The tensor to estimate size for

        Returns:
            Estimated size in bytes
        """
        element_size = tensor.element_size()
        num_elements = tensor.numel()
        return element_size * num_elements

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compress a tensor based on the configured compression method.

        Args:
            tensor: The tensor to compress

        Returns:
            Compressed tensor
        """
        if not self.compression_enabled:
            return tensor

        if self.compression_method == "fp16":
            return tensor.to(torch.float16)
        elif self.compression_method == "int8":
            # Simple quantization to int8
            min_val, max_val = tensor.min(), tensor.max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            quantized = ((tensor - min_val) / range_val * 255).round().clamp(0, 255).byte()
            return torch.cat([quantized.flatten(), torch.tensor([min_val, max_val], dtype=tensor.dtype)])
        elif self.compression_method == "sparse":
            # Simple sparsification - keep only top 50% values
            threshold = torch.quantile(tensor.abs().flatten(), 0.5)
            mask = tensor.abs() >= threshold
            sparse_tensor = tensor * mask
            return sparse_tensor
        else:
            return tensor

    def _decompress_tensor(self, compressed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decompress a tensor based on the configured compression method.

        Args:
            compressed_tensor: The compressed tensor to decompress

        Returns:
            Decompressed tensor
        """
        if not self.compression_enabled:
            return compressed_tensor

        if self.compression_method == "fp16":
            return compressed_tensor.to(self.precision)
        elif self.compression_method == "int8":
            # Extract min/max values and reconstruct
            min_val = compressed_tensor[-2].item()
            max_val = compressed_tensor[-1].item()
            quantized = compressed_tensor[:-2].view(compressed_tensor.shape[:-2] + (-1,)).char()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            decompressed = (quantized.float() / 255.0) * range_val + min_val
            return decompressed.view(compressed_tensor.shape[:-2] + (-1,))[:-2]
        elif self.compression_method == "sparse":
            return compressed_tensor
        else:
            return compressed_tensor

    def _evict_if_needed(self):
        """
        Evict entries from the cache if the size exceeds the maximum.
        """
        while self.cache_size > self.max_cache_size and len(self.cache) > 0:
            if self.eviction_policy == EvictionPolicy.LRU:
                # Remove least recently used
                key, _ = self.cache.popitem(last=False)
            elif self.eviction_policy == EvictionPolicy.FIFO:
                # Remove first inserted
                oldest_time = min(self.insertion_times.values())
                key_to_remove = None
                for key, ins_time in self.insertion_times.items():
                    if ins_time == oldest_time:
                        key_to_remove = key
                        break
                if key_to_remove:
                    del self.cache[key_to_remove]
                    del self.insertion_times[key_to_remove]
                    if key_to_remove in self.access_counts:
                        del self.access_counts[key_to_remove]
            elif self.eviction_policy == EvictionPolicy.LFU:
                # Remove least frequently used
                min_count = min(self.access_counts.values())
                key_to_remove = None
                for key, count in self.access_counts.items():
                    if count == min_count:
                        key_to_remove = key
                        break
                if key_to_remove:
                    del self.cache[key_to_remove]
                    del self.access_counts[key_to_remove]
                    if key_to_remove in self.insertion_times:
                        del self.insertion_times[key_to_remove]

            # Update cache size
            if key_to_remove in self.cache:
                self.cache_size -= self._estimate_tensor_size(self.cache[key_to_remove])

    def put(self, prefix_key: str, prefix_data: torch.Tensor):
        """
        Put a prefix into the cache.

        Args:
            prefix_key: Key for the prefix
            prefix_data: Prefix data to cache
        """
        with self.lock:
            # Check if prefix is long enough to cache
            if prefix_data.size(-2) < self.min_prefix_length:
                return  # Don't cache short prefixes

            # Check if prefix is too long
            if prefix_data.size(-2) > self.max_prefix_length:
                return  # Don't cache very long prefixes

            # Compress the data
            compressed_data = self._compress_tensor(prefix_data)

            # Calculate size
            estimated_size = self._estimate_tensor_size(compressed_data)

            # Check if this is a warm prefix (seen multiple times)
            if prefix_key in self.access_counts:
                self.access_counts[prefix_key] += 1
            else:
                self.access_counts[prefix_key] = 1
                self.insertion_times[prefix_key] = time.time()

            # Only cache if we've seen this prefix enough times
            if self.access_counts[prefix_key] < self.warmup_threshold:
                return

            # Remove old entry if exists
            if prefix_key in self.cache:
                old_size = self._estimate_tensor_size(self.cache[prefix_key])
                self.cache_size -= old_size

            # Add to cache
            self.cache[prefix_key] = compressed_data
            self.cache_size += estimated_size

            # Update access time for LRU
            if prefix_key in self.cache:
                self.cache.move_to_end(prefix_key)

            # Evict if needed
            self._evict_if_needed()

    def get(self, prefix_key: str) -> Optional[torch.Tensor]:
        """
        Get a prefix from the cache.

        Args:
            prefix_key: Key for the prefix

        Returns:
            Cached prefix data or None if not found
        """
        with self.lock:
            if prefix_key not in self.cache:
                return None

            # Update access counts for LFU
            if prefix_key in self.access_counts:
                self.access_counts[prefix_key] += 1

            # Update insertion time for FIFO (to maintain order)
            if prefix_key in self.insertion_times:
                self.insertion_times[prefix_key] = time.time()

            # Move to end for LRU
            self.cache.move_to_end(prefix_key)

            # Decompress and return
            compressed_data = self.cache[prefix_key]
            return self._decompress_tensor(compressed_data)

    def prefetch(self, current_prefix: str) -> Optional[torch.Tensor]:
        """
        Prefetch likely-to-be-used prefixes based on the current prefix.

        Args:
            current_prefix: Current prefix to base prefetching on

        Returns:
            Prefetched data if available
        """
        if not self.enable_prefetching:
            return None

        # This is a simple implementation - in a real system, this would
        # use more sophisticated prediction algorithms
        potential_next_prefix = current_prefix + "_next"
        return self.get(potential_next_prefix)

    def clear(self):
        """
        Clear the cache.
        """
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.insertion_times.clear()
            self.cache_size = 0


def apply_prefix_cache_to_model(model: nn.Module, config: PrefixCacheConfig) -> nn.Module:
    """
    Apply prefix caching to the model.

    Args:
        model: The model to apply prefix caching to
        config: Prefix cache configuration

    Returns:
        Model with prefix caching applied
    """
    # Create and attach the prefix cache manager to the model
    model.prefix_cache_manager = PrefixCacheManager(config)

    return model


def create_prefix_cache_for_glm47(config: GLM47FlashConfig) -> PrefixCacheManager:
    """
    Create a prefix cache manager specifically configured for GLM-4.7.

    Args:
        config: GLM-4.7 configuration

    Returns:
        Prefix cache manager
    """
    prefix_config = PrefixCacheConfig(
        max_cache_size=config.prefix_cache_max_size,
        cache_precision=_str_to_dtype(config.prefix_cache_precision),
        compression_enabled=config.prefix_cache_compression_enabled,
        eviction_policy=_str_to_eviction_policy(config.prefix_cache_eviction_policy),
        enable_prefetching=config.prefix_cache_enable_prefetching,
        prefetch_distance=config.prefix_cache_prefetch_distance,
        max_prefix_length=config.prefix_cache_max_prefix_length,
        min_prefix_length=config.prefix_cache_min_prefix_length,
        cache_warmup_threshold=config.prefix_cache_warmup_threshold,
    )

    return PrefixCacheManager(prefix_config)


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string to torch dtype.

    Args:
        dtype_str: String representation of dtype

    Returns:
        Corresponding torch dtype
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8
    }
    return dtype_map.get(dtype_str, torch.float16)


def _str_to_eviction_policy(policy_str: str) -> EvictionPolicy:
    """
    Convert string to eviction policy.

    Args:
        policy_str: String representation of eviction policy

    Returns:
        Corresponding eviction policy
    """
    policy_map = {
        "lru": EvictionPolicy.LRU,
        "fifo": EvictionPolicy.FIFO,
        "lfu": EvictionPolicy.LFU
    }
    return policy_map.get(policy_str, EvictionPolicy.LRU)


__all__ = [
    "EvictionPolicy",
    "PrefixCacheConfig",
    "PrefixCacheManager",
    "apply_prefix_cache_to_model",
    "create_prefix_cache_for_glm47"
]