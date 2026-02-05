"""
Qwen3-Coder-30B Intelligent Cache System

This module implements an intelligent cache system with predictive and advanced caching policies
for the Qwen3-Coder-30B model. The system includes predictive caching, intelligent eviction policies,
and adaptive cache management.
"""

import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class CachePolicy(Enum):
    """
    Enum for different cache policies.
    """
    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"
    PREDICTIVE = "predictive"
    INTELLIGENT = "intelligent"


@dataclass
class IntelligentCacheConfig:
    """
    Configuration for intelligent caching.
    """
    # Basic cache settings
    max_cache_size: int = 1024 * 1024 * 512  # 512MB for larger model
    cache_precision: torch.dtype = torch.float16
    compression_enabled: bool = True
    compression_method: str = "intelligent"  # Options: "fp16", "int8", "sparse", "intelligent"

    # Policy settings
    cache_policy: CachePolicy = CachePolicy.INTELLIGENT
    enable_prefetching: bool = True
    prefetch_distance: int = 1

    # Size constraints
    max_prefix_length: int = 4096  # Larger for coding tasks
    min_prefix_length: int = 8

    # Warmup and prediction settings
    cache_warmup_threshold: int = 2  # Lower for coding tasks
    prediction_horizon: int = 15  # Number of steps to predict ahead
    prediction_confidence_threshold: float = 0.65  # Slightly lower for coding patterns

    # Adaptive settings
    enable_adaptive_eviction: bool = True
    enable_adaptive_prefetching: bool = True
    adaptive_window_size: int = 200  # Larger window for coding patterns

    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 100  # Log performance every N operations


class AccessPatternPredictor:
    """
    Predicts access patterns based on historical data.
    """
    def __init__(self, config: IntelligentCacheConfig):
        self.config = config
        self.access_history = []  # List of (timestamp, key, action) tuples
        self.pattern_model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def record_access(self, key: str, action: str = "access"):
        """
        Record an access event.

        Args:
            key: The key that was accessed
            action: The type of action ('access', 'insert', 'evict')
        """
        self.access_history.append((time.time(), key, action))

        # Keep only recent history to avoid memory issues
        if len(self.access_history) > self.config.adaptive_window_size * 10:
            self.access_history = self.access_history[-self.config.adaptive_window_size * 10:]

    def predict_next_accesses(self) -> List[Tuple[str, float]]:
        """
        Predict the next accesses based on historical patterns.

        Returns:
            List of (key, confidence) tuples for predicted accesses
        """
        if len(self.access_history) < 10:  # Need sufficient history
            return []

        # Simple prediction based on recent access patterns
        recent_accesses = self.access_history[-self.config.adaptive_window_size:]

        # Count frequency of accesses
        access_freq = defaultdict(int)
        for _, key, _ in recent_accesses:
            access_freq[key] += 1

        # Calculate relative frequencies as confidence scores
        total_accesses = len(recent_accesses)
        predictions = []
        for key, freq in access_freq.items():
            confidence = freq / total_accesses
            if confidence >= self.config.prediction_confidence_threshold:
                predictions.append((key, confidence))

        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:self.config.prediction_horizon]


class PerformanceMonitor:
    """
    Monitors cache performance metrics.
    """
    def __init__(self, config: IntelligentCacheConfig):
        self.config = config
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.prefetch_hit_count = 0
        self.prefetch_miss_count = 0
        self.total_operations = 0
        self.last_log_time = time.time()

    def record_hit(self, is_prefetch: bool = False):
        """Record a cache hit."""
        self.hit_count += 1
        if is_prefetch:
            self.prefetch_hit_count += 1
        self.total_operations += 1

    def record_miss(self, is_prefetch: bool = False):
        """Record a cache miss."""
        self.miss_count += 1
        if is_prefetch:
            self.prefetch_miss_count += 1
        self.total_operations += 1

    def record_eviction(self):
        """Record a cache eviction."""
        self.eviction_count += 1

    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        prefetch_hit_rate = (
            self.prefetch_hit_count / (self.prefetch_hit_count + self.prefetch_miss_count)
            if (self.prefetch_hit_count + self.prefetch_miss_count) > 0
            else 0
        )

        return {
            "hit_rate": hit_rate,
            "miss_rate": 1 - hit_rate,
            "prefetch_hit_rate": prefetch_hit_rate,
            "eviction_count": self.eviction_count,
            "total_operations": self.total_operations,
            "cache_size_utilization": 0  # Will be filled by cache manager
        }

    def should_log(self) -> bool:
        """Check if it's time to log performance."""
        return self.total_operations % self.config.performance_log_interval == 0


class IntelligentCacheManager:
    """
    Advanced cache manager with predictive and intelligent policies for Qwen3-Coder-30B.
    """

    def __init__(self, config: IntelligentCacheConfig):
        self.config = config
        self.cache = OrderedDict()  # Main cache storage
        self.access_times = {}  # Last access time for each key
        self.access_counts = {}  # Frequency of access for each key
        self.insertion_times = {}  # When each item was inserted
        self.cache_size = 0  # Current cache size in bytes
        self.predictor = AccessPatternPredictor(config)
        self.performance_monitor = PerformanceMonitor(config)

        # Thread lock for thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "predictions_made": 0,
            "predictions_correct": 0
        }

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
        if not self.config.compression_enabled:
            return tensor

        if self.config.compression_method == "fp16":
            return tensor.to(torch.float16)
        elif self.config.compression_method == "int8":
            # Simple quantization to int8
            min_val, max_val = tensor.min(), tensor.max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            quantized = (
                ((tensor - min_val) / range_val * 255).round().clamp(0, 255).byte()
            )
            # Store min/max values along with quantized tensor
            return {
                'quantized': quantized,
                'min_val': min_val,
                'max_val': max_val
            }
        elif self.config.compression_method == "sparse":
            # Simple sparsification - keep only top 50% values
            threshold = torch.quantile(tensor.abs().flatten(), 0.5)
            mask = tensor.abs() >= threshold
            sparse_tensor = tensor * mask
            return {'sparse': sparse_tensor, 'mask': mask}
        elif self.config.compression_method == "intelligent":
            # Intelligent compression based on tensor properties
            # Use different compression for different tensor types
            if tensor.numel() > 1000:  # Large tensors get quantized
                min_val, max_val = tensor.min(), tensor.max()
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1.0
                quantized = (
                    ((tensor - min_val) / range_val * 255).round().clamp(0, 255).byte()
                )
                return {
                    'type': 'quantized',
                    'data': quantized,
                    'min_val': min_val,
                    'max_val': max_val
                }
            else:  # Small tensors stay in original precision
                return {'type': 'original', 'data': tensor}
        else:
            return tensor

    def _decompress_tensor(self, compressed_tensor) -> torch.Tensor:
        """
        Decompress a tensor based on the configured compression method.

        Args:
            compressed_tensor: The compressed tensor to decompress

        Returns:
            Decompressed tensor
        """
        if not self.config.compression_enabled:
            return compressed_tensor

        if self.config.compression_method == "fp16":
            return compressed_tensor.to(self.config.cache_precision)
        elif self.config.compression_method == "int8":
            # Extract min/max values and reconstruct
            min_val = compressed_tensor['min_val']
            max_val = compressed_tensor['max_val']
            quantized = compressed_tensor['quantized'].float()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            decompressed = (quantized / 255.0) * range_val + min_val
            return decompressed
        elif self.config.compression_method == "sparse":
            # Reconstruct from sparse representation
            return compressed_tensor['sparse']
        elif self.config.compression_method == "intelligent":
            # Decompress based on stored type
            if compressed_tensor['type'] == 'quantized':
                min_val = compressed_tensor['min_val']
                max_val = compressed_tensor['max_val']
                quantized = compressed_tensor['data'].float()
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = 1.0
                decompressed = (quantized / 255.0) * range_val + min_val
                return decompressed
            else:  # original type
                return compressed_tensor['data']
        else:
            return compressed_tensor

    def _evict_by_policy(self) -> bool:
        """
        Evict entries from the cache based on the selected policy.

        Returns:
            True if eviction was successful, False otherwise
        """
        if len(self.cache) == 0:
            return False

        if self.config.cache_policy == CachePolicy.LRU:
            # Remove least recently used
            key, _ = self.cache.popitem(last=False)
        elif self.config.cache_policy == CachePolicy.FIFO:
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
                if key_to_remove in self.access_times:
                    del self.access_times[key_to_remove]
        elif self.config.cache_policy == CachePolicy.LFU:
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
                if key_to_remove in self.access_times:
                    del self.access_times[key_to_remove]
                if key_to_remove in self.insertion_times:
                    del self.insertion_times[key_to_remove]
        elif self.config.cache_policy in [CachePolicy.PREDICTIVE, CachePolicy.INTELLIGENT]:
            # For predictive and intelligent policies, evict based on prediction
            # Remove items that are least likely to be accessed soon
            if self.config.cache_policy == CachePolicy.INTELLIGENT:
                # Use predictor to identify least valuable items
                predicted_accesses = self.predictor.predict_next_accesses()
                predicted_keys = {key for key, _ in predicted_accesses}

                # Find items not in predicted accesses
                non_predicted_items = [
                    (key, self.access_times.get(key, float('-inf')))
                    for key in self.cache.keys()
                    if key not in predicted_keys
                ]

                if non_predicted_items:
                    # Remove the one that was accessed longest ago
                    key_to_remove = min(non_predicted_items, key=lambda x: x[1])[0]
                    del self.cache[key_to_remove]
                    if key_to_remove in self.access_counts:
                        del self.access_counts[key_to_remove]
                    if key_to_remove in self.access_times:
                        del self.access_times[key_to_remove]
                    if key_to_remove in self.insertion_times:
                        del self.insertion_times[key_to_remove]
                else:
                    # If all items are predicted to be accessed, use LRU as fallback
                    key, _ = self.cache.popitem(last=False)
            else:  # PREDICTIVE
                # Just use LRU as a simple predictive approach
                key, _ = self.cache.popitem(last=False)
        else:
            # Default to LRU
            key, _ = self.cache.popitem(last=False)

        # Update cache size
        if 'key_to_remove' in locals():
            if key_to_remove in self.cache:
                removed_size = self._estimate_tensor_size(self.cache[key_to_remove])
                self.cache_size -= removed_size
        else:
            # For LRU case
            if key in self.cache:
                removed_size = self._estimate_tensor_size(self.cache[key])
                self.cache_size -= removed_size

        self.stats["evictions"] += 1
        self.performance_monitor.record_eviction()
        return True

    def _evict_if_needed(self):
        """
        Evict entries from the cache if the size exceeds the maximum.
        """
        while self.cache_size > self.config.max_cache_size and len(self.cache) > 0:
            if not self._evict_by_policy():
                break  # Prevent infinite loop if eviction fails

    def put(self, key: str, value: torch.Tensor, force_insert: bool = False):
        """
        Put a value into the cache.

        Args:
            key: Key for the value
            value: Value to cache (tensor)
            force_insert: Whether to force insertion even if size constraints would prevent it
        """
        with self.lock:
            # Check if value is long enough to cache
            if value.size(-2) < self.config.min_prefix_length and not force_insert:
                return  # Don't cache short sequences

            # Check if value is too long
            if value.size(-2) > self.config.max_prefix_length and not force_insert:
                return  # Don't cache very long sequences

            # Compress the data
            compressed_data = self._compress_tensor(value)

            # Calculate size
            estimated_size = self._estimate_tensor_size(value)  # Use original size for estimation

            # Check if this is a warm key (seen multiple times)
            if key in self.access_counts:
                self.access_counts[key] += 1
            else:
                self.access_counts[key] = 1
                self.insertion_times[key] = time.time()

            # Only cache if we've seen this key enough times or forced insertion
            if self.access_counts[key] < self.config.cache_warmup_threshold and not force_insert:
                return

            # Remove old entry if exists
            if key in self.cache:
                old_size = self._estimate_tensor_size(self.cache[key])
                self.cache_size -= old_size
                del self.cache[key]

            # Add to cache
            self.cache[key] = compressed_data
            self.cache_size += estimated_size
            self.access_times[key] = time.time()

            # Update access time for LRU
            if key in self.cache:
                self.cache.move_to_end(key)

            # Record access for prediction
            self.predictor.record_access(key, "insert")

            # Evict if needed
            self._evict_if_needed()

            # Trigger adaptive prefetching if enabled
            if self.config.enable_adaptive_prefetching:
                self._adaptive_prefetch(key)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a value from the cache.

        Args:
            key: Key for the value

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key not in self.cache:
                self.performance_monitor.record_miss()
                return None

            # Update access counts and times
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_times[key] = time.time()

            # Move to end for LRU
            self.cache.move_to_end(key)

            # Record access for prediction
            self.predictor.record_access(key, "access")

            # Decompress and return
            compressed_data = self.cache[key]
            result = self._decompress_tensor(compressed_data)

            self.stats["hits"] += 1
            self.performance_monitor.record_hit()
            return result

    def _adaptive_prefetch(self, current_key: str):
        """
        Perform adaptive prefetching based on access patterns.

        Args:
            current_key: The key that was just accessed
        """
        if not self.config.enable_prefetching:
            return

        # Get predictions for next likely accesses
        predictions = self.predictor.predict_next_accesses()

        for pred_key, confidence in predictions:
            if confidence >= self.config.prediction_confidence_threshold:
                # Prefetch this key if it's related to the current access
                # In a real implementation, this would involve more sophisticated logic
                # to determine related keys
                pass

    def prefetch(self, key: str) -> Optional[torch.Tensor]:
        """
        Prefetch a value based on prediction.

        Args:
            key: Key to prefetch

        Returns:
            Prefetched value if available
        """
        if not self.config.enable_prefetching:
            return None

        # This is a simple implementation - in a real system, this would
        # use more sophisticated prediction algorithms
        result = self.get(key)
        if result is not None:
            self.performance_monitor.record_hit(is_prefetch=True)
        else:
            self.performance_monitor.record_miss(is_prefetch=True)

        return result

    def predict_and_prefetch(self) -> List[str]:
        """
        Predict next accesses and prefetch them.

        Returns:
            List of keys that were prefetched
        """
        if not self.config.enable_prefetching:
            return []

        predictions = self.predictor.predict_next_accesses()
        prefetched_keys = []

        for pred_key, confidence in predictions:
            if confidence >= self.config.prediction_confidence_threshold:
                # Attempt to prefetch
                self.prefetch(pred_key)
                prefetched_keys.append(pred_key)

        self.stats["predictions_made"] += len(predictions)
        return prefetched_keys

    def clear(self):
        """
        Clear the cache.
        """
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.insertion_times.clear()
            self.cache_size = 0
            self.stats = {k: 0 for k in self.stats}

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                **self.stats,
                "hit_rate": hit_rate,
                "miss_rate": 1 - hit_rate,
                "current_size_bytes": self.cache_size,
                "max_size_bytes": self.config.max_cache_size,
                "size_utilization": self.cache_size / self.config.max_cache_size if self.config.max_cache_size > 0 else 0,
                "num_cached_items": len(self.cache),
                "performance_metrics": self.performance_monitor.get_metrics()
            }


def apply_intelligent_caching_to_model(model: nn.Module, config: IntelligentCacheConfig) -> nn.Module:
    """
    Apply intelligent caching to the model.

    Args:
        model: The model to apply caching to
        config: Intelligent cache configuration

    Returns:
        Model with intelligent caching applied
    """
    # Create and attach the intelligent cache manager to the model
    model.intelligent_cache_manager = IntelligentCacheManager(config)

    return model


def create_intelligent_cache_for_qwen3_coder(config) -> IntelligentCacheManager:
    """
    Create an intelligent cache manager specifically configured for Qwen3-Coder-30B.

    Args:
        config: Qwen3-Coder-30B configuration

    Returns:
        Intelligent cache manager
    """
    intelligent_config = IntelligentCacheConfig(
        max_cache_size=getattr(config, 'intelligent_cache_max_size', 1024 * 1024 * 512),
        cache_precision=_str_to_dtype(getattr(config, 'intelligent_cache_precision', 'float16')),
        compression_enabled=getattr(config, 'intelligent_cache_compression_enabled', True),
        compression_method=getattr(config, 'intelligent_cache_compression_method', 'intelligent'),
        cache_policy=_str_to_cache_policy(getattr(config, 'intelligent_cache_policy', 'intelligent')),
        enable_prefetching=getattr(config, 'intelligent_cache_enable_prefetching', True),
        prefetch_distance=getattr(config, 'intelligent_cache_prefetch_distance', 1),
        max_prefix_length=getattr(config, 'intelligent_cache_max_prefix_length', 4096),
        min_prefix_length=getattr(config, 'intelligent_cache_min_prefix_length', 8),
        cache_warmup_threshold=getattr(config, 'intelligent_cache_warmup_threshold', 2),
        prediction_horizon=getattr(config, 'intelligent_cache_prediction_horizon', 15),
        prediction_confidence_threshold=getattr(config, 'intelligent_cache_prediction_confidence_threshold', 0.65),
        enable_adaptive_eviction=getattr(config, 'intelligent_cache_enable_adaptive_eviction', True),
        enable_adaptive_prefetching=getattr(config, 'intelligent_cache_enable_adaptive_prefetching', True),
        adaptive_window_size=getattr(config, 'intelligent_cache_adaptive_window_size', 200),
        enable_performance_monitoring=getattr(config, 'intelligent_cache_enable_performance_monitoring', True),
        performance_log_interval=getattr(config, 'intelligent_cache_performance_log_interval', 100)
    )

    return IntelligentCacheManager(intelligent_config)


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
        "int8": torch.int8,
    }
    return dtype_map.get(dtype_str, torch.float16)


def _str_to_cache_policy(policy_str: str) -> CachePolicy:
    """
    Convert string to cache policy.

    Args:
        policy_str: String representation of cache policy

    Returns:
        Corresponding cache policy
    """
    policy_map = {
        "lru": CachePolicy.LRU,
        "fifo": CachePolicy.FIFO,
        "lfu": CachePolicy.LFU,
        "predictive": CachePolicy.PREDICTIVE,
        "intelligent": CachePolicy.INTELLIGENT,
    }
    return policy_map.get(policy_str, CachePolicy.INTELLIGENT)


__all__ = [
    "CachePolicy",
    "IntelligentCacheConfig",
    "IntelligentCacheManager",
    "AccessPatternPredictor",
    "PerformanceMonitor",
    "apply_intelligent_caching_to_model",
    "create_intelligent_cache_for_qwen3_coder",
]