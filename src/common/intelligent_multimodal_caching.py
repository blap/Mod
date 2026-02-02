"""
Generic Intelligent Multimodal Caching System

This module implements a generic intelligent caching system for multimodal models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations and caching strategies.
"""

import hashlib
import logging
import os
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class CacheEvictionPolicy(Enum):
    """Enumeration of cache eviction policies."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    PREDICTIVE = "predictive"


class CacheEntryType(Enum):
    """Enumeration of cache entry types."""

    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE_PAIR = "text_image_pair"
    VISION_ENCODER_OUTPUT = "vision_encoder_output"
    LANGUAGE_ENCODER_OUTPUT = "language_encoder_output"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    FUSION_OUTPUT = "fusion_output"


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""

    data: Any
    timestamp: float
    access_count: int
    entry_type: CacheEntryType
    size_bytes: int
    similarity_hash: str
    ttl: Optional[float] = None  # Time-to-live in seconds
    priority: float = 0.5  # Priority for retention (0.0 to 1.0)


class GenericIntelligentMultimodalCache:
    """
    Generic intelligent caching system for multimodal data with adaptive strategies.

    This cache system is designed for vision-language models with support for caching
    of text, image, and cross-modal attention states with intelligent eviction based on
    access patterns and content similarity.
    """

    def __init__(
        self,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.PREDICTIVE,
        enable_similarity_caching: bool = True,
        similarity_threshold: float = 0.8,
        enable_ttl: bool = True,
        default_ttl: float = 3600.0,  # 1 hour default TTL
        enable_compression: bool = True,
        compression_ratio: float = 0.5,
    ):
        """
        Initialize the generic intelligent multimodal cache.

        Args:
            max_size_bytes: Maximum cache size in bytes
            eviction_policy: Policy for evicting cache entries
            enable_similarity_caching: Whether to enable similarity-based caching
            similarity_threshold: Threshold for considering content similar
            enable_ttl: Whether to enable time-to-live for cache entries
            default_ttl: Default TTL in seconds
            enable_compression: Whether to enable compression for cached data
            compression_ratio: Target compression ratio (0.0 to 1.0)
        """
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.eviction_policy = eviction_policy
        self.enable_similarity_caching = enable_similarity_caching
        self.similarity_threshold = similarity_threshold
        self.enable_ttl = enable_ttl
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.compression_ratio = compression_ratio

        # Initialize cache storage based on policy
        if eviction_policy == CacheEvictionPolicy.LRU:
            self.cache = OrderedDict()
        else:
            self.cache = {}

        # Track access patterns for predictive eviction
        self.access_history = defaultdict(list)
        self.access_frequency = defaultdict(int)

        # Track content similarity groups
        self.similarity_groups = defaultdict(set)

        logger.info(
            f"Initialized GenericIntelligentMultimodalCache with max_size={max_size_bytes/(1024**3):.2f}GB, "
            f"policy={eviction_policy.value}, similarity_caching={enable_similarity_caching}"
        )

    def _calculate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate the size of a tensor in bytes."""
        if tensor is None:
            return 0
        return tensor.numel() * tensor.element_size()

    def _calculate_size(self, data: Any) -> int:
        """Calculate the approximate size of cached data in bytes."""
        if isinstance(data, torch.Tensor):
            return self._calculate_tensor_size(data)
        elif isinstance(data, (list, tuple)):
            return sum(self._calculate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._calculate_size(value) for value in data.values())
        elif isinstance(data, (str, int, float)):
            return len(str(data).encode("utf-8")) if isinstance(data, str) else 8
        elif isinstance(data, Image.Image):
            # Approximate size based on image dimensions and mode
            return data.width * data.height * len(data.mode)  # bytes per pixel
        else:
            # For other types, try to get size with pickle
            try:
                return len(pickle.dumps(data))
            except:
                return 0

    def _generate_similarity_hash(self, data: Any) -> str:
        """Generate a similarity hash for the given data."""
        if isinstance(data, torch.Tensor):
            # For tensors, use a hash of the flattened tensor with reduced precision
            flat_tensor = data.flatten().cpu().numpy()
            # Use only a sample of values to reduce computation
            if len(flat_tensor) > 1000:
                sample_indices = np.linspace(0, len(flat_tensor) - 1, 1000, dtype=int)
                flat_tensor = flat_tensor[sample_indices]
            # Round to reduce sensitivity to small numerical differences
            rounded_tensor = np.round(flat_tensor, decimals=2)
            return hashlib.sha256(rounded_tensor.tobytes()).hexdigest()
        elif isinstance(data, (str, int, float)):
            return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
        elif isinstance(data, Image.Image):
            # Resize image to a standard size for hashing
            resized_img = data.resize((224, 224)).convert("RGB")
            img_array = np.array(resized_img)
            return hashlib.sha256(img_array.tobytes()).hexdigest()
        else:
            # For other types, use pickle hash
            try:
                pickled_data = pickle.dumps(data)
                return hashlib.sha256(pickled_data).hexdigest()
            except:
                return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if not self.enable_ttl or entry.ttl is None:
            return False
        return time.time() - entry.timestamp > entry.ttl

    def _compress_data(self, data: Any) -> Any:
        """Compress data if compression is enabled."""
        if not self.enable_compression:
            return data

        # For tensors, apply quantization-based compression
        if isinstance(data, torch.Tensor):
            # Quantize to reduce precision (simulating compression)
            if data.dtype == torch.float32:
                return data.half()  # Convert to float16
            elif data.dtype == torch.float64:
                return data.float()  # Convert to float32
            else:
                return data
        elif isinstance(data, (list, tuple)):
            return type(data)(self._compress_data(item) for item in data)
        elif isinstance(data, dict):
            return {key: self._compress_data(value) for key, value in data.items()}
        else:
            # For other types, return as is (compression not applicable)
            return data

    def _decompress_data(self, data: Any) -> Any:
        """Decompress data if compression was applied."""
        if not self.enable_compression:
            return data

        # For tensors, decompress by restoring precision
        if isinstance(data, torch.Tensor):
            if data.dtype == torch.float16:
                return data.float()  # Convert back to float32
            else:
                return data
        elif isinstance(data, (list, tuple)):
            return type(data)(self._decompress_data(item) for item in data)
        elif isinstance(data, dict):
            return {key: self._decompress_data(value) for key, value in data.items()}
        else:
            return data

    def _evict_entries(self, needed_space: int):
        """Evict entries to make space for new data."""
        if needed_space <= 0:
            return

        # Sort entries based on eviction policy
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # For LRU, entries are already ordered in OrderedDict
            while (
                self.current_size_bytes + needed_space > self.max_size_bytes
                and len(self.cache) > 0
            ):
                key, entry = self.cache.popitem(last=False)
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted LRU entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # For FIFO, sort by timestamp
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].timestamp)
            while (
                self.current_size_bytes + needed_space > self.max_size_bytes
                and len(self.cache) > 0
            ):
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted FIFO entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # For LFU, sort by access count
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].access_count)
            while (
                self.current_size_bytes + needed_space > self.max_size_bytes
                and len(self.cache) > 0
            ):
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted LFU entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.PREDICTIVE:
            # For predictive, consider access patterns and priority
            # Entries with lower access frequency and lower priority are evicted first
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (self.access_frequency[x[0]], x[1].priority),
            )
            while (
                self.current_size_bytes + needed_space > self.max_size_bytes
                and len(self.cache) > 0
            ):
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted predictive entry: {key}")

    def put(
        self,
        key: str,
        data: Any,
        entry_type: CacheEntryType,
        ttl: Optional[float] = None,
        priority: float = 0.5,
    ):
        """
        Put data into the cache with metadata.

        Args:
            key: Cache key
            data: Data to cache
            entry_type: Type of the cached entry
            ttl: Time-to-live in seconds (None for default)
            priority: Priority for retention (0.0 to 1.0)
        """
        # Calculate size and similarity hash
        original_size = self._calculate_size(data)
        similarity_hash = self._generate_similarity_hash(data)

        # Compress data if enabled
        compressed_data = self._compress_data(data)
        compressed_size = self._calculate_size(compressed_data)

        # Check if we need to evict entries
        needed_space = compressed_size
        if key in self.cache:
            # If key exists, account for the difference in size
            needed_space = needed_space - self.cache[key].size_bytes

        if self.current_size_bytes + needed_space > self.max_size_bytes:
            self._evict_entries(needed_space)

        # Create cache entry
        entry = CacheEntry(
            data=compressed_data,
            timestamp=time.time(),
            access_count=1,
            entry_type=entry_type,
            size_bytes=compressed_size,
            similarity_hash=similarity_hash,
            ttl=ttl or self.default_ttl,
            priority=priority,
        )

        # Add to cache
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Remove old key if exists to update position
            if key in self.cache:
                del self.cache[key]
            self.cache[key] = entry
        else:
            self.cache[key] = entry

        # Update size tracking
        self.current_size_bytes += compressed_size

        # Update access history
        self.access_history[key].append(time.time())
        self.access_frequency[key] += 1

        # Update similarity groups if enabled
        if self.enable_similarity_caching:
            for existing_key, existing_entry in self.cache.items():
                if existing_key != key and existing_entry.entry_type == entry_type:
                    similarity_score = self._calculate_similarity(
                        entry.similarity_hash, existing_entry.similarity_hash
                    )
                    if similarity_score > self.similarity_threshold:
                        self.similarity_groups[existing_key].add(key)
                        self.similarity_groups[key].add(existing_key)

        logger.debug(
            f"Put entry in cache: {key}, type: {entry_type.value}, size: {compressed_size/(1024**2):.2f}MB, "
            f"priority: {priority:.2f}"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Get data from the cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check if expired
        if self._is_expired(entry):
            self.delete(key)
            return None

        # Update access count and history
        entry.access_count += 1
        self.access_history[key].append(time.time())
        self.access_frequency[key] += 1

        # For LRU policy, move to end
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            self.cache.move_to_end(key)

        # Decompress data if needed
        decompressed_data = self._decompress_data(entry.data)

        logger.debug(
            f"Get entry from cache: {key}, type: {entry.entry_type.value}, "
            f"access_count: {entry.access_count}"
        )

        return decompressed_data

    def delete(self, key: str):
        """Delete an entry from the cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes

            # Remove from similarity groups
            for group_key in self.similarity_groups[key]:
                self.similarity_groups[group_key].discard(key)
            del self.similarity_groups[key]

            # Remove from cache
            if self.eviction_policy == CacheEvictionPolicy.LRU:
                del self.cache[key]
            else:
                del self.cache[key]

            logger.debug(f"Deleted entry from cache: {key}")

    def get_by_similarity(
        self, data: Any, entry_type: CacheEntryType, threshold: Optional[float] = None
    ) -> Optional[Tuple[str, Any]]:
        """
        Get data from cache based on similarity to the provided data.

        Args:
            data: Data to compare for similarity
            entry_type: Type of entries to search for
            threshold: Similarity threshold (uses default if None)

        Returns:
            Tuple of (key, data) if similar entry found, None otherwise
        """
        if not self.enable_similarity_caching:
            return None

        threshold = threshold or self.similarity_threshold
        target_hash = self._generate_similarity_hash(data)

        for key, entry in self.cache.items():
            if entry.entry_type == entry_type and not self._is_expired(entry):
                similarity_score = self._calculate_similarity(
                    target_hash, entry.similarity_hash
                )
                if similarity_score > threshold:
                    # Update access count and history
                    entry.access_count += 1
                    self.access_history[key].append(time.time())
                    self.access_frequency[key] += 1

                    # For LRU policy, move to end
                    if self.eviction_policy == CacheEvictionPolicy.LRU:
                        self.cache.move_to_end(key)

                    # Decompress data if needed
                    decompressed_data = self._decompress_data(entry.data)

                    logger.debug(
                        f"Similarity hit: {key}, similarity: {similarity_score:.3f}"
                    )
                    return key, decompressed_data

        return None

    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes."""
        # For now, use exact match (will be improved in future versions)
        # In a more advanced implementation, we would use techniques like:
        # - Hamming distance for binary hashes
        # - Cosine similarity for embedding vectors
        # - Jaccard similarity for sets of features
        if hash1 == hash2:
            return 1.0
        else:
            # For now, return 0.0 for non-exact matches
            # In a future implementation, we could compute actual similarity
            # between hash representations using techniques like Hamming distance
            return 0.0

    def clear_expired(self):
        """Clear expired entries from the cache."""
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values() if self._is_expired(entry)
        )
        active_entries = total_entries - expired_entries

        # Calculate average access frequency
        avg_access_freq = (
            sum(self.access_frequency.values()) / len(self.access_frequency)
            if self.access_frequency
            else 0
        )

        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "current_size_bytes": self.current_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "usage_percentage": (
                (self.current_size_bytes / self.max_size_bytes) * 100
                if self.max_size_bytes > 0
                else 0
            ),
            "average_access_frequency": avg_access_freq,
            "eviction_policy": self.eviction_policy.value,
            "compression_enabled": self.enable_compression,
            "similarity_caching_enabled": self.enable_similarity_caching,
        }

    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()
        self.access_history.clear()
        self.access_frequency.clear()
        self.similarity_groups.clear()
        self.current_size_bytes = 0
        logger.info("Cache cleared")


class GenericIntelligentCachingManager:
    """
    Generic caching manager for multimodal models.
    Specific models should extend this class with their own optimizations.
    """

    def __init__(
        self,
        cache_size_gb: float = 2.0,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.PREDICTIVE,
        enable_similarity_caching: bool = True,
        similarity_threshold: float = 0.85,
        enable_ttl: bool = True,
        default_ttl: float = 7200.0,  # 2 hours
        enable_compression: bool = True,
        compression_ratio: float = 0.6,
    ):
        """
        Initialize the generic intelligent caching manager.

        Args:
            cache_size_gb: Cache size in gigabytes
            eviction_policy: Policy for evicting cache entries
            enable_similarity_caching: Whether to enable similarity-based caching
            similarity_threshold: Threshold for considering content similar
            enable_ttl: Whether to enable time-to-live for cache entries
            default_ttl: Default TTL in seconds
            enable_compression: Whether to enable compression for cached data
            compression_ratio: Target compression ratio (0.0 to 1.0)
        """
        cache_size_bytes = int(cache_size_gb * 1024 * 1024 * 1024)
        self.cache = GenericIntelligentMultimodalCache(
            max_size_bytes=cache_size_bytes,
            eviction_policy=eviction_policy,
            enable_similarity_caching=enable_similarity_caching,
            similarity_threshold=similarity_threshold,
            enable_ttl=enable_ttl,
            default_ttl=default_ttl,
            enable_compression=enable_compression,
            compression_ratio=compression_ratio,
        )

        # Track specific types of cached data
        self.text_cache_keys = set()
        self.image_cache_keys = set()
        self.vision_encoder_cache_keys = set()
        self.language_encoder_cache_keys = set()
        self.cross_modal_cache_keys = set()
        self.fusion_cache_keys = set()

        logger.info("Generic Intelligent Caching Manager initialized")

    def cache_text_input(
        self, text: str, processed_tensor: torch.Tensor, priority: float = 0.5
    ):
        """
        Cache processed text input.

        Args:
            text: Original text input
            processed_tensor: Processed tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        key = f"text_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=processed_tensor,
            entry_type=CacheEntryType.TEXT,
            priority=priority,
        )
        self.text_cache_keys.add(key)
        logger.debug(f"Cached text input: {key}")

    def cache_image_input(
        self, image: Image.Image, processed_tensor: torch.Tensor, priority: float = 0.5
    ):
        """
        Cache processed image input.

        Args:
            image: Original image input
            processed_tensor: Processed tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        # Convert image to bytes for hashing
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        key = f"image_{hashlib.sha256(img_bytes).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=processed_tensor,
            entry_type=CacheEntryType.IMAGE,
            priority=priority,
        )
        self.image_cache_keys.add(key)
        logger.debug(f"Cached image input: {key}")

    def cache_text_image_pair(
        self, text: str, image: Image.Image, processed_pair: Any, priority: float = 0.7
    ):
        """
        Cache processed text-image pair.

        Args:
            text: Text input
            image: Image input
            processed_pair: Processed paired representation
            priority: Priority for retention (0.0 to 1.0)
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()[:8]
        key = f"pair_{text_hash}_{img_hash}"
        self.cache.put(
            key=key,
            data=processed_pair,
            entry_type=CacheEntryType.TEXT_IMAGE_PAIR,
            priority=priority,
        )
        self.cross_modal_cache_keys.add(key)
        logger.debug(f"Cached text-image pair: {key}")

    def get_cached_text_input(self, text: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached text input.

        Args:
            text: Original text input

        Returns:
            Cached tensor or None if not found
        """
        key = f"text_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_image_input(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        Retrieve cached image input.

        Args:
            image: Original image input

        Returns:
            Cached tensor or None if not found
        """
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        key = f"image_{hashlib.sha256(img_bytes).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_text_image_pair(
        self, text: str, image: Image.Image
    ) -> Optional[Any]:
        """
        Retrieve cached text-image pair.

        Args:
            text: Text input
            image: Image input

        Returns:
            Cached paired representation or None if not found
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()[:8]
        key = f"pair_{text_hash}_{img_hash}"
        return self.cache.get(key)

    def find_similar_text(self, text: str) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Find similar text in cache.

        Args:
            text: Text to compare

        Returns:
            Tuple of (key, cached_tensor) if similar text found, None otherwise
        """
        return self.cache.get_by_similarity(text, CacheEntryType.TEXT)

    def find_similar_image(
        self, image: Image.Image
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Find similar image in cache.

        Args:
            image: Image to compare

        Returns:
            Tuple of (key, cached_tensor) if similar image found, None otherwise
        """
        return self.cache.get_by_similarity(image, CacheEntryType.IMAGE)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        base_stats = self.cache.get_cache_stats()

        # Add model-specific cache breakdown
        base_stats.update(
            {
                "text_cache_entries": len(self.text_cache_keys),
                "image_cache_entries": len(self.image_cache_keys),
                "vision_encoder_cache_entries": len(self.vision_encoder_cache_keys),
                "language_encoder_cache_entries": len(self.language_encoder_cache_keys),
                "cross_modal_cache_entries": len(self.cross_modal_cache_keys),
                "fusion_cache_entries": len(self.fusion_cache_keys),
            }
        )

        return base_stats

    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.text_cache_keys.clear()
        self.image_cache_keys.clear()
        self.vision_encoder_cache_keys.clear()
        self.language_encoder_cache_keys.clear()
        self.cross_modal_cache_keys.clear()
        self.fusion_cache_keys.clear()
        logger.info("Generic cache cleared")


def create_generic_intelligent_caching_manager(
    cache_size_gb: float = 2.0,
) -> GenericIntelligentCachingManager:
    """
    Factory function to create a generic intelligent caching manager.

    Args:
        cache_size_gb: Cache size in gigabytes

    Returns:
        GenericIntelligentCachingManager: The created caching manager
    """
    return GenericIntelligentCachingManager(cache_size_gb=cache_size_gb)


class Qwen3VL2BIntelligentCachingManager(GenericIntelligentCachingManager):
    """
    Specialized caching manager for Qwen3-VL-2B model with multimodal optimizations.
    """

    def __init__(
        self,
        cache_size_gb: float = 2.0,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.PREDICTIVE,
        enable_similarity_caching: bool = True,
        similarity_threshold: float = 0.85,
        enable_ttl: bool = True,
        default_ttl: float = 7200.0,  # 2 hours for Qwen3-VL-2B
        enable_compression: bool = True,
        compression_ratio: float = 0.6,
    ):
        """
        Initialize the Qwen3-VL-2B intelligent caching manager.

        Args:
            cache_size_gb: Cache size in gigabytes
            eviction_policy: Policy for evicting cache entries
            enable_similarity_caching: Whether to enable similarity-based caching
            similarity_threshold: Threshold for considering content similar
            enable_ttl: Whether to enable time-to-live for cache entries
            default_ttl: Default TTL in seconds
            enable_compression: Whether to enable compression for cached data
            compression_ratio: Target compression ratio (0.0 to 1.0)
        """
        super().__init__(
            cache_size_gb=cache_size_gb,
            eviction_policy=eviction_policy,
            enable_similarity_caching=enable_similarity_caching,
            similarity_threshold=similarity_threshold,
            enable_ttl=enable_ttl,
            default_ttl=default_ttl,
            enable_compression=enable_compression,
            compression_ratio=compression_ratio,
        )

        # Track specific types of cached data for Qwen3-VL-2B
        self.text_cache_keys = set()
        self.image_cache_keys = set()
        self.vision_encoder_cache_keys = set()
        self.language_encoder_cache_keys = set()
        self.cross_modal_cache_keys = set()
        self.fusion_cache_keys = set()

        logger.info("Qwen3-VL-2B Intelligent Caching Manager initialized")

    def cache_text_input(
        self, text: str, processed_tensor: torch.Tensor, priority: float = 0.5
    ):
        """
        Cache processed text input for Qwen3-VL-2B model.

        Args:
            text: Original text input
            processed_tensor: Processed tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        key = f"text_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=processed_tensor,
            entry_type=CacheEntryType.TEXT,
            priority=priority,
        )
        self.text_cache_keys.add(key)
        logger.debug(f"Cached text input: {key}")

    def cache_image_input(
        self, image: Image.Image, processed_tensor: torch.Tensor, priority: float = 0.5
    ):
        """
        Cache processed image input for Qwen3-VL-2B model.

        Args:
            image: Original image input
            processed_tensor: Processed tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        # Convert image to bytes for hashing
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        key = f"image_{hashlib.sha256(img_bytes).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=processed_tensor,
            entry_type=CacheEntryType.IMAGE,
            priority=priority,
        )
        self.image_cache_keys.add(key)
        logger.debug(f"Cached image input: {key}")

    def cache_text_image_pair(
        self, text: str, image: Image.Image, processed_pair: Any, priority: float = 0.7
    ):
        """
        Cache processed text-image pair for Qwen3-VL-2B model.

        Args:
            text: Text input
            image: Image input
            processed_pair: Processed paired representation
            priority: Priority for retention (0.0 to 1.0)
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()[:8]
        key = f"pair_{text_hash}_{img_hash}"
        self.cache.put(
            key=key,
            data=processed_pair,
            entry_type=CacheEntryType.TEXT_IMAGE_PAIR,
            priority=priority,
        )
        self.cross_modal_cache_keys.add(key)
        logger.debug(f"Cached text-image pair: {key}")

    def get_cached_text_input(self, text: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached text input for Qwen3-VL-2B model.

        Args:
            text: Original text input

        Returns:
            Cached tensor or None if not found
        """
        key = f"text_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_image_input(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        Retrieve cached image input for Qwen3-VL-2B model.

        Args:
            image: Original image input

        Returns:
            Cached tensor or None if not found
        """
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        key = f"image_{hashlib.sha256(img_bytes).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_text_image_pair(
        self, text: str, image: Image.Image
    ) -> Optional[Any]:
        """
        Retrieve cached text-image pair for Qwen3-VL-2B model.

        Args:
            text: Text input
            image: Image input

        Returns:
            Cached paired representation or None if not found
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()[:8]
        key = f"pair_{text_hash}_{img_hash}"
        return self.cache.get(key)

    def find_similar_text(self, text: str) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Find similar text in cache for Qwen3-VL-2B model.

        Args:
            text: Text to compare

        Returns:
            Tuple of (key, cached_tensor) if similar text found, None otherwise
        """
        return self.cache.get_by_similarity(text, CacheEntryType.TEXT)

    def find_similar_image(
        self, image: Image.Image
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Find similar image in cache for Qwen3-VL-2B model.

        Args:
            image: Image to compare

        Returns:
            Tuple of (key, cached_tensor) if similar image found, None otherwise
        """
        return self.cache.get_by_similarity(image, CacheEntryType.IMAGE)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for Qwen3-VL-2B model.

        Returns:
            Dictionary containing cache statistics
        """
        base_stats = self.cache.get_cache_stats()

        # Add model-specific cache breakdown
        base_stats.update(
            {
                "text_cache_entries": len(self.text_cache_keys),
                "image_cache_entries": len(self.image_cache_keys),
                "vision_encoder_cache_entries": len(self.vision_encoder_cache_keys),
                "language_encoder_cache_entries": len(self.language_encoder_cache_keys),
                "cross_modal_cache_entries": len(self.cross_modal_cache_keys),
                "fusion_cache_entries": len(self.fusion_cache_keys),
            }
        )

        return base_stats

    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.text_cache_keys.clear()
        self.image_cache_keys.clear()
        self.vision_encoder_cache_keys.clear()
        self.language_encoder_cache_keys.clear()
        self.cross_modal_cache_keys.clear()
        self.fusion_cache_keys.clear()
        logger.info("Qwen3-VL-2B cache cleared")


def create_qwen3_vl_intelligent_caching_manager(
    cache_size_gb: float = 2.0,
) -> Qwen3VL2BIntelligentCachingManager:
    """
    Factory function to create a Qwen3-VL-2B intelligent caching manager.

    Args:
        cache_size_gb: Cache size in gigabytes

    Returns:
        Qwen3VL2BIntelligentCachingManager: The created caching manager
    """
    return Qwen3VL2BIntelligentCachingManager(cache_size_gb=cache_size_gb)


def apply_intelligent_multimodal_caching_to_model(
    model: nn.Module, caching_manager: GenericIntelligentCachingManager
) -> nn.Module:
    """
    Apply intelligent multimodal caching to the model.

    Args:
        model: The model to optimize
        caching_manager: The caching manager to use

    Returns:
        Model with caching capabilities
    """
    logger.info("Applying generic intelligent multimodal caching to model...")

    # Add caching manager to model
    model._caching_manager = caching_manager

    # Add caching methods to model if they don't exist
    if not hasattr(model, "cache_text_input"):
        model.cache_text_input = caching_manager.cache_text_input

    if not hasattr(model, "cache_image_input"):
        model.cache_image_input = caching_manager.cache_image_input

    if not hasattr(model, "cache_text_image_pair"):
        model.cache_text_image_pair = caching_manager.cache_text_image_pair

    if not hasattr(model, "get_cached_text_input"):
        model.get_cached_text_input = caching_manager.get_cached_text_input

    if not hasattr(model, "get_cached_image_input"):
        model.get_cached_image_input = caching_manager.get_cached_image_input

    if not hasattr(model, "get_cached_text_image_pair"):
        model.get_cached_text_image_pair = caching_manager.get_cached_text_image_pair

    if not hasattr(model, "find_similar_text"):
        model.find_similar_text = caching_manager.find_similar_text

    if not hasattr(model, "find_similar_image"):
        model.find_similar_image = caching_manager.find_similar_image

    logger.info("Generic intelligent multimodal caching applied successfully")
    return model


__all__ = [
    "CacheEvictionPolicy",
    "CacheEntryType",
    "CacheEntry",
    "GenericIntelligentMultimodalCache",
    "GenericIntelligentCachingManager",
    "Qwen3VL2BIntelligentCachingManager",
    "create_generic_intelligent_caching_manager",
    "create_qwen3_vl_intelligent_caching_manager",
    "apply_intelligent_multimodal_caching_to_model",
]
