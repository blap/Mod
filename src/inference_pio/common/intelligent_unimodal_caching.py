"""
Intelligent Unimodal Caching System for Text Models

This module implements an intelligent caching system specifically designed for unimodal text models
like GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b. The system optimizes caching for
textual data with adaptive strategies based on access patterns, content similarity, and language-specific
optimizations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import hashlib
import pickle
import os
import time
from collections import OrderedDict, defaultdict
import numpy as np
from io import StringIO
import logging
from dataclasses import dataclass
from enum import Enum
import re


logger = logging.getLogger(__name__)


class CacheEvictionPolicy(Enum):
    """Enumeration of cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    PREDICTIVE = "predictive"
    CUSTOM = "custom"


class CacheEntryType(Enum):
    """Enumeration of cache entry types for unimodal text models."""
    TEXT_INPUT = "text_input"
    TOKENIZED_INPUT = "tokenized_input"
    EMBEDDING_OUTPUT = "embedding_output"
    ATTENTION_OUTPUT = "attention_output"
    FFN_OUTPUT = "ffn_output"
    LAYER_OUTPUT = "layer_output"
    PREFIX_CACHE = "prefix_cache"
    KV_CACHE = "kv_cache"
    DECODER_STATE = "decoder_state"
    CONTEXT_WINDOW = "context_window"


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
    language_specific_features: Optional[Dict[str, Any]] = None  # Language-specific features


class IntelligentUnimodalCache:
    """
    Intelligent caching system for unimodal text data with adaptive strategies.

    This cache system is optimized for text models like GLM-4-7, Qwen3-4b-instruct-2507,
    and Qwen3-coder-30b, supporting caching of various text processing stages with
    intelligent eviction based on access patterns, content similarity, and language-specific features.
    """

    def __init__(self,
                 max_size_bytes: int = 512 * 1024 * 1024,  # 512MB default
                 eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.PREDICTIVE,
                 enable_similarity_caching: bool = True,
                 similarity_threshold: float = 0.8,
                 enable_ttl: bool = True,
                 default_ttl: float = 1800.0,  # 30 minutes default TTL
                 enable_compression: bool = True,
                 compression_ratio: float = 0.5,
                 enable_language_specific_optimizations: bool = True,
                 language_model_type: str = "general"):
        """
        Initialize the intelligent unimodal cache.

        Args:
            max_size_bytes: Maximum cache size in bytes
            eviction_policy: Policy for evicting cache entries
            enable_similarity_caching: Whether to enable similarity-based caching
            similarity_threshold: Threshold for considering content similar
            enable_ttl: Whether to enable time-to-live for cache entries
            default_ttl: Default TTL in seconds
            enable_compression: Whether to enable compression for cached data
            compression_ratio: Target compression ratio (0.0 to 1.0)
            enable_language_specific_optimizations: Whether to enable language-specific optimizations
            language_model_type: Type of language model (affects optimizations)
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
        self.enable_language_specific_optimizations = enable_language_specific_optimizations
        self.language_model_type = language_model_type

        # Initialize cache storage based on policy
        if eviction_policy == CacheEvictionPolicy.LRU:
            self.cache = OrderedDict()
        else:
            self.cache = {}

        # Track access patterns for predictive eviction
        self.access_history = defaultdict(list)
        self.access_frequency = defaultdict(int)
        self.access_intervals = defaultdict(list)  # Track intervals between accesses

        # Track content similarity groups
        self.similarity_groups = defaultdict(set)

        # Track language-specific features
        self.language_features = defaultdict(dict)

        logger.info(f"Initialized IntelligentUnimodalCache with max_size={max_size_bytes/(1024**3):.2f}GB, "
                   f"policy={eviction_policy.value}, similarity_caching={enable_similarity_caching}, "
                   f"model_type={language_model_type}")

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
            return len(str(data).encode('utf-8')) if isinstance(data, str) else 8
        else:
            # For other types, try to get size with pickle
            try:
                return len(pickle.dumps(data))
            except:
                return 0

    def _extract_language_features(self, text: str) -> Dict[str, Any]:
        """Extract language-specific features from text."""
        if not self.enable_language_specific_optimizations:
            return {}

        features = {}
        
        # Basic linguistic features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character-level features
        features['char_types'] = len(set(text.lower()))
        features['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        features['upper_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        
        # Language-specific patterns (for code models)
        if 'code' in self.language_model_type.lower() or 'coder' in self.language_model_type.lower():
            features['has_code_patterns'] = bool(re.search(r'\b(def|class|import|from|function|var|let|const)\b', text))
            features['bracket_balance'] = text.count('{') - text.count('}')
            features['indentation_level'] = len(text) - len(text.lstrip(' \t')) if text.strip() else 0
            
        # Language-specific patterns (for instruction models)
        if 'instruct' in self.language_model_type.lower():
            features['question_marks'] = text.count('?')
            features['exclamation_marks'] = text.count('!')
            features['colon_count'] = text.count(':')
            
        return features

    def _generate_similarity_hash(self, data: Any) -> str:
        """Generate a similarity hash for the given data."""
        if isinstance(data, torch.Tensor):
            # For tensors, use a hash of the flattened tensor with reduced precision
            flat_tensor = data.flatten().cpu().numpy()
            # Use only a sample of values to reduce computation
            if len(flat_tensor) > 1000:
                sample_indices = np.linspace(0, len(flat_tensor)-1, 1000, dtype=int)
                flat_tensor = flat_tensor[sample_indices]
            # Round to reduce sensitivity to small numerical differences
            rounded_tensor = np.round(flat_tensor, decimals=2)
            return hashlib.sha256(rounded_tensor.tobytes()).hexdigest()
        elif isinstance(data, str):
            # For text, use both raw content and language features
            if self.enable_language_specific_optimizations:
                features = self._extract_language_features(data)
                feature_str = str(sorted(features.items()))
                combined_content = data + feature_str
                return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()
            else:
                return hashlib.sha256(data.encode('utf-8')).hexdigest()
        elif isinstance(data, (int, float)):
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
        else:
            # For other types, use pickle hash
            try:
                pickled_data = pickle.dumps(data)
                return hashlib.sha256(pickled_data).hexdigest()
            except:
                return hashlib.sha256(str(data).encode('utf-8')).hexdigest()

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

    def _calculate_access_predictiveness(self, key: str) -> float:
        """Calculate predictiveness score for an entry based on access patterns."""
        if key not in self.access_history or len(self.access_history[key]) < 2:
            return 0.0

        # Calculate average interval between accesses
        intervals = [self.access_history[key][i+1] - self.access_history[key][i] 
                     for i in range(len(self.access_history[key])-1)]
        
        if not intervals:
            return 0.0

        avg_interval = sum(intervals) / len(intervals)
        time_since_last = time.time() - self.access_history[key][-1]

        # Higher score if access is expected soon (time_since_last < avg_interval)
        if avg_interval > 0:
            return max(0.0, 1.0 - (time_since_last / avg_interval))
        else:
            return 1.0

    def _evict_entries(self, needed_space: int):
        """Evict entries to make space for new data."""
        if needed_space <= 0:
            return

        # Sort entries based on eviction policy
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # For LRU, entries are already ordered in OrderedDict
            while self.current_size_bytes + needed_space > self.max_size_bytes and len(self.cache) > 0:
                key, entry = self.cache.popitem(last=False)
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted LRU entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # For FIFO, sort by timestamp
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].timestamp)
            while self.current_size_bytes + needed_space > self.max_size_bytes and len(self.cache) > 0:
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted FIFO entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # For LFU, sort by access count
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].access_count)
            while self.current_size_bytes + needed_space > self.max_size_bytes and len(self.cache) > 0:
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted LFU entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.PREDICTIVE:
            # For predictive, consider access patterns and priority
            # Entries with lower access predictiveness and lower priority are evicted first
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (self._calculate_access_predictiveness(x[0]), x[1].priority)
            )
            while self.current_size_bytes + needed_space > self.max_size_bytes and len(self.cache) > 0:
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted predictive entry: {key}")

        elif self.eviction_policy == CacheEvictionPolicy.CUSTOM:
            # Custom eviction based on language-specific features and access patterns
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (
                    self._calculate_access_predictiveness(x[0]),
                    x[1].priority,
                    self._calculate_language_relevance(x[0], x[1])
                )
            )
            while self.current_size_bytes + needed_space > self.max_size_bytes and len(self.cache) > 0:
                key, entry = sorted_entries.pop(0)
                del self.cache[key]
                self.current_size_bytes -= entry.size_bytes
                logger.debug(f"Evicted custom entry: {key}")

    def _calculate_language_relevance(self, key: str, entry: CacheEntry) -> float:
        """Calculate language-specific relevance for custom eviction."""
        if not self.enable_language_specific_optimizations:
            return 0.0

        # For code models, prioritize entries with code patterns
        if 'code' in self.language_model_type.lower() or 'coder' in self.language_model_type.lower():
            if entry.language_specific_features and entry.language_specific_features.get('has_code_patterns', False):
                return 1.0
            else:
                return 0.0

        # For instruction models, prioritize entries with instruction patterns
        if 'instruct' in self.language_model_type.lower():
            if entry.language_specific_features and entry.language_specific_features.get('question_marks', 0) > 0:
                return 1.0
            else:
                return 0.0

        # General relevance calculation
        return 0.5

    def put(self, key: str, data: Any, entry_type: CacheEntryType, 
            ttl: Optional[float] = None, priority: float = 0.5, text_content: Optional[str] = None):
        """
        Put data into the cache with metadata.

        Args:
            key: Cache key
            data: Data to cache
            entry_type: Type of the cached entry
            ttl: Time-to-live in seconds (None for default)
            priority: Priority for retention (0.0 to 1.0)
            text_content: Original text content (for language feature extraction)
        """
        # Calculate size and similarity hash
        original_size = self._calculate_size(data)
        similarity_hash = self._generate_similarity_hash(data)

        # Extract language-specific features if applicable
        language_features = None
        if text_content and self.enable_language_specific_optimizations:
            language_features = self._extract_language_features(text_content)

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
            language_specific_features=language_features
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
                    similarity_score = self._calculate_similarity(entry.similarity_hash, existing_entry.similarity_hash)
                    if similarity_score > self.similarity_threshold:
                        self.similarity_groups[existing_key].add(key)
                        self.similarity_groups[key].add(existing_key)

        logger.debug(f"Put entry in cache: {key}, type: {entry_type.value}, size: {compressed_size/(1024**2):.2f}MB, "
                    f"priority: {priority:.2f}")

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

        logger.debug(f"Get entry from cache: {key}, type: {entry.entry_type.value}, "
                    f"access_count: {entry.access_count}")

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

    def get_by_similarity(self, data: Any, entry_type: CacheEntryType, 
                         text_content: Optional[str] = None, threshold: Optional[float] = None) -> Optional[Tuple[str, Any]]:
        """
        Get data from cache based on similarity to the provided data.

        Args:
            data: Data to compare for similarity
            entry_type: Type of entries to search for
            text_content: Original text content (for language feature extraction)
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
                similarity_score = self._calculate_similarity(target_hash, entry.similarity_hash)
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

                    logger.debug(f"Similarity hit: {key}, similarity: {similarity_score:.3f}")
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
        expired_entries = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        active_entries = total_entries - expired_entries

        # Calculate average access frequency
        avg_access_freq = sum(self.access_frequency.values()) / len(self.access_frequency) if self.access_frequency else 0

        # Calculate cache hit rate estimation
        total_accesses = sum(self.access_frequency.values())
        hit_rate = avg_access_freq / (total_accesses / total_entries) if total_entries > 0 and total_accesses > 0 else 0

        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "current_size_bytes": self.current_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "usage_percentage": (self.current_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
            "average_access_frequency": avg_access_freq,
            "estimated_hit_rate": hit_rate,
            "eviction_policy": self.eviction_policy.value,
            "compression_enabled": self.enable_compression,
            "similarity_caching_enabled": self.enable_similarity_caching,
            "language_specific_optimizations": self.enable_language_specific_optimizations,
            "model_type": self.language_model_type
        }

    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()
        self.access_history.clear()
        self.access_frequency.clear()
        self.similarity_groups.clear()
        self.current_size_bytes = 0
        logger.info("Cache cleared")


class UnimodalCachingManager:
    """
    Specialized caching manager for unimodal text models with language-specific optimizations.
    """

    def __init__(self,
                 cache_size_mb: float = 512.0,
                 eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.PREDICTIVE,
                 enable_similarity_caching: bool = True,
                 similarity_threshold: float = 0.85,
                 enable_ttl: bool = True,
                 default_ttl: float = 3600.0,  # 1 hour for text models
                 enable_compression: bool = True,
                 compression_ratio: float = 0.6,
                 language_model_type: str = "general"):
        """
        Initialize the unimodal caching manager.

        Args:
            cache_size_mb: Cache size in megabytes
            eviction_policy: Policy for evicting cache entries
            enable_similarity_caching: Whether to enable similarity-based caching
            similarity_threshold: Threshold for considering content similar
            enable_ttl: Whether to enable time-to-live for cache entries
            default_ttl: Default TTL in seconds
            enable_compression: Whether to enable compression for cached data
            compression_ratio: Target compression ratio (0.0 to 1.0)
            language_model_type: Type of language model (affects optimizations)
        """
        cache_size_bytes = int(cache_size_mb * 1024 * 1024)
        self.cache = IntelligentUnimodalCache(
            max_size_bytes=cache_size_bytes,
            eviction_policy=eviction_policy,
            enable_similarity_caching=enable_similarity_caching,
            similarity_threshold=similarity_threshold,
            enable_ttl=enable_ttl,
            default_ttl=default_ttl,
            enable_compression=enable_compression,
            compression_ratio=compression_ratio,
            enable_language_specific_optimizations=True,
            language_model_type=language_model_type
        )

        # Track specific types of cached data
        self.text_input_cache_keys = set()
        self.tokenized_input_cache_keys = set()
        self.embedding_cache_keys = set()
        self.attention_cache_keys = set()
        self.ffn_cache_keys = set()
        self.layer_cache_keys = set()
        self.prefix_cache_keys = set()
        self.kv_cache_keys = set()
        self.decoder_state_cache_keys = set()
        self.context_window_cache_keys = set()

        logger.info(f"Unimodal Caching Manager initialized for {language_model_type} model")

    def cache_text_input(self, text: str, processed_tensor: torch.Tensor, priority: float = 0.5):
        """
        Cache processed text input for unimodal model.

        Args:
            text: Original text input
            processed_tensor: Processed tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        key = f"text_input_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=processed_tensor,
            entry_type=CacheEntryType.TEXT_INPUT,
            priority=priority,
            text_content=text
        )
        self.text_input_cache_keys.add(key)
        logger.debug(f"Cached text input: {key}")

    def cache_tokenized_input(self, text: str, tokenized_tensor: torch.Tensor, priority: float = 0.6):
        """
        Cache tokenized text input for unimodal model.

        Args:
            text: Original text input
            tokenized_tensor: Tokenized tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        key = f"tokenized_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=tokenized_tensor,
            entry_type=CacheEntryType.TOKENIZED_INPUT,
            priority=priority,
            text_content=text
        )
        self.tokenized_input_cache_keys.add(key)
        logger.debug(f"Cached tokenized input: {key}")

    def cache_embedding_output(self, text: str, embedding_tensor: torch.Tensor, priority: float = 0.7):
        """
        Cache embedding layer output for unimodal model.

        Args:
            text: Original text input
            embedding_tensor: Embedding tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        key = f"embedding_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=embedding_tensor,
            entry_type=CacheEntryType.EMBEDDING_OUTPUT,
            priority=priority,
            text_content=text
        )
        self.embedding_cache_keys.add(key)
        logger.debug(f"Cached embedding output: {key}")

    def cache_attention_output(self, text: str, layer_idx: int, attention_tensor: torch.Tensor, priority: float = 0.8):
        """
        Cache attention layer output for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the attention layer
            attention_tensor: Attention tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"attn_{text_hash}_layer{layer_idx}"
        self.cache.put(
            key=key,
            data=attention_tensor,
            entry_type=CacheEntryType.ATTENTION_OUTPUT,
            priority=priority,
            text_content=text
        )
        self.attention_cache_keys.add(key)
        logger.debug(f"Cached attention output for layer {layer_idx}: {key}")

    def cache_ffn_output(self, text: str, layer_idx: int, ffn_tensor: torch.Tensor, priority: float = 0.75):
        """
        Cache FFN layer output for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the FFN layer
            ffn_tensor: FFN tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"ffn_{text_hash}_layer{layer_idx}"
        self.cache.put(
            key=key,
            data=ffn_tensor,
            entry_type=CacheEntryType.FFN_OUTPUT,
            priority=priority,
            text_content=text
        )
        self.ffn_cache_keys.add(key)
        logger.debug(f"Cached FFN output for layer {layer_idx}: {key}")

    def cache_layer_output(self, text: str, layer_idx: int, layer_tensor: torch.Tensor, priority: float = 0.85):
        """
        Cache transformer layer output for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the transformer layer
            layer_tensor: Layer tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"layer_{text_hash}_layer{layer_idx}"
        self.cache.put(
            key=key,
            data=layer_tensor,
            entry_type=CacheEntryType.LAYER_OUTPUT,
            priority=priority,
            text_content=text
        )
        self.layer_cache_keys.add(key)
        logger.debug(f"Cached layer output for layer {layer_idx}: {key}")

    def cache_prefix(self, prefix_text: str, prefix_tensor: torch.Tensor, priority: float = 0.9):
        """
        Cache prefix for unimodal model.

        Args:
            prefix_text: Prefix text
            prefix_tensor: Prefix tensor representation
            priority: Priority for retention (0.0 to 1.0)
        """
        key = f"prefix_{hashlib.sha256(prefix_text.encode('utf-8')).hexdigest()[:16]}"
        self.cache.put(
            key=key,
            data=prefix_tensor,
            entry_type=CacheEntryType.PREFIX_CACHE,
            priority=priority,
            text_content=prefix_text
        )
        self.prefix_cache_keys.add(key)
        logger.debug(f"Cached prefix: {key}")

    def cache_kv_cache(self, text: str, layer_idx: int, kv_cache: torch.Tensor, priority: float = 0.95):
        """
        Cache KV cache for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the layer
            kv_cache: KV cache tensor
            priority: Priority for retention (0.0 to 1.0)
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"kv_cache_{text_hash}_layer{layer_idx}"
        self.cache.put(
            key=key,
            data=kv_cache,
            entry_type=CacheEntryType.KV_CACHE,
            priority=priority,
            text_content=text
        )
        self.kv_cache_keys.add(key)
        logger.debug(f"Cached KV cache for layer {layer_idx}: {key}")

    def get_cached_text_input(self, text: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached text input for unimodal model.

        Args:
            text: Original text input

        Returns:
            Cached tensor or None if not found
        """
        key = f"text_input_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_tokenized_input(self, text: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached tokenized input for unimodal model.

        Args:
            text: Original text input

        Returns:
            Cached tensor or None if not found
        """
        key = f"tokenized_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_embedding_output(self, text: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached embedding output for unimodal model.

        Args:
            text: Original text input

        Returns:
            Cached tensor or None if not found
        """
        key = f"embedding_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_attention_output(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve cached attention output for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the attention layer

        Returns:
            Cached tensor or None if not found
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"attn_{text_hash}_layer{layer_idx}"
        return self.cache.get(key)

    def get_cached_ffn_output(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve cached FFN output for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the FFN layer

        Returns:
            Cached tensor or None if not found
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"ffn_{text_hash}_layer{layer_idx}"
        return self.cache.get(key)

    def get_cached_layer_output(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve cached transformer layer output for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the transformer layer

        Returns:
            Cached tensor or None if not found
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"layer_{text_hash}_layer{layer_idx}"
        return self.cache.get(key)

    def get_cached_prefix(self, prefix_text: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached prefix for unimodal model.

        Args:
            prefix_text: Prefix text

        Returns:
            Cached tensor or None if not found
        """
        key = f"prefix_{hashlib.sha256(prefix_text.encode('utf-8')).hexdigest()[:16]}"
        return self.cache.get(key)

    def get_cached_kv_cache(self, text: str, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve cached KV cache for unimodal model.

        Args:
            text: Original text input
            layer_idx: Index of the layer

        Returns:
            Cached tensor or None if not found
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
        key = f"kv_cache_{text_hash}_layer{layer_idx}"
        return self.cache.get(key)

    def find_similar_text(self, text: str) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Find similar text in cache for unimodal model.

        Args:
            text: Text to compare

        Returns:
            Tuple of (key, cached_tensor) if similar text found, None otherwise
        """
        return self.cache.get_by_similarity(text, CacheEntryType.TEXT_INPUT, text_content=text)

    def find_similar_tokenized(self, text: str) -> Optional[Tuple[str, torch.Tensor]]:
        """
        Find similar tokenized text in cache for unimodal model.

        Args:
            text: Text to compare

        Returns:
            Tuple of (key, cached_tensor) if similar tokenized text found, None otherwise
        """
        return self.cache.get_by_similarity(text, CacheEntryType.TOKENIZED_INPUT, text_content=text)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for unimodal model.

        Returns:
            Dictionary containing cache statistics
        """
        base_stats = self.cache.get_cache_stats()

        # Add model-specific cache breakdown
        base_stats.update({
            "text_input_cache_entries": len(self.text_input_cache_keys),
            "tokenized_input_cache_entries": len(self.tokenized_input_cache_keys),
            "embedding_cache_entries": len(self.embedding_cache_keys),
            "attention_cache_entries": len(self.attention_cache_keys),
            "ffn_cache_entries": len(self.ffn_cache_keys),
            "layer_cache_entries": len(self.layer_cache_keys),
            "prefix_cache_entries": len(self.prefix_cache_keys),
            "kv_cache_entries": len(self.kv_cache_keys),
            "decoder_state_cache_entries": len(self.decoder_state_cache_keys),
            "context_window_cache_entries": len(self.context_window_cache_keys),
        })

        return base_stats

    def clear_cache(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.text_input_cache_keys.clear()
        self.tokenized_input_cache_keys.clear()
        self.embedding_cache_keys.clear()
        self.attention_cache_keys.clear()
        self.ffn_cache_keys.clear()
        self.layer_cache_keys.clear()
        self.prefix_cache_keys.clear()
        self.kv_cache_keys.clear()
        self.decoder_state_cache_keys.clear()
        self.context_window_cache_keys.clear()
        logger.info("Unimodal cache cleared")


def create_unimodal_caching_manager(cache_size_mb: float = 512.0, 
                                  language_model_type: str = "general") -> UnimodalCachingManager:
    """
    Factory function to create a unimodal caching manager.

    Args:
        cache_size_mb: Cache size in megabytes
        language_model_type: Type of language model (affects optimizations)

    Returns:
        Unimodal caching manager
    """
    return UnimodalCachingManager(cache_size_mb=cache_size_mb, language_model_type=language_model_type)


def apply_intelligent_unimodal_caching_to_model(model: nn.Module,
                                              caching_manager: UnimodalCachingManager) -> nn.Module:
    """
    Apply intelligent unimodal caching to a text model.

    Args:
        model: The text model to optimize
        caching_manager: The caching manager to use

    Returns:
        Model with caching capabilities
    """
    logger.info(f"Applying intelligent unimodal caching to {caching_manager.cache.language_model_type} model...")

    # Add caching manager to model
    model._caching_manager = caching_manager

    # Add caching methods to model if they don't exist
    if not hasattr(model, 'cache_text_input'):
        model.cache_text_input = caching_manager.cache_text_input

    if not hasattr(model, 'cache_tokenized_input'):
        model.cache_tokenized_input = caching_manager.cache_tokenized_input

    if not hasattr(model, 'cache_embedding_output'):
        model.cache_embedding_output = caching_manager.cache_embedding_output

    if not hasattr(model, 'cache_attention_output'):
        model.cache_attention_output = caching_manager.cache_attention_output

    if not hasattr(model, 'cache_ffn_output'):
        model.cache_ffn_output = caching_manager.cache_ffn_output

    if not hasattr(model, 'cache_layer_output'):
        model.cache_layer_output = caching_manager.cache_layer_output

    if not hasattr(model, 'cache_prefix'):
        model.cache_prefix = caching_manager.cache_prefix

    if not hasattr(model, 'cache_kv_cache'):
        model.cache_kv_cache = caching_manager.cache_kv_cache

    if not hasattr(model, 'get_cached_text_input'):
        model.get_cached_text_input = caching_manager.get_cached_text_input

    if not hasattr(model, 'get_cached_tokenized_input'):
        model.get_cached_tokenized_input = caching_manager.get_cached_tokenized_input

    if not hasattr(model, 'get_cached_embedding_output'):
        model.get_cached_embedding_output = caching_manager.get_cached_embedding_output

    if not hasattr(model, 'get_cached_attention_output'):
        model.get_cached_attention_output = caching_manager.get_cached_attention_output

    if not hasattr(model, 'get_cached_ffn_output'):
        model.get_cached_ffn_output = caching_manager.get_cached_ffn_output

    if not hasattr(model, 'get_cached_layer_output'):
        model.get_cached_layer_output = caching_manager.get_cached_layer_output

    if not hasattr(model, 'get_cached_prefix'):
        model.get_cached_prefix = caching_manager.get_cached_prefix

    if not hasattr(model, 'get_cached_kv_cache'):
        model.get_cached_kv_cache = caching_manager.get_cached_kv_cache

    if not hasattr(model, 'find_similar_text'):
        model.find_similar_text = caching_manager.find_similar_text

    if not hasattr(model, 'find_similar_tokenized'):
        model.find_similar_tokenized = caching_manager.find_similar_tokenized

    logger.info("Intelligent unimodal caching applied successfully")
    return model


__all__ = [
    "CacheEvictionPolicy",
    "CacheEntryType",
    "CacheEntry",
    "IntelligentUnimodalCache",
    "UnimodalCachingManager",
    "create_unimodal_caching_manager",
    "apply_intelligent_unimodal_caching_to_model"
]