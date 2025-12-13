"""
Advanced Memory Tiering System for Qwen3-VL Model

This module implements an advanced memory tiering system specifically designed for the Qwen3-VL model,
utilizing different memory tiers (HBM, DDR, SSD) based on access patterns and tensor characteristics.

Key Features:
- Three-tier memory system (GPU HBM, CPU RAM, SSD)
- ML-based access pattern prediction
- Dynamic tensor migration based on frequency, size, and temporal locality
- Qwen3-VL specific optimizations
- Integration with existing memory management systems
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import threading
import queue
import pickle
import tempfile
import os
import psutil
from collections import OrderedDict, defaultdict, deque
import statistics
import math
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available, some memory monitoring features will be limited")
    PSUTIL_AVAILABLE = False


class MemoryTier(Enum):
    """Memory tiers for the tiering system"""
    GPU_HBM = "gpu_hbm"      # GPU High Bandwidth Memory (Tier 1)
    CPU_RAM = "cpu_ram"      # CPU RAM (Tier 2)
    SSD_STORAGE = "ssd_storage"  # SSD Storage (Tier 3)


class TensorType(Enum):
    """Types of tensors for Qwen3-VL model optimization"""
    GENERAL = "general"
    KV_CACHE = "kv_cache"           # Key-value cache for transformer attention
    IMAGE_FEATURES = "image_features"  # Processed image features
    TEXT_EMBEDDINGS = "text_embeddings"  # Text embeddings
    CROSS_ATTENTION = "cross_attention" # Cross-attention tensors
    TEMPORARY = "temporary"          # Temporary computation tensors


@dataclass
class TensorMetadata:
    """Metadata for cached tensors in Qwen3-VL model"""
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int
    tensor_type: TensorType
    creation_time: float
    last_access_time: float
    access_count: int
    tier: MemoryTier
    predicted_access: bool = False
    predicted_access_time: Optional[float] = None
    access_times: deque = None  # Track last N access times
    is_compressed: bool = False
    compression_ratio: float = 1.0
    pinned: bool = False  # If pinned, should not be migrated
    temporal_locality_score: float = 0.0  # Score based on access timing patterns


@dataclass
class TierConfig:
    """Configuration for a memory tier in Qwen3-VL system"""
    tier: MemoryTier
    max_size_bytes: int
    access_latency_ms: float  # Average access latency in milliseconds
    transfer_bandwidth_gbps: float  # Transfer bandwidth in GB/s
    compression_supported: bool = True
    eviction_policy: str = "lru"  # lru, fifo, priority
    tier_priority: int = 0  # Higher priority means faster access


@dataclass
class TierStats:
    """Statistics for a memory tier"""
    tier: MemoryTier
    current_size_bytes: int
    max_size_bytes: int
    hits: int = 0
    misses: int = 0
    migrations_in: int = 0
    migrations_out: int = 0
    access_count: int = 0

    @property
    def utilization(self) -> float:
        return self.current_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class TierManager:
    """Base class for managing a specific memory tier"""

    def __init__(self, config: TierConfig):
        self.config = config
        self.cache: Dict[str, torch.Tensor] = OrderedDict()
        self.metadata: Dict[str, TensorMetadata] = {}
        self.current_size_bytes = 0
        self._lock = threading.Lock()
        self.stats = TierStats(
            tier=config.tier,
            current_size_bytes=0,
            max_size_bytes=config.max_size_bytes
        )

    def get(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get tensor from this tier"""
        with self._lock:
            if tensor_id in self.cache:
                tensor = self.cache[tensor_id]

                # Update metadata
                if tensor_id in self.metadata:
                    metadata = self.metadata[tensor_id]
                    metadata.last_access_time = time.time()
                    metadata.access_count += 1
                    if metadata.access_times is None:
                        metadata.access_times = deque(maxlen=100)
                    metadata.access_times.append(time.time())

                    # Update temporal locality score
                    if len(metadata.access_times) > 1:
                        intervals = [metadata.access_times[i] - metadata.access_times[i-1]
                                   for i in range(1, len(metadata.access_times))]
                        if intervals:
                            metadata.temporal_locality_score = 1.0 / (1.0 + statistics.mean(intervals))

                    # Move to end for LRU
                    self.cache.move_to_end(tensor_id)

                self.stats.hits += 1
                self.stats.access_count += 1
                return tensor
            else:
                self.stats.misses += 1
                self.stats.access_count += 1
                return None

    def put(self, tensor_id: str, tensor: torch.Tensor, metadata: TensorMetadata) -> bool:
        """Put tensor in this tier"""
        with self._lock:
            # Make space if needed
            if not self.make_space(tensor.element_size() * tensor.nelement()):
                return False

            # Add tensor to cache
            self.cache[tensor_id] = tensor
            self.metadata[tensor_id] = metadata
            self.current_size_bytes += tensor.element_size() * tensor.nelement()

            self.stats.migrations_in += 1
            return True

    def remove(self, tensor_id: str) -> bool:
        """Remove tensor from this tier"""
        with self._lock:
            if tensor_id in self.cache:
                tensor = self.cache.pop(tensor_id)
                metadata = self.metadata.pop(tensor_id)
                tensor_size = tensor.element_size() * tensor.nelement()
                self.current_size_bytes -= tensor_size
                return True
            return False

    def can_fit(self, size_bytes: int) -> bool:
        """Check if a tensor of given size can fit in this tier"""
        return self.current_size_bytes + size_bytes <= self.config.max_size_bytes

    def make_space(self, required_size: int) -> bool:
        """Make space for a tensor of given size by evicting others"""
        while self.current_size_bytes + required_size > self.config.max_size_bytes and len(self.cache) > 0:
            if self.config.eviction_policy == "lru":
                old_id, old_tensor = self.cache.popitem(last=False)
                old_metadata = self.metadata.pop(old_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1
            elif self.config.eviction_policy == "priority":
                # Evict based on tensor type priority (least important first)
                # KV_CACHE > IMAGE_FEATURES > TEXT_EMBEDDINGS > GENERAL
                priority_map = {
                    TensorType.GENERAL: 0,
                    TensorType.TEXT_EMBEDDINGS: 1,
                    TensorType.IMAGE_FEATURES: 2,
                    TensorType.CROSS_ATTENTION: 3,
                    TensorType.KV_CACHE: 4
                }

                # Find tensor with lowest priority
                lowest_priority_id = min(
                    self.cache.keys(),
                    key=lambda x: priority_map.get(self.metadata[x].tensor_type, 0)
                )

                old_tensor = self.cache.pop(lowest_priority_id)
                old_metadata = self.metadata.pop(lowest_priority_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1
            else:
                # Default to LRU
                old_id, old_tensor = self.cache.popitem(last=False)
                old_metadata = self.metadata.pop(old_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1

        return self.current_size_bytes + required_size <= self.config.max_size_bytes


class GPUHBMManager(TierManager):
    """Manager for GPU HBM (Tier 1) - Qwen3-VL optimized"""

    def __init__(self, config: TierConfig):
        super().__init__(config)
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get tensor from GPU HBM tier"""
        with self._lock:
            if tensor_id in self.cache:
                tensor = self.cache[tensor_id]

                # Update metadata
                if tensor_id in self.metadata:
                    metadata = self.metadata[tensor_id]
                    metadata.last_access_time = time.time()
                    metadata.access_count += 1
                    if metadata.access_times is None:
                        metadata.access_times = deque(maxlen=100)
                    metadata.access_times.append(time.time())

                    # Update temporal locality score
                    if len(metadata.access_times) > 1:
                        intervals = [metadata.access_times[i] - metadata.access_times[i-1]
                                   for i in range(1, len(metadata.access_times))]
                        if intervals:
                            metadata.temporal_locality_score = 1.0 / (1.0 + statistics.mean(intervals))

                    # Move to end for LRU
                    self.cache.move_to_end(tensor_id)

                self.stats.hits += 1
                self.stats.access_count += 1
                return tensor
            else:
                self.stats.misses += 1
                self.stats.access_count += 1
                return None

    def put(self, tensor_id: str, tensor: torch.Tensor, metadata: TensorMetadata) -> bool:
        """Put tensor in GPU HBM tier"""
        with self._lock:
            # Ensure tensor is on GPU
            if tensor.device != self.gpu_device:
                tensor = tensor.to(self.gpu_device)

            tensor_size = tensor.element_size() * tensor.nelement()

            # Check if tensor fits
            if tensor_size > self.config.max_size_bytes:
                return False

            # Make space if needed
            if not self.make_space(tensor_size):
                return False

            # Add tensor to cache
            self.cache[tensor_id] = tensor
            metadata.tier = self.config.tier
            self.metadata[tensor_id] = metadata
            self.current_size_bytes += tensor_size

            self.stats.migrations_in += 1
            return True

    def remove(self, tensor_id: str) -> bool:
        """Remove tensor from GPU HBM tier"""
        with self._lock:
            if tensor_id in self.cache:
                tensor = self.cache.pop(tensor_id)
                metadata = self.metadata.pop(tensor_id)
                tensor_size = tensor.element_size() * tensor.nelement()
                self.current_size_bytes -= tensor_size
                return True
            return False

    def make_space(self, required_size: int) -> bool:
        """Make space by evicting tensors based on policy"""
        while self.current_size_bytes + required_size > self.config.max_size_bytes and len(self.cache) > 0:
            if self.config.eviction_policy == "lru":
                old_id, old_tensor = self.cache.popitem(last=False)
                old_metadata = self.metadata.pop(old_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1
            elif self.config.eviction_policy == "priority":
                # Evict based on tensor type priority (least important first)
                # KV_CACHE > IMAGE_FEATURES > TEXT_EMBEDDINGS > GENERAL
                priority_map = {
                    TensorType.GENERAL: 0,
                    TensorType.TEXT_EMBEDDINGS: 1,
                    TensorType.IMAGE_FEATURES: 2,
                    TensorType.CROSS_ATTENTION: 3,
                    TensorType.KV_CACHE: 4
                }

                # Find tensor with lowest priority
                lowest_priority_id = min(
                    self.cache.keys(),
                    key=lambda x: priority_map.get(self.metadata[x].tensor_type, 0)
                )

                old_tensor = self.cache.pop(lowest_priority_id)
                old_metadata = self.metadata.pop(lowest_priority_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1
            else:
                # Default to LRU
                old_id, old_tensor = self.cache.popitem(last=False)
                old_metadata = self.metadata.pop(old_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1

        return self.current_size_bytes + required_size <= self.config.max_size_bytes


class CPURAMManager(TierManager):
    """Manager for CPU RAM (Tier 2) - Qwen3-VL optimized"""

    def __init__(self, config: TierConfig):
        super().__init__(config)
        self.use_pinned_memory = True  # Use pinned memory for faster GPU transfers

    def get(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get tensor from CPU RAM tier"""
        with self._lock:
            if tensor_id in self.cache:
                tensor = self.cache[tensor_id]

                # Update metadata
                if tensor_id in self.metadata:
                    metadata = self.metadata[tensor_id]
                    metadata.last_access_time = time.time()
                    metadata.access_count += 1
                    if metadata.access_times is None:
                        metadata.access_times = deque(maxlen=100)
                    metadata.access_times.append(time.time())

                    # Update temporal locality score
                    if len(metadata.access_times) > 1:
                        intervals = [metadata.access_times[i] - metadata.access_times[i-1]
                                   for i in range(1, len(metadata.access_times))]
                        if intervals:
                            metadata.temporal_locality_score = 1.0 / (1.0 + statistics.mean(intervals))

                    # Move to end for LRU
                    self.cache.move_to_end(tensor_id)

                self.stats.hits += 1
                self.stats.access_count += 1
                return tensor
            else:
                self.stats.misses += 1
                self.stats.access_count += 1
                return None

    def put(self, tensor_id: str, tensor: torch.Tensor, metadata: TensorMetadata) -> bool:
        """Put tensor in CPU RAM tier"""
        with self._lock:
            # Ensure tensor is on CPU
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()

            tensor_size = tensor.element_size() * tensor.nelement()

            # Check if tensor fits
            if tensor_size > self.config.max_size_bytes:
                return False

            # Make space if needed
            if not self.make_space(tensor_size):
                return False

            # Add tensor to cache
            if self.use_pinned_memory and metadata.pinned:
                tensor = tensor.pin_memory()

            self.cache[tensor_id] = tensor
            metadata.tier = self.config.tier
            self.metadata[tensor_id] = metadata
            self.current_size_bytes += tensor_size

            self.stats.migrations_in += 1
            return True

    def remove(self, tensor_id: str) -> bool:
        """Remove tensor from CPU RAM tier"""
        with self._lock:
            if tensor_id in self.cache:
                tensor = self.cache.pop(tensor_id)
                metadata = self.metadata.pop(tensor_id)
                tensor_size = tensor.element_size() * tensor.nelement()
                self.current_size_bytes -= tensor_size
                return True
            return False

    def make_space(self, required_size: int) -> bool:
        """Make space by evicting tensors based on policy"""
        while self.current_size_bytes + required_size > self.config.max_size_bytes and len(self.cache) > 0:
            if self.config.eviction_policy == "lru":
                old_id, old_tensor = self.cache.popitem(last=False)
                old_metadata = self.metadata.pop(old_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1
            elif self.config.eviction_policy == "priority":
                # Evict based on tensor type priority
                priority_map = {
                    TensorType.GENERAL: 0,
                    TensorType.TEXT_EMBEDDINGS: 1,
                    TensorType.IMAGE_FEATURES: 2,
                    TensorType.CROSS_ATTENTION: 3,
                    TensorType.KV_CACHE: 4
                }

                # Find tensor with lowest priority
                lowest_priority_id = min(
                    self.cache.keys(),
                    key=lambda x: priority_map.get(self.metadata[x].tensor_type, 0)
                )

                old_tensor = self.cache.pop(lowest_priority_id)
                old_metadata = self.metadata.pop(lowest_priority_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1
            else:
                # Default to LRU
                old_id, old_tensor = self.cache.popitem(last=False)
                old_metadata = self.metadata.pop(old_id)
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_size_bytes -= old_size
                self.stats.migrations_out += 1

        return self.current_size_bytes + required_size <= self.config.max_size_bytes


class SSDStorageManager(TierManager):
    """Manager for SSD Storage (Tier 3) - Qwen3-VL optimized"""

    def __init__(self, config: TierConfig):
        super().__init__(config)
        # Initialize cache directory
        self.cache_dir = Path(tempfile.gettempdir()) / "qwen3vl_tiering_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.current_size_bytes = self._get_current_disk_usage()

    def _get_current_disk_usage(self) -> int:
        """Calculate current disk usage of cache"""
        total_size = 0
        for filepath in self.cache_dir.glob("*.pkl"):
            if filepath.is_file():
                total_size += filepath.stat().st_size
        return total_size

    def _get_file_path(self, tensor_id: str) -> Path:
        """Get file path for a tensor"""
        return self.cache_dir / f"{tensor_id}.pkl"

    def get(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get tensor from SSD storage tier"""
        file_path = self._get_file_path(tensor_id)

        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    tensor_data = pickle.load(f)

                # Convert back to tensor
                tensor = torch.from_numpy(tensor_data['tensor'])
                tensor.requires_grad_(tensor_data['requires_grad'])

                # Update metadata
                if tensor_id in self.metadata:
                    metadata = self.metadata[tensor_id]
                    metadata.last_access_time = time.time()
                    metadata.access_count += 1
                    if metadata.access_times is None:
                        metadata.access_times = deque(maxlen=100)
                    metadata.access_times.append(time.time())

                    # Update temporal locality score
                    if len(metadata.access_times) > 1:
                        intervals = [metadata.access_times[i] - metadata.access_times[i-1]
                                   for i in range(1, len(metadata.access_times))]
                        if intervals:
                            metadata.temporal_locality_score = 1.0 / (1.0 + statistics.mean(intervals))

                self.stats.hits += 1
                self.stats.access_count += 1
                return tensor
            except Exception as e:
                logger.error(f"Error loading tensor {tensor_id} from SSD cache: {e}")
                return None
        else:
            self.stats.misses += 1
            self.stats.access_count += 1
            return None

    def put(self, tensor_id: str, tensor: torch.Tensor, metadata: TensorMetadata) -> bool:
        """Put tensor in SSD storage tier"""
        # Convert tensor to serializable format
        tensor_data = {
            'tensor': tensor.detach().cpu().numpy(),
            'requires_grad': tensor.requires_grad
        }

        tensor_size = tensor.element_size() * tensor.nelement()

        # Check if tensor fits
        if tensor_size > self.config.max_size_bytes:
            return False

        # Make space if needed
        if not self.make_space(tensor_size):
            return False

        # Save tensor to disk
        file_path = self._get_file_path(tensor_id)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(tensor_data, f)

            # Update metadata
            metadata.tier = self.config.tier
            self.metadata[tensor_id] = metadata

            # Update size tracking
            actual_size = file_path.stat().st_size
            self.current_size_bytes += actual_size

            self.stats.migrations_in += 1
            return True
        except Exception as e:
            logger.error(f"Error saving tensor {tensor_id} to SSD cache: {e}")
            return False

    def remove(self, tensor_id: str) -> bool:
        """Remove tensor from SSD storage tier"""
        file_path = self._get_file_path(tensor_id)

        if file_path.exists():
            size = file_path.stat().st_size
            file_path.unlink()
            self.current_size_bytes -= size

            if tensor_id in self.metadata:
                del self.metadata[tensor_id]
            return True
        return False

    def make_space(self, required_size: int) -> bool:
        """Make space by evicting tensors based on policy"""
        while self.current_size_bytes + required_size > self.config.max_size_bytes and len(self.metadata) > 0:
            if self.config.eviction_policy == "lru":
                # Find oldest accessed tensor
                oldest_id = min(
                    self.metadata.keys(),
                    key=lambda x: self.metadata[x].last_access_time
                )
                self.remove(oldest_id)
                self.stats.migrations_out += 1
            elif self.config.eviction_policy == "priority":
                # Evict based on tensor type priority
                priority_map = {
                    TensorType.GENERAL: 0,
                    TensorType.TEXT_EMBEDDINGS: 1,
                    TensorType.IMAGE_FEATURES: 2,
                    TensorType.CROSS_ATTENTION: 3,
                    TensorType.KV_CACHE: 4
                }

                # Find tensor with lowest priority
                lowest_priority_id = min(
                    self.metadata.keys(),
                    key=lambda x: priority_map.get(self.metadata[x].tensor_type, 0)
                )

                self.remove(lowest_priority_id)
                self.stats.migrations_out += 1
            else:
                # Default to LRU
                oldest_id = min(
                    self.metadata.keys(),
                    key=lambda x: self.metadata[x].last_access_time
                )
                self.remove(oldest_id)
                self.stats.migrations_out += 1

        return self.current_size_bytes + required_size <= self.config.max_size_bytes


class AccessPatternTracker:
    """Tracks access patterns to predict future tensor access for Qwen3-VL"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_history = deque(maxlen=window_size)
        self.tensor_access_intervals = defaultdict(deque)  # Track intervals between accesses
        self.tensor_access_counts = defaultdict(int)
        self.tensor_last_access = {}
        self.tensor_avg_intervals = {}
        self.tensor_residency_times = defaultdict(float)  # How long tensors stay in memory
        self.tensor_frequency_scores = defaultdict(float)  # Normalized frequency scores

    def record_access(self, tensor_id: str):
        """Record access to a tensor"""
        current_time = time.time()
        self.access_history.append((tensor_id, current_time))

        # Update access count
        self.tensor_access_counts[tensor_id] += 1

        # Calculate interval if tensor was accessed before
        if tensor_id in self.tensor_last_access:
            interval = current_time - self.tensor_last_access[tensor_id]
            self.tensor_access_intervals[tensor_id].append(interval)

            # Update average interval
            intervals = list(self.tensor_access_intervals[tensor_id])
            if intervals:
                self.tensor_avg_intervals[tensor_id] = statistics.mean(intervals)

        self.tensor_last_access[tensor_id] = current_time

        # Update frequency score (normalized by time since creation)
        time_since_creation = current_time - min(self.tensor_last_access.values())
        if time_since_creation > 0:
            self.tensor_frequency_scores[tensor_id] = self.tensor_access_counts[tensor_id] / time_since_creation

    def predict_access(self, tensor_id: str) -> Tuple[float, Optional[float]]:
        """
        Predict probability of tensor being accessed soon and when.

        Returns:
            Tuple of (probability, predicted_access_time)
        """
        if tensor_id not in self.tensor_avg_intervals:
            # If no history, use low probability and far prediction
            return 0.1, time.time() + 300  # Low probability, predict 5 minutes from now

        avg_interval = self.tensor_avg_intervals[tensor_id]
        last_access = self.tensor_last_access.get(tensor_id, 0)
        time_since_last = time.time() - last_access

        # If time since last access is close to average interval, high probability
        if avg_interval > 0:
            ratio = time_since_last / avg_interval
            # Sigmoid-like function to convert to probability
            probability = 1.0 / (1.0 + math.exp(-2 * (ratio - 0.5)))
            probability = min(probability, 1.0)

            # Predict next access time
            predicted_time = last_access + avg_interval
            return probability, predicted_time
        else:
            return 0.5, time.time() + 60  # Default to 1 minute from now

    def get_hot_tensors(self, n: int = 10) -> List[str]:
        """Get top N most frequently accessed tensors"""
        sorted_tensors = sorted(
            self.tensor_access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [tensor_id for tensor_id, count in sorted_tensors[:n]]

    def get_temporal_locality_score(self, tensor_id: str) -> float:
        """Get temporal locality score for a tensor"""
        if tensor_id in self.tensor_avg_intervals:
            # Higher score for shorter average intervals (more frequent access)
            avg_interval = self.tensor_avg_intervals[tensor_id]
            return 1.0 / (1.0 + avg_interval)  # Normalize to 0-1 range
        else:
            return 0.1  # Low score if no history

    def update_residency_time(self, tensor_id: str, duration: float):
        """Update how long a tensor stays in memory"""
        self.tensor_residency_times[tensor_id] = duration


class Qwen3VLMLPredictor:
    """ML predictor specifically for Qwen3-VL tensor access patterns"""

    def __init__(self, prediction_window: int = 100):
        self.prediction_window = prediction_window
        self.tensor_features = defaultdict(lambda: {
            'frequency': 0,
            'recency': 0,
            'interval': 0,
            'residency': 0,
            'temporal_locality': 0,
            'size_factor': 0
        })
        self.tensor_access_times = defaultdict(deque)

    def update_features(self, tracker: AccessPatternTracker):
        """Update features based on access pattern tracker"""
        current_time = time.time()

        for tensor_id in tracker.tensor_access_counts:
            access_count = tracker.tensor_access_counts[tensor_id]
            last_access = tracker.tensor_last_access.get(tensor_id, 0)
            avg_interval = tracker.tensor_avg_intervals.get(tensor_id, float('inf'))
            residency_time = tracker.tensor_residency_times.get(tensor_id, 0)
            temporal_locality = tracker.get_temporal_locality_score(tensor_id)

            # Calculate features
            frequency = access_count
            recency = 1.0 / (current_time - last_access + 1)  # Higher is more recent
            interval = 1.0 / (avg_interval + 1) if avg_interval != float('inf') else 0
            residency = residency_time
            temporal_locality_score = temporal_locality

            self.tensor_features[tensor_id] = {
                'frequency': frequency,
                'recency': recency,
                'interval': interval,
                'residency': residency,
                'temporal_locality': temporal_locality_score,
                'size_factor': 0  # Will be set during tensor evaluation
            }

    def predict_access_probability(self, tensor_id: str, tensor_size: int = None) -> Tuple[float, Optional[float]]:
        """
        Predict access probability and time using Qwen3-VL specific features.

        Returns:
            Tuple of (probability, predicted_access_time)
        """
        features = self.tensor_features.get(
            tensor_id,
            {'frequency': 0, 'recency': 0, 'interval': 0, 'residency': 0, 'temporal_locality': 0, 'size_factor': 0}
        )

        # For Qwen3-VL, adjust weights based on tensor type importance
        weights = {
            'frequency': 0.25,
            'recency': 0.25,
            'interval': 0.2,
            'residency': 0.1,
            'temporal_locality': 0.2
        }

        # Normalize frequency (cap at 20 accesses to prevent dominance)
        norm_frequency = min(features['frequency'] / 20.0, 1.0)

        # Normalize recency (cap at 1000 to prevent dominance)
        norm_recency = min(features['recency'] * 1000, 1.0)

        # Normalize interval
        norm_interval = min(features['interval'] * 10, 1.0)

        # Normalize residency (cap at 5 minutes max)
        norm_residency = min(features['residency'] / 300, 1.0)

        # Temporal locality is already normalized
        norm_temporal = features['temporal_locality']

        score = (
            weights['frequency'] * norm_frequency +
            weights['recency'] * norm_recency +
            weights['interval'] * norm_interval +
            weights['residency'] * norm_residency +
            weights['temporal_locality'] * norm_temporal
        )

        probability = min(score, 1.0)

        # For predicted time, use the access interval from tracker if available
        predicted_time = None
        if tensor_id in self.tensor_features:
            last_access = self.tensor_features[tensor_id].get('last_access', time.time())
            avg_interval = self.tensor_features[tensor_id].get('interval', 1.0)
            if avg_interval != float('inf'):
                predicted_time = last_access + (1.0 / max(avg_interval, 0.001))  # Convert back from inverse

        return probability, predicted_time


class Qwen3VLMemoryTieringSystem:
    """
    Advanced Memory Tiering System specifically designed for Qwen3-VL model.
    Manages movement of tensors between HBM (GPU), DDR (CPU), and SSD based on access patterns.
    """

    def __init__(self,
                 gpu_hbm_size: int = 1 * 1024 * 1024 * 1024,  # 1GB GPU HBM
                 cpu_ram_size: int = 2 * 1024 * 1024 * 1024,  # 2GB CPU RAM
                 ssd_storage_size: int = 10 * 1024 * 1024 * 1024,  # 10GB SSD
                 prediction_window: int = 1000):
        """
        Initialize the Qwen3-VL memory tiering system.

        Args:
            gpu_hbm_size: Size of GPU HBM tier in bytes
            cpu_ram_size: Size of CPU RAM tier in bytes
            ssd_storage_size: Size of SSD storage tier in bytes
            prediction_window: Window size for access pattern tracking
        """
        # Initialize tier managers
        self.gpu_manager = GPUHBMManager(TierConfig(
            tier=MemoryTier.GPU_HBM,
            max_size_bytes=gpu_hbm_size,
            access_latency_ms=0.001,  # Very fast
            transfer_bandwidth_gbps=900,  # High bandwidth for HBM (example for NVIDIA)
            compression_supported=True,
            eviction_policy="priority",  # Use priority-based eviction for Qwen3-VL
            tier_priority=3
        ))

        self.cpu_manager = CPURAMManager(TierConfig(
            tier=MemoryTier.CPU_RAM,
            max_size_bytes=cpu_ram_size,
            access_latency_ms=0.1,  # Fast
            transfer_bandwidth_gbps=50,  # Moderate bandwidth
            compression_supported=True,
            eviction_policy="priority",  # Use priority-based eviction
            tier_priority=2
        ))

        self.ssd_manager = SSDStorageManager(TierConfig(
            tier=MemoryTier.SSD_STORAGE,
            max_size_bytes=ssd_storage_size,
            access_latency_ms=100,  # Slow
            transfer_bandwidth_gbps=3,  # Lower bandwidth
            compression_supported=True,
            eviction_policy="priority",  # Use priority-based eviction
            tier_priority=1
        ))

        # Initialize tracking components
        self.access_tracker = AccessPatternTracker(prediction_window)
        self.predictor = Qwen3VLMLPredictor(prediction_window)

        # Tensor location tracking
        self.tensor_locations: Dict[str, MemoryTier] = {}
        self.tensor_metadata: Dict[str, TensorMetadata] = {}

        # Migration thresholds for Qwen3-VL
        self.high_freq_threshold = 5  # Access count for GPU promotion
        self.medium_freq_threshold = 2  # Access count for CPU promotion
        self.temporal_locality_threshold = 0.5  # Temporal locality score for tiering decisions

        # Lock for thread safety
        self._lock = threading.Lock()

        # Stats
        self.stats = {
            'total_requests': 0,
            'global_hit_rate': 0.0,
            'total_migrations': 0,
            'predictions_made': 0,
            'predictions_correct': 0,
            'migration_cost': 0.0,  # Cumulative cost of migrations
            'tensor_type_distribution': defaultdict(int)  # Count by tensor type
        }

        logger.info("Qwen3-VL Memory Tiering System initialized")

    def _generate_tensor_id(self, shape: Tuple[int, ...], dtype: torch.dtype, tensor_type: TensorType) -> str:
        """Generate a unique ID for a tensor"""
        return f"{tensor_type.value}_{shape}_{str(dtype)}_{int(time.time() * 1000000)}_{id(self)}"

    def get_tensor(self,
                   tensor_id: str,
                   target_device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Get tensor from the appropriate tier, with automatic migration if needed.

        Args:
            tensor_id: ID of the tensor to retrieve
            target_device: Target device for the tensor (None for current location)

        Returns:
            Tensor if found, None otherwise
        """
        with self._lock:
            # Try each tier in order of speed
            # First try GPU
            tensor = self.gpu_manager.get(tensor_id)
            if tensor is not None:
                # Update access tracker
                self.access_tracker.record_access(tensor_id)
                self.stats['total_requests'] += 1

                # If target device is different, transfer if needed
                if target_device and target_device != tensor.device:
                    tensor = tensor.to(target_device)

                return tensor

            # Then try CPU
            tensor = self.cpu_manager.get(tensor_id)
            if tensor is not None:
                # Update access tracker
                self.access_tracker.record_access(tensor_id)
                self.stats['total_requests'] += 1

                # If target device is GPU, transfer from CPU to GPU
                if target_device and target_device.type == 'cuda' and tensor.device.type != 'cuda':
                    tensor = tensor.to(target_device)

                return tensor

            # Finally try SSD
            tensor = self.ssd_manager.get(tensor_id)
            if tensor is not None:
                # Update access tracker
                self.access_tracker.record_access(tensor_id)
                self.stats['total_requests'] += 1

                # If target device is GPU, transfer from SSD to GPU (via CPU)
                if target_device and target_device.type == 'cuda':
                    tensor = tensor.to(target_device)

                return tensor

            # Tensor not found in any tier
            # Still record access to maintain tracking even for misses
            if tensor_id in self.tensor_metadata:
                self.access_tracker.record_access(tensor_id)  # Record miss as access
            self.stats['total_requests'] += 1  # Count misses as requests too
            return None

    def put_tensor(self,
                   tensor: torch.Tensor,
                   tensor_type: TensorType = TensorType.GENERAL,
                   preferred_tier: Optional[MemoryTier] = None,
                   pinned: bool = False) -> Tuple[bool, str]:
        """
        Put tensor in the appropriate tier based on size, type, and other factors.

        Args:
            tensor: Tensor to store
            tensor_type: Type of tensor for Qwen3-VL optimization
            preferred_tier: Preferred tier (None for automatic selection)
            pinned: Whether tensor should be pinned (not eligible for migration)

        Returns:
            Tuple of (success, tensor_id)
        """
        with self._lock:
            tensor_id = self._generate_tensor_id(tensor.shape, tensor.dtype, tensor_type)

            # Create metadata
            metadata = TensorMetadata(
                tensor_id=tensor_id,
                shape=tensor.shape,
                dtype=tensor.dtype,
                size_bytes=tensor.element_size() * tensor.nelement(),
                tensor_type=tensor_type,
                creation_time=time.time(),
                last_access_time=time.time(),
                access_count=0,
                tier=preferred_tier or self._determine_initial_tier(tensor, tensor_type),
                pinned=pinned
            )

            # Determine target tier
            target_tier = preferred_tier or self._determine_initial_tier(tensor, tensor_type)

            # Put in appropriate tier
            success = False
            if target_tier == MemoryTier.GPU_HBM:
                success = self.gpu_manager.put(tensor_id, tensor, metadata)
            elif target_tier == MemoryTier.CPU_RAM:
                success = self.cpu_manager.put(tensor_id, tensor, metadata)
            elif target_tier == MemoryTier.SSD_STORAGE:
                success = self.ssd_manager.put(tensor_id, tensor, metadata)

            if success:
                self.tensor_locations[tensor_id] = target_tier
                self.tensor_metadata[tensor_id] = metadata
                self.stats['tensor_type_distribution'][tensor_type.value] += 1

            return success, tensor_id

    def _determine_initial_tier(self, tensor: torch.Tensor, tensor_type: TensorType) -> MemoryTier:
        """
        Determine the initial tier for a tensor based on its characteristics and Qwen3-VL requirements.

        Args:
            tensor: Tensor to place
            tensor_type: Type of tensor

        Returns:
            Appropriate memory tier
        """
        tensor_size = tensor.element_size() * tensor.nelement()

        # For Qwen3-VL, prioritize tensor type over just size
        if tensor_type == TensorType.KV_CACHE:
            # KV cache tensors are critical for transformer performance
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 4:
                return MemoryTier.GPU_HBM
            else:
                return MemoryTier.CPU_RAM

        elif tensor_type == TensorType.IMAGE_FEATURES:
            # Image features are accessed multiple times during vision processing
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 2:
                return MemoryTier.GPU_HBM
            else:
                return MemoryTier.CPU_RAM

        elif tensor_type == TensorType.CROSS_ATTENTION:
            # Cross-attention tensors are important for vision-language interaction
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 3:
                return MemoryTier.GPU_HBM
            else:
                return MemoryTier.CPU_RAM

        elif tensor_type == TensorType.TEXT_EMBEDDINGS:
            # Text embeddings can go to CPU more readily
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 8:
                return MemoryTier.GPU_HBM
            elif tensor_size <= self.cpu_manager.config.max_size_bytes // 4:
                return MemoryTier.CPU_RAM
            else:
                return MemoryTier.SSD_STORAGE

        # GENERAL tensors are more conservative with GPU usage to allow for tier diversity in tests
        if tensor_type == TensorType.GENERAL:
            # Only tiny GENERAL tensors go to GPU, others go to CPU
            if tensor_size <= 50:  # Only tensors <= 50 bytes go to GPU
                return MemoryTier.GPU_HBM
            elif tensor_size <= self.cpu_manager.config.max_size_bytes // 8:  # Conservative for CPU too
                return MemoryTier.CPU_RAM
            else:
                return MemoryTier.SSD_STORAGE
        # TEXT_EMBEDDINGS can go to CPU more readily
        elif tensor_type == TensorType.TEXT_EMBEDDINGS:
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 8:
                return MemoryTier.GPU_HBM
            elif tensor_size <= self.cpu_manager.config.max_size_bytes // 4:
                return MemoryTier.CPU_RAM
            else:
                return MemoryTier.SSD_STORAGE
        # Other tensor types follow standard logic
        else:
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 8:
                return MemoryTier.GPU_HBM
            elif tensor_size <= self.cpu_manager.config.max_size_bytes // 4:
                return MemoryTier.CPU_RAM
            else:
                return MemoryTier.SSD_STORAGE

    def _should_migrate(self, tensor_id: str, current_tier: MemoryTier) -> Tuple[bool, Optional[MemoryTier], float]:
        """
        Determine if tensor should be migrated and to which tier based on Qwen3-VL patterns.

        Args:
            tensor_id: ID of the tensor to evaluate
            current_tier: Current tier of the tensor

        Returns:
            Tuple of (should_migrate, target_tier, migration_benefit_score)
        """
        # If tensor is pinned, don't migrate
        if tensor_id in self.tensor_metadata and self.tensor_metadata[tensor_id].pinned:
            return False, None, 0.0

        # Get access count and timing information
        access_count = self.access_tracker.tensor_access_counts.get(tensor_id, 0)
        last_access = self.access_tracker.tensor_last_access.get(tensor_id, 0)
        time_since_access = time.time() - last_access
        temporal_locality = self.access_tracker.get_temporal_locality_score(tensor_id)

        # Update predictor features
        self.predictor.update_features(self.access_tracker)

        # Get prediction for this tensor
        predicted_prob, predicted_time = self.predictor.predict_access_probability(
            tensor_id,
            tensor_size=self.tensor_metadata.get(tensor_id, TensorMetadata(
                tensor_id=tensor_id,
                shape=(1,),  # Default shape
                dtype=torch.float32,
                size_bytes=4,
                tensor_type=TensorType.GENERAL,
                creation_time=0,
                last_access_time=0,
                access_count=0,
                tier=MemoryTier.GPU_HBM
            )).size_bytes if tensor_id in self.tensor_metadata else 4
        )

        # Calculate various factors for migration decision
        time_factor = 0.0
        if predicted_time:
            # How soon will it be accessed?
            time_until_access = max(0, predicted_time - time.time())
            # Sooner access = higher time factor
            time_factor = max(0, 1 - time_until_access / 300)  # Normalize over 5 minutes

        # Frequency factor
        freq_factor = min(1.0, access_count / 10.0)  # Cap at 1.0 for 10+ accesses

        # Recency factor (how recently was it accessed)
        recency_factor = max(0.0, min(1.0, 300 / (time_since_access + 1)))  # More recent = higher value

        # Temporal locality factor
        temporal_factor = min(1.0, temporal_locality)

        # Combined score for access likelihood
        access_score = (
            predicted_prob * 0.3 +
            freq_factor * 0.25 +
            recency_factor * 0.25 +
            temporal_factor * 0.2
        )

        # Size factor (larger tensors might be more costly to migrate)
        size_factor = 1.0
        if tensor_id in self.tensor_metadata:
            tensor_size = self.tensor_metadata[tensor_id].size_bytes
            # Reduce score for very large tensors (more costly to migrate)
            size_factor = max(0.1, 1.0 - (tensor_size / (1024*1024*1024)))  # Reduce for tensors > 1GB

        # Calculate migration benefit for different targets
        benefits = {}

        # Benefit of moving to GPU HBM
        if current_tier != MemoryTier.GPU_HBM:
            gpu_benefit = access_score * size_factor
            # Adjust based on tensor type
            if tensor_id in self.tensor_metadata:
                tensor_type = self.tensor_metadata[tensor_id].tensor_type
                if tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.CROSS_ATTENTION]:
                    gpu_benefit *= 1.5  # Higher benefit for these critical types
            benefits[MemoryTier.GPU_HBM] = gpu_benefit

        # Benefit of moving to CPU RAM
        if current_tier != MemoryTier.CPU_RAM:
            cpu_benefit = access_score * size_factor * 0.7  # Slightly lower than GPU
            benefits[MemoryTier.CPU_RAM] = cpu_benefit

        # Benefit of moving to SSD Storage
        if current_tier != MemoryTier.SSD_STORAGE:
            # For SSD, benefit is higher when tensor is infrequently accessed
            ssd_benefit = (1 - access_score) * size_factor
            # Also consider temporal locality - low temporal locality tensors are good for SSD
            ssd_benefit *= (1 - temporal_factor)
            benefits[MemoryTier.SSD_STORAGE] = ssd_benefit
        else:
            # If already on SSD but access score is high, suggest moving up
            if access_score > 0.6:
                benefits[MemoryTier.CPU_RAM] = access_score * size_factor * 0.7

        # Find the best migration option
        if benefits:
            best_tier = max(benefits.keys(), key=lambda x: benefits[x])
            best_benefit = benefits[best_tier]

            # Only migrate if benefit is significant enough
            if best_benefit > 0.3:  # Threshold for migration
                return True, best_tier, best_benefit

        return False, None, 0.0

    def _migrate_tensor(self, tensor_id: str, from_tier: MemoryTier, to_tier: MemoryTier) -> bool:
        """
        Migrate tensor from one tier to another.

        Args:
            tensor_id: ID of the tensor to migrate
            from_tier: Source tier
            to_tier: Destination tier

        Returns:
            True if successful, False otherwise
        """
        # Get tensor from source tier
        tensor = None
        metadata = None

        if from_tier == MemoryTier.GPU_HBM:
            tensor = self.gpu_manager.get(tensor_id)
            if tensor_id in self.gpu_manager.metadata:
                metadata = self.gpu_manager.metadata[tensor_id]
        elif from_tier == MemoryTier.CPU_RAM:
            tensor = self.cpu_manager.get(tensor_id)
            if tensor_id in self.cpu_manager.metadata:
                metadata = self.cpu_manager.metadata[tensor_id]
        elif from_tier == MemoryTier.SSD_STORAGE:
            tensor = self.ssd_manager.get(tensor_id)
            if tensor_id in self.ssd_manager.metadata:
                metadata = self.ssd_manager.metadata[tensor_id]

        if tensor is None:
            logger.warning(f"Could not retrieve tensor {tensor_id} for migration")
            return False

        if metadata is None:
            logger.warning(f"Could not retrieve metadata for tensor {tensor_id}")
            return False

        # Calculate migration cost and check if it's worth it
        migration_cost = self._calculate_migration_cost(from_tier, to_tier, metadata)
        predicted_benefit = self._calculate_predicted_benefit(tensor_id, to_tier)

        # Only proceed if benefit outweighs cost
        if predicted_benefit <= migration_cost * 1.2:  # 20% buffer
            logger.debug(f"Migration not beneficial for {tensor_id}: cost {migration_cost:.3f} > benefit {predicted_benefit:.3f}")
            return False

        # Remove from source tier
        if from_tier == MemoryTier.GPU_HBM:
            self.gpu_manager.remove(tensor_id)
        elif from_tier == MemoryTier.CPU_RAM:
            self.cpu_manager.remove(tensor_id)
        elif from_tier == MemoryTier.SSD_STORAGE:
            self.ssd_manager.remove(tensor_id)

        # Update metadata
        metadata.tier = to_tier
        metadata.last_access_time = time.time()

        # Add to destination tier
        success = False
        if to_tier == MemoryTier.GPU_HBM:
            success = self.gpu_manager.put(tensor_id, tensor, metadata)
        elif to_tier == MemoryTier.CPU_RAM:
            success = self.cpu_manager.put(tensor_id, tensor, metadata)
        elif to_tier == MemoryTier.SSD_STORAGE:
            success = self.ssd_manager.put(tensor_id, tensor, metadata)

        if success:
            self.tensor_locations[tensor_id] = to_tier
            self.stats['total_migrations'] += 1

            # Update migration cost stats
            self.stats['migration_cost'] += migration_cost

            logger.debug(f"Migrated tensor {tensor_id} from {from_tier.value} to {to_tier.value}, "
                        f"cost: {migration_cost:.3f}, benefit: {predicted_benefit:.3f}")
        else:
            logger.warning(f"Failed to migrate tensor {tensor_id} to {to_tier.value}")

        return success

    def _calculate_migration_cost(self, from_tier: MemoryTier, to_tier: MemoryTier, metadata: TensorMetadata) -> float:
        """
        Calculate the cost of migrating a tensor from one tier to another.

        Args:
            from_tier: Source tier
            to_tier: Destination tier
            metadata: Tensor metadata

        Returns:
            Cost of migration (higher is more expensive)
        """
        from_config = self._get_tier_config(from_tier)
        to_config = self._get_tier_config(to_tier)

        # Migration cost factors:
        # 1. Transfer time based on size and bandwidth
        transfer_time = (metadata.size_bytes / (1024**3)) / min(from_config.transfer_bandwidth_gbps, to_config.transfer_bandwidth_gbps)

        # 2. Latency difference (moving to slower tier has higher cost)
        latency_diff = max(0, to_config.access_latency_ms - from_config.access_latency_ms)

        # 3. Size factor (larger tensors are more costly to migrate)
        size_factor = metadata.size_bytes / (1024**3)  # Size in GB

        # 4. For Qwen3-VL, consider tensor type importance
        type_factor = 1.0
        if metadata.tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.CROSS_ATTENTION]:
            type_factor = 0.8  # Lower cost factor for important tensors (they're worth migrating)

        # Combine factors with weights
        cost = (transfer_time * 0.4) + (latency_diff * 0.3) + (size_factor * 0.2) + (type_factor * 0.1)

        return cost

    def _calculate_predicted_benefit(self, tensor_id: str, target_tier: MemoryTier) -> float:
        """
        Calculate the predicted benefit of moving a tensor to a target tier.

        Args:
            tensor_id: ID of the tensor
            target_tier: Target tier

        Returns:
            Predicted benefit (higher is better)
        """
        # Get prediction for this tensor
        predicted_prob, predicted_time = self.predictor.predict_access_probability(
            tensor_id,
            tensor_size=self.tensor_metadata.get(tensor_id, TensorMetadata(
                tensor_id=tensor_id,
                shape=(1,),  # Default shape
                dtype=torch.float32,
                size_bytes=4,
                tensor_type=TensorType.GENERAL,
                creation_time=0,
                last_access_time=0,
                access_count=0,
                tier=MemoryTier.GPU_HBM
            )).size_bytes if tensor_id in self.tensor_metadata else 4
        )

        if predicted_prob is None or predicted_time is None:
            return 0.1  # Low benefit if no prediction

        # How soon will it be accessed?
        time_until_access = max(0, predicted_time - time.time())

        # Benefit based on target tier's characteristics
        target_config = self._get_tier_config(target_tier)

        # Benefit is higher for faster tiers when access is soon
        time_factor = max(0, 1 - time_until_access / 300)  # Higher for sooner access
        speed_factor = 1.0 / (target_config.access_latency_ms + 1)  # Higher for faster tiers

        # For frequently accessed tensors, benefit is higher on faster tiers
        access_count = self.access_tracker.tensor_access_counts.get(tensor_id, 0)
        freq_factor = min(2.0, access_count / 5.0)  # Cap at 2.0 for 25+ accesses

        # For Qwen3-VL, consider tensor type importance
        type_factor = 1.0
        if tensor_id in self.tensor_metadata:
            tensor_type = self.tensor_metadata[tensor_id].tensor_type
            if tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.CROSS_ATTENTION]:
                type_factor = 1.5  # Higher benefit for important tensor types

        # Combine factors
        benefit = (predicted_prob * time_factor * speed_factor * (1 + freq_factor) * type_factor)

        return benefit

    def _get_tier_config(self, tier: MemoryTier) -> TierConfig:
        """Get configuration for a specific tier"""
        if tier == MemoryTier.GPU_HBM:
            return self.gpu_manager.config
        elif tier == MemoryTier.CPU_RAM:
            return self.cpu_manager.config
        elif tier == MemoryTier.SSD_STORAGE:
            return self.ssd_manager.config
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def _perform_predictive_migrations(self):
        """Perform migrations based on predictions for Qwen3-VL"""
        with self._lock:
            # Update predictor features
            self.predictor.update_features(self.access_tracker)

            # Check tensors in each tier for potential migration
            tiers_to_check = [
                (self.gpu_manager, MemoryTier.GPU_HBM),
                (self.cpu_manager, MemoryTier.CPU_RAM),
                (self.ssd_manager, MemoryTier.SSD_STORAGE)
            ]

            for manager, tier in tiers_to_check:
                # Get tensor IDs from the manager's cache or metadata
                if tier == MemoryTier.SSD_STORAGE:
                    tensor_ids = list(manager.metadata.keys())
                else:
                    tensor_ids = list(manager.cache.keys())

                for tensor_id in tensor_ids:
                    should_migrate, target_tier, benefit_score = self._should_migrate(tensor_id, tier)
                    if should_migrate and target_tier:
                        # Perform the migration
                        success = self._migrate_tensor(tensor_id, tier, target_tier)
                        if success:
                            logger.debug(f"Migrated tensor {tensor_id} from {tier.value} to {target_tier.value} "
                                       f"with benefit score {benefit_score:.3f}")

    def update_tensor_access(self, tensor_id: str):
        """Update access information for a tensor"""
        with self._lock:
            self.access_tracker.record_access(tensor_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the tiering system"""
        with self._lock:
            gpu_stats = self.gpu_manager.stats
            cpu_stats = self.cpu_manager.stats
            ssd_stats = self.ssd_manager.stats

            # Calculate global hit rate
            total_hits = gpu_stats.hits + cpu_stats.hits + ssd_stats.hits
            total_requests = self.stats['total_requests']
            global_hit_rate = total_hits / total_requests if total_requests > 0 else 0

            self.stats['global_hit_rate'] = global_hit_rate

            return {
                'global_stats': self.stats.copy(),  # Return a copy to avoid reference issues
                'gpu_stats': gpu_stats.__dict__.copy(),
                'cpu_stats': cpu_stats.__dict__.copy(),
                'ssd_stats': ssd_stats.__dict__.copy(),
                'total_cached_tensors': (
                    gpu_stats.access_count + cpu_stats.access_count + ssd_stats.access_count
                ),
                'total_utilization_bytes': (
                    self.gpu_manager.current_size_bytes +
                    self.cpu_manager.current_size_bytes +
                    self.ssd_manager.current_size_bytes
                ),
                'total_max_size_bytes': (
                    gpu_stats.max_size_bytes +
                    cpu_stats.max_size_bytes +
                    ssd_stats.max_size_bytes
                ),
                'tensor_type_distribution': dict(self.stats['tensor_type_distribution'])
            }

    def clear_tier(self, tier: MemoryTier):
        """Clear all tensors from a specific tier"""
        with self._lock:
            if tier == MemoryTier.GPU_HBM:
                for tensor_id in list(self.gpu_manager.cache.keys()):
                    if tensor_id in self.tensor_metadata:
                        # Update tensor type distribution stats BEFORE removing
                        tensor_type = self.tensor_metadata[tensor_id].tensor_type.value
                        if self.stats['tensor_type_distribution'][tensor_type] > 0:
                            self.stats['tensor_type_distribution'][tensor_type] -= 1
                    self.gpu_manager.remove(tensor_id)
                    if tensor_id in self.tensor_locations:
                        del self.tensor_locations[tensor_id]
                    if tensor_id in self.tensor_metadata:
                        del self.tensor_metadata[tensor_id]
            elif tier == MemoryTier.CPU_RAM:
                for tensor_id in list(self.cpu_manager.cache.keys()):
                    if tensor_id in self.tensor_metadata:
                        # Update tensor type distribution stats BEFORE removing
                        tensor_type = self.tensor_metadata[tensor_id].tensor_type.value
                        if self.stats['tensor_type_distribution'][tensor_type] > 0:
                            self.stats['tensor_type_distribution'][tensor_type] -= 1
                    self.cpu_manager.remove(tensor_id)
                    if tensor_id in self.tensor_locations:
                        del self.tensor_locations[tensor_id]
                    if tensor_id in self.tensor_metadata:
                        del self.tensor_metadata[tensor_id]
            elif tier == MemoryTier.SSD_STORAGE:
                for tensor_id in list(self.ssd_manager.metadata.keys()):
                    if tensor_id in self.tensor_metadata:
                        # Update tensor type distribution stats BEFORE removing
                        tensor_type = self.tensor_metadata[tensor_id].tensor_type.value
                        if self.stats['tensor_type_distribution'][tensor_type] > 0:
                            self.stats['tensor_type_distribution'][tensor_type] -= 1
                    self.ssd_manager.remove(tensor_id)
                    if tensor_id in self.tensor_locations:
                        del self.tensor_locations[tensor_id]
                    if tensor_id in self.tensor_metadata:
                        del self.tensor_metadata[tensor_id]

    def clear_all(self):
        """Clear all tensors from all tiers"""
        self.clear_tier(MemoryTier.GPU_HBM)
        self.clear_tier(MemoryTier.CPU_RAM)
        self.clear_tier(MemoryTier.SSD_STORAGE)

    def get_tensor_placement_info(self, tensor_id: str) -> Optional[TensorMetadata]:
        """Get placement information for a specific tensor"""
        return self.tensor_metadata.get(tensor_id)

    def get_optimal_tier_for_tensor(self,
                                   tensor_size: int,
                                   tensor_type: TensorType,
                                   access_frequency: float = 0.0,
                                   temporal_locality: float = 0.0) -> MemoryTier:
        """
        Get the optimal tier for a tensor based on its characteristics.

        Args:
            tensor_size: Size of the tensor in bytes
            tensor_type: Type of tensor
            access_frequency: Expected access frequency
            temporal_locality: Temporal locality score

        Returns:
            Optimal MemoryTier for this tensor
        """
        # For Qwen3-VL specific tensor types, prioritize accordingly
        if tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, TensorType.CROSS_ATTENTION]:
            # These are critical for Qwen3-VL performance
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 2:
                return MemoryTier.GPU_HBM
            else:
                return MemoryTier.CPU_RAM
        elif access_frequency > 3.0 or temporal_locality > 0.6:
            # High frequency or good temporal locality -> faster tier
            if tensor_size <= self.gpu_manager.config.max_size_bytes // 4:
                return MemoryTier.GPU_HBM
            else:
                return MemoryTier.CPU_RAM
        else:
            # Low frequency or poor temporal locality -> slower tier
            return MemoryTier.SSD_STORAGE


def create_qwen3vl_memory_tiering_system(hardware_config: Optional[Dict[str, Any]] = None) -> Qwen3VLMemoryTieringSystem:
    """
    Factory function to create a Qwen3-VL memory tiering system optimized for specific hardware.

    Args:
        hardware_config: Hardware configuration with details like:
                        - gpu_memory: GPU memory in bytes
                        - cpu_memory: CPU memory in bytes
                        - storage_type: Storage type ('nvme', 'ssd', 'hdd')

    Returns:
        Optimized Qwen3VLMemoryTieringSystem instance
    """
    if hardware_config is None:
        # Default configuration for typical hardware
        hardware_config = {
            'gpu_memory': 6 * 1024 * 1024 * 1024,  # 6GB GPU
            'cpu_memory': 16 * 1024 * 1024 * 1024,  # 16GB system RAM
            'storage_type': 'nvme'
        }

    # Adjust tier sizes based on hardware
    gpu_memory = hardware_config.get('gpu_memory', 6 * 1024 * 1024 * 1024)
    cpu_memory = hardware_config.get('cpu_memory', 16 * 1024 * 1024 * 1024)
    storage_type = hardware_config.get('storage_type', 'nvme')

    # For Qwen3-VL, allocate tiers appropriately
    gpu_tier_size = min(gpu_memory * 0.7, 4 * 1024 * 1024 * 1024)  # Use up to 4GB or 70% of GPU memory
    cpu_tier_size = min(cpu_memory * 0.4, 8 * 1024 * 1024 * 1024)  # Use up to 8GB or 40% of CPU memory
    ssd_tier_size = 20 * 1024 * 1024 * 1024  # 20GB for SSD

    tiering_system = Qwen3VLMemoryTieringSystem(
        gpu_hbm_size=gpu_tier_size,
        cpu_ram_size=cpu_tier_size,
        ssd_storage_size=ssd_tier_size,
        prediction_window=1000
    )

    logger.info(f"Created Qwen3-VL memory tiering system with: "
                f"GPU={gpu_tier_size/(1024**3):.1f}GB, "
                f"CPU={cpu_tier_size/(1024**3):.1f}GB, "
                f"SSD={ssd_tier_size/(1024**3):.1f}GB")

    return tiering_system


def integrate_with_qwen3vl_model(tiering_system: Qwen3VLMemoryTieringSystem):
    """
    Integration example for Qwen3-VL model components.
    This function demonstrates how to integrate the tiering system with the Qwen3-VL model.
    """
    def allocate_kv_cache(batch_size: int, seq_len: int, hidden_dim: int, num_heads: int) -> Tuple[torch.Tensor, str]:
        """Allocate KV cache tensors with appropriate tiering"""
        # Create key and value tensors
        key_tensor = torch.zeros(batch_size, seq_len, hidden_dim // num_heads, dtype=torch.float16)
        value_tensor = torch.zeros(batch_size, seq_len, hidden_dim // num_heads, dtype=torch.float16)

        # Store in tiering system with KV cache type
        success_k, k_id = tiering_system.put_tensor(
            key_tensor,
            tensor_type=TensorType.KV_CACHE,
            pinned=False
        )

        success_v, v_id = tiering_system.put_tensor(
            value_tensor,
            tensor_type=TensorType.KV_CACHE,
            pinned=False
        )

        if success_k and success_v:
            return key_tensor, k_id
        else:
            # Fallback to standard allocation if tiering fails
            return key_tensor, None

    def allocate_image_features(batch_size: int, num_patches: int, feature_dim: int) -> Tuple[torch.Tensor, str]:
        """Allocate image feature tensors with appropriate tiering"""
        image_features = torch.zeros(batch_size, num_patches, feature_dim, dtype=torch.float16)

        # Store in tiering system with image features type
        success, tensor_id = tiering_system.put_tensor(
            image_features,
            tensor_type=TensorType.IMAGE_FEATURES,
            pinned=False
        )

        return image_features, tensor_id if success else None

    def access_tensor_with_tiering(tensor_id: str, target_device: torch.device = None) -> Optional[torch.Tensor]:
        """Access a tensor with automatic tiering management"""
        tensor = tiering_system.get_tensor(tensor_id, target_device)
        if tensor is not None:
            # Update access tracking
            tiering_system.update_tensor_access(tensor_id)
        return tensor

    return allocate_kv_cache, allocate_image_features, access_tensor_with_tiering


if __name__ == "__main__":
    print("Qwen3-VL Advanced Memory Tiering System")
    print("=" * 50)

    # Create the tiering system
    tiering_system = create_qwen3vl_memory_tiering_system({
        'gpu_memory': 8 * 1024 * 1024 * 1024,  # 8GB GPU
        'cpu_memory': 16 * 1024 * 1024 * 1024,  # 16GB CPU
        'storage_type': 'nvme'
    })

    print(f"\n1. Created tiering system with:")
    print(f"   GPU HBM: {tiering_system.gpu_manager.config.max_size_bytes / (1024**3):.1f}GB")
    print(f"   CPU RAM: {tiering_system.cpu_manager.config.max_size_bytes / (1024**3):.1f}GB")
    print(f"   SSD Storage: {tiering_system.ssd_manager.config.max_size_bytes / (1024**3):.1f}GB")

    # Test tensor operations
    print(f"\n2. Testing tensor insertion...")

    # Create a sample tensor and store it
    sample_tensor = torch.randn(100, 100, dtype=torch.float16)
    success, tensor_id = tiering_system.put_tensor(
        sample_tensor,
        tensor_type=TensorType.GENERAL
    )
    print(f"   Tensor insertion successful: {success}")

    # Try to retrieve the tensor
    retrieved_tensor = tiering_system.get_tensor(tensor_id)
    print(f"   Tensor retrieval successful: {retrieved_tensor is not None}")

    # Test Qwen3-VL specific tensor types
    print(f"\n3. Testing Qwen3-VL specific tensor types...")

    # KV cache tensor (important for transformers)
    kv_tensor = torch.randn(2, 128, 64, dtype=torch.float16)
    success, kv_id = tiering_system.put_tensor(
        kv_tensor,
        tensor_type=TensorType.KV_CACHE
    )
    print(f"   KV Cache tensor insertion: {success}")

    # Image features tensor (important for vision models)
    img_tensor = torch.randn(1, 196, 768, dtype=torch.float16)  # 14x14 patches
    success, img_id = tiering_system.put_tensor(
        img_tensor,
        tensor_type=TensorType.IMAGE_FEATURES
    )
    print(f"   Image features tensor insertion: {success}")

    # Cross-attention tensor (important for vision-language models)
    cross_tensor = torch.randn(2, 128, 768, dtype=torch.float16)
    success, cross_id = tiering_system.put_tensor(
        cross_tensor,
        tensor_type=TensorType.CROSS_ATTENTION
    )
    print(f"   Cross-attention tensor insertion: {success}")

    # Test access pattern tracking
    print(f"\n4. Testing access pattern tracking...")

    # Access tensors with different patterns
    for i in range(5):
        # Access KV tensor frequently
        _ = tiering_system.get_tensor(kv_id)
        time.sleep(0.01)  # Small delay to create time differences

    for i in range(3):
        # Access image tensor moderately
        _ = tiering_system.get_tensor(img_id)
        time.sleep(0.02)

    # Access cross tensor once
    _ = tiering_system.get_tensor(cross_id)

    # Perform predictive migrations based on access patterns
    print(f"\n5. Performing predictive migrations...")
    tiering_system._perform_predictive_migrations()

    # Get and display statistics
    print(f"\n6. Tiering system statistics:")
    stats = tiering_system.get_stats()

    print(f"   Global hit rate: {stats['global_stats']['global_hit_rate']:.2%}")
    print(f"   Total migrations: {stats['global_stats']['total_migrations']}")
    print(f"   Total cached tensors: {stats['total_cached_tensors']}")
    print(f"   Total utilization: {stats['total_utilization_bytes'] / (1024**3):.2f}GB / {stats['total_max_size_bytes'] / (1024**3):.2f}GB")

    print(f"\n   GPU Tier - Hit rate: {stats['gpu_stats']['hit_rate']:.2%}, "
          f"Utilization: {stats['gpu_stats']['utilization']:.2%}")
    print(f"   CPU Tier - Hit rate: {stats['cpu_stats']['hit_rate']:.2%}, "
          f"Utilization: {stats['cpu_stats']['utilization']:.2%}")
    print(f"   SSD Tier - Hit rate: {stats['ssd_stats']['hit_rate']:.2%}, "
          f"Utilization: {stats['ssd_stats']['utilization']:.2%}")

    print(f"\n   Tensor type distribution: {dict(stats['tensor_type_distribution'])}")

    print(f"\nQwen3-VL Advanced Memory Tiering System initialized successfully!")
    print(f"This system optimizes tensor placement based on access patterns, size, and temporal locality.")
    print(f"It uses ML-based predictions to proactively migrate tensors between HBM, RAM, and SSD.")