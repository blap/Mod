"""
Advanced Disk Offloading System for Inference-PIO

This module implements a sophisticated disk offloading system that uses the hard drive
as an extension of RAM, creating an optimized paging system that moves model parts
between disk and memory based on predictive algorithms and advanced memory management strategies.
"""

import gc
import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch import Tensor

# Import sklearn modules with fallback
try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None
    StandardScaler = None
    RandomForestRegressor = None
    KMeans = None


logger = logging.getLogger(__name__)


class OffloadPriority(Enum):
    """Priority levels for disk offloading."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class OffloadStrategy(Enum):
    """Strategies for disk offloading."""

    LRU = "lru"
    FIFO = "fifo"
    PRIORITY = "priority"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    CLUSTER_BASED = "cluster_based"


class ModelComponentType(Enum):
    """Types of model components for specialized offloading."""

    ATTENTION_WEIGHTS = "attention_weights"
    MLP_WEIGHTS = "mlp_weights"
    EMBEDDINGS = "embeddings"
    ACTIVATIONS = "activations"
    KV_CACHE = "kv_cache"
    GRADIENTS = "gradients"
    OPTIMIZER_STATE = "optimizer_state"
    OTHER = "other"


@dataclass
class OffloadPage:
    """Represents a memory page for disk offloading."""

    id: str
    tensor: Optional[Tensor] = None
    device: Optional[str] = None
    size_bytes: int = 0
    priority: OffloadPriority = OffloadPriority.MEDIUM
    last_access_time: float = 0.0
    pinned: bool = False
    file_path: Optional[str] = None  # Path on disk when offloaded
    access_pattern: str = "unknown"  # Pattern of access (sequential, random, etc.)
    predicted_next_access: float = 0.0  # Predicted next access time
    component_type: ModelComponentType = ModelComponentType.OTHER
    access_frequency: float = 0.0  # Number of accesses per unit time
    temporal_locality: float = 0.0  # Measure of how recently accessed
    spatial_locality: float = 0.0  # Measure of how close to other accessed components
    reuse_probability: float = 0.0  # Probability of being accessed again
    offload_history: List[float] = None  # Times when this page was offloaded
    restore_history: List[float] = None  # Times when this page was restored
    hash_value: str = ""  # Hash of the tensor for integrity checking

    def __post_init__(self):
        if self.offload_history is None:
            self.offload_history = []
        if self.restore_history is None:
            self.restore_history = []
        if self.hash_value == "":
            self.hash_value = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of the tensor for integrity checking."""
        if self.tensor is not None:
            # Compute hash of tensor data
            tensor_bytes = self.tensor.detach().cpu().numpy().tobytes()
            return hashlib.sha256(tensor_bytes).hexdigest()
        return ""


class AccessPattern(Enum):
    """Types of access patterns for model components."""

    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORARY = "temporary"
    FREQUENT = "frequent"
    RARE = "rare"


class DiskOffloader:
    """
    Advanced disk offloading system that manages moving model components
    between RAM and disk based on predictive algorithms and advanced memory management strategies.
    """

    def __init__(
        self,
        max_memory_ratio: float = 0.8,
        offload_directory: Optional[str] = None,
        page_size_mb: int = 16,
        eviction_policy: str = "predictive",
        prediction_horizon: int = 30,
        enable_clustering: bool = True,
        cluster_count: int = 5,
        enable_adaptive: bool = True,
        adaptive_threshold: float = 0.75,
    ):
        """
        Initialize the disk offloading system.

        Args:
            max_memory_ratio: Maximum ratio of system memory to use (0.0 to 1.0)
            offload_directory: Directory for offload files (default: temporary directory)
            page_size_mb: Size of memory pages in MB
            eviction_policy: Page eviction policy ("lru", "fifo", "priority", "predictive", "adaptive", "cluster_based")
            prediction_horizon: Time horizon (in seconds) for memory predictions
            enable_clustering: Whether to enable clustering-based offloading
            cluster_count: Number of clusters for clustering-based offloading
            enable_adaptive: Whether to enable adaptive offloading
            adaptive_threshold: Threshold for adaptive offloading decisions
        """
        self.max_memory_ratio = max_memory_ratio
        self.page_size_bytes = page_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.prediction_horizon = prediction_horizon
        self.enable_clustering = enable_clustering
        self.cluster_count = cluster_count
        self.enable_adaptive = enable_adaptive
        self.adaptive_threshold = adaptive_threshold

        # Set up offload directory
        if offload_directory:
            self.offload_directory = Path(offload_directory)
        else:
            self.offload_directory = Path(tempfile.mkdtemp(prefix="pio_offload_"))

        self.offload_directory.mkdir(parents=True, exist_ok=True)

        # Track offload pages
        self.pages: Dict[str, OffloadPage] = {}
        self.ram_pages: List[str] = []  # Pages currently in RAM
        self.disk_pages: List[str] = []  # Pages currently on disk
        self.access_times: Dict[str, float] = {}  # Last access times for LRU

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "pages_offloaded": 0,
            "pages_restored": 0,
            "page_faults": 0,
            "total_pages": 0,
            "ram_pages": 0,
            "disk_pages": 0,
            "peak_memory_used": 0,
            "total_offloaded_bytes": 0,
            "total_restored_bytes": 0,
            "clustering_enabled": enable_clustering,
            "adaptive_enabled": enable_adaptive,
            "offload_strategy": eviction_policy,
        }

        # Prediction components
        self.memory_predictor = MemoryPredictor(window_size=100)
        self.access_analyzer = AccessPatternAnalyzer(history_size=1000)
        self.component_predictor = ComponentAccessPredictor()

        # Clustering for similar components
        if self.enable_clustering and SKLEARN_AVAILABLE:
            self.clustering_model = KMeans(
                n_clusters=self.cluster_count, random_state=42
            )
        else:
            self.clustering_model = None

        # Component type-specific strategies
        self.component_strategies = {
            ModelComponentType.KV_CACHE: OffloadStrategy.PREDICTIVE,
            ModelComponentType.ACTIVATIONS: OffloadStrategy.ADAPTIVE,
            ModelComponentType.ATTENTION_WEIGHTS: OffloadStrategy.PRIORITY,
            ModelComponentType.MLP_WEIGHTS: OffloadStrategy.PRIORITY,
            ModelComponentType.EMBEDDINGS: OffloadStrategy.LRU,
            ModelComponentType.GRADIENTS: OffloadStrategy.FIFO,
            ModelComponentType.OPTIMIZER_STATE: OffloadStrategy.PRIORITY,
            ModelComponentType.OTHER: OffloadStrategy.PREDICTIVE,
        }

        # Background thread for proactive offloading
        self.proactive_thread = None
        self.stop_proactive_thread = threading.Event()

        logger.info(
            f"Advanced disk offloading system initialized with max_memory_ratio={max_memory_ratio}, "
            f"offload_directory={self.offload_directory}, page_size={page_size_mb}MB, "
            f"prediction_horizon={prediction_horizon}s, strategy={eviction_policy}, "
            f"clustering={enable_clustering}, adaptive={enable_adaptive}"
        )

    def get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        memory = psutil.virtual_memory()
        return int(memory.available)

    def get_total_memory(self) -> int:
        """Get total system memory in bytes."""
        memory = psutil.virtual_memory()
        return int(memory.total)

    def get_current_memory_usage(self) -> int:
        """Get current memory usage by this process in bytes."""
        process = psutil.Process()
        return process.memory_info().rss

    def is_memory_pressure_high(self) -> bool:
        """Check if memory pressure is high."""
        memory = psutil.virtual_memory()
        return memory.percent > (self.max_memory_ratio * 100)

    def _update_predictions(self):
        """Update memory usage predictions."""
        current_time = time.time()
        current_memory = self.get_current_memory_usage()
        self.memory_predictor.record_memory_usage(current_time, current_memory)

    def allocate_page(
        self,
        tensor: Tensor,
        page_id: str,
        priority: OffloadPriority = OffloadPriority.MEDIUM,
        access_pattern: AccessPattern = AccessPattern.RANDOM,
        component_type: ModelComponentType = ModelComponentType.OTHER,
    ) -> bool:
        """
        Allocate a memory page for a tensor.

        Args:
            tensor: The tensor to store in the page
            page_id: Unique identifier for the page
            priority: Priority level for the page
            access_pattern: Expected access pattern for the tensor
            component_type: Type of model component for specialized handling

        Returns:
            True if allocation was successful, False otherwise
        """
        with self.lock:
            if page_id in self.pages:
                logger.warning(f"Page {page_id} already exists, overwriting")
                self.deallocate_page(page_id)

            size_bytes = tensor.element_size() * tensor.nelement()

            page = OffloadPage(
                id=page_id,
                tensor=tensor,
                device=str(tensor.device),
                size_bytes=size_bytes,
                priority=priority,
                last_access_time=time.time(),
                access_pattern=access_pattern.value,
                component_type=component_type,
            )

            self.pages[page_id] = page
            self.ram_pages.append(page_id)
            self.access_times[page_id] = page.last_access_time

            self.stats["total_pages"] += 1
            self.stats["ram_pages"] += 1

            # Update predictions
            self._update_predictions()

            # Record access pattern for prediction
            self.access_analyzer.record_access(
                page_id, time.time(), access_pattern.value
            )

            # Update component-specific access patterns
            self.component_predictor.record_component_access(
                page_id, time.time(), access_pattern.value
            )

            # Check if we need to offload pages due to memory pressure
            self._handle_memory_pressure()

            logger.debug(
                f"Allocated page {page_id} ({size_bytes} bytes) in RAM, component_type={component_type.value}"
            )
            return True

    def deallocate_page(self, page_id: str) -> bool:
        """
        Deallocate a memory page.

        Args:
            page_id: ID of the page to deallocate

        Returns:
            True if deallocation was successful, False otherwise
        """
        with self.lock:
            if page_id not in self.pages:
                logger.warning(f"Page {page_id} does not exist")
                return False

            page = self.pages[page_id]

            # Remove from appropriate list
            if page_id in self.ram_pages:
                self.ram_pages.remove(page_id)
                self.stats["ram_pages"] -= 1
            elif page_id in self.disk_pages:
                self.disk_pages.remove(page_id)
                self.stats["disk_pages"] -= 1

                # Delete the offload file if it exists
                if page.file_path and os.path.exists(page.file_path):
                    try:
                        os.remove(page.file_path)
                    except OSError as e:
                        logger.error(
                            f"Failed to delete offload file {page.file_path}: {e}"
                        )

            # Remove from tracking
            if page_id in self.access_times:
                del self.access_times[page_id]

            # Delete the page
            del self.pages[page_id]

            logger.debug(f"Deallocated page {page_id}")
            return True

    def offload_page_to_disk(self, page_id: str) -> bool:
        """
        Offload a page from RAM to disk.

        Args:
            page_id: ID of the page to offload

        Returns:
            True if offload was successful, False otherwise
        """
        with self.lock:
            if page_id not in self.pages:
                logger.error(f"Page {page_id} does not exist")
                return False

            page = self.pages[page_id]

            if page_id not in self.ram_pages:
                logger.warning(f"Page {page_id} is not in RAM, nothing to offload")
                return True  # Already on disk or invalid state

            # Create offload file path
            offload_file = self.offload_directory / f"page_{page_id}.pkl"

            try:
                # Save tensor to disk
                with open(offload_file, "wb") as f:
                    pickle.dump(
                        {
                            "tensor": (
                                page.tensor.cpu() if page.tensor is not None else None
                            ),
                            "device": page.device,
                        },
                        f,
                    )

                # Update page info
                page.file_path = str(offload_file)
                page.tensor = None  # Free RAM

                # Move from RAM to disk tracking
                self.ram_pages.remove(page_id)
                self.disk_pages.append(page_id)

                self.stats["pages_offloaded"] += 1
                self.stats["ram_pages"] -= 1
                self.stats["disk_pages"] += 1
                self.stats["total_offloaded_bytes"] += page.size_bytes

                logger.debug(f"Offloaded page {page_id} to disk: {offload_file}")
                return True

            except Exception as e:
                logger.error(f"Failed to offload page {page_id} to disk: {e}")
                return False

    def restore_page_to_ram(self, page_id: str) -> bool:
        """
        Restore a page from disk to RAM.

        Args:
            page_id: ID of the page to restore

        Returns:
            True if restore was successful, False otherwise
        """
        with self.lock:
            if page_id not in self.pages:
                logger.error(f"Page {page_id} does not exist")
                return False

            page = self.pages[page_id]

            if page_id not in self.disk_pages:
                logger.warning(f"Page {page_id} is not on disk, nothing to restore")
                return True  # Already in RAM or invalid state

            if not page.file_path or not os.path.exists(page.file_path):
                logger.error(
                    f"Offload file for page {page_id} does not exist: {page.file_path}"
                )
                return False

            try:
                # Load tensor from disk
                with open(page.file_path, "rb") as f:
                    data = pickle.load(f)
                    tensor = data["tensor"]
                    original_device = data["device"]

                # Move tensor to the appropriate device
                if tensor is not None and original_device:
                    device = torch.device(original_device)
                    tensor = tensor.to(device)

                # Update page info
                page.tensor = tensor
                page.device = original_device

                # Move from disk to RAM tracking
                self.disk_pages.remove(page_id)
                self.ram_pages.append(page_id)

                self.stats["pages_restored"] += 1
                self.stats["ram_pages"] += 1
                self.stats["disk_pages"] -= 1
                self.stats["total_restored_bytes"] += page.size_bytes

                logger.debug(f"Restored page {page_id} to RAM from: {page.file_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to restore page {page_id} to RAM: {e}")
                return False

    def access_page(self, page_id: str) -> Optional[Tensor]:
        """
        Access a page, ensuring it's in RAM. This may trigger offloads/restores.

        Args:
            page_id: ID of the page to access

        Returns:
            The tensor if successful, None otherwise
        """
        with self.lock:
            if page_id not in self.pages:
                logger.error(f"Page {page_id} does not exist")
                return None

            page = self.pages[page_id]
            current_time = time.time()
            page.last_access_time = current_time
            self.access_times[page_id] = page.last_access_time

            # Record access for prediction
            self.access_analyzer.record_access(
                page_id, current_time, page.access_pattern
            )

            # If page is on disk, restore it to RAM
            if page_id in self.disk_pages:
                self.stats["page_faults"] += 1
                if not self.restore_page_to_ram(page_id):
                    logger.error(f"Failed to restore page {page_id} to RAM")
                    return None

            # Update peak memory usage
            current_usage = self.get_current_memory_usage()
            if current_usage > self.stats["peak_memory_used"]:
                self.stats["peak_memory_used"] = current_usage

            return page.tensor

    def _handle_memory_pressure(self):
        """Handle memory pressure by offloading pages if needed."""
        if not self.is_memory_pressure_high():
            return

        # Determine the strategy to use based on configuration
        strategy = self._get_current_strategy()

        # Get pages to consider for offloading
        pages_to_consider = []
        for page_id in self.ram_pages:
            page = self.pages[page_id]
            # Don't offload pinned pages
            if not page.pinned:
                pages_to_consider.append((page_id, page))

        if not pages_to_consider:
            return

        # Sort pages based on the selected strategy
        if strategy == OffloadStrategy.LRU:
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif strategy == OffloadStrategy.FIFO:
            # For FIFO, we'd need to track insertion order, so we'll use access time as proxy
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif strategy == OffloadStrategy.PRIORITY:
            # Evict lowest priority pages first
            pages_to_consider.sort(key=lambda x: x[1].priority.value)
        elif strategy == OffloadStrategy.PREDICTIVE:
            # Use predictive algorithm to determine which pages to offload
            current_time = time.time()
            pages_to_consider.sort(
                key=lambda x: self._calculate_prediction_score(x[0], current_time)
            )
        elif strategy == OffloadStrategy.ADAPTIVE:
            # Use adaptive algorithm that considers multiple factors
            current_time = time.time()
            pages_to_consider.sort(
                key=lambda x: self._calculate_adaptive_score(x[0], current_time)
            )
        elif (
            strategy == OffloadStrategy.CLUSTER_BASED
            and self.clustering_model is not None
        ):
            # Use clustering-based offloading
            pages_to_consider = self._get_cluster_based_offload_order(pages_to_consider)
        else:
            # Default to LRU
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])

        # Offload pages until memory pressure is relieved
        # Limit the number of iterations to prevent infinite loops
        max_offloads_per_call = min(
            len(pages_to_consider), 10
        )  # Limit to 10 offloads per call
        offloads_performed = 0

        for page_id, page in pages_to_consider:
            if (
                not self.is_memory_pressure_high()
                or offloads_performed >= max_offloads_per_call
            ):
                break

            if self.offload_page_to_disk(page_id):
                logger.debug(
                    f"Evicted page {page_id} due to memory pressure using {strategy.value} strategy"
                )
                offloads_performed += 1
            else:
                logger.warning(f"Failed to evict page {page_id} due to memory pressure")

    def _get_current_strategy(self) -> OffloadStrategy:
        """Get the current offloading strategy based on configuration."""
        if self.eviction_policy == "adaptive" and self.enable_adaptive:
            return OffloadStrategy.ADAPTIVE
        elif self.eviction_policy == "cluster_based" and self.enable_clustering:
            return OffloadStrategy.CLUSTER_BASED
        elif self.eviction_policy == "lru":
            return OffloadStrategy.LRU
        elif self.eviction_policy == "fifo":
            return OffloadStrategy.FIFO
        elif self.eviction_policy == "priority":
            return OffloadStrategy.PRIORITY
        else:  # Default to predictive
            return OffloadStrategy.PREDICTIVE

    def _calculate_prediction_score(self, page_id: str, current_time: float) -> float:
        """
        Calculate a score for predictive eviction. Lower scores mean higher likelihood of eviction.

        Args:
            page_id: ID of the page to evaluate
            current_time: Current time for prediction

        Returns:
            Score for the page (lower means more likely to be evicted)
        """
        # Get access score from the analyzer
        access_score = self.access_analyzer.get_access_score(page_id, current_time)

        # Get the page object
        page = self.pages[page_id]

        # Combine factors:
        # - Higher access score means less likely to be evicted (higher score = lower eviction probability)
        # - Higher priority means less likely to be evicted
        # - Larger size might make it more likely to be evicted under memory pressure
        priority_factor = (
            page.priority.value
        )  # Higher priority = less likely to be evicted
        size_factor = page.size_bytes / (
            1024 * 1024
        )  # Size in MB, larger = more likely to be evicted

        # Calculate final score (lower score = more likely to be evicted)
        # Access score is inverted because high access score means keep in memory
        final_score = -access_score + (5 - priority_factor) + (size_factor / 100.0)

        return final_score

    def _calculate_adaptive_score(self, page_id: str, current_time: float) -> float:
        """
        Calculate a score for adaptive eviction considering multiple factors.

        Args:
            page_id: ID of the page to evaluate
            current_time: Current time for prediction

        Returns:
            Score for the page (lower means more likely to be evicted)
        """
        page = self.pages[page_id]

        # Get various scores
        access_score = self.access_analyzer.get_access_score(page_id, current_time)
        component_score = self.component_predictor.get_component_priority(
            page_id, current_time
        )

        # Calculate temporal locality (how recently accessed)
        time_since_access = current_time - page.last_access_time
        temporal_locality = max(
            0, 1.0 - (time_since_access / 60.0)
        )  # Decay over 1 minute

        # Calculate access frequency (how often accessed)
        access_frequency = self.access_analyzer.get_access_frequency(
            page_id, current_time
        )

        # Calculate reuse probability
        reuse_prob = self._estimate_reuse_probability(page_id, current_time)

        # Combine factors with adaptive weights
        # Higher values mean less likely to be evicted
        combined_score = (
            0.3 * access_score  # 30% access patterns
            + 0.2 * component_score  # 20% component type importance
            + 0.2 * temporal_locality  # 20% recent access
            + 0.15 * access_frequency  # 15% access frequency
            + 0.15 * reuse_prob  # 15% reuse probability
        )

        # Factor in priority and size
        priority_factor = page.priority.value / 4.0  # Normalize to 0-1
        size_factor = 1.0 - min(
            1.0, page.size_bytes / (100 * 1024 * 1024)
        )  # Inverse effect, larger = less priority

        final_score = combined_score * priority_factor * size_factor

        # Return negative score because lower scores mean higher eviction probability
        return -final_score

    def _estimate_reuse_probability(self, page_id: str, current_time: float) -> float:
        """
        Estimate the probability that a page will be reused.

        Args:
            page_id: ID of the page to evaluate
            current_time: Current time for prediction

        Returns:
            Estimated reuse probability (0.0 to 1.0)
        """
        page = self.pages[page_id]

        # Calculate based on access history
        if len(page.offload_history) > 0:
            # If page has been offloaded before, check how often it gets accessed again
            recent_accesses = [
                t for t in page.restore_history if current_time - t < 300
            ]  # Last 5 minutes
            return min(
                1.0, len(recent_accesses) * 0.5
            )  # More recent accesses = higher prob

        # Calculate based on access pattern
        access_freq = self.access_analyzer.get_access_frequency(page_id, current_time)
        return min(1.0, access_freq * 10)  # Higher frequency = higher probability

    def _get_cluster_based_offload_order(
        self, pages_to_consider: List[Tuple[str, OffloadPage]]
    ) -> List[Tuple[str, OffloadPage]]:
        """
        Get offload order based on clustering of similar components.

        Args:
            pages_to_consider: List of (page_id, page) tuples to consider

        Returns:
            Ordered list of (page_id, page) tuples for offloading
        """
        if not pages_to_consider or self.clustering_model is None:
            return pages_to_consider

        # Extract features for clustering
        features = []
        page_ids = []
        for page_id, page in pages_to_consider:
            # Create feature vector based on page characteristics
            feature_vector = [
                page.size_bytes / (1024 * 1024),  # Size in MB
                page.priority.value / 4.0,  # Normalized priority
                page.access_frequency,  # Access frequency
                page.temporal_locality,  # Temporal locality
                (
                    1 if page.component_type == ModelComponentType.KV_CACHE else 0
                ),  # Is KV cache
                (
                    1 if page.component_type == ModelComponentType.ACTIVATIONS else 0
                ),  # Is activations
            ]
            features.append(feature_vector)
            page_ids.append(page_id)

        if not features:
            return pages_to_consider

        # Perform clustering
        try:
            clusters = self.clustering_model.fit_predict(features)

            # Group pages by cluster
            cluster_groups = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_groups[cluster_id].append(
                    (page_ids[i], self.pages[page_ids[i]])
                )

            # Sort clusters by some criterion (e.g., average size, access frequency)
            sorted_clusters = sorted(
                cluster_groups.items(), key=lambda x: self._cluster_priority_score(x[1])
            )

            # Within each cluster, sort by individual priority
            result = []
            for cluster_id, cluster_pages in sorted_clusters:
                cluster_pages.sort(
                    key=lambda x: self._calculate_prediction_score(x[0], time.time())
                )
                result.extend(cluster_pages)

            return result
        except Exception as e:
            logger.warning(
                f"Clustering failed, falling back to predictive strategy: {e}"
            )
            # Fall back to predictive sorting
            current_time = time.time()
            pages_to_consider.sort(
                key=lambda x: self._calculate_prediction_score(x[0], current_time)
            )
            return pages_to_consider

    def _cluster_priority_score(
        self, cluster_pages: List[Tuple[str, OffloadPage]]
    ) -> float:
        """
        Calculate a priority score for a cluster of pages.

        Args:
            cluster_pages: List of (page_id, page) tuples in the cluster

        Returns:
            Priority score for the cluster
        """
        if not cluster_pages:
            return 0.0

        # Calculate average characteristics of the cluster
        avg_size = sum(page.size_bytes for _, page in cluster_pages) / len(
            cluster_pages
        )
        avg_priority = sum(page.priority.value for _, page in cluster_pages) / len(
            cluster_pages
        )
        avg_access_freq = sum(page.access_frequency for _, page in cluster_pages) / len(
            cluster_pages
        )

        # Lower priority score means higher likelihood of offloading
        return avg_size * 0.4 + (5 - avg_priority) * 0.4 + (1 - avg_access_freq) * 0.2

    def get_page_stats(self) -> Dict[str, Any]:
        """Get statistics about memory pages."""
        with self.lock:
            total_size = sum(page.size_bytes for page in self.pages.values())
            ram_size = sum(self.pages[pid].size_bytes for pid in self.ram_pages)
            disk_size = sum(self.pages[pid].size_bytes for pid in self.disk_pages)

            return {
                "total_pages": len(self.pages),
                "ram_pages": len(self.ram_pages),
                "disk_pages": len(self.disk_pages),
                "total_size_bytes": total_size,
                "ram_size_bytes": ram_size,
                "disk_size_bytes": disk_size,
                "stats": self.stats.copy(),
            }

    def start_proactive_management(self, interval: float = 5.0):
        """
        Start proactive memory management in a background thread.

        Args:
            interval: Time interval (in seconds) between checks
        """
        if self.proactive_thread is not None and self.proactive_thread.is_alive():
            logger.warning("Proactive memory management already running")
            return

        self.stop_proactive_thread.clear()
        self.proactive_thread = threading.Thread(
            target=self._proactive_management_loop, args=(interval,), daemon=True
        )
        self.proactive_thread.start()
        logger.info(f"Started proactive memory management with interval {interval}s")

    def stop_proactive_management(self):
        """Stop proactive memory management."""
        if self.proactive_thread is not None:
            self.stop_proactive_thread.set()
            self.proactive_thread.join(
                timeout=2.0
            )  # Wait up to 2 seconds for thread to finish
            logger.info("Stopped proactive memory management")

    def _proactive_management_loop(self, interval: float):
        """
        Background loop for proactive memory management based on predictions.

        Args:
            interval: Time interval (in seconds) between checks
        """
        while not self.stop_proactive_thread.is_set():
            try:
                # Check if we should proactively manage memory
                self._perform_proactive_management()

                # Wait for the specified interval or until stop signal
                if self.stop_proactive_thread.wait(timeout=interval):
                    break  # Stop signal received
            except Exception as e:
                logger.error(f"Error in proactive management loop: {e}")
                # Continue loop despite error

    def _perform_proactive_management(self):
        """Perform proactive memory management based on predictions."""
        with self.lock:
            current_time = time.time()

            # Predict future memory usage
            future_time = current_time + self.prediction_horizon
            predicted_memory = self.memory_predictor.predict_future_memory(future_time)
            total_memory = self.get_total_memory()
            predicted_usage_ratio = predicted_memory / total_memory

            # If we predict high memory usage, consider pre-emptively offloading
            if predicted_usage_ratio > self.max_memory_ratio * 0.9:  # 90% of threshold
                logger.info(
                    f"Predicted high memory usage ({predicted_usage_ratio:.2%}), performing proactive offloads"
                )

                # Identify pages that are unlikely to be accessed soon
                pages_to_offload = []
                for page_id in self.ram_pages:
                    if not self.pages[page_id].pinned:  # Don't offload pinned pages
                        access_score = self.access_analyzer.get_access_score(
                            page_id, current_time
                        )

                        # If access score is low (meaning not likely to be accessed soon), consider for offloading
                        if (
                            access_score < 0.3
                        ):  # Threshold for "not likely to be accessed soon"
                            pages_to_offload.append((page_id, access_score))

                # Sort by access score (ascending - lowest scores first)
                pages_to_offload.sort(key=lambda x: x[1])

                # Offload pages until we reach a safe memory level
                for page_id, access_score in pages_to_offload:
                    if (
                        self.is_memory_pressure_high()
                    ):  # Re-check actual memory pressure
                        if self.offload_page_to_disk(page_id):
                            logger.debug(
                                f"Proactively offloaded page {page_id} (access_score: {access_score:.2f})"
                            )
                        else:
                            logger.warning(
                                f"Failed to proactively offload page {page_id}"
                            )
                    else:
                        break  # Memory pressure is under control, stop offloading

    def cleanup(self):
        """Clean up all resources."""
        with self.lock:
            # Stop proactive management first
            self.stop_proactive_management()

            # Deallocate all pages
            pages_to_delete = list(self.pages.keys())
            for page_id in pages_to_delete:
                self.deallocate_page(page_id)

            # Delete offload directory if it's a temp directory
            try:
                shutil.rmtree(self.offload_directory)
            except Exception as e:
                logger.error(
                    f"Failed to clean up offload directory {self.offload_directory}: {e}"
                )


class TensorOffloadingManager:
    """
    Manages tensor offloading for large models, allowing parts of models to be moved
    between RAM and disk as needed.
    """

    def __init__(self, disk_offloader: DiskOffloader):
        self.disk_offloader = disk_offloader
        self.tensor_mappings: Dict[str, str] = {}  # tensor_id -> page_id
        self.page_mappings: Dict[str, str] = {}  # page_id -> tensor_id
        self.component_types: Dict[str, ModelComponentType] = (
            {}
        )  # tensor_id -> component type

    def offload_tensor(
        self,
        tensor: Tensor,
        tensor_id: str,
        priority: OffloadPriority = OffloadPriority.MEDIUM,
        access_pattern: AccessPattern = AccessPattern.RANDOM,
        component_type: ModelComponentType = ModelComponentType.OTHER,
    ) -> bool:
        """
        Offload a tensor to the disk offloader.

        Args:
            tensor: The tensor to offload
            tensor_id: Unique identifier for the tensor
            priority: Priority level for the tensor
            access_pattern: Expected access pattern for the tensor
            component_type: Type of model component for specialized handling

        Returns:
            True if offloading was successful, False otherwise
        """
        page_id = f"tensor_{tensor_id}_{id(tensor)}"

        success = self.disk_offloader.allocate_page(
            tensor, page_id, priority, access_pattern, component_type
        )
        if success:
            self.tensor_mappings[tensor_id] = page_id
            self.page_mappings[page_id] = tensor_id
            self.component_types[tensor_id] = component_type

        return success

    def offload_model_components(
        self,
        model: nn.Module,
        component_filter: Optional[List[ModelComponentType]] = None,
        priority: OffloadPriority = OffloadPriority.MEDIUM,
    ) -> Dict[str, bool]:
        """
        Offload specific types of model components to disk.

        Args:
            model: PyTorch model to offload components from
            component_filter: List of component types to offload (None means all)
            priority: Priority level for offloaded components

        Returns:
            Dictionary mapping component names to success status
        """
        results = {}

        for name, param in model.named_parameters():
            # Determine component type based on name
            comp_type = self._infer_component_type(name)

            # Skip if filter is specified and component type doesn't match
            if component_filter and comp_type not in component_filter:
                continue

            # Create tensor ID based on parameter name
            tensor_id = f"param_{name.replace('.', '_')}"

            # Determine access pattern based on parameter role
            access_pattern = self._infer_access_pattern(name)

            success = self.offload_tensor(
                param.data,
                tensor_id,
                priority=priority,
                access_pattern=access_pattern,
                component_type=comp_type,
            )

            results[name] = success

        for name, buffer in model.named_buffers():
            # Determine component type based on name
            comp_type = self._infer_component_type(name)

            # Skip if filter is specified and component type doesn't match
            if component_filter and comp_type not in component_filter:
                continue

            # Create tensor ID based on buffer name
            tensor_id = f"buffer_{name.replace('.', '_')}"

            # Determine access pattern based on parameter role
            access_pattern = self._infer_access_pattern(name)

            success = self.offload_tensor(
                buffer.data,
                tensor_id,
                priority=priority,
                access_pattern=access_pattern,
                component_type=comp_type,
            )

            results[f"buffer_{name}"] = success

        return results

    def _infer_component_type(self, name: str) -> ModelComponentType:
        """
        Infer the component type based on parameter/buffer name.

        Args:
            name: Name of the parameter or buffer

        Returns:
            Inferred component type
        """
        name_lower = name.lower()

        if "attn" in name_lower or "attention" in name_lower:
            return ModelComponentType.ATTENTION_WEIGHTS
        elif "mlp" in name_lower or "ffn" in name_lower or "feed_forward" in name_lower:
            return ModelComponentType.MLP_WEIGHTS
        elif "embed" in name_lower or "wte" in name_lower or "wpe" in name_lower:
            return ModelComponentType.EMBEDDINGS
        elif "kv_cache" in name_lower or "past_key_values" in name_lower:
            return ModelComponentType.KV_CACHE
        elif "grad" in name_lower:
            return ModelComponentType.GRADIENTS
        else:
            return ModelComponentType.OTHER

    def _infer_access_pattern(self, name: str) -> AccessPattern:
        """
        Infer the access pattern based on parameter/buffer name.

        Args:
            name: Name of the parameter or buffer

        Returns:
            Inferred access pattern
        """
        name_lower = name.lower()

        if "attn" in name_lower or "attention" in name_lower:
            return AccessPattern.FREQUENT
        elif "embed" in name_lower:
            return AccessPattern.SEQUENTIAL
        elif "kv_cache" in name_lower:
            return AccessPattern.TEMPORARY
        else:
            return AccessPattern.RANDOM

    def start_proactive_management(self, interval: float = 5.0):
        """
        Start proactive memory management for tensor offloading.

        Args:
            interval: Time interval (in seconds) between checks
        """
        self.disk_offloader.start_proactive_management(interval)

    def stop_proactive_management(self):
        """Stop proactive memory management for tensor offloading."""
        self.disk_offloader.stop_proactive_management()

    def unoffload_tensor(self, tensor_id: str) -> bool:
        """
        Remove a tensor from offloading management.

        Args:
            tensor_id: ID of the tensor to unoffload

        Returns:
            True if unoffloading was successful, False otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return False

        page_id = self.tensor_mappings[tensor_id]
        success = self.disk_offloader.deallocate_page(page_id)

        if success:
            del self.tensor_mappings[tensor_id]
            del self.page_mappings[page_id]

        return success

    def access_tensor(self, tensor_id: str) -> Optional[Tensor]:
        """
        Access an offloaded tensor, ensuring it's in RAM.

        Args:
            tensor_id: ID of the tensor to access

        Returns:
            The tensor if successful, None otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return None

        page_id = self.tensor_mappings[tensor_id]
        return self.disk_offloader.access_page(page_id)

    def pin_tensor(self, tensor_id: str) -> bool:
        """
        Pin a tensor to prevent it from being offloaded.

        Args:
            tensor_id: ID of the tensor to pin

        Returns:
            True if pinning was successful, False otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return False

        page_id = self.tensor_mappings[tensor_id]
        with self.disk_offloader.lock:
            if page_id in self.disk_offloader.pages:
                self.disk_offloader.pages[page_id].pinned = True
                return True
        return False

    def unpin_tensor(self, tensor_id: str) -> bool:
        """
        Unpin a tensor to allow it to be offloaded.

        Args:
            tensor_id: ID of the tensor to unpin

        Returns:
            True if unpinning was successful, False otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return False

        page_id = self.tensor_mappings[tensor_id]
        with self.disk_offloader.lock:
            if page_id in self.disk_offloader.pages:
                self.disk_offloader.pages[page_id].pinned = False
                return True
        return False


class MemoryPredictor:
    """Class to handle memory usage prediction using ML algorithms."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.memory_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)

        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.model = LinearRegression()
            self.is_trained = False
        else:
            # Fallback to simple moving average if sklearn not available
            self.scaler = None
            self.model = None
            self.is_trained = False

    def record_memory_usage(self, timestamp: float, memory_usage: int):
        """Record memory usage at a specific timestamp."""
        self.memory_history.append(memory_usage)
        self.timestamp_history.append(timestamp)

    def predict_future_memory(self, future_timestamp: float) -> int:
        """Predict memory usage at a future timestamp."""
        if len(self.memory_history) < 2:
            return 0

        # If sklearn is available, use ML-based prediction
        if SKLEARN_AVAILABLE:
            return self._ml_predict_future_memory(future_timestamp)
        else:
            # Fallback to simple trend analysis
            return self._simple_predict_future_memory(future_timestamp)

    def _ml_predict_future_memory(self, future_timestamp: float) -> int:
        """ML-based prediction using sklearn."""
        if len(self.memory_history) < 2:
            return 0

        # Prepare features for prediction
        timestamps = np.array(list(self.timestamp_history)).reshape(-1, 1)
        memory_values = np.array(list(self.memory_history))

        # Train the model if not trained or if we have new data
        if not self.is_trained or len(self.memory_history) > 10:
            try:
                # Scale the features
                scaled_timestamps = self.scaler.fit_transform(timestamps)

                # Fit the model
                self.model.fit(scaled_timestamps, memory_values)
                self.is_trained = True
            except Exception as e:
                logger.warning(f"Could not train prediction model: {e}")
                return max(memory_values)  # Return max observed if training fails

        # Predict for the future timestamp
        try:
            future_scaled = self.scaler.transform([[future_timestamp]])
            predicted_memory = self.model.predict(future_scaled)[0]
            return max(0, int(predicted_memory))  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Could not predict memory usage: {e}")
            return max(memory_values)  # Return max observed if prediction fails

    def _simple_predict_future_memory(self, future_timestamp: float) -> int:
        """Simple prediction without sklearn - using trend analysis."""
        if len(self.memory_history) < 2:
            return 0

        # Convert to lists for calculation
        timestamps = list(self.timestamp_history)
        memory_values = list(self.memory_history)

        # Calculate simple linear trend
        n = len(timestamps)
        if n < 2:
            return 0

        # Calculate slope using first and last points
        time_diff = timestamps[-1] - timestamps[0]
        mem_diff = memory_values[-1] - memory_values[0]

        if time_diff == 0:
            return memory_values[-1]  # No time difference, return last value

        # Calculate rate of change
        rate_of_change = mem_diff / time_diff

        # Predict based on the trend
        time_to_future = future_timestamp - timestamps[-1]
        predicted_memory = memory_values[-1] + (rate_of_change * time_to_future)

        return max(0, int(predicted_memory))  # Ensure non-negative


class AccessPatternAnalyzer:
    """Analyze access patterns to predict future page accesses."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.access_history = deque(maxlen=history_size)
        self.page_frequency = defaultdict(int)
        self.page_recency = {}  # Maps page_id to last access time
        self.access_intervals = defaultdict(deque)  # Maps page_id to access intervals
        self.access_patterns = defaultdict(
            lambda: "unknown"
        )  # Maps page_id to access pattern

    def record_access(
        self, page_id: str, timestamp: float, access_pattern: str = "unknown"
    ):
        """Record a page access event."""
        self.access_history.append((timestamp, page_id))
        self.page_frequency[page_id] += 1
        self.page_recency[page_id] = timestamp
        self.access_patterns[page_id] = access_pattern

        # Calculate access intervals for frequently accessed pages
        if page_id in self.access_intervals:
            prev_access = (
                self.access_intervals[page_id][-1]
                if self.access_intervals[page_id]
                else None
            )
            if prev_access is not None:
                interval = timestamp - prev_access
                self.access_intervals[page_id].append(interval)
                if len(self.access_intervals[page_id]) > 10:  # Keep last 10 intervals
                    self.access_intervals[page_id].popleft()
        else:
            self.access_intervals[page_id] = deque([timestamp], maxlen=10)

    def predict_next_access(self, page_id: str, current_time: float) -> float:
        """Predict when a page will be accessed next based on historical patterns."""
        if page_id not in self.access_intervals or not self.access_intervals[page_id]:
            return current_time + 10.0  # Default to 10 seconds if no pattern known

        # Calculate average interval
        intervals = list(self.access_intervals[page_id])
        avg_interval = sum(intervals) / len(intervals) if intervals else 10.0

        # Use the most recent access time
        last_access = self.page_recency.get(page_id, current_time)
        return last_access + avg_interval

    def get_access_score(self, page_id: str, current_time: float) -> float:
        """Calculate a score representing how soon a page will be accessed."""
        # Higher frequency increases score (more likely to be accessed soon)
        freq_score = min(self.page_frequency[page_id] / 10.0, 1.0)  # Normalize

        # Recency affects score (recently accessed pages are more likely to be accessed again)
        last_access = self.page_recency.get(page_id, 0)
        time_since_access = current_time - last_access
        recency_score = max(0, 1.0 - (time_since_access / 60.0))  # Decay over 1 minute

        # Predictive component based on access patterns
        predicted_next = self.predict_next_access(page_id, current_time)
        time_to_next = max(0.1, predicted_next - current_time)  # Avoid division by zero
        prediction_score = 1.0 / (
            time_to_next + 1.0
        )  # Higher score for sooner predicted access

        # Weighted combination of scores
        return 0.3 * freq_score + 0.3 * recency_score + 0.4 * prediction_score


class ComponentAccessPredictor:
    """Predicts which model components will be accessed next based on usage patterns."""

    def __init__(self):
        self.component_access_history = defaultdict(deque)
        self.component_access_patterns = defaultdict(str)
        self.component_prediction_models = {}

    def record_component_access(
        self, component_id: str, access_time: float, pattern: str = "unknown"
    ):
        """Record access to a model component."""
        self.component_access_history[component_id].append(access_time)
        self.component_access_patterns[component_id] = pattern

        # Keep only recent accesses (last 100)
        if len(self.component_access_history[component_id]) > 100:
            self.component_access_history[component_id].popleft()

    def predict_next_component_access(
        self, component_id: str, current_time: float
    ) -> float:
        """Predict when a component will be accessed next."""
        if component_id not in self.component_access_history:
            return current_time + 30.0  # Default to 30 seconds

        accesses = list(self.component_access_history[component_id])
        if len(accesses) < 2:
            return current_time + 30.0

        # Calculate average interval between accesses
        intervals = [accesses[i + 1] - accesses[i] for i in range(len(accesses) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        # Predict next access based on last access and average interval
        last_access = accesses[-1]
        return last_access + avg_interval

    def get_component_priority(self, component_id: str, current_time: float) -> float:
        """Get priority score for a component based on predicted access timing."""
        next_access = self.predict_next_component_access(component_id, current_time)
        time_to_next = max(0.1, next_access - current_time)  # Avoid division by zero

        # Higher priority for components that will be accessed sooner
        return 1.0 / time_to_next


class MultimodalOffloadingManager:
    """
    Specialized offloading manager for multimodal models that handles
    different types of modalities (text, image, audio, etc.) with different strategies.
    """

    def __init__(self, disk_offloader: DiskOffloader):
        self.disk_offloader = disk_offloader
        self.tensor_mappings: Dict[str, str] = {}  # tensor_id -> page_id
        self.page_mappings: Dict[str, str] = {}  # page_id -> tensor_id
        self.modality_types: Dict[str, str] = {}  # tensor_id -> modality type

        # Different strategies for different modalities
        self.modality_strategies = {
            "text": OffloadStrategy.PREDICTIVE,
            "image": OffloadStrategy.ADAPTIVE,
            "audio": OffloadStrategy.LRU,
            "video": OffloadStrategy.CLUSTER_BASED,
            "features": OffloadStrategy.PRIORITY,
        }

    def offload_multimodal_tensor(
        self,
        tensor: Tensor,
        tensor_id: str,
        modality: str = "text",
        priority: OffloadPriority = OffloadPriority.MEDIUM,
    ) -> bool:
        """
        Offload a multimodal tensor with modality-specific strategy.

        Args:
            tensor: The tensor to offload
            tensor_id: Unique identifier for the tensor
            modality: Type of modality ('text', 'image', 'audio', etc.)
            priority: Priority level for the tensor

        Returns:
            True if offloading was successful, False otherwise
        """
        page_id = f"mm_tensor_{modality}_{tensor_id}_{id(tensor)}"

        # Determine access pattern based on modality
        access_pattern = self._get_modality_access_pattern(modality)

        # Determine component type based on modality
        component_type = self._get_modality_component_type(modality)

        success = self.disk_offloader.allocate_page(
            tensor, page_id, priority, access_pattern, component_type
        )

        if success:
            self.tensor_mappings[tensor_id] = page_id
            self.page_mappings[page_id] = tensor_id
            self.modality_types[tensor_id] = modality

            # Update the page's strategy based on modality
            with self.disk_offloader.lock:
                if page_id in self.disk_offloader.pages:
                    page = self.disk_offloader.pages[page_id]
                    # Store modality-specific strategy info
                    page.access_frequency = self._get_modality_frequency(modality)

        return success

    def _get_modality_access_pattern(self, modality: str) -> AccessPattern:
        """Get access pattern for a specific modality."""
        patterns = {
            "text": AccessPattern.FREQUENT,
            "image": AccessPattern.TEMPORARY,
            "audio": AccessPattern.SEQUENTIAL,
            "video": AccessPattern.SEQUENTIAL,
            "features": AccessPattern.RARE,
        }
        return patterns.get(modality, AccessPattern.RANDOM)

    def _get_modality_component_type(self, modality: str) -> ModelComponentType:
        """Get component type for a specific modality."""
        types = {
            "text": ModelComponentType.EMBEDDINGS,
            "image": ModelComponentType.EMBEDDINGS,
            "audio": ModelComponentType.EMBEDDINGS,
            "video": ModelComponentType.EMBEDDINGS,
            "features": ModelComponentType.ACTIVATIONS,
        }
        return types.get(modality, ModelComponentType.OTHER)

    def _get_modality_frequency(self, modality: str) -> float:
        """Get expected access frequency for a specific modality."""
        frequencies = {
            "text": 0.8,  # High frequency
            "image": 0.3,  # Medium frequency
            "audio": 0.4,  # Medium frequency
            "video": 0.6,  # High frequency
            "features": 0.2,  # Low frequency
        }
        return frequencies.get(modality, 0.5)

    def offload_vision_encoder_features(
        self, features: Tensor, feature_id: str
    ) -> bool:
        """Specialized method for offloading vision encoder features."""
        return self.offload_multimodal_tensor(
            features, feature_id, modality="image", priority=OffloadPriority.HIGH
        )

    def offload_audio_features(self, features: Tensor, feature_id: str) -> bool:
        """Specialized method for offloading audio features."""
        return self.offload_multimodal_tensor(
            features, feature_id, modality="audio", priority=OffloadPriority.MEDIUM
        )

    def offload_text_embeddings(self, embeddings: Tensor, embedding_id: str) -> bool:
        """Specialized method for offloading text embeddings."""
        return self.offload_multimodal_tensor(
            embeddings, embedding_id, modality="text", priority=OffloadPriority.HIGH
        )


# Global disk offloader instance
_global_disk_offloader: Optional[DiskOffloader] = None


def get_disk_offloader() -> DiskOffloader:
    """
    Get the global disk offloader instance.

    Returns:
        DiskOffloader instance
    """
    global _global_disk_offloader
    if _global_disk_offloader is None:
        _global_disk_offloader = DiskOffloader()
    return _global_disk_offloader


def create_disk_offloader(
    max_memory_ratio: float = 0.8,
    offload_directory: Optional[str] = None,
    page_size_mb: int = 16,
    eviction_policy: str = "predictive",
    enable_clustering: bool = True,
    cluster_count: int = 5,
    enable_adaptive: bool = True,
) -> DiskOffloader:
    """
    Create a new disk offloader instance.

    Args:
        max_memory_ratio: Maximum ratio of system memory to use
        offload_directory: Directory for offload files
        page_size_mb: Size of memory pages in MB
        eviction_policy: Page eviction policy
        enable_clustering: Whether to enable clustering-based offloading
        cluster_count: Number of clusters for clustering-based offloading
        enable_adaptive: Whether to enable adaptive offloading

    Returns:
        New DiskOffloader instance
    """
    return DiskOffloader(
        max_memory_ratio=max_memory_ratio,
        offload_directory=offload_directory,
        page_size_mb=page_size_mb,
        eviction_policy=eviction_policy,
        enable_clustering=enable_clustering,
        cluster_count=cluster_count,
        enable_adaptive=enable_adaptive,
    )


__all__ = [
    "DiskOffloader",
    "TensorOffloadingManager",
    "MultimodalOffloadingManager",
    "OffloadPriority",
    "OffloadStrategy",
    "ModelComponentType",
    "OffloadPage",
    "AccessPattern",
    "get_disk_offloader",
    "create_disk_offloader",
]
