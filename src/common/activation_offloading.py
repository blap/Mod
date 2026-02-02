"""
Activation Offloading System for Inference-PIO

This module implements an activation offloading system that moves intermediate
activations to disk during inference and reloads them when needed in subsequent steps.
"""

import gc
import hashlib
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
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ActivationPriority(Enum):
    """Priority levels for activation offloading."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ActivationPage:
    """Represents a memory page for activation offloading."""

    id: str
    activation: Optional[Tensor] = None
    device: Optional[str] = None
    size_bytes: int = 0
    priority: ActivationPriority = ActivationPriority.MEDIUM
    last_access_time: float = 0.0
    pinned: bool = False
    file_path: Optional[str] = None  # Path on disk when offloaded
    access_pattern: str = "unknown"  # Pattern of access (sequential, random, etc.)
    predicted_next_access: float = 0.0  # Predicted next access time
    layer_index: int = -1  # Index of the layer this activation belongs to
    sequence_position: int = -1  # Position in the sequence
    is_intermediate: bool = True  # Whether this is an intermediate activation


class ActivationAccessPattern(Enum):
    """Types of access patterns for activations."""

    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORARY = "temporary"
    FREQUENT = "frequent"
    RARE = "rare"
    FORWARD_ONLY = "forward_only"  # Used once in forward pass, then discarded
    BACKWARD_REQUIRED = "backward_required"  # Needed for backward pass


class ActivationOffloader:
    """
    Advanced activation offloading system that manages moving intermediate activations
    between RAM and disk based on predictive algorithms and memory pressure.
    """

    def __init__(
        self,
        max_memory_ratio: float = 0.7,
        offload_directory: Optional[str] = None,
        page_size_mb: int = 8,
        eviction_policy: str = "predictive",
        prediction_horizon: int = 30,
        activation_cache_size: int = 100,
    ):
        """
        Initialize the activation offloading system.

        Args:
            max_memory_ratio: Maximum ratio of system memory to use (0.0 to 1.0)
            offload_directory: Directory for offload files (default: temporary directory)
            page_size_mb: Size of memory pages in MB
            eviction_policy: Page eviction policy ("lru", "fifo", "priority", "predictive")
            prediction_horizon: Time horizon (in seconds) for memory predictions
            activation_cache_size: Number of recently accessed activations to cache
        """
        self.max_memory_ratio = max_memory_ratio
        self.page_size_bytes = page_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.prediction_horizon = prediction_horizon

        # Set up offload directory
        if offload_directory:
            self.offload_directory = Path(offload_directory)
        else:
            self.offload_directory = Path(
                tempfile.mkdtemp(prefix="pio_activation_offload_")
            )

        self.offload_directory.mkdir(parents=True, exist_ok=True)

        # Track activation pages
        self.activations: Dict[str, ActivationPage] = {}
        self.ram_activations: List[str] = []  # Activations currently in RAM
        self.disk_activations: List[str] = []  # Activations currently on disk
        self.access_times: Dict[str, float] = {}  # Last access times for LRU
        self.activation_cache: deque = deque(
            maxlen=activation_cache_size
        )  # Recently accessed activations

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "activations_offloaded": 0,
            "activations_restored": 0,
            "activation_faults": 0,
            "total_activations": 0,
            "ram_activations": 0,
            "disk_activations": 0,
            "peak_memory_used": 0,
            "total_offloaded_bytes": 0,
            "total_restored_bytes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Prediction components
        self.memory_predictor = ActivationMemoryPredictor(window_size=100)
        self.access_analyzer = ActivationAccessPatternAnalyzer(history_size=1000)
        self.activation_predictor = ActivationAccessPredictor()

        # Background thread for proactive offloading
        self.proactive_thread = None
        self.stop_proactive_thread = threading.Event()

        logger.info(
            f"Activation offloading system initialized with max_memory_ratio={max_memory_ratio}, "
            f"offload_directory={self.offload_directory}, page_size={page_size_mb}MB, "
            f"prediction_horizon={prediction_horizon}s"
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

    def allocate_activation(
        self,
        activation: Tensor,
        activation_id: str,
        priority: ActivationPriority = ActivationPriority.MEDIUM,
        access_pattern: ActivationAccessPattern = ActivationAccessPattern.TEMPORARY,
        layer_index: int = -1,
        sequence_position: int = -1,
    ) -> bool:
        """
        Allocate space for an activation tensor.

        Args:
            activation: The activation tensor to store
            activation_id: Unique identifier for the activation
            priority: Priority level for the activation
            access_pattern: Expected access pattern for the activation
            layer_index: Index of the layer this activation belongs to
            sequence_position: Position in the sequence

        Returns:
            True if allocation was successful, False otherwise
        """
        with self.lock:
            if activation_id in self.activations:
                logger.warning(
                    f"Activation {activation_id} already exists, overwriting"
                )
                self.deallocate_activation(activation_id)

            size_bytes = activation.element_size() * activation.nelement()

            activation_page = ActivationPage(
                id=activation_id,
                activation=activation,
                device=str(activation.device),
                size_bytes=size_bytes,
                priority=priority,
                last_access_time=time.time(),
                access_pattern=access_pattern.value,
                layer_index=layer_index,
                sequence_position=sequence_position,
                is_intermediate=True,  # By default, assume it's intermediate
            )

            self.activations[activation_id] = activation_page
            self.ram_activations.append(activation_id)
            self.access_times[activation_id] = activation_page.last_access_time

            self.stats["total_activations"] += 1
            self.stats["ram_activations"] += 1

            # Update predictions
            self._update_predictions()

            # Record access pattern for prediction
            self.access_analyzer.record_access(
                activation_id, time.time(), access_pattern.value
            )

            # Check if we need to offload activations due to memory pressure
            self._handle_memory_pressure()

            logger.debug(
                f"Allocated activation {activation_id} ({size_bytes} bytes) in RAM"
            )
            return True

    def deallocate_activation(self, activation_id: str) -> bool:
        """
        Deallocate an activation.

        Args:
            activation_id: ID of the activation to deallocate

        Returns:
            True if deallocation was successful, False otherwise
        """
        with self.lock:
            if activation_id not in self.activations:
                logger.warning(f"Activation {activation_id} does not exist")
                return False

            activation_page = self.activations[activation_id]

            # Remove from appropriate list
            if activation_id in self.ram_activations:
                self.ram_activations.remove(activation_id)
                self.stats["ram_activations"] -= 1
            elif activation_id in self.disk_activations:
                self.disk_activations.remove(activation_id)
                self.stats["disk_activations"] -= 1

                # Delete the offload file if it exists
                if activation_page.file_path and os.path.exists(
                    activation_page.file_path
                ):
                    try:
                        os.remove(activation_page.file_path)
                    except OSError as e:
                        logger.error(
                            f"Failed to delete offload file {activation_page.file_path}: {e}"
                        )

            # Remove from tracking
            if activation_id in self.access_times:
                del self.access_times[activation_id]

            # Remove from cache if present
            if activation_id in self.activation_cache:
                self.activation_cache.remove(activation_id)

            # Delete the activation
            del self.activations[activation_id]

            logger.debug(f"Deallocated activation {activation_id}")
            return True

    def offload_activation_to_disk(self, activation_id: str) -> bool:
        """
        Offload an activation from RAM to disk.

        Args:
            activation_id: ID of the activation to offload

        Returns:
            True if offload was successful, False otherwise
        """
        with self.lock:
            if activation_id not in self.activations:
                logger.error(f"Activation {activation_id} does not exist")
                return False

            activation_page = self.activations[activation_id]

            if activation_id not in self.ram_activations:
                logger.warning(
                    f"Activation {activation_id} is not in RAM, nothing to offload"
                )
                return True  # Already on disk or invalid state

            # Create offload file path
            offload_file = self.offload_directory / f"activation_{activation_id}.pkl"

            try:
                # Save activation to disk
                with open(offload_file, "wb") as f:
                    pickle.dump(
                        {
                            "activation": (
                                activation_page.activation.cpu()
                                if activation_page.activation is not None
                                else None
                            ),
                            "device": activation_page.device,
                        },
                        f,
                    )

                # Update activation info
                activation_page.file_path = str(offload_file)
                activation_page.activation = None  # Free RAM

                # Move from RAM to disk tracking
                self.ram_activations.remove(activation_id)
                self.disk_activations.append(activation_id)

                self.stats["activations_offloaded"] += 1
                self.stats["ram_activations"] -= 1
                self.stats["disk_activations"] += 1
                self.stats["total_offloaded_bytes"] += activation_page.size_bytes

                logger.debug(
                    f"Offloaded activation {activation_id} to disk: {offload_file}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to offload activation {activation_id} to disk: {e}"
                )
                return False

    def restore_activation_to_ram(self, activation_id: str) -> bool:
        """
        Restore an activation from disk to RAM.

        Args:
            activation_id: ID of the activation to restore

        Returns:
            True if restore was successful, False otherwise
        """
        with self.lock:
            if activation_id not in self.activations:
                logger.error(f"Activation {activation_id} does not exist")
                return False

            activation_page = self.activations[activation_id]

            if activation_id not in self.disk_activations:
                logger.warning(
                    f"Activation {activation_id} is not on disk, nothing to restore"
                )
                return True  # Already in RAM or invalid state

            if not activation_page.file_path or not os.path.exists(
                activation_page.file_path
            ):
                logger.error(
                    f"Offload file for activation {activation_id} does not exist: {activation_page.file_path}"
                )
                return False

            try:
                # Load activation from disk
                with open(activation_page.file_path, "rb") as f:
                    data = pickle.load(f)
                    activation = data["activation"]
                    original_device = data["device"]

                # Move activation to the appropriate device
                if activation is not None and original_device:
                    device = torch.device(original_device)
                    activation = activation.to(device)

                # Update activation info
                activation_page.activation = activation
                activation_page.device = original_device

                # Move from disk to RAM tracking
                self.disk_activations.remove(activation_id)
                self.ram_activations.append(activation_id)

                self.stats["activations_restored"] += 1
                self.stats["ram_activations"] += 1
                self.stats["disk_activations"] -= 1
                self.stats["total_restored_bytes"] += activation_page.size_bytes

                logger.debug(
                    f"Restored activation {activation_id} to RAM from: {activation_page.file_path}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to restore activation {activation_id} to RAM: {e}"
                )
                return False

    def access_activation(self, activation_id: str) -> Optional[Tensor]:
        """
        Access an activation, ensuring it's in RAM. This may trigger offloads/restores.

        Args:
            activation_id: ID of the activation to access

        Returns:
            The activation tensor if successful, None otherwise
        """
        with self.lock:
            # Check cache first
            if activation_id in self.activation_cache:
                self.stats["cache_hits"] += 1
                # Move to end to show it's most recently used
                self.activation_cache.remove(activation_id)
                self.activation_cache.append(activation_id)

                if activation_id in self.activations:
                    return self.activations[activation_id].activation
                else:
                    return None

            self.stats["cache_misses"] += 1

            if activation_id not in self.activations:
                logger.error(f"Activation {activation_id} does not exist")
                return None

            activation_page = self.activations[activation_id]
            current_time = time.time()
            activation_page.last_access_time = current_time
            self.access_times[activation_id] = activation_page.last_access_time

            # Record access for prediction
            self.access_analyzer.record_access(
                activation_id, current_time, activation_page.access_pattern
            )

            # If activation is on disk, restore it to RAM
            if activation_id in self.disk_activations:
                self.stats["activation_faults"] += 1
                if not self.restore_activation_to_ram(activation_id):
                    logger.error(f"Failed to restore activation {activation_id} to RAM")
                    return None

            # Add to cache
            if activation_id not in self.activation_cache:
                self.activation_cache.append(activation_id)

            # Update peak memory usage
            current_usage = self.get_current_memory_usage()
            if current_usage > self.stats["peak_memory_used"]:
                self.stats["peak_memory_used"] = current_usage

            return activation_page.activation

    def _handle_memory_pressure(self):
        """Handle memory pressure by offloading activations if needed."""
        if not self.is_memory_pressure_high():
            return

        # Sort activations by eviction criteria
        activations_to_consider = []
        for activation_id in self.ram_activations:
            activation_page = self.activations[activation_id]
            activations_to_consider.append((activation_id, activation_page))

        # Sort based on eviction policy
        if self.eviction_policy == "lru":
            activations_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif self.eviction_policy == "fifo":
            # For FIFO, we'd need to track insertion order, so we'll use access time as proxy
            activations_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif self.eviction_policy == "priority":
            # Evict lowest priority activations first
            activations_to_consider.sort(key=lambda x: x[1].priority.value)
        elif self.eviction_policy == "predictive":
            # Use predictive algorithm to determine which activations to offload
            current_time = time.time()
            activations_to_consider.sort(
                key=lambda x: self._calculate_prediction_score(x[0], current_time)
            )
        else:
            # Default to LRU
            activations_to_consider.sort(key=lambda x: self.access_times[x[0]])

        # Offload activations until memory pressure is relieved
        # Limit the number of iterations to prevent infinite loops
        max_offloads_per_call = min(
            len(activations_to_consider), 10
        )  # Limit to 10 offloads per call
        offloads_performed = 0

        for activation_id, activation_page in activations_to_consider:
            if (
                not self.is_memory_pressure_high()
                or offloads_performed >= max_offloads_per_call
            ):
                break

            # Don't offload pinned activations
            if activation_page.pinned:
                continue

            # Don't offload activations marked as critical
            if activation_page.priority == ActivationPriority.CRITICAL:
                continue

            if self.offload_activation_to_disk(activation_id):
                logger.debug(
                    f"Evicted activation {activation_id} due to memory pressure"
                )
                offloads_performed += 1
            else:
                logger.warning(
                    f"Failed to evict activation {activation_id} due to memory pressure"
                )

    def _calculate_prediction_score(
        self, activation_id: str, current_time: float
    ) -> float:
        """
        Calculate a score for predictive eviction. Lower scores mean higher likelihood of eviction.

        Args:
            activation_id: ID of the activation to evaluate
            current_time: Current time for prediction

        Returns:
            Score for the activation (lower means more likely to be evicted)
        """
        # Get access score from the analyzer
        access_score = self.access_analyzer.get_access_score(
            activation_id, current_time
        )

        # Get the activation object
        activation_page = self.activations[activation_id]

        # Combine factors:
        # - Higher access score means less likely to be evicted (higher score = lower eviction probability)
        # - Higher priority means less likely to be evicted
        # - Larger size might make it more likely to be evicted under memory pressure
        priority_factor = (
            activation_page.priority.value
        )  # Higher priority = less likely to be evicted
        size_factor = activation_page.size_bytes / (
            1024 * 1024
        )  # Size in MB, larger = more likely to be evicted

        # Calculate final score (lower score = more likely to be evicted)
        # Access score is inverted because high access score means keep in memory
        final_score = -access_score + (5 - priority_factor) + (size_factor / 100.0)

        return final_score

    def get_activation_stats(self) -> Dict[str, Any]:
        """Get statistics about activation pages."""
        with self.lock:
            total_size = sum(
                activation.size_bytes for activation in self.activations.values()
            )
            ram_size = sum(
                self.activations[aid].size_bytes for aid in self.ram_activations
            )
            disk_size = sum(
                self.activations[aid].size_bytes for aid in self.disk_activations
            )

            return {
                "total_activations": len(self.activations),
                "ram_activations": len(self.ram_activations),
                "disk_activations": len(self.disk_activations),
                "total_size_bytes": total_size,
                "ram_size_bytes": ram_size,
                "disk_size_bytes": disk_size,
                "cache_size": len(self.activation_cache),
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]),
                "stats": self.stats.copy(),
            }

    def start_proactive_management(self, interval: float = 5.0):
        """
        Start proactive memory management in a background thread.

        Args:
            interval: Time interval (in seconds) between checks
        """
        if self.proactive_thread is not None and self.proactive_thread.is_alive():
            logger.warning("Proactive activation management already running")
            return

        self.stop_proactive_thread.clear()
        self.proactive_thread = threading.Thread(
            target=self._proactive_management_loop, args=(interval,), daemon=True
        )
        self.proactive_thread.start()
        logger.info(
            f"Started proactive activation management with interval {interval}s"
        )

    def stop_proactive_management(self):
        """Stop proactive memory management."""
        if self.proactive_thread is not None:
            self.stop_proactive_thread.set()
            self.proactive_thread.join(
                timeout=2.0
            )  # Wait up to 2 seconds for thread to finish
            logger.info("Stopped proactive activation management")

    def _proactive_management_loop(self, interval: float):
        """
        Background loop for proactive activation management based on predictions.

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
        """Perform proactive activation management based on predictions."""
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

                # Identify activations that are unlikely to be accessed soon
                activations_to_offload = []
                for activation_id in self.ram_activations:
                    if not self.activations[
                        activation_id
                    ].pinned:  # Don't offload pinned activations
                        access_score = self.access_analyzer.get_access_score(
                            activation_id, current_time
                        )

                        # If access score is low (meaning not likely to be accessed soon), consider for offloading
                        if (
                            access_score < 0.3
                        ):  # Threshold for "not likely to be accessed soon"
                            activations_to_offload.append((activation_id, access_score))

                # Sort by access score (ascending - lowest scores first)
                activations_to_offload.sort(key=lambda x: x[1])

                # Offload activations until we reach a safe memory level
                for activation_id, access_score in activations_to_offload:
                    if (
                        self.is_memory_pressure_high()
                    ):  # Re-check actual memory pressure
                        if self.offload_activation_to_disk(activation_id):
                            logger.debug(
                                f"Proactively offloaded activation {activation_id} (access_score: {access_score:.2f})"
                            )
                        else:
                            logger.warning(
                                f"Failed to proactively offload activation {activation_id}"
                            )
                    else:
                        break  # Memory pressure is under control, stop offloading

    def cleanup(self):
        """Clean up all resources."""
        with self.lock:
            # Stop proactive management first
            self.stop_proactive_management()

            # Deallocate all activations
            activations_to_delete = list(self.activations.keys())
            for activation_id in activations_to_delete:
                self.deallocate_activation(activation_id)

            # Delete offload directory if it's a temp directory
            try:
                shutil.rmtree(self.offload_directory)
            except Exception as e:
                logger.error(
                    f"Failed to clean up offload directory {self.offload_directory}: {e}"
                )


class ActivationOffloadingManager:
    """
    Manages activation offloading for large models, allowing intermediate activations
    to be moved between RAM and disk as needed during inference.
    """

    def __init__(self, activation_offloader: ActivationOffloader):
        self.activation_offloader = activation_offloader
        self.activation_mappings: Dict[str, str] = {}  # activation_id -> page_id
        self.page_mappings: Dict[str, str] = {}  # page_id -> activation_id

    def offload_activation(
        self,
        activation: Tensor,
        activation_id: str,
        priority: ActivationPriority = ActivationPriority.MEDIUM,
        access_pattern: ActivationAccessPattern = ActivationAccessPattern.TEMPORARY,
        layer_index: int = -1,
        sequence_position: int = -1,
    ) -> bool:
        """
        Offload an activation to the disk offloader.

        Args:
            activation: The activation tensor to offload
            activation_id: Unique identifier for the activation
            priority: Priority level for the activation
            access_pattern: Expected access pattern for the activation
            layer_index: Index of the layer this activation belongs to
            sequence_position: Position in the sequence

        Returns:
            True if offloading was successful, False otherwise
        """
        page_id = f"activation_{activation_id}_{id(activation)}"

        success = self.activation_offloader.allocate_activation(
            activation,
            page_id,
            priority,
            access_pattern,
            layer_index,
            sequence_position,
        )
        if success:
            self.activation_mappings[activation_id] = page_id
            self.page_mappings[page_id] = activation_id

        return success

    def start_proactive_management(self, interval: float = 5.0):
        """
        Start proactive memory management for activation offloading.

        Args:
            interval: Time interval (in seconds) between checks
        """
        self.activation_offloader.start_proactive_management(interval)

    def stop_proactive_management(self):
        """Stop proactive memory management for activation offloading."""
        self.activation_offloader.stop_proactive_management()

    def unoffload_activation(self, activation_id: str) -> bool:
        """
        Remove an activation from offloading management.

        Args:
            activation_id: ID of the activation to unoffload

        Returns:
            True if unoffloading was successful, False otherwise
        """
        if activation_id not in self.activation_mappings:
            return False

        page_id = self.activation_mappings[activation_id]
        success = self.activation_offloader.deallocate_activation(page_id)

        if success:
            del self.activation_mappings[activation_id]
            del self.page_mappings[page_id]

        return success

    def access_activation(self, activation_id: str) -> Optional[Tensor]:
        """
        Access an offloaded activation, ensuring it's in RAM.

        Args:
            activation_id: ID of the activation to access

        Returns:
            The activation tensor if successful, None otherwise
        """
        if activation_id not in self.activation_mappings:
            return None

        page_id = self.activation_mappings[activation_id]
        return self.activation_offloader.access_activation(page_id)

    def pin_activation(self, activation_id: str) -> bool:
        """
        Pin an activation to prevent it from being offloaded.

        Args:
            activation_id: ID of the activation to pin

        Returns:
            True if pinning was successful, False otherwise
        """
        if activation_id not in self.activation_mappings:
            return False

        page_id = self.activation_mappings[activation_id]
        with self.activation_offloader.lock:
            if page_id in self.activation_offloader.activations:
                self.activation_offloader.activations[page_id].pinned = True
                return True
        return False

    def unpin_activation(self, activation_id: str) -> bool:
        """
        Unpin an activation to allow it to be offloaded.

        Args:
            activation_id: ID of the activation to unpin

        Returns:
            True if unpinning was successful, False otherwise
        """
        if activation_id not in self.activation_mappings:
            return False

        page_id = self.activation_mappings[activation_id]
        with self.activation_offloader.lock:
            if page_id in self.activation_offloader.activations:
                self.activation_offloader.activations[page_id].pinned = False
                return True
        return False

    def get_activation_priority(self, activation_id: str) -> ActivationPriority:
        """
        Get the priority of an activation.

        Args:
            activation_id: ID of the activation

        Returns:
            Priority level of the activation
        """
        if activation_id not in self.activation_mappings:
            return ActivationPriority.MEDIUM

        page_id = self.activation_mappings[activation_id]
        with self.activation_offloader.lock:
            if page_id in self.activation_offloader.activations:
                return self.activation_offloader.activations[page_id].priority
        return ActivationPriority.MEDIUM

    def set_activation_priority(
        self, activation_id: str, priority: ActivationPriority
    ) -> bool:
        """
        Set the priority of an activation.

        Args:
            activation_id: ID of the activation
            priority: New priority level

        Returns:
            True if priority was set successfully, False otherwise
        """
        if activation_id not in self.activation_mappings:
            return False

        page_id = self.activation_mappings[activation_id]
        with self.activation_offloader.lock:
            if page_id in self.activation_offloader.activations:
                self.activation_offloader.activations[page_id].priority = priority
                return True
        return False


class ActivationMemoryPredictor:
    """Class to handle activation memory usage prediction using ML algorithms."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.memory_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)

    def record_memory_usage(self, timestamp: float, memory_usage: int):
        """Record memory usage at a specific timestamp."""
        self.memory_history.append(memory_usage)
        self.timestamp_history.append(timestamp)

    def predict_future_memory(self, future_timestamp: float) -> int:
        """Predict memory usage at a future timestamp."""
        if len(self.memory_history) < 2:
            return 0

        # Calculate simple linear trend
        timestamps = list(self.timestamp_history)
        memory_values = list(self.memory_history)

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


class ActivationAccessPatternAnalyzer:
    """Analyze activation access patterns to predict future accesses."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.access_history = deque(maxlen=history_size)
        self.activation_frequency = defaultdict(int)
        self.activation_recency = {}  # Maps activation_id to last access time
        self.access_intervals = defaultdict(
            deque
        )  # Maps activation_id to access intervals
        self.access_patterns = defaultdict(
            lambda: "unknown"
        )  # Maps activation_id to access pattern

    def record_access(
        self, activation_id: str, timestamp: float, access_pattern: str = "unknown"
    ):
        """Record an activation access event."""
        self.access_history.append((timestamp, activation_id))
        self.activation_frequency[activation_id] += 1
        self.activation_recency[activation_id] = timestamp
        self.access_patterns[activation_id] = access_pattern

        # Calculate access intervals for frequently accessed activations
        if activation_id in self.access_intervals:
            prev_access = (
                self.access_intervals[activation_id][-1]
                if self.access_intervals[activation_id]
                else None
            )
            if prev_access is not None:
                interval = timestamp - prev_access
                self.access_intervals[activation_id].append(interval)
                if (
                    len(self.access_intervals[activation_id]) > 10
                ):  # Keep last 10 intervals
                    self.access_intervals[activation_id].popleft()
        else:
            self.access_intervals[activation_id] = deque([timestamp], maxlen=10)

    def predict_next_access(self, activation_id: str, current_time: float) -> float:
        """Predict when an activation will be accessed next based on historical patterns."""
        if (
            activation_id not in self.access_intervals
            or not self.access_intervals[activation_id]
        ):
            return current_time + 10.0  # Default to 10 seconds if no pattern known

        # Calculate average interval
        intervals = list(self.access_intervals[activation_id])
        avg_interval = sum(intervals) / len(intervals) if intervals else 10.0

        # Use the most recent access time
        last_access = self.activation_recency.get(activation_id, current_time)
        return last_access + avg_interval

    def get_access_score(self, activation_id: str, current_time: float) -> float:
        """Calculate a score representing how soon an activation will be accessed."""
        # Higher frequency increases score (more likely to be accessed soon)
        freq_score = min(
            self.activation_frequency[activation_id] / 10.0, 1.0
        )  # Normalize

        # Recency affects score (recently accessed activations are more likely to be accessed again)
        last_access = self.activation_recency.get(activation_id, 0)
        time_since_access = current_time - last_access
        recency_score = max(0, 1.0 - (time_since_access / 60.0))  # Decay over 1 minute

        # Predictive component based on access patterns
        predicted_next = self.predict_next_access(activation_id, current_time)
        time_to_next = max(0.1, predicted_next - current_time)  # Avoid division by zero
        prediction_score = 1.0 / (
            time_to_next + 1.0
        )  # Higher score for sooner predicted access

        # Weighted combination of scores
        return 0.3 * freq_score + 0.3 * recency_score + 0.4 * prediction_score


class ActivationAccessPredictor:
    """Predicts which activations will be accessed next based on usage patterns."""

    def __init__(self):
        self.activation_access_history = defaultdict(deque)
        self.activation_access_patterns = defaultdict(str)
        self.activation_prediction_models = {}

    def record_activation_access(
        self, activation_id: str, access_time: float, pattern: str = "unknown"
    ):
        """Record access to an activation."""
        self.activation_access_history[activation_id].append(access_time)
        self.activation_access_patterns[activation_id] = pattern

        # Keep only recent accesses (last 100)
        if len(self.activation_access_history[activation_id]) > 100:
            self.activation_access_history[activation_id].popleft()

    def predict_next_activation_access(
        self, activation_id: str, current_time: float
    ) -> float:
        """Predict when an activation will be accessed next."""
        if activation_id not in self.activation_access_history:
            return current_time + 30.0  # Default to 30 seconds

        accesses = list(self.activation_access_history[activation_id])
        if len(accesses) < 2:
            return current_time + 30.0

        # Calculate average interval between accesses
        intervals = [accesses[i + 1] - accesses[i] for i in range(len(accesses) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        # Predict next access based on last access and average interval
        last_access = accesses[-1]
        return last_access + avg_interval

    def get_activation_priority(self, activation_id: str, current_time: float) -> float:
        """Get priority score for an activation based on predicted access timing."""
        next_access = self.predict_next_activation_access(activation_id, current_time)
        time_to_next = max(0.1, next_access - current_time)  # Avoid division by zero

        # Higher priority for activations that will be accessed sooner
        return 1.0 / time_to_next


# Global activation offloader instance
_global_activation_offloader: Optional[ActivationOffloader] = None


def get_activation_offloader() -> ActivationOffloader:
    """
    Get the global activation offloader instance.

    Returns:
        ActivationOffloader instance
    """
    global _global_activation_offloader
    if _global_activation_offloader is None:
        _global_activation_offloader = ActivationOffloader()
    return _global_activation_offloader


def create_activation_offloader(
    max_memory_ratio: float = 0.7,
    offload_directory: Optional[str] = None,
    page_size_mb: int = 8,
    eviction_policy: str = "predictive",
) -> ActivationOffloader:
    """
    Create a new activation offloader instance.

    Args:
        max_memory_ratio: Maximum ratio of system memory to use
        offload_directory: Directory for offload files
        page_size_mb: Size of memory pages in MB
        eviction_policy: Page eviction policy

    Returns:
        New ActivationOffloader instance
    """
    return ActivationOffloader(
        max_memory_ratio=max_memory_ratio,
        offload_directory=offload_directory,
        page_size_mb=page_size_mb,
        eviction_policy=eviction_policy,
    )


__all__ = [
    "ActivationOffloader",
    "ActivationOffloadingManager",
    "ActivationPriority",
    "ActivationPage",
    "ActivationAccessPattern",
    "get_activation_offloader",
    "create_activation_offloader",
]
