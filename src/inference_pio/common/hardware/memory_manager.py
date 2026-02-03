"""
Memory Manager for Smart Swap and Tensor Paging System

This module implements a sophisticated memory management system that handles
swapping and paging operations for large language models, enabling efficient
use of both RAM and disk storage.
"""

import gc
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch import Tensor

# Import sklearn modules with fallback
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None
    StandardScaler = None


logger = logging.getLogger(__name__)


class MemoryPriority(Enum):
    """Priority levels for memory management."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryPage:
    """Represents a memory page for tensor paging."""

    id: str
    tensor: Optional[Tensor] = None
    device: Optional[str] = None
    size_bytes: int = 0
    priority: MemoryPriority = MemoryPriority.MEDIUM
    last_access_time: float = 0.0
    pinned: bool = False
    file_path: Optional[str] = None  # Path on disk when swapped out


class MemoryPrediction:
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

    def record_access(self, page_id: str, timestamp: float):
        """Record a page access event."""
        self.access_history.append((timestamp, page_id))
        self.page_frequency[page_id] += 1
        self.page_recency[page_id] = timestamp

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


class MemoryManager:
    """
    Advanced memory manager that handles swapping and paging operations
    for large language models, optimizing memory usage between RAM and disk.
    """

    def __init__(
        self,
        max_memory_ratio: float = 0.8,
        swap_directory: Optional[str] = None,
        page_size_mb: int = 16,
        eviction_policy: str = "lru",
        prediction_horizon: int = 30,
    ):
        """
        Initialize the memory manager.

        Args:
            max_memory_ratio: Maximum ratio of system memory to use (0.0 to 1.0)
            swap_directory: Directory for swap files (default: temporary directory)
            page_size_mb: Size of memory pages in MB
            eviction_policy: Page eviction policy ("lru", "fifo", "priority", "predictive")
            prediction_horizon: Time horizon (in seconds) for memory predictions
        """
        self.max_memory_ratio = max_memory_ratio
        self.page_size_bytes = page_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.prediction_horizon = prediction_horizon

        # Set up swap directory
        if swap_directory:
            self.swap_directory = Path(swap_directory)
        else:
            self.swap_directory = Path(tempfile.mkdtemp(prefix="pio_swap_"))

        self.swap_directory.mkdir(parents=True, exist_ok=True)

        # Track memory pages
        self.pages: Dict[str, MemoryPage] = {}
        self.ram_pages: List[str] = []  # Pages currently in RAM
        self.disk_pages: List[str] = []  # Pages currently on disk
        self.access_times: Dict[str, float] = {}  # Last access times for LRU

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "pages_swapped_in": 0,
            "pages_swapped_out": 0,
            "page_faults": 0,
            "total_pages": 0,
            "ram_pages": 0,
            "disk_pages": 0,
            "peak_memory_used": 0,
        }

        # Prediction components
        self.memory_predictor = MemoryPrediction(window_size=100)
        self.access_analyzer = AccessPatternAnalyzer(history_size=1000)

        # Background thread for proactive memory management
        self.proactive_thread = None
        self.stop_proactive_thread = threading.Event()

        logger.info(
            f"Memory manager initialized with max_memory_ratio={max_memory_ratio}, "
            f"swap_directory={self.swap_directory}, page_size={page_size_mb}MB, "
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

    def allocate_page(
        self,
        tensor: Tensor,
        page_id: str,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
    ) -> bool:
        """
        Allocate a memory page for a tensor.

        Args:
            tensor: The tensor to store in the page
            page_id: Unique identifier for the page
            priority: Priority level for the page

        Returns:
            True if allocation was successful, False otherwise
        """
        with self.lock:
            if page_id in self.pages:
                logger.warning(f"Page {page_id} already exists, overwriting")
                self.deallocate_page(page_id)

            size_bytes = tensor.element_size() * tensor.nelement()

            page = MemoryPage(
                id=page_id,
                tensor=tensor,
                device=str(tensor.device),
                size_bytes=size_bytes,
                priority=priority,
                last_access_time=time.time(),
            )

            self.pages[page_id] = page
            self.ram_pages.append(page_id)
            self.access_times[page_id] = page.last_access_time

            self.stats["total_pages"] += 1
            self.stats["ram_pages"] += 1

            # Update predictions
            self._update_predictions()

            # Check if we need to evict pages due to memory pressure
            self._handle_memory_pressure()

            logger.debug(f"Allocated page {page_id} ({size_bytes} bytes) in RAM")
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

                # Delete the swap file if it exists
                if page.file_path and os.path.exists(page.file_path):
                    try:
                        os.remove(page.file_path)
                    except OSError as e:
                        logger.error(
                            f"Failed to delete swap file {page.file_path}: {e}"
                        )

            # Remove from tracking
            if page_id in self.access_times:
                del self.access_times[page_id]

            # Delete the page
            del self.pages[page_id]

            logger.debug(f"Deallocated page {page_id}")
            return True

    def swap_page_to_disk(self, page_id: str) -> bool:
        """
        Swap a page from RAM to disk.

        Args:
            page_id: ID of the page to swap

        Returns:
            True if swap was successful, False otherwise
        """
        with self.lock:
            if page_id not in self.pages:
                logger.error(f"Page {page_id} does not exist")
                return False

            page = self.pages[page_id]

            if page_id not in self.ram_pages:
                logger.warning(f"Page {page_id} is not in RAM, nothing to swap")
                return True  # Already on disk or invalid state

            # Create swap file path
            swap_file = self.swap_directory / f"page_{page_id}.pkl"

            try:
                # Save tensor to disk
                with open(swap_file, "wb") as f:
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
                page.file_path = str(swap_file)
                page.tensor = None  # Free RAM

                # Move from RAM to disk tracking
                self.ram_pages.remove(page_id)
                self.disk_pages.append(page_id)

                self.stats["pages_swapped_out"] += 1
                self.stats["ram_pages"] -= 1
                self.stats["disk_pages"] += 1

                logger.debug(f"Swapped page {page_id} to disk: {swap_file}")
                return True

            except Exception as e:
                logger.error(f"Failed to swap page {page_id} to disk: {e}")
                return False

    def swap_page_to_ram(self, page_id: str) -> bool:
        """
        Swap a page from disk to RAM.

        Args:
            page_id: ID of the page to swap

        Returns:
            True if swap was successful, False otherwise
        """
        with self.lock:
            if page_id not in self.pages:
                logger.error(f"Page {page_id} does not exist")
                return False

            page = self.pages[page_id]

            if page_id not in self.disk_pages:
                logger.warning(f"Page {page_id} is not on disk, nothing to swap")
                return True  # Already in RAM or invalid state

            if not page.file_path or not os.path.exists(page.file_path):
                logger.error(
                    f"Swap file for page {page_id} does not exist: {page.file_path}"
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

                self.stats["pages_swapped_in"] += 1
                self.stats["ram_pages"] += 1
                self.stats["disk_pages"] -= 1

                logger.debug(f"Swapped page {page_id} to RAM from: {page.file_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to swap page {page_id} to RAM: {e}")
                return False

    def access_page(self, page_id: str) -> Optional[Tensor]:
        """
        Access a page, ensuring it's in RAM. This may trigger swaps.

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
            self.access_analyzer.record_access(page_id, current_time)

            # If page is on disk, swap it to RAM
            if page_id in self.disk_pages:
                self.stats["page_faults"] += 1
                if not self.swap_page_to_ram(page_id):
                    logger.error(f"Failed to swap page {page_id} to RAM")
                    return None

            # Update peak memory usage
            current_usage = self.get_current_memory_usage()
            if current_usage > self.stats["peak_memory_used"]:
                self.stats["peak_memory_used"] = current_usage

            return page.tensor

    def _handle_memory_pressure(self):
        """Handle memory pressure by swapping out pages if needed."""
        if not self.is_memory_pressure_high():
            return

        # Sort pages by eviction criteria
        pages_to_consider = []
        for page_id in self.ram_pages:
            page = self.pages[page_id]
            pages_to_consider.append((page_id, page))

        # Sort based on eviction policy
        if self.eviction_policy == "lru":
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif self.eviction_policy == "fifo":
            # For FIFO, we'd need to track insertion order, so we'll use access time as proxy
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif self.eviction_policy == "priority":
            # Evict lowest priority pages first
            pages_to_consider.sort(key=lambda x: x[1].priority.value)
        elif self.eviction_policy == "predictive":
            # Use predictive algorithm to determine which pages to evict
            current_time = time.time()
            pages_to_consider.sort(
                key=lambda x: self._calculate_prediction_score(x[0], current_time)
            )
        else:
            # Default to LRU
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])

        # Swap out pages until memory pressure is relieved
        # Limit the number of iterations to prevent infinite loops
        max_swaps_per_call = min(
            len(pages_to_consider), 10
        )  # Limit to 10 swaps per call
        swaps_performed = 0

        for page_id, page in pages_to_consider:
            if (
                not self.is_memory_pressure_high()
                or swaps_performed >= max_swaps_per_call
            ):
                break

            # Don't swap pinned pages
            if page.pinned:
                continue

            if self.swap_page_to_disk(page_id):
                logger.debug(f"Evicted page {page_id} due to memory pressure")
                swaps_performed += 1
            else:
                logger.warning(f"Failed to evict page {page_id} due to memory pressure")

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

            # If we predict high memory usage, consider pre-emptively swapping
            if predicted_usage_ratio > self.max_memory_ratio * 0.9:  # 90% of threshold
                logger.info(
                    f"Predicted high memory usage ({predicted_usage_ratio:.2%}), performing proactive swaps"
                )

                # Identify pages that are unlikely to be accessed soon
                pages_to_swap = []
                for page_id in self.ram_pages:
                    if not self.pages[page_id].pinned:  # Don't swap pinned pages
                        access_score = self.access_analyzer.get_access_score(
                            page_id, current_time
                        )

                        # If access score is low (meaning not likely to be accessed soon), consider for swapping
                        if (
                            access_score < 0.3
                        ):  # Threshold for "not likely to be accessed soon"
                            pages_to_swap.append((page_id, access_score))

                # Sort by access score (ascending - lowest scores first)
                pages_to_swap.sort(key=lambda x: x[1])

                # Swap out pages until we reach a safe memory level
                for page_id, access_score in pages_to_swap:
                    if (
                        self.is_memory_pressure_high()
                    ):  # Re-check actual memory pressure
                        if self.swap_page_to_disk(page_id):
                            logger.debug(
                                f"Proactively swapped page {page_id} (access_score: {access_score:.2f})"
                            )
                        else:
                            logger.warning(f"Failed to proactively swap page {page_id}")
                    else:
                        break  # Memory pressure is under control, stop swapping

    def cleanup(self):
        """Clean up all resources."""
        with self.lock:
            # Stop proactive management first
            self.stop_proactive_management()

            # Deallocate all pages
            pages_to_delete = list(self.pages.keys())
            for page_id in pages_to_delete:
                self.deallocate_page(page_id)

            # Delete swap directory if it's a temp directory
            try:
                shutil.rmtree(self.swap_directory)
            except Exception as e:
                logger.error(
                    f"Failed to clean up swap directory {self.swap_directory}: {e}"
                )


class TensorPagingManager:
    """
    Manages tensor paging for large models, allowing parts of models to be moved
    between RAM and disk as needed.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.tensor_mappings: Dict[str, str] = {}  # tensor_id -> page_id
        self.page_mappings: Dict[str, str] = {}  # page_id -> tensor_id

    def page_tensor(
        self,
        tensor: Tensor,
        tensor_id: str,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
    ) -> bool:
        """
        Page a tensor to the memory manager.

        Args:
            tensor: The tensor to page
            tensor_id: Unique identifier for the tensor
            priority: Priority level for the tensor

        Returns:
            True if paging was successful, False otherwise
        """
        page_id = f"tensor_{tensor_id}_{id(tensor)}"

        success = self.memory_manager.allocate_page(tensor, page_id, priority)
        if success:
            self.tensor_mappings[tensor_id] = page_id
            self.page_mappings[page_id] = tensor_id

        return success

    def start_proactive_management(self, interval: float = 5.0):
        """
        Start proactive memory management for tensor paging.

        Args:
            interval: Time interval (in seconds) between checks
        """
        self.memory_manager.start_proactive_management(interval)

    def stop_proactive_management(self):
        """Stop proactive memory management for tensor paging."""
        self.memory_manager.stop_proactive_management()

    def unpage_tensor(self, tensor_id: str) -> bool:
        """
        Remove a tensor from paging management.

        Args:
            tensor_id: ID of the tensor to unpage

        Returns:
            True if unpaging was successful, False otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return False

        page_id = self.tensor_mappings[tensor_id]
        success = self.memory_manager.deallocate_page(page_id)

        if success:
            del self.tensor_mappings[tensor_id]
            del self.page_mappings[page_id]

        return success

    def access_tensor(self, tensor_id: str) -> Optional[Tensor]:
        """
        Access a paged tensor, ensuring it's in RAM.

        Args:
            tensor_id: ID of the tensor to access

        Returns:
            The tensor if successful, None otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return None

        page_id = self.tensor_mappings[tensor_id]
        return self.memory_manager.access_page(page_id)

    def pin_tensor(self, tensor_id: str) -> bool:
        """
        Pin a tensor to prevent it from being swapped out.

        Args:
            tensor_id: ID of the tensor to pin

        Returns:
            True if pinning was successful, False otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return False

        page_id = self.tensor_mappings[tensor_id]
        with self.memory_manager.lock:
            if page_id in self.memory_manager.pages:
                self.memory_manager.pages[page_id].pinned = True
                return True
        return False

    def unpin_tensor(self, tensor_id: str) -> bool:
        """
        Unpin a tensor to allow it to be swapped out.

        Args:
            tensor_id: ID of the tensor to unpin

        Returns:
            True if unpinning was successful, False otherwise
        """
        if tensor_id not in self.tensor_mappings:
            return False

        page_id = self.tensor_mappings[tensor_id]
        with self.memory_manager.lock:
            if page_id in self.memory_manager.pages:
                self.memory_manager.pages[page_id].pinned = False
                return True
        return False


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """
    Get the global memory manager instance.

    Returns:
        MemoryManager instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def create_memory_manager(
    max_memory_ratio: float = 0.8,
    swap_directory: Optional[str] = None,
    page_size_mb: int = 16,
    eviction_policy: str = "lru",
) -> MemoryManager:
    """
    Create a new memory manager instance.

    Args:
        max_memory_ratio: Maximum ratio of system memory to use
        swap_directory: Directory for swap files
        page_size_mb: Size of memory pages in MB
        eviction_policy: Page eviction policy

    Returns:
        New MemoryManager instance
    """
    return MemoryManager(
        max_memory_ratio=max_memory_ratio,
        swap_directory=swap_directory,
        page_size_mb=page_size_mb,
        eviction_policy=eviction_policy,
    )


__all__ = [
    "MemoryManager",
    "TensorPagingManager",
    "MemoryPriority",
    "MemoryPage",
    "get_memory_manager",
    "create_memory_manager",
]
