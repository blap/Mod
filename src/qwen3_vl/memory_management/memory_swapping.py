"""
Advanced Memory Swapping System for SSD with Memory Pressure Monitoring

Implements an advanced memory swapping system optimized for NVMe SSD storage
with intelligent algorithms and integration with existing cache and compression systems.
Designed for Intel i5-10210U + NVIDIA SM61 hardware.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging
import pickle
import tempfile
from pathlib import Path
import psutil
import heapq
from collections import OrderedDict, defaultdict
import gc
from abc import ABC, abstractmethod


class MemoryPressureLevel(Enum):
    """Levels of memory pressure"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SwapAlgorithm(Enum):
    """Available swapping algorithms"""
    LRU = "lru"
    CLOCK = "clock"
    ADAPTIVE = "adaptive"


class MemoryRegionType(Enum):
    """Types of memory regions"""
    TENSOR_DATA = "tensor_data"
    ACTIVATION_BUFFER = "activation_buffer"
    KV_CACHE = "kv_cache"
    TEMPORARY = "temporary"


@dataclass
class MemoryBlock:
    """Represents a memory block that can be swapped"""
    id: str
    ptr: int
    size: int
    region_type: MemoryRegionType
    allocated: bool
    timestamp: float
    last_access_time: float
    access_count: int
    is_swapped: bool
    swap_location: Optional[str]
    ref_count: int = 1
    pinned: bool = False  # If pinned, should not be swapped


@dataclass
class SwapStats:
    """Statistics for swapping operations"""
    total_swapped_out: int = 0
    total_swapped_in: int = 0
    total_swapped_bytes: int = 0
    swap_out_time: float = 0.0
    swap_in_time: float = 0.0
    total_swaps: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    swap_efficiency: float = 0.0

    def calculate_efficiency(self) -> float:
        """Calculate swap efficiency as ratio of useful swaps to total operations"""
        total_ops = self.cache_hits + self.cache_misses
        if total_ops == 0:
            return 0.0
        return self.cache_hits / total_ops


class MemoryPressureMonitor:
    """Monitors memory pressure on both RAM and GPU"""

    def __init__(self, ram_thresholds: Tuple[float, float, float] = (0.7, 0.85, 0.95),
                 gpu_thresholds: Tuple[float, float, float] = (0.7, 0.85, 0.95)):
        """
        Initialize memory pressure monitor

        Args:
            ram_thresholds: Thresholds for (medium, high, critical) RAM pressure (0.0-1.0)
            gpu_thresholds: Thresholds for (medium, high, critical) GPU pressure (0.0-1.0)
        """
        self.ram_thresholds = ram_thresholds
        self.gpu_thresholds = gpu_thresholds
        self.pressure_history = []
        self.history_size = 10  # Keep last 10 readings

    def get_ram_pressure(self) -> Tuple[MemoryPressureLevel, float]:
        """
        Get current RAM pressure level and usage percentage

        Returns:
            Tuple of (pressure_level, usage_percentage)
        """
        memory_percent = psutil.virtual_memory().percent / 100.0

        if memory_percent < self.ram_thresholds[0]:
            level = MemoryPressureLevel.LOW
        elif memory_percent < self.ram_thresholds[1]:
            level = MemoryPressureLevel.MEDIUM
        elif memory_percent < self.ram_thresholds[2]:
            level = MemoryPressureLevel.HIGH
        else:
            level = MemoryPressureLevel.CRITICAL

        return level, memory_percent

    def get_gpu_pressure(self) -> Tuple[MemoryPressureLevel, float]:
        """
        Get current GPU pressure level and usage percentage

        Returns:
            Tuple of (pressure_level, usage_percentage)
        """
        if not torch.cuda.is_available():
            return MemoryPressureLevel.LOW, 0.0

        # Get GPU memory stats
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)

        usage_percent = reserved_memory / total_memory if total_memory > 0 else 0.0

        if usage_percent < self.gpu_thresholds[0]:
            level = MemoryPressureLevel.LOW
        elif usage_percent < self.gpu_thresholds[1]:
            level = MemoryPressureLevel.MEDIUM
        elif usage_percent < self.gpu_thresholds[2]:
            level = MemoryPressureLevel.HIGH
        else:
            level = MemoryPressureLevel.CRITICAL

        return level, usage_percent

    def get_overall_pressure(self) -> Tuple[MemoryPressureLevel, float]:
        """
        Get overall memory pressure considering both RAM and GPU

        Returns:
            Tuple of (pressure_level, max_usage_percentage)
        """
        ram_level, ram_usage = self.get_ram_pressure()
        gpu_level, gpu_usage = self.get_gpu_pressure()

        # Return the highest pressure level and the corresponding usage
        levels = [MemoryPressureLevel.LOW, MemoryPressureLevel.MEDIUM,
                 MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]

        ram_idx = levels.index(ram_level)
        gpu_idx = levels.index(gpu_level)

        max_idx = max(ram_idx, gpu_idx)
        max_level = levels[max_idx]

        # Return the usage that corresponds to the highest pressure level
        max_usage = ram_usage if ram_idx >= gpu_idx else gpu_usage

        return max_level, max_usage


class BaseSwapAlgorithm(ABC):
    """Base class for swapping algorithms"""

    def __init__(self):
        self.blocks = OrderedDict()
        self.access_log = {}

    def add_block(self, block: MemoryBlock):
        """Add a block to the algorithm's tracking"""
        self.blocks[block.id] = block
        self.access_log[block.id] = block.last_access_time

    def remove_block(self, block_id: str):
        """Remove a block from tracking"""
        if block_id in self.blocks:
            del self.blocks[block_id]
        if block_id in self.access_log:
            del self.access_log[block_id]

    def access_block(self, block_id: str):
        """Record access to a block"""
        if block_id in self.blocks:
            self.blocks[block_id].last_access_time = time.time()
            self.access_log[block_id] = self.blocks[block_id].last_access_time

    @abstractmethod
    def select_victim(self) -> Optional[MemoryBlock]:
        """Select a block to swap out"""
        pass


class LRUSwapAlgorithm(BaseSwapAlgorithm):
    """Least Recently Used swapping algorithm"""

    def select_victim(self) -> Optional[MemoryBlock]:
        """Select the least recently used block to swap out"""
        if not self.blocks:
            return None

        # Find the block with the oldest access time (non-pinned)
        non_pinned_blocks = {bid: block for bid, block in self.blocks.items() if not block.pinned}

        if not non_pinned_blocks:
            return None  # All blocks are pinned

        # Find the oldest access time among non-pinned blocks
        oldest_block_id = min(non_pinned_blocks.keys(), key=lambda k: self.access_log[k])
        return self.blocks[oldest_block_id]


class ClockSwapAlgorithm(BaseSwapAlgorithm):
    """Clock (Second Chance) swapping algorithm"""

    def __init__(self):
        super().__init__()
        self.hand = 0
        self.block_list = []

    def add_block(self, block: MemoryBlock):
        """Add a block to the algorithm's tracking"""
        super().add_block(block)
        if block.id not in self.block_list:
            self.block_list.append(block.id)
        # Initialize with reference bit = True (recently accessed)
        self.access_log[block.id] = (block.last_access_time, True)  # (time, reference_bit)

    def remove_block(self, block_id: str):
        """Remove a block from tracking"""
        super().remove_block(block_id)
        if block_id in self.block_list:
            self.block_list.remove(block_id)
        # Adjust hand if necessary
        if self.hand >= len(self.block_list) and self.block_list:
            self.hand = len(self.block_list) - 1

    def access_block(self, block_id: str):
        """Record access to a block (set reference bit)"""
        if block_id in self.access_log and block_id in self.blocks:
            current_time, _ = self.access_log[block_id]
            self.access_log[block_id] = (time.time(), True)  # Set reference bit
            self.blocks[block_id].last_access_time = time.time()

    def select_victim(self) -> Optional[MemoryBlock]:
        """Select a block to swap out using clock algorithm"""
        if not self.block_list:
            return None

        start_hand = self.hand
        iterations = 0  # Prevent infinite loops
        max_iterations = len(self.block_list) * 2  # Give each block at most 2 chances

        while iterations < max_iterations:
            if not self.block_list:  # Check if list is empty after updates
                return None

            # Ensure hand is within bounds
            self.hand = self.hand % len(self.block_list) if self.block_list else 0
            if not self.block_list:  # Double-check after modulo
                return None

            block_id = self.block_list[self.hand]

            # Check if block still exists (in case it was removed)
            if block_id not in self.blocks:
                # Remove from block_list and continue
                self.block_list.remove(block_id)
                if self.hand >= len(self.block_list) and self.block_list:
                    self.hand = len(self.block_list) - 1
                iterations += 1
                continue

            block = self.blocks[block_id]

            # Don't swap pinned blocks
            if block.pinned:
                # Move hand forward and continue
                self.hand = (self.hand + 1) % len(self.block_list) if self.block_list else 0
                iterations += 1
                continue

            if block_id not in self.access_log:
                # Block was removed from access_log, skip
                self.hand = (self.hand + 1) % len(self.block_list) if self.block_list else 0
                iterations += 1
                continue

            access_time, ref_bit = self.access_log[block_id]

            if ref_bit:
                # Give second chance, clear reference bit
                self.access_log[block_id] = (access_time, False)
                self.hand = (self.hand + 1) % len(self.block_list) if self.block_list else 0
            else:
                # This is our victim
                victim = block
                # Move hand for next selection
                self.hand = (self.hand + 1) % len(self.block_list) if self.block_list else 0
                return victim

            iterations += 1

        # If we've gone through all blocks and couldn't find a victim,
        # return any non-pinned block as fallback
        for block_id in self.block_list:
            if block_id in self.blocks and not self.blocks[block_id].pinned:
                self.hand = (self.hand + 1) % len(self.block_list) if self.block_list else 0
                return self.blocks[block_id]

        return None


class AdaptiveSwapAlgorithm(BaseSwapAlgorithm):
    """
    Adaptive swapping algorithm that combines multiple factors:
    - Recency (like LRU)
    - Frequency of access
    - Size of block (prefer swapping larger blocks)
    """

    def select_victim(self) -> Optional[MemoryBlock]:
        """Select a block based on adaptive algorithm"""
        if not self.blocks:
            return None

        # Calculate score for each block
        # Score = (time_since_access * access_frequency) / size_factor
        scores = {}
        current_time = time.time()

        for block_id, block in self.blocks.items():
            # Don't swap pinned blocks
            if block.pinned:
                continue

            time_since_access = current_time - block.last_access_time
            access_frequency = block.access_count / max(1, time_since_access)
            size_factor = block.size / (1024 * 1024)  # Size in MB

            # Calculate score (higher score = more likely to be swapped)
            score = (time_since_access * access_frequency) / max(1, size_factor)
            scores[block_id] = score

        if not scores:
            return None

        # Select block with lowest score (least valuable to keep in memory)
        victim_id = min(scores.keys(), key=lambda k: scores[k])
        return self.blocks[victim_id]


class NVMeOptimizer:
    """Optimizes swapping operations for NVMe SSD performance"""

    def __init__(self, block_size: int = 4 * 1024 * 1024):  # 4MB blocks
        """
        Initialize NVMe optimizer

        Args:
            block_size: Size of blocks for swapping operations
        """
        self.block_size = block_size
        self.swap_directory = Path(tempfile.gettempdir()) / "qwen3vl_swap"
        self.swap_directory.mkdir(exist_ok=True)
        self.active_swaps = 0
        self.max_concurrent_swaps = 4

    def swap_out_to_file(self, block_id: str, tensor_data: Any) -> bool:
        """
        Swap out a block to NVMe file storage

        Args:
            block_id: ID of the block to swap
            tensor_data: Data to swap out

        Returns:
            True if successful, False otherwise
        """
        swap_file = self.swap_directory / f"{block_id}.swap"

        start_time = time.time()
        try:
            # Serialize and save to NVMe
            with open(swap_file, 'wb') as f:
                pickle.dump(tensor_data, f)

            swap_time = time.time() - start_time
            logging.debug(f"Swapped out block {block_id} to {swap_file}, time: {swap_time:.4f}s")
            return True
        except Exception as e:
            logging.error(f"Failed to swap out block {block_id}: {e}")
            return False

    def swap_in_from_file(self, block_id: str) -> Optional[Any]:
        """
        Swap in a block from NVMe file storage

        Args:
            block_id: ID of the block to swap in

        Returns:
            Data if successful, None otherwise
        """
        swap_file = self.swap_directory / f"{block_id}.swap"

        start_time = time.time()
        try:
            # Load from NVMe and deserialize
            with open(swap_file, 'rb') as f:
                tensor_data = pickle.load(f)

            swap_time = time.time() - start_time
            logging.debug(f"Swapped in block {block_id} from {swap_file}, time: {swap_time:.4f}s")
            return tensor_data
        except Exception as e:
            logging.error(f"Failed to swap in block {block_id}: {e}")
            return None

    def cleanup_swap_file(self, block_id: str):
        """Remove swap file after block is no longer needed"""
        swap_file = self.swap_directory / f"{block_id}.swap"
        try:
            if swap_file.exists():
                swap_file.unlink()
        except Exception as e:
            logging.error(f"Failed to clean up swap file for {block_id}: {e}")

    def get_swap_stats(self) -> Dict[str, Any]:
        """Get statistics about swap operations"""
        return {
            'active_swaps': self.active_swaps,
            'max_concurrent_swaps': self.max_concurrent_swaps,
            'swap_directory': str(self.swap_directory),
            'swap_files_count': len(list(self.swap_directory.glob("*.swap")))
        }


class AdvancedMemorySwapper:
    """Main class for advanced memory swapping system"""

    def __init__(self,
                 swap_algorithm: SwapAlgorithm = SwapAlgorithm.ADAPTIVE,
                 swap_threshold: float = 0.8,  # Start swapping at 80% memory usage
                 max_swap_size: int = 2 * 1024 * 1024 * 1024,  # 2GB max swap space
                 nvme_optimizer: Optional[NVMeOptimizer] = None):
        """
        Initialize advanced memory swapper

        Args:
            swap_algorithm: Algorithm to use for selecting blocks to swap
            swap_threshold: Memory usage threshold to trigger swapping (0.0-1.0)
            max_swap_size: Maximum amount of memory to use for swapping (bytes)
            nvme_optimizer: NVMe optimizer instance (creates one if None)
        """
        self.swap_algorithm = swap_algorithm
        self.swap_threshold = swap_threshold
        self.max_swap_size = max_swap_size
        self.nvme_optimizer = nvme_optimizer or NVMeOptimizer()

        # Initialize the appropriate swapping algorithm
        if swap_algorithm == SwapAlgorithm.LRU:
            self.algorithm = LRUSwapAlgorithm()
        elif swap_algorithm == SwapAlgorithm.CLOCK:
            self.algorithm = ClockSwapAlgorithm()
        elif swap_algorithm == SwapAlgorithm.ADAPTIVE:
            self.algorithm = AdaptiveSwapAlgorithm()
        else:
            raise ValueError(f"Unsupported swap algorithm: {swap_algorithm}")

        # Memory blocks tracking
        self.blocks: Dict[str, MemoryBlock] = {}
        self.block_lock = threading.RLock()

        # Memory tracking
        self.current_swap_size = 0
        self.max_swap_utilization = 0

        # Statistics
        self.stats = SwapStats()

        # Memory pressure monitor
        self.pressure_monitor = MemoryPressureMonitor()

        # Access pattern tracker
        self.access_patterns = defaultdict(list)
        self.pattern_analysis_window = 100  # Analyze last 100 accesses

        logging.info(f"AdvancedMemorySwapper initialized with {swap_algorithm.value} algorithm")

    def register_memory_block(self, block_id: str, size: int,
                            region_type: MemoryRegionType = MemoryRegionType.TENSOR_DATA,
                            pinned: bool = False) -> MemoryBlock:
        """
        Register a memory block for potential swapping

        Args:
            block_id: Unique identifier for the block
            size: Size of the block in bytes
            region_type: Type of memory region
            pinned: If True, block will not be swapped out

        Returns:
            MemoryBlock object
        """
        with self.block_lock:
            if block_id in self.blocks:
                logging.warning(f"Block {block_id} already registered")
                return self.blocks[block_id]

            block = MemoryBlock(
                id=block_id,
                ptr=0,  # Placeholder, actual address not needed for our system
                size=size,
                region_type=region_type,
                allocated=True,
                timestamp=time.time(),
                last_access_time=time.time(),
                access_count=1,
                is_swapped=False,
                swap_location=None,
                pinned=pinned
            )

            self.blocks[block_id] = block
            self.algorithm.add_block(block)

            logging.debug(f"Registered memory block {block_id}, size: {size} bytes")
            return block

    def unregister_memory_block(self, block_id: str) -> bool:
        """
        Unregister a memory block (and clean up if it was swapped)

        Args:
            block_id: ID of the block to unregister

        Returns:
            True if successful
        """
        with self.block_lock:
            if block_id not in self.blocks:
                return False

            block = self.blocks[block_id]

            # If block was swapped, clean up the swap file
            if block.is_swapped and block.swap_location:
                self.nvme_optimizer.cleanup_swap_file(block_id)

            # Remove from algorithm tracking
            self.algorithm.remove_block(block_id)

            # Update stats
            if block.is_swapped:
                self.current_swap_size -= block.size
                self.stats.total_swapped_out -= 1

            del self.blocks[block_id]
            logging.debug(f"Unregistered memory block {block_id}")
            return True

    def access_memory_block(self, block_id: str) -> Optional[MemoryBlock]:
        """
        Record access to a memory block, potentially swapping it in if needed

        Args:
            block_id: ID of the block being accessed

        Returns:
            MemoryBlock if found and accessible, None otherwise
        """
        with self.block_lock:
            if block_id not in self.blocks:
                logging.warning(f"Access to unregistered block {block_id}")
                return None

            block = self.blocks[block_id]

            # Record the access
            block.last_access_time = time.time()
            block.access_count += 1
            self.algorithm.access_block(block_id)

            # Add to access pattern tracker
            self.access_patterns[block_id].append(time.time())
            if len(self.access_patterns[block_id]) > self.pattern_analysis_window:
                self.access_patterns[block_id].pop(0)

            # If block is swapped out, we need to swap it back in
            if block.is_swapped:
                logging.debug(f"Block {block_id} is swapped out, swapping in...")

                start_time = time.time()
                # Load the data from swap file
                tensor_data = self.nvme_optimizer.swap_in_from_file(block_id)

                if tensor_data is not None:
                    # Successfully loaded, update block status
                    block.is_swapped = False
                    block.swap_location = None
                    self.current_swap_size -= block.size

                    self.stats.total_swapped_in += 1
                    self.stats.cache_misses += 1  # This counts as a cache miss
                else:
                    logging.error(f"Failed to swap in block {block_id}")
                    return None

                self.stats.swap_in_time += time.time() - start_time

                logging.debug(f"Swapped in block {block_id}")
            else:
                self.stats.cache_hits += 1  # This counts as a cache hit

            return block

    def should_swap(self) -> bool:
        """
        Determine if swapping should be triggered based on memory pressure

        Returns:
            True if swapping should occur
        """
        pressure_level, usage = self.pressure_monitor.get_overall_pressure()

        # Trigger swapping based on pressure level or usage threshold
        return (pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL] or
                usage >= self.swap_threshold)

    def perform_swapping(self) -> int:
        """
        Perform swapping operation by selecting and swapping out blocks

        Returns:
            Number of blocks swapped out
        """
        if not self.should_swap():
            return 0

        blocks_swapped = 0
        start_time = time.time()

        # Determine how much memory we need to free
        pressure_level, usage = self.pressure_monitor.get_overall_pressure()
        target_free = 0

        if pressure_level == MemoryPressureLevel.HIGH:
            target_free = int(self.max_swap_size * 0.1)  # Free 10% of max swap size
        elif pressure_level == MemoryPressureLevel.CRITICAL:
            target_free = int(self.max_swap_size * 0.3)  # Free 30% of max swap size

        current_free = self.max_swap_size - self.current_swap_size
        needed_to_free = max(0, target_free - current_free)

        while self.current_swap_size < self.max_swap_size and blocks_swapped < 5:  # Max 5 blocks per cycle
            victim = self.algorithm.select_victim()

            if victim is None or victim.pinned:
                logging.debug("No suitable victim found for swapping")
                break

            # Perform the swap
            if self._swap_out_block(victim):
                blocks_swapped += 1
                if needed_to_free > 0 and self.current_swap_size >= needed_to_free:
                    break
            else:
                logging.warning(f"Failed to swap out block {victim.id}")
                break

        self.stats.swap_out_time += time.time() - start_time
        self.stats.total_swaps += blocks_swapped

        logging.debug(f"Performed swapping: {blocks_swapped} blocks swapped out")
        return blocks_swapped

    def _swap_out_block(self, block: MemoryBlock) -> bool:
        """
        Actually swap out a block to NVMe storage

        Args:
            block: Block to swap out

        Returns:
            True if successful
        """
        if block.pinned:
            logging.warning(f"Attempted to swap pinned block {block.id}")
            return False

        if self.current_swap_size + block.size > self.max_swap_size:
            logging.warning(f"Not enough swap space for block {block.id}")
            return False

        # In a real implementation, we would save the actual tensor data
        # For this example, we'll just simulate the operation with a placeholder
        start_time = time.time()

        # Create a placeholder for the tensor data (in real implementation, this would be the actual tensor)
        tensor_data = f"tensor_data_for_{block.id}"

        # Perform the swap operation
        swap_success = self.nvme_optimizer.swap_out_to_file(block.id, tensor_data)

        if swap_success:
            # Update block status
            block.is_swapped = True
            block.swap_location = f"nvme://{self.nvme_optimizer.swap_directory}/{block.id}.swap"
            self.current_swap_size += block.size

            # Update statistics
            self.stats.total_swapped_out += 1
            self.stats.total_swapped_bytes += block.size
            self.stats.swap_out_time += time.time() - start_time

            if self.current_swap_size > self.max_swap_utilization:
                self.max_swap_utilization = self.current_swap_size

            logging.debug(f"Successfully swapped out block {block.id}, size: {block.size} bytes")
            return True
        else:
            logging.error(f"Failed to swap out block {block.id}")
            return False

    def get_access_pattern_priority(self, block_id: str) -> float:
        """
        Determine priority for a block based on access patterns

        Args:
            block_id: ID of the block

        Returns:
            Priority score (higher = more important to keep in memory)
        """
        if block_id not in self.access_patterns:
            return 0.5  # Default priority

        accesses = self.access_patterns[block_id]
        if len(accesses) < 2:
            return 0.5

        # Calculate access frequency in the last window
        time_span = accesses[-1] - accesses[0]
        frequency = len(accesses) / max(1, time_span)  # Accesses per second

        # Calculate recency (more recent accesses get higher priority)
        time_since_last = time.time() - accesses[-1]
        recency_factor = max(0.1, 1.0 - (time_since_last / 300))  # Decay over 5 minutes

        # Combine factors for priority score
        priority = min(1.0, (frequency * 0.6 + recency_factor * 0.4))
        return priority

    def analyze_access_patterns(self) -> Dict[str, Any]:
        """
        Analyze access patterns to optimize swapping decisions

        Returns:
            Dictionary with pattern analysis results
        """
        analysis = {
            'most_frequently_accessed': [],
            'least_frequently_accessed': [],
            'temporal_locality': 0.0,
            'spatial_locality': 0.0
        }

        # Sort blocks by access frequency
        block_access_counts = [(bid, block.access_count)
                              for bid, block in self.blocks.items()
                              if not block.pinned]
        block_access_counts.sort(key=lambda x: x[1], reverse=True)

        analysis['most_frequently_accessed'] = block_access_counts[:5]  # Top 5
        analysis['least_frequently_accessed'] = block_access_counts[-5:]  # Bottom 5

        # Calculate temporal locality score
        total_blocks = len([b for b in self.blocks.values() if not b.pinned])
        if total_blocks > 0:
            recently_accessed = len([b for b in self.blocks.values()
                                   if not b.pinned and
                                   time.time() - b.last_access_time < 60])  # Last minute
            analysis['temporal_locality'] = recently_accessed / total_blocks

        return analysis

    def get_swapping_efficiency(self) -> Dict[str, float]:
        """
        Get efficiency metrics for the swapping system

        Returns:
            Dictionary with efficiency metrics
        """
        total_ops = self.stats.cache_hits + self.stats.cache_misses
        hit_rate = self.stats.cache_hits / total_ops if total_ops > 0 else 0.0

        avg_swap_out_time = (self.stats.swap_out_time / self.stats.total_swapped_out
                           if self.stats.total_swapped_out > 0 else 0.0)
        avg_swap_in_time = (self.stats.swap_in_time / self.stats.total_swapped_in
                          if self.stats.total_swapped_in > 0 else 0.0)

        return {
            'hit_rate': hit_rate,
            'avg_swap_out_time': avg_swap_out_time,
            'avg_swap_in_time': avg_swap_in_time,
            'total_swapped_GB': self.stats.total_swapped_bytes / (1024**3),
            'current_swap_utilization_GB': self.current_swap_size / (1024**3),
            'max_swap_utilization_GB': self.max_swap_utilization / (1024**3)
        }

    def integrate_with_compression(self, compression_manager):
        """
        Integrate with memory compression system for additional optimization

        Args:
            compression_manager: Instance of MemoryCompressionManager
        """
        self.compression_manager = compression_manager
        logging.info("Integrated with compression system")

    def integrate_with_cache(self, cache_manager):
        """
        Integrate with hierarchical cache system

        Args:
            cache_manager: Instance of HierarchicalCompressionCache or similar
        """
        self.cache_manager = cache_manager
        logging.info("Integrated with cache system")

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the swapping system

        Returns:
            Dictionary with system status information
        """
        pressure_level, pressure_usage = self.pressure_monitor.get_overall_pressure()
        return {
            'algorithm': self.swap_algorithm.value,
            'swap_threshold': self.swap_threshold,
            'max_swap_size_GB': self.max_swap_size / (1024**3),
            'current_swap_size_GB': self.current_swap_size / (1024**3),
            'total_registered_blocks': len(self.blocks),
            'swapped_blocks': len([b for b in self.blocks.values() if b.is_swapped]),
            'pinned_blocks': len([b for b in self.blocks.values() if b.pinned]),
            'pressure_level': pressure_level.value,
            'pressure_usage': pressure_usage,
            'nvme_stats': self.nvme_optimizer.get_swap_stats(),
            'efficiency_metrics': self.get_swapping_efficiency()
        }

    def cleanup(self):
        """Clean up all registered blocks and swap files"""
        with self.block_lock:
            # Unregister all blocks
            for block_id in list(self.blocks.keys()):
                self.unregister_memory_block(block_id)

            # Perform final garbage collection
            gc.collect()


def create_advanced_memory_swapper(hardware_config: Optional[Dict[str, Any]] = None) -> AdvancedMemorySwapper:
    """
    Factory function to create an optimized swapping system for specific hardware

    Args:
        hardware_config: Hardware configuration with details like:
                        - cpu_model: CPU model string
                        - gpu_model: GPU model string
                        - memory_size: Total system memory in bytes
                        - storage_type: Storage type ('nvme', 'ssd', 'hdd')

    Returns:
        Optimized AdvancedMemorySwapper instance
    """
    if hardware_config is None:
        hardware_config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
            'storage_type': 'nvme'
        }

    # Determine optimal settings based on hardware
    memory_size = hardware_config.get('memory_size', 8 * 1024 * 1024 * 1024)
    storage_type = hardware_config.get('storage_type', 'nvme')

    # Adjust swap size based on available memory
    swap_size = min(memory_size * 2, 4 * 1024 * 1024 * 1024)  # Max 4GB swap

    # Choose algorithm based on storage type
    if storage_type == 'nvme':
        algorithm = SwapAlgorithm.ADAPTIVE  # NVMe can handle more sophisticated algorithms
        threshold = 0.75  # Start swapping a bit earlier for NVMe
    else:
        algorithm = SwapAlgorithm.LRU  # Simpler algorithm for slower storage
        threshold = 0.85  # Start swapping later for slower storage

    # Create the swapper with hardware-optimized settings
    swapper = AdvancedMemorySwapper(
        swap_algorithm=algorithm,
        swap_threshold=threshold,
        max_swap_size=swap_size,
        nvme_optimizer=NVMeOptimizer()
    )

    logging.info(f"Created optimized swapping system for {hardware_config.get('cpu_model', 'unknown')} "
                f"with {storage_type.upper()} storage")

    return swapper


# Example usage and integration
def integrate_with_qwen3_vl_swapping(swapper: AdvancedMemorySwapper,
                                   memory_optimizer,
                                   compression_manager):
    """
    Example of how to integrate the swapping system with Qwen3-VL components
    """
    # Integrate with existing components
    swapper.integrate_with_compression(compression_manager)

    # Example: Register tensor allocations with the swapper
    def optimized_tensor_allocation(shape, dtype=torch.float32, tensor_type="general", pinned=False):
        # Allocate tensor using existing memory optimizer
        tensor = memory_optimizer.allocate_tensor(shape, dtype, tensor_type)

        # Register the allocation with the swapping system
        block_id = f"tensor_{id(tensor)}_{int(time.time() * 1000000)}"
        size = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        swapper.register_memory_block(block_id, size,
                                    getattr(MemoryRegionType, tensor_type.upper().replace('-', '_'), MemoryRegionType.TENSOR_DATA),
                                    pinned)

        return tensor, block_id

    # Example: Access tensor with swapping awareness
    def access_tensor_with_swapping(tensor, block_id):
        # Record access in swapping system
        accessed_block = swapper.access_memory_block(block_id)

        # If block was swapped in, update tensor reference if needed
        if accessed_block and accessed_block.is_swapped:
            # In real implementation, this would reload the tensor data
            # For now, we just return the tensor as-is
            logging.debug(f"Block {block_id} was swapped in during access")
        elif accessed_block is None:
            logging.warning(f"Could not access block {block_id}")

        return tensor

    # Example: Perform swapping when needed
    def check_and_swap():
        return swapper.perform_swapping()

    return optimized_tensor_allocation, access_tensor_with_swapping, check_and_swap

