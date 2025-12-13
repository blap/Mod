"""
Thread-Safe Memory Management System for Qwen3-VL Model
with proper locking mechanisms and race condition prevention
"""

import math
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import time
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorType(Enum):
    """Enumeration for different tensor types in the Qwen3-VL model"""
    KV_CACHE = "kv_cache"
    IMAGE_FEATURES = "image_features"
    TEXT_EMBEDDINGS = "text_embeddings"
    GRADIENTS = "gradients"
    ACTIVATIONS = "activations"
    PARAMETERS = "parameters"


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool with thread safety"""
    start_addr: int
    size: int
    is_free: bool
    tensor_type: Optional[TensorType] = None
    tensor_id: Optional[str] = None
    timestamp: float = 0.0
    ref_count: int = 0
    pinned: bool = False  # If true, block won't be swapped out

    def __hash__(self):
        """Make MemoryBlock hashable for use in sets"""
        return hash((self.start_addr, self.size))

    def __eq__(self, other):
        """Define equality for use in sets"""
        if not isinstance(other, MemoryBlock):
            return False
        return self.start_addr == other.start_addr and self.size == other.size


class ThreadSafeBuddyAllocator:
    """
    Thread-safe Buddy Allocation algorithm implementation with proper locking
    """
    def __init__(self, total_size: int, min_block_size: int = 256):
        """
        Initialize the thread-safe buddy allocator

        Args:
            total_size: Total memory pool size in bytes
            min_block_size: Minimum block size in bytes (must be power of 2)
        """
        # Input validation
        if not isinstance(total_size, int) or total_size <= 0:
            raise ValueError(f"total_size must be a positive integer, got {total_size}")
        if not isinstance(min_block_size, int) or min_block_size <= 0:
            raise ValueError(f"min_block_size must be a positive integer, got {min_block_size}")

        # Validate that min_block_size is a power of 2
        if min_block_size & (min_block_size - 1) != 0:
            raise ValueError(f"min_block_size must be a power of 2, got {min_block_size}")

        # Ensure total_size is at least as large as min_block_size
        if total_size < min_block_size:
            raise ValueError(f"total_size ({total_size}) must be at least min_block_size ({min_block_size})")

        # Ensure total_size and min_block_size are powers of 2
        self.total_size = self._next_power_of_2(total_size)
        self.min_block_size = self._next_power_of_2(min_block_size)

        # Calculate number of levels in the buddy tree
        # Level 0 = blocks of size min_block_size
        # Level n = blocks of size min_block_size * 2^n
        self.levels = int(math.log2(self.total_size // self.min_block_size)) + 1

        # Initialize free blocks organized by order (size = min_block_size * 2^order)
        self.free_blocks: Dict[int, set] = {i: set() for i in range(self.levels)}

        # Initially, the entire pool is one free block at the highest level
        initial_block = MemoryBlock(
            start_addr=0,
            size=self.total_size,
            is_free=True,
            timestamp=time.time()
        )
        self.free_blocks[self.levels - 1].add(initial_block)

        # Keep track of allocated blocks
        self.allocated_blocks: Dict[int, MemoryBlock] = {}

        # Thread safety - using RLock for reentrant locking
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            'total_allocated': 0,
            'total_freed': 0,
            'allocation_count': 0,
            'deallocation_count': 0
        }

        # Cache for size-to-level mapping to avoid repeated calculations
        self._size_to_level_cache: Dict[int, int] = {}

    def _next_power_of_2(self, x: int) -> int:
        """Return the next number that is a power of 2"""
        if not isinstance(x, int) or x < 0:
            raise ValueError(f"x must be a non-negative integer, got {x}")
        if x <= 1:
            return 1
        return 2 ** ((x - 1).bit_length())

    def _size_to_level(self, size: int) -> int:
        """Convert a size to the corresponding level in the buddy tree"""
        # Use cache to avoid repeated calculations
        if size in self._size_to_level_cache:
            return self._size_to_level_cache[size]

        # Round size up to the next power of 2
        actual_size = self._next_power_of_2(max(size, self.min_block_size))

        # Calculate the level based on the size
        # Level 0 = smallest blocks, Level n = largest blocks
        level = int(math.log2(actual_size // self.min_block_size))
        level = max(0, min(level, self.levels - 1))  # Ensure bounds

        self._size_to_level_cache[size] = level
        return level

    def _get_buddy_addr(self, addr: int, size: int) -> int:
        """Get the address of the buddy block"""
        # In a buddy system, the buddy of a block at address addr with size size
        # is located at address addr XOR size (this flips the block bit)
        return addr ^ size

    def allocate(self, size: int, tensor_type: TensorType, tensor_id: str) -> Optional[MemoryBlock]:
        """
        Thread-safe allocation of a memory block of at least the requested size

        Args:
            size: Size in bytes
            tensor_type: Type of tensor that will be stored
            tensor_id: Unique ID for the tensor

        Returns:
            Allocated MemoryBlock or None if allocation fails
        """
        # Input validation
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size}")
        if not isinstance(tensor_type, TensorType):
            raise ValueError(f"tensor_type must be a TensorType, got {tensor_type}")
        if not isinstance(tensor_id, str) or not tensor_id:
            raise ValueError(f"tensor_id must be a non-empty string, got {tensor_id}")

        with self._lock:
            try:
                # Round up size to next power of 2 and minimum block size
                actual_size = self._next_power_of_2(max(size, self.min_block_size))
                req_order = int(math.log2(actual_size // self.min_block_size))

                # Find a free block of at least the required order
                for current_level in range(req_order, self.levels):
                    if self.free_blocks[current_level]:
                        # Found a block, allocate it
                        block = self.free_blocks[current_level].pop()

                        # Split the block until we get the right size
                        current_addr = block.start_addr
                        current_size = block.size
                        while current_level > req_order:
                            # Split the block in half - put the second half in the lower order list
                            current_size //= 2
                            buddy_addr = current_addr + current_size

                            # Add the buddy block to the lower level
                            buddy_block = MemoryBlock(
                                start_addr=buddy_addr,
                                size=current_size,
                                is_free=True,
                                timestamp=time.time()
                            )
                            self.free_blocks[current_level - 1].add(buddy_block)
                            current_level -= 1

                        # Finalize the allocated block
                        allocated_block = MemoryBlock(
                            start_addr=current_addr,
                            size=current_size,
                            is_free=False,
                            tensor_type=tensor_type,
                            tensor_id=tensor_id,
                            timestamp=time.time(),
                            ref_count=1
                        )

                        # Track the allocated block
                        self.allocated_blocks[current_addr] = allocated_block

                        # Update statistics
                        self.stats['total_allocated'] += current_size
                        self.stats['allocation_count'] += 1

                        return allocated_block

                # No suitable block found
                return None
            except Exception as e:
                logger.error(f"Error during allocation: {e}")
                raise RuntimeError(f"Error during allocation: {e}") from e

    def deallocate(self, block: MemoryBlock) -> None:
        """
        Thread-safe deallocation of a memory block and attempt to merge with buddies
        """
        if not isinstance(block, MemoryBlock):
            raise ValueError(f"block must be a MemoryBlock, got {type(block)}")
        if not isinstance(block.start_addr, int) or block.start_addr < 0:
            raise ValueError(f"block.start_addr must be a non-negative integer, got {block.start_addr}")

        with self._lock:
            if block.start_addr not in self.allocated_blocks:
                raise ValueError("Attempting to deallocate a block not allocated by this allocator")

            # Mark block as free
            block.is_free = True
            block.tensor_type = None
            block.tensor_id = None
            block.ref_count = 0

            # Remove from allocated blocks
            del self.allocated_blocks[block.start_addr]

            # Update statistics
            self.stats['total_freed'] += block.size
            self.stats['deallocation_count'] += 1

            # Try to merge with buddies
            self._merge_buddies(block)

    def _merge_buddies(self, block: MemoryBlock) -> None:
        """
        Attempt to merge the block with its buddies recursively with thread safety
        """
        current_addr = block.start_addr
        current_size = block.size
        current_order = int(math.log2(current_size // self.min_block_size))

        # Continue merging while possible
        while current_order < self.levels - 1:
            buddy_addr = self._get_buddy_addr(current_addr, current_size)

            # Find buddy in the same level
            buddy_found = None
            for potential_buddy in self.free_blocks[current_order]:
                if potential_buddy.start_addr == buddy_addr and potential_buddy.is_free:
                    buddy_found = potential_buddy
                    break

            if buddy_found is None:
                # Buddy not free, add current block to free list and stop
                current_block = MemoryBlock(
                    start_addr=current_addr,
                    size=current_size,
                    is_free=True,
                    timestamp=time.time()
                )
                self.free_blocks[current_order].add(current_block)
                break

            # Remove both blocks from this order
            self.free_blocks[current_order].remove(buddy_found)

            # Calculate address of the merged block (the lower address becomes the new block)
            if current_addr < buddy_addr:
                current_addr = current_addr
            else:
                current_addr = buddy_addr

            # Increase size and order
            current_size *= 2
            current_order += 1

        # Add merged block to the appropriate level
        merged_block = MemoryBlock(
            start_addr=current_addr,
            size=current_size,
            is_free=True,
            timestamp=time.time()
        )
        self.free_blocks[current_order].add(merged_block)

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get memory allocation statistics with thread safety
        """
        with self._lock:
            total_free = sum((1 << order) * len(blocks) for order, blocks in self.free_blocks.items())
            total_allocated = sum(block.size for block in self.allocated_blocks.values())

            # Calculate fragmentation ratio
            largest_free_block = 0
            for order, blocks in self.free_blocks.items():
                if blocks:
                    block_size = 1 << order
                    largest_free_block = max(largest_free_block, block_size)

            fragmentation_ratio = 0.0
            if total_free > 0 and largest_free_block > 0:
                fragmentation_ratio = 1.0 - (largest_free_block / total_free)

            return {
                'total_allocated': total_allocated,
                'total_free': total_free,
                'largest_free_block': largest_free_block,
                'fragmentation_ratio': fragmentation_ratio,
                'allocated_blocks': len(self.allocated_blocks),
                'num_free_blocks': sum(len(blocks) for blocks in self.free_blocks.values()),
                **self.stats
            }

    def defragment(self) -> Dict[str, Union[int, float]]:
        """
        Perform memory defragmentation with thread safety
        """
        with self._lock:
            pre_stats = self.get_stats()

            # In a real system, this would move allocated blocks to consolidate free space
            # For this implementation, we'll focus on ensuring proper merging of free blocks
            # that might have been missed due to timing issues

            # The current _merge_buddies implementation already handles merging,
            # so we'll just return stats showing the current state
            post_stats = self.get_stats()

            return {
                'pre_defrag_stats': pre_stats,
                'post_defrag_stats': post_stats,
                'defragmentation_improvement': post_stats['largest_free_block'] - pre_stats['largest_free_block']
            }


class ThreadSafeMemoryPool:
    """
    Thread-safe specialized memory pool for a specific tensor type
    """
    def __init__(self, tensor_type: TensorType, pool_size: int, min_block_size: int = 256):
        self.tensor_type = tensor_type
        self.pool_size = pool_size
        self.min_block_size = min_block_size
        self.allocator = ThreadSafeBuddyAllocator(pool_size, min_block_size)
        self.active_allocations: Dict[str, MemoryBlock] = {}

        # Thread safety for allocation tracking
        self._allocation_lock = threading.RLock()

        # Statistics
        self.utilization_ratio = 0.0
        self.fragmentation_ratio = 0.0
        self.last_compaction_time = time.time()

        # Thresholds for compaction
        self.fragmentation_threshold = 0.3  # Compact when fragmentation exceeds 30%
        self.utilization_threshold = 0.8    # Compact when utilization exceeds 80%

    def allocate(self, size: int, tensor_id: str) -> Optional[MemoryBlock]:
        """
        Thread-safe allocation of memory block for the specified tensor
        """
        with self._allocation_lock:
            block = self.allocator.allocate(size, self.tensor_type, tensor_id)
            if block:
                self.active_allocations[tensor_id] = block
                self._update_stats()
            return block

    def deallocate(self, tensor_id: str) -> bool:
        """
        Thread-safe deallocation of memory block associated with tensor_id
        """
        with self._allocation_lock:
            if tensor_id not in self.active_allocations:
                return False

            block = self.active_allocations[tensor_id]
            self.allocator.deallocate(block)
            del self.active_allocations[tensor_id]
            self._update_stats()
            return True

    def _update_stats(self):
        """Update utilization and fragmentation statistics with thread safety"""
        allocator_stats = self.allocator.get_stats()
        total_allocated = sum(block.size for block in self.allocator.allocated_blocks.values())
        self.utilization_ratio = min(1.0, total_allocated / self.pool_size if self.pool_size > 0 else 0)

        # Calculate fragmentation more accurately
        free_block_sizes = []
        for level_blocks in self.allocator.free_blocks.values():
            for block in level_blocks:
                free_block_sizes.append(block.size)

        if free_block_sizes:
            total_free = sum(free_block_sizes)
            largest_free_block = max(free_block_sizes)
            self.fragmentation_ratio = max(0, 1 - (largest_free_block / total_free))
        else:
            self.fragmentation_ratio = 0  # Pool completely utilized

    def should_compact(self) -> bool:
        """
        Determine if the pool should be compacted based on thresholds with thread safety
        """
        with self._allocation_lock:
            return (self.fragmentation_ratio > self.fragmentation_threshold or
                    self.utilization_ratio > self.utilization_threshold)

    def compact(self) -> bool:
        """
        Perform intelligent memory pool compaction with thread safety
        """
        with self._allocation_lock:
            # In a real implementation, this would involve moving allocated blocks
            # and updating all references to those blocks. For this implementation,
            # we'll focus on the defragmentation aspect.
            defrag_result = self.allocator.defragment()
            self.last_compaction_time = time.time()
            self._update_stats()

            logger.debug(f"Compacted {self.tensor_type.value} pool: fragmentation reduced from "
                        f"{defrag_result['pre_defrag_stats']['fragmentation_ratio']:.3f} to "
                        f"{defrag_result['post_defrag_stats']['fragmentation_ratio']:.3f}")
            return True

    def get_pool_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get pool statistics with thread safety
        """
        with self._allocation_lock:
            allocator_stats = self.allocator.get_stats()
            return {
                **allocator_stats,
                'pool_size': self.pool_size,
                'utilization_ratio': self.utilization_ratio,
                'fragmentation_ratio': self.fragmentation_ratio,
                'active_allocations': len(self.active_allocations),
                'tensor_type': self.tensor_type.value
            }


class ThreadSafeHardwareOptimizer:
    """
    Thread-safe hardware-specific optimizer for Intel i5-10210U + NVIDIA SM61 + NVMe SSD
    """
    def __init__(self):
        # Intel i5-10210U specifications (4 cores, 8 threads)
        self.cpu_cores = 4
        self.cpu_threads = 8  # With hyperthreading
        self.cpu_l1_cache_size = 32 * 1024  # 32KB per core
        self.cpu_l2_cache_size = 256 * 1024  # 256KB per core
        self.cpu_l3_cache_size = 6 * 1024 * 1024  # 6MB shared

        # NVIDIA SM61 (Maxwell architecture) specifications
        self.gpu_compute_capability = (6, 1)  # SM61
        self.max_threads_per_block = 1024
        self.max_shared_memory_per_block = 48 * 1024  # 48KB
        self.warp_size = 32  # Threads per warp

        # NVMe SSD specifications
        self.ssd_read_speed = 3500 * 1024 * 1024  # 3500 MB/s
        self.ssd_write_speed = 3000 * 1024 * 1024  # 3000 MB/s
        self.ssd_latency = 0.000025  # 25μs typical latency

        # Thread safety for hardware-specific optimizations
        self._lock = threading.RLock()

    def get_optimal_block_size(self, tensor_type: TensorType) -> int:
        """
        Determine optimal block size for a specific tensor type considering hardware characteristics
        """
        with self._lock:
            # Different tensor types have different access patterns and optimal block sizes
            if tensor_type == TensorType.KV_CACHE:
                # KV cache is frequently accessed, optimize for cache efficiency
                return min(self.cpu_l2_cache_size, 1024*1024)  # 1MB max for KV cache blocks
            elif tensor_type == TensorType.IMAGE_FEATURES:
                # Image features are typically processed in blocks, optimize for spatial locality
                return min(self.cpu_l2_cache_size * 2, 2*1024*1024)  # 2MB max for image features
            elif tensor_type == TensorType.TEXT_EMBEDDINGS:
                # Text embeddings often processed sequentially, optimize for memory bandwidth
                return min(self.cpu_l1_cache_size * 4, 512*1024)  # 512KB max for text embeddings
            elif tensor_type == TensorType.GRADIENTS:
                # Gradients are updated frequently, optimize for write efficiency
                return min(self.cpu_l2_cache_size, 1024*1024)  # 1MB max for gradients
            elif tensor_type == TensorType.ACTIVATIONS:
                # Activations are temporary, optimize for quick allocation/deallocation
                return min(self.cpu_l2_cache_size, 1024*1024)  # 1MB max for activations
            elif tensor_type == TensorType.PARAMETERS:
                # Parameters are large and accessed frequently, optimize for cache hits
                return min(self.cpu_l3_cache_size // 4, 4*1024*1024)  # 4MB max for parameters
            else:
                return self.cpu_l2_cache_size  # Default to L2 cache size

    def should_use_gpu_memory(self, tensor_type: TensorType, size: int) -> bool:
        """
        Determine if a tensor should be allocated in GPU memory
        """
        with self._lock:
            # For SM61, consider tensor type and size for GPU memory allocation
            if tensor_type in [TensorType.IMAGE_FEATURES, TensorType.ACTIVATIONS]:
                # These tensor types benefit from GPU parallel processing
                return size < self.max_shared_memory_per_block * 10  # Use GPU for reasonably sized tensors
            elif tensor_type == TensorType.KV_CACHE:
                # KV cache might be too large for GPU memory, consider carefully
                return size < self.max_shared_memory_per_block * 5  # More conservative for KV cache
            else:
                # Other tensor types might not benefit as much from GPU memory
                return size < self.max_shared_memory_per_block * 2

    def get_thread_optimizations(self) -> Dict[str, int]:
        """
        Get threading optimization parameters based on CPU capabilities
        """
        with self._lock:
            # For i5-10210U with 4 cores and 8 threads
            # Limit concurrent cache operations to avoid thread contention
            return {
                'max_cache_threads': min(4, self.cpu_threads // 2),  # Use up to half the threads
                'prefetch_worker_threads': 2,  # Dedicated threads for prefetching
                'migration_worker_threads': 1,  # Dedicated thread for migrations
                'io_worker_threads': 2  # Threads for SSD I/O operations
            }


class ThreadSafeMemoryPoolingSystem:
    """
    Advanced thread-safe memory pooling system for Qwen3-VL model
    Optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD
    """
    def __init__(self,
                 kv_cache_size: int = 512*1024*1024,      # 512MB
                 image_features_size: int = 512*1024*1024,  # 512MB
                 text_embeddings_size: int = 256*1024*1024,  # 256MB
                 gradients_size: int = 512*1024*1024,      # 512MB
                 activations_size: int = 256*1024*1024,    # 256MB
                 parameters_size: int = 1024*1024*1024,    # 1GB
                 min_block_size: int = 256):
        """
        Initialize the thread-safe memory pooling system with specialized pools

        Args:
            kv_cache_size: Size of pool for KV cache tensors
            image_features_size: Size of pool for image feature tensors
            text_embeddings_size: Size of pool for text embedding tensors
            gradients_size: Size of pool for gradient tensors
            activations_size: Size of pool for activation tensors
            parameters_size: Size of pool for parameter tensors
            min_block_size: Minimum block size in bytes
        """
        # Initialize system lock first
        self._system_lock = threading.RLock()

        # Create hardware optimizer
        self.hardware_optimizer = ThreadSafeHardwareOptimizer()

        # Create specialized pools for different tensor types with thread safety
        self.pools: Dict[TensorType, ThreadSafeMemoryPool] = {}

        # Adjust pool sizes based on hardware capabilities
        adjusted_kv_size = self._adjust_pool_size(TensorType.KV_CACHE, kv_cache_size)
        adjusted_img_size = self._adjust_pool_size(TensorType.IMAGE_FEATURES, image_features_size)
        adjusted_text_size = self._adjust_pool_size(TensorType.TEXT_EMBEDDINGS, text_embeddings_size)
        adjusted_grad_size = self._adjust_pool_size(TensorType.GRADIENTS, gradients_size)
        adjusted_act_size = self._adjust_pool_size(TensorType.ACTIVATIONS, activations_size)
        adjusted_param_size = self._adjust_pool_size(TensorType.PARAMETERS, parameters_size)

        self.pools[TensorType.KV_CACHE] = ThreadSafeMemoryPool(TensorType.KV_CACHE, adjusted_kv_size, min_block_size)
        self.pools[TensorType.IMAGE_FEATURES] = ThreadSafeMemoryPool(TensorType.IMAGE_FEATURES, adjusted_img_size, min_block_size)
        self.pools[TensorType.TEXT_EMBEDDINGS] = ThreadSafeMemoryPool(TensorType.TEXT_EMBEDDINGS, adjusted_text_size, min_block_size)
        self.pools[TensorType.GRADIENTS] = ThreadSafeMemoryPool(TensorType.GRADIENTS, adjusted_grad_size, min_block_size)
        self.pools[TensorType.ACTIVATIONS] = ThreadSafeMemoryPool(TensorType.ACTIVATIONS, adjusted_act_size, min_block_size)
        self.pools[TensorType.PARAMETERS] = ThreadSafeMemoryPool(TensorType.PARAMETERS, adjusted_param_size, min_block_size)

        # Statistics
        self.stats = {
            'total_allocated': 0,
            'total_freed': 0,
            'peak_utilization': 0.0,
            'total_fragmentation': 0.0
        }

        # Histogram for allocation patterns
        self.allocation_histogram: Dict[TensorType, List[int]] = defaultdict(list)

        # Start background optimization threads
        self._start_background_optimizations()

    def _adjust_pool_size(self, tensor_type: TensorType, original_size: int) -> int:
        """
        Adjust pool size based on hardware characteristics with thread safety
        """
        with self._system_lock:
            optimal_block_size = self.hardware_optimizer.get_optimal_block_size(tensor_type)

            # Adjust size to be a multiple of optimal block size
            adjusted_size = ((original_size // optimal_block_size) + 1) * optimal_block_size

            # Ensure it doesn't exceed hardware limitations
            if tensor_type == TensorType.KV_CACHE:
                # Limit KV cache to prevent memory exhaustion
                max_kv_size = self.hardware_optimizer.cpu_l3_cache_size * 3
                adjusted_size = min(adjusted_size, max_kv_size)

            return adjusted_size

    def _start_background_optimizations(self):
        """
        Start background threads for ongoing optimizations
        """
        def run_background_optimizations():
            while True:
                try:
                    time.sleep(1.0)  # Check every second
                    self._perform_background_optimizations()
                except Exception as e:
                    logger.error(f"Error in background optimization thread: {e}")
                    time.sleep(5.0)  # Wait longer before retrying

        # Start background thread
        bg_thread = threading.Thread(target=run_background_optimizations, daemon=True)
        bg_thread.start()

    def _perform_background_optimizations(self):
        """
        Perform background optimizations like auto-compaction
        """
        with self._system_lock:
            # Check if any pool needs compaction
            for tensor_type, pool in self.pools.items():
                if pool.should_compact():
                    pool.compact()

            # Update system-wide statistics
            total_allocated = sum(
                sum(block.size for block in pool.allocator.allocated_blocks.values())
                for pool in self.pools.values()
            )
            total_pool_size = sum(pool.pool_size for pool in self.pools.values())

            if total_pool_size > 0:
                utilization = total_allocated / total_pool_size
                self.stats['peak_utilization'] = max(self.stats['peak_utilization'], utilization)

    def allocate(self, tensor_type: TensorType, size: int, tensor_id: str) -> Optional[MemoryBlock]:
        """
        Thread-safe allocation of memory for a tensor of the specified type

        Args:
            tensor_type: Type of tensor to allocate
            size: Size in bytes
            tensor_id: Unique identifier for the tensor

        Returns:
            Allocated MemoryBlock or None if allocation fails
        """
        with self._system_lock:
            pool = self.pools.get(tensor_type)
            if not pool:
                raise ValueError(f"Unsupported tensor type: {tensor_type}")

            # Determine if GPU memory should be used
            if self.hardware_optimizer.should_use_gpu_memory(tensor_type, size):
                # In a real implementation, this would handle GPU memory allocation
                # For this example, we'll proceed with CPU allocation but note the intent
                logger.debug(f"Would use GPU memory for {tensor_type.value} tensor of size {size}")

            # Record allocation in histogram for optimization
            self.allocation_histogram[tensor_type].append(size)

            block = pool.allocate(size, tensor_id)
            if block:
                self.stats['total_allocated'] += 1
                # Update peak utilization
                peak = max(self.stats['peak_utilization'], pool.utilization_ratio)
                self.stats['peak_utilization'] = peak

            return block

    def deallocate(self, tensor_type: TensorType, tensor_id: str) -> bool:
        """
        Thread-safe deallocation of memory for a tensor of the specified type

        Args:
            tensor_type: Type of tensor to deallocate
            tensor_id: Unique identifier for the tensor

        Returns:
            True if deallocation was successful, False otherwise
        """
        with self._system_lock:
            pool = self.pools.get(tensor_type)
            if not pool:
                return False

            success = pool.deallocate(tensor_id)
            if success:
                self.stats['total_freed'] += 1
            return success

    def get_pool_stats(self, tensor_type: TensorType) -> Dict[str, Union[int, float]]:
        """
        Get statistics for a specific pool with thread safety
        """
        with self._system_lock:
            pool = self.pools.get(tensor_type)
            if not pool:
                return {}

            return pool.get_pool_stats()

    def get_system_stats(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get overall system statistics with thread safety
        """
        with self._system_lock:
            total_pool_size = sum(pool.pool_size for pool in self.pools.values())
            total_allocated = sum(
                sum(block.size for block in pool.allocator.allocated_blocks.values())
                for pool in self.pools.values()
            )

            utilization = total_allocated / total_pool_size if total_pool_size > 0 else 0

            # Calculate average fragmentation
            avg_fragmentation = sum(
                pool.fragmentation_ratio for pool in self.pools.values()
            ) / len(self.pools) if self.pools else 0

            # Get hardware-specific stats
            hw_stats = {
                'cpu_cores': self.hardware_optimizer.cpu_cores,
                'cpu_threads': self.hardware_optimizer.cpu_threads,
                'gpu_compute_capability': self.hardware_optimizer.gpu_compute_capability,
                'max_threads_per_block': self.hardware_optimizer.max_threads_per_block,
                'max_shared_memory_per_block_kb': self.hardware_optimizer.max_shared_memory_per_block // 1024
            }

            return {
                **self.stats,
                'overall_utilization': utilization,
                'average_fragmentation': avg_fragmentation,
                'total_pools': len(self.pools),
                'hardware_stats': hw_stats,
                'allocation_histogram': {k.value: v for k, v in self.allocation_histogram.items()}
            }

    def compact_memory(self) -> bool:
        """
        Perform intelligent memory compaction to reduce fragmentation with thread safety
        """
        with self._system_lock:
            # Compact each pool individually
            for pool in self.pools.values():
                if pool.should_compact():
                    pool.compact()

            # Update statistics after compaction
            total_allocated = sum(
                sum(block.size for block in pool.allocator.allocated_blocks.values())
                for pool in self.pools.values()
            )
            total_pool_size = sum(pool.pool_size for pool in self.pools.values())

            if total_pool_size > 0:
                utilization = total_allocated / total_pool_size
                self.stats['peak_utilization'] = max(self.stats['peak_utilization'], utilization)

            # Calculate new average fragmentation
            avg_fragmentation = sum(
                max(0, pool.fragmentation_ratio) for pool in self.pools.values()
            ) / len(self.pools) if self.pools else 0
            self.stats['total_fragmentation'] = avg_fragmentation

            logger.debug(f"Memory compaction completed. Avg fragmentation: {avg_fragmentation:.3f}")
            return True


# Example usage and demonstration
if __name__ == "__main__":
    print("Initializing Thread-Safe Advanced Memory Pooling System for Qwen3-VL...")
    
    # Create the thread-safe memory system
    memory_system = ThreadSafeMemoryPoolingSystem(
        kv_cache_size=1024*1024*256,      # 256MB
        image_features_size=1024*1024*128,  # 128MB
        text_embeddings_size=1024*1024*64,  # 64MB
        gradients_size=1024*1024*256,      # 256MB
        activations_size=1024*1024*128,    # 128MB
        parameters_size=1024*1024*512,     # 512MB
        min_block_size=256
    )
    
    print("Memory system initialized successfully!")
    print("\nHardware-specific optimizations:")
    hw_optimizer = ThreadSafeHardwareOptimizer()
    print(f"  - CPU: {hw_optimizer.cpu_cores} cores, {hw_optimizer.cpu_threads} threads")
    print(f"  - GPU Compute Capability: {hw_optimizer.gpu_compute_capability}")
    print(f"  - Max shared memory per block: {hw_optimizer.max_shared_memory_per_block // 1024} KB")
    print(f"  - Max threads per block: {hw_optimizer.max_threads_per_block}")
    
    print("\n1. Allocating different types of tensors...")
    
    # Allocate KV cache blocks
    kv_block = memory_system.allocate(TensorType.KV_CACHE, 1024*1024*10, "kv_tensor_1")  # 10MB
    if kv_block:
        print(f"  Allocated KV cache block: {kv_block.size} bytes at address {kv_block.start_addr}")
    
    # Allocate image feature blocks
    img_block = memory_system.allocate(TensorType.IMAGE_FEATURES, 1024*1024*5, "img_tensor_1")  # 5MB
    if img_block:
        print(f"  Allocated image features block: {img_block.size} bytes at address {img_block.start_addr}")
    
    # Allocate text embedding blocks
    text_block = memory_system.allocate(TensorType.TEXT_EMBEDDINGS, 1024*1024*2, "text_tensor_1")  # 2MB
    if text_block:
        print(f"  Allocated text embeddings block: {text_block.size} bytes at address {text_block.start_addr}")
    
    print("\n2. Checking pool statistics...")
    kv_stats = memory_system.get_pool_stats(TensorType.KV_CACHE)
    print(f"  KV Cache Pool Stats: Utilization={kv_stats['utilization_ratio']:.3f}, "
          f"Fragmentation={kv_stats['fragmentation_ratio']:.3f}")
    
    img_stats = memory_system.get_pool_stats(TensorType.IMAGE_FEATURES)
    print(f"  Image Features Pool Stats: Utilization={img_stats['utilization_ratio']:.3f}, "
          f"Fragmentation={img_stats['fragmentation_ratio']:.3f}")
    
    print("\n3. Performing memory compaction...")
    memory_system.compact_memory()
    
    print("\n4. Getting system-wide statistics...")
    system_stats = memory_system.get_system_stats()
    print(f"  Overall Utilization: {system_stats['overall_utilization']:.3f}")
    print(f"  Average Fragmentation: {system_stats['average_fragmentation']:.3f}")
    print(f"  Total Allocated Operations: {system_stats['total_allocated']}")
    print(f"  Peak Utilization: {system_stats['peak_utilization']:.3f}")
    
    print("\n5. Deallocating tensors...")
    memory_system.deallocate(TensorType.KV_CACHE, "kv_tensor_1")
    memory_system.deallocate(TensorType.IMAGE_FEATURES, "img_tensor_1")
    memory_system.deallocate(TensorType.TEXT_EMBEDDINGS, "text_tensor_1")
    print("  All test tensors deallocated")
    
    print("\nThread-Safe Advanced Memory Pooling System demo completed successfully!")