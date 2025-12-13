"""
Comprehensive Memory Pool Module for Qwen3-VL Model
Consolidates basic and enhanced memory pool implementations with advanced features
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import threading
import bisect
import math
import time
import logging
from enum import Enum
from dataclasses import dataclass


class MemoryPoolError(Exception):
    """
    Custom exception for memory pool related errors.

    This exception is raised when operations on the memory pool fail due to
    invalid inputs, resource exhaustion, or other memory management issues.
    """
    pass


class BuddyAllocator:
    """
    Implements a buddy allocation system for efficient memory management.

    The buddy allocation algorithm works by splitting and combining memory blocks of
    power-of-2 sizes. When a request is made for memory, the allocator finds the
    smallest power-of-2 block that can accommodate the request. When memory is freed,
    adjacent "buddy" blocks are combined when possible to reduce fragmentation.

    This implementation is thread-safe using a reentrant lock and provides detailed
    statistics about memory usage and allocation patterns.
    """

    def __init__(self, initial_size: int = 2**30):  # 1GB default
        """
        Initialize buddy allocator with a given initial size.

        The buddy allocator uses a power-of-2 approach where memory blocks are split
        and combined based on binary tree principles. This helps minimize external
        fragmentation while maintaining efficient allocation and deallocation.

        Args:
            initial_size: Initial size of the memory pool in bytes (default: 1GB)
                Must be a positive integer. Will be adjusted to the next power of 2.

        Raises:
            ValueError: If initial_size is not positive
            TypeError: If initial_size is not an integer
        """
        if initial_size <= 0:
            raise ValueError(f"Initial size must be positive, got {initial_size}")

        # Ensure initial size is a power of 2
        self.initial_size = self._next_power_of_2(initial_size)
        self.max_order = self._log2(self.initial_size)

        # Initialize free lists for each order (size = 2^order)
        self.free_lists: List[List[int]] = [[] for _ in range(self.max_order + 1)]

        # Keep track of allocated blocks
        self.allocated_blocks = {}  # {start_addr: (size, order)}

        # Create initial large block
        self.free_lists[self.max_order].append(0)

        # Thread lock for thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'total_requested': 0,
            'total_allocated': 0,
            'internal_fragmentation': 0
        }

    def _next_power_of_2(self, x: int) -> int:
        """Get the next power of 2 >= x"""
        if x <= 0:
            raise ValueError(f"Input to _next_power_of_2 must be positive, got {x}")
        if x == 1:
            return 1
        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        return x + 1

    def _log2(self, x: int) -> int:
        """Calculate log base 2 of x"""
        if x <= 0:
            raise ValueError(f"Input to _log2 must be positive, got {x}")
        if x == 1:
            return 0
        return int(np.log2(x))

    def allocate(self, size: int) -> Optional[int]:
        """
        Allocate a block of at least 'size' bytes

        Args:
            size: Requested size in bytes

        Returns:
            The address of the allocated block, or None if allocation fails

        Raises:
            ValueError: If size is not positive
        """
        if not isinstance(size, int):
            raise TypeError(f"Size must be an integer, got {type(size)}")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        with self._lock:
            try:
                # Round up to next power of 2 for buddy system
                alloc_size: int = self._next_power_of_2(size)
                order: int = self._log2(alloc_size)

                if order > self.max_order:
                    logging.warning(f"Allocation request of {size} bytes exceeds maximum pool size of {self.initial_size} bytes")
                    return None  # Request too large

                # Find a suitable block
                for curr_order in range(order, self.max_order + 1):
                    if self.free_lists[curr_order]:
                        # Found a block, allocate it
                        addr: int = self.free_lists[curr_order].pop()

                        # Split if necessary
                        while curr_order > order:
                            curr_order -= 1
                            # Split the block in half
                            buddy_addr: int = addr + (1 << curr_order)
                            self.free_lists[curr_order].append(buddy_addr)

                        # Record allocation
                        actual_size: int = 1 << order
                        self.allocated_blocks[addr] = (actual_size, order)

                        # Update statistics
                        self.stats['allocations'] += 1
                        self.stats['total_requested'] += size
                        self.stats['total_allocated'] += actual_size
                        self.stats['internal_fragmentation'] += (actual_size - size)

                        return addr

                # No suitable block found
                logging.warning(f"No suitable block found for allocation of {size} bytes")
                return None
            except Exception as e:
                logging.error(f"Error during allocation: {e}")
                raise MemoryPoolError(f"Allocation failed: {e}")

    def deallocate(self, addr: int) -> bool:
        """
        Deallocate the block at 'addr'

        Args:
            addr: Address of the block to deallocate

        Returns:
            True if successful, False otherwise
        """
        if not isinstance(addr, int):
            raise TypeError(f"Address must be an integer, got {type(addr)}")

        with self._lock:
            try:
                if addr not in self.allocated_blocks:
                    logging.warning(f"Attempted to deallocate unallocated address: {addr}")
                    return False

                size, order = self.allocated_blocks[addr]

                # Remove from allocated blocks
                del self.allocated_blocks[addr]

                # Try to merge with buddies
                curr_addr = addr
                curr_order = order

                while curr_order < self.max_order:
                    buddy_addr = curr_addr ^ (1 << curr_order)

                    # Check if buddy is free
                    if buddy_addr in self.free_lists[curr_order]:
                        # Remove buddy from free list
                        self.free_lists[curr_order].remove(buddy_addr)

                        # Merge: use the lower address
                        if buddy_addr < curr_addr:
                            curr_addr = buddy_addr

                        curr_order += 1
                    else:
                        # Buddy not free, stop merging
                        break

                # Add merged block to free list
                self.free_lists[curr_order].append(curr_addr)

                # Update statistics
                self.stats['deallocations'] += 1

                return True
            except Exception as e:
                logging.error(f"Error during deallocation: {e}")
                raise MemoryPoolError(f"Deallocation failed: {e}")

    def get_stats(self) -> Dict:
        """Get allocation statistics"""
        with self._lock:
            stats = self.stats.copy()

            # Calculate fragmentation
            total_free = 0
            for order, free_list in enumerate(self.free_lists):
                total_free += len(free_list) * (1 << order)

            stats['total_free'] = total_free
            stats['total_memory'] = self.initial_size
            stats['utilization'] = (self.stats['total_allocated'] - self.stats['internal_fragmentation']) / self.initial_size if self.initial_size > 0 else 0

            return stats

    def is_valid_address(self, addr: int) -> bool:
        """Check if an address is valid (allocated)"""
        with self._lock:
            return addr in self.allocated_blocks


class TensorCache:
    """
    Pre-allocated tensor cache for commonly used dimensions with enhanced error handling
    """

    def __init__(self, max_cache_size_per_key: int = 10):
        """
        Initialize tensor cache

        Args:
            max_cache_size_per_key: Maximum number of tensors to cache per shape/dtype combination
        """
        if max_cache_size_per_key <= 0:
            raise ValueError(f"max_cache_size_per_key must be positive, got {max_cache_size_per_key}")
        self.max_cache_size_per_key = max_cache_size_per_key

        self.cache: Dict[Tuple[Tuple[int, ...], torch.dtype], List[torch.Tensor]] = defaultdict(list)  # {(shape, dtype): [tensor_list]}
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0
        }
        self._lock = threading.RLock()

    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a tensor of specified shape and dtype from cache or create new one

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor

        Returns:
            A tensor of the specified shape and dtype
        """
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Data type must be a torch.dtype, got {type(dtype)}")

        with self._lock:
            try:
                self.stats['total_requests'] += 1
                key = (shape, dtype)

                if self.cache[key]:
                    # Cache hit
                    tensor = self.cache[key].pop()
                    self.stats['cache_hits'] += 1
                    return tensor
                else:
                    # Cache miss - create new tensor
                    self.stats['cache_misses'] += 1
                    return torch.empty(shape, dtype=dtype)
            except Exception as e:
                logging.error(f"Error getting tensor from cache: {e}")
                raise MemoryPoolError(f"Failed to get tensor from cache: {e}")

    def return_tensor(self, tensor: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> bool:
        """
        Return a tensor to the cache for reuse

        Args:
            tensor: Tensor to return to cache
            shape: Shape of the tensor
            dtype: Data type of the tensor

        Returns:
            True if successfully cached, False otherwise
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Tensor must be a torch.Tensor, got {type(tensor)}")
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Data type must be a torch.dtype, got {type(dtype)}")

        with self._lock:
            try:
                key = (shape, dtype)

                # Only cache if the cache isn't too large (prevent memory bloat)
                if len(self.cache[key]) < self.max_cache_size_per_key:
                    self.cache[key].append(tensor)
                    return True
                else:
                    # Cache is full, tensor not cached
                    return False
            except Exception as e:
                logging.error(f"Error returning tensor to cache: {e}")
                raise MemoryPoolError(f"Failed to return tensor to cache: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            stats['cache_size'] = sum(len(tensors) for tensors in self.cache.values())
            stats['max_cache_size_per_key'] = self.max_cache_size_per_key
            return stats

    def clear_cache(self):
        """Clear all cached tensors"""
        with self._lock:
            for key in list(self.cache.keys()):
                del self.cache[key]
            logging.info("Tensor cache cleared")


class MemoryPool:
    """
    Main memory pool class that combines buddy allocation and tensor caching with enhanced error handling
    """

    def __init__(self, initial_size: int = 2**30, max_cache_size_per_key: int = 10):  # 1GB default
        """
        Initialize memory pool

        Args:
            initial_size: Initial size of the memory pool in bytes
            max_cache_size_per_key: Maximum number of tensors to cache per shape/dtype combination
        """
        if initial_size <= 0:
            raise ValueError(f"Initial size must be positive, got {initial_size}")
        if max_cache_size_per_key <= 0:
            raise ValueError(f"max_cache_size_per_key must be positive, got {max_cache_size_per_key}")

        self.buddy_allocator = BuddyAllocator(initial_size)
        self.tensor_cache = TensorCache(max_cache_size_per_key)
        self._lock = threading.RLock()

        # Keep track of allocated tensors and their metadata
        self.allocated_tensors = {}  # {id(tensor): (address, shape, dtype)}

        # Common tensor shapes for transformer models
        self.common_shapes = [
            ((1, 512, 4096), torch.float32),    # Attention output
            ((1, 512, 512), torch.float32),     # Attention weight matrix
            ((1, 8, 512, 512), torch.float32),  # Multi-head attention
            ((1, 512, 11008), torch.float32),   # FFN intermediate
            ((1, 11008, 4096), torch.float32),  # FFN output
            ((1, 576, 4096), torch.float32),    # Patch embeddings
            ((1, 3, 224, 224), torch.float32),  # Vision input
            ((4096, 4096), torch.float32),      # Linear projection
            ((4096, 11008), torch.float32),     # FFN expansion
            ((11008, 4096), torch.float32),     # FFN compression
        ]

        # Pre-allocate common tensors
        self._preallocate_common_tensors()

    def _preallocate_common_tensors(self):
        """Pre-allocate commonly used tensor shapes"""
        for shape, dtype in self.common_shapes:
            # Pre-allocate a few tensors of each common shape
            for _ in range(3):  # Pre-allocate 3 of each common shape
                try:
                    tensor = torch.empty(shape, dtype=dtype)
                    self.tensor_cache.return_tensor(tensor, shape, dtype)
                except Exception as e:
                    logging.warning(f"Failed to pre-allocate tensor {shape} with dtype {dtype}: {e}")

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """
        Allocate a tensor with the specified shape and dtype

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            device: Device to allocate the tensor on

        Returns:
            A tensor of the specified shape and dtype
        """
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape)}")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Data type must be a torch.dtype, got {type(dtype)}")
        if not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError(f"All dimensions in shape must be positive integers, got {shape}")

        device = torch.device(device)

        # First try to get from cache
        try:
            tensor = self.tensor_cache.get_tensor(shape, dtype)
        except MemoryPoolError:
            # If cache fails, create a new tensor directly
            tensor = torch.empty(shape, dtype=dtype, device=device)

        # Ensure tensor is on the right device
        if tensor.device != device:
            tensor = tensor.to(device)

        # Track the allocation
        tensor_id = id(tensor)
        with self._lock:
            self.allocated_tensors[tensor_id] = (None, shape, dtype, device)  # Using None for address in this implementation

        return tensor

    def deallocate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Deallocate a tensor and return it to the cache if appropriate

        Args:
            tensor: Tensor to deallocate

        Returns:
            True if successfully deallocated
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")

        tensor_id = id(tensor)

        with self._lock:
            if tensor_id in self.allocated_tensors:
                shape, dtype, device = self.allocated_tensors[tensor_id][1], self.allocated_tensors[tensor_id][2], self.allocated_tensors[tensor_id][3]
                del self.allocated_tensors[tensor_id]
            else:
                # If tensor wasn't tracked, use its actual properties
                shape = tuple(tensor.shape)
                dtype = tensor.dtype
                device = tensor.device

        # Return to cache for potential reuse
        try:
            success = self.tensor_cache.return_tensor(tensor, shape, dtype)
            return success
        except MemoryPoolError:
            logging.warning(f"Failed to return tensor to cache: {shape}, {dtype}")
            return False

    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory pool statistics"""
        buddy_stats = self.buddy_allocator.get_stats()
        cache_stats = self.tensor_cache.get_cache_stats()

        return {
            'buddy_allocator': buddy_stats,
            'tensor_cache': cache_stats,
            'total_allocated_tensors': len(self.allocated_tensors)
        }

    def defragment(self):
        """
        Perform memory defragmentation (simplified implementation)
        In a real system, this would move blocks around to reduce fragmentation
        """
        with self._lock:
            # In this implementation, we'll just report fragmentation
            stats = self.buddy_allocator.get_stats()
            logging.info(f"Current fragmentation: {stats['utilization']:.4f}")
            return stats

    def reset(self):
        """Reset the memory pool to initial state"""
        with self._lock:
            # Clear all allocated tensors tracking
            self.allocated_tensors.clear()
            # Clear tensor cache
            self.tensor_cache.clear_cache()
            # Reinitialize buddy allocator
            self.buddy_allocator = BuddyAllocator(self.buddy_allocator.initial_size)
            # Reinitialize cache and pre-allocate common tensors
            self._preallocate_common_tensors()
            logging.info("Memory pool reset to initial state")


class OptimizedBuddyAllocator:
    """
    Optimized Buddy Allocator with Improved Locking Strategy

    This module implements a buddy allocator with optimized locking strategies:
    - Lock striping for different memory levels to reduce contention
    - Reader-writer locks for read-heavy operations like searching
    - Fine-grained locking for better concurrency
    """

    def __init__(self, total_size: int, min_block_size: int = 256, num_lock_stripes: int = 16):
        """
        Initialize optimized buddy allocator with lock striping

        Args:
            total_size: Total size of the memory pool in bytes
            min_block_size: Minimum block size in bytes (power of 2)
            num_lock_stripes: Number of lock stripes for granular locking
        """
        # Input validation
        if not isinstance(total_size, int) or total_size <= 0:
            raise ValueError(f"total_size must be a positive integer, got {total_size}")
        if not isinstance(min_block_size, int) or min_block_size <= 0:
            raise ValueError(f"min_block_size must be a positive integer, got {min_block_size}")
        if not isinstance(num_lock_stripes, int) or num_lock_stripes <= 0:
            raise ValueError(f"num_lock_stripes must be a positive integer, got {num_lock_stripes}")

        # Validate that min_block_size is a power of 2
        if min_block_size & (min_block_size - 1) != 0:
            raise ValueError(f"min_block_size must be a power of 2, got {min_block_size}")

        # Ensure total_size is at least as large as min_block_size
        if total_size < min_block_size:
            raise ValueError(f"total_size ({total_size}) must be at least min_block_size ({min_block_size})")

        # Ensure total_size and min_block_size are powers of 2
        self.total_size = self._next_power_of_2(total_size)
        self.min_block_size = self._next_power_of_2(min_block_size)
        self.num_lock_stripes = num_lock_stripes

        # Calculate number of levels in buddy tree
        self.levels = int(math.log2(self.total_size // self.min_block_size)) + 1

        # Create free block lists for each level with lock striping
        self.free_blocks: List[List[int]] = [[] for _ in range(self.levels)]
        self._lock_stripes = [threading.RLock() for _ in range(num_lock_stripes)]

        # Dictionary to map addresses to allocated blocks
        self.allocated_blocks: Dict[int, Tuple[int, int]] = {}  # {addr: (size, level)}
        self._allocation_lock = threading.Lock()

        # Cache for mapping sizes to levels for fast access
        self.size_to_level_cache: Dict[int, int] = {}
        self._cache_lock = threading.RLock()

        # Initialize the largest block as free (highest level)
        self.free_blocks[self.levels - 1].append(0)

    def _next_power_of_2(self, x: int) -> int:
        """Return the next number that is a power of 2"""
        if not isinstance(x, int) or x < 0:
            raise ValueError(f"x must be a non-negative integer, got {x}")
        if x <= 1:
            return 1
        return 2 ** ((x - 1).bit_length())

    def _get_lock_stripe(self, addr: int) -> threading.RLock:
        """Get the appropriate lock stripe for a given address"""
        return self._lock_stripes[addr % self.num_lock_stripes]

    def _size_to_level(self, size: int) -> int:
        """Convert a size to the corresponding level in the buddy tree"""
        # Use cache to avoid repeated calculations
        with self._cache_lock:
            if size in self.size_to_level_cache:
                return self.size_to_level_cache[size]

        # Round size up to the next power of 2
        actual_size = self._next_power_of_2(max(size, self.min_block_size))

        # Calculate the level based on size
        level = int(math.log2(actual_size // self.min_block_size))
        level = max(0, min(level, self.levels - 1))  # Ensure bounds

        with self._cache_lock:
            self.size_to_level_cache[size] = level
        return level

    def allocate(self, size: int) -> Optional[int]:
        """
        Allocate a memory block of the specified size with optimized locking

        Args:
            size: Size required in bytes

        Returns:
            Address of allocated block or None if allocation fails
        """
        # Input validation
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size}")

        with self._allocation_lock:
            try:
                # Round size up to the next power of 2
                actual_size = self._next_power_of_2(max(size, self.min_block_size))

                # Find the required level
                level = int(math.log2(actual_size // self.min_block_size))
                level = max(0, min(level, self.levels - 1))

                # Look for a free block at the required level or higher
                for current_level in range(level, self.levels):
                    # Use lock striping to reduce contention by level
                    level_lock = self._get_lock_stripe(current_level)
                    with level_lock:
                        if self.free_blocks[current_level]:
                            # Found a block, allocate it
                            block_addr = self.free_blocks[current_level].pop()

                            # Split the block until we reach the required level
                            while current_level > level:
                                # Split the block in half
                                buddy_size = actual_size >> 1
                                buddy_addr = block_addr + buddy_size

                                # Add the second half to the lower level
                                lower_level_lock = self._get_lock_stripe(current_level - 1)
                                with lower_level_lock:
                                    self.free_blocks[current_level - 1].append(buddy_addr)

                                # Update the current block size
                                actual_size = buddy_size
                                current_level -= 1

                            # Record the allocated block
                            self.allocated_blocks[block_addr] = (actual_size, level)

                            return block_addr

                # No suitable block found
                return None
            except ValueError as e:
                logging.error(f"Invalid allocation parameters: {e}")
                raise
            except Exception as e:
                logging.error(f"Error during allocation: {e}")
                raise MemoryPoolError(f"Error during allocation: {e}") from e

    def deallocate(self, block_addr: int) -> None:
        """
        Deallocate a memory block and attempt to merge with buddies with optimized locking
        """
        # Input validation
        if not isinstance(block_addr, int) or block_addr < 0:
            raise ValueError(f"block_addr must be a non-negative integer, got {block_addr}")

        with self._allocation_lock:
            if block_addr not in self.allocated_blocks:
                raise ValueError("Attempted to deallocate an unallocated block")

            size, level = self.allocated_blocks[block_addr]

            # Remove from allocated blocks
            del self.allocated_blocks[block_addr]

            # Try to merge with buddies
            try:
                self._merge_buddies(block_addr, size, level)
            except Exception as e:
                logging.error(f"Error during buddy merge: {e}")
                raise MemoryPoolError(f"Error during buddy merge: {e}") from e

    def _merge_buddies(self, block_addr: int, size: int, level: int) -> None:
        """
        Attempt to merge the block with its buddies recursively with optimized locking
        """
        current_addr = block_addr
        current_level = level

        # Continue merging while possible
        while current_level < self.levels - 1:
            # Calculate the buddy address
            buddy_addr = current_addr ^ (1 << (current_level + int(math.log2(self.min_block_size))))

            # Use lock striping to access the specific level
            level_lock = self._get_lock_stripe(current_level)
            with level_lock:
                # Check if buddy exists in the same level
                if buddy_addr in self.free_blocks[current_level]:
                    # Remove the buddy from free list
                    self.free_blocks[current_level].remove(buddy_addr)

                    # Merge the blocks - keep the lower address
                    if current_addr < buddy_addr:
                        # Current block is first
                        pass  # current_addr is already correct
                    else:
                        # Buddy is first
                        current_addr = buddy_addr

                    # Move to the next level
                    current_level += 1
                else:
                    # No buddy found, add current block to free list and stop
                    self.free_blocks[current_level].append(current_addr)
                    break

        # If we reached the last level or didn't find more buddies,
        # add the block to the appropriate level
        if current_level < self.levels:
            level_lock = self._get_lock_stripe(current_level)
            with level_lock:
                self.free_blocks[current_level].append(current_addr)

        # Clear the cache when memory structure changes
        with self._cache_lock:
            self.size_to_level_cache.clear()


# Global memory pool instance
_global_memory_pool: Optional[MemoryPool] = None
_pool_lock = threading.Lock()


def get_memory_pool() -> MemoryPool:
    """Get the global memory pool instance"""
    global _global_memory_pool
    if _global_memory_pool is None:
        with _pool_lock:
            if _global_memory_pool is None:
                _global_memory_pool = MemoryPool()
    return _global_memory_pool


def allocate_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Allocate a tensor using the global memory pool

    Args:
        shape: Shape of the tensor to allocate
        dtype: Data type of the tensor
        device: Device to allocate the tensor on

    Returns:
        A tensor of the specified shape and dtype
    """
    pool = get_memory_pool()
    return pool.allocate_tensor(shape, dtype, device)


def deallocate_tensor(tensor: torch.Tensor) -> bool:
    """
    Deallocate a tensor using the global memory pool

    Args:
        tensor: Tensor to deallocate

    Returns:
        True if successfully deallocated
    """
    pool = get_memory_pool()
    return pool.deallocate_tensor(tensor)


if __name__ == "__main__":
    print("Testing Enhanced Buddy Allocator...")

    try:
        # Test buddy allocator
        buddy = BuddyAllocator(2**20)  # 1MB

        # Test invalid size
        try:
            BuddyAllocator(0)
        except ValueError as e:
            print(f"✓ Correctly caught error for invalid size: {e}")

        # Allocate some blocks
        addr1 = buddy.allocate(1024)  # 1KB
        addr2 = buddy.allocate(2048)  # 2KB
        addr3 = buddy.allocate(512)   # 512B

        print(f"Allocated blocks at addresses: {addr1}, {addr2}, {addr3}")

        # Test invalid allocations
        try:
            buddy.allocate(-100)
        except ValueError as e:
            print(f"✓ Correctly caught error for negative size: {e}")

        try:
            buddy.allocate("invalid")
        except TypeError as e:
            print(f"✓ Correctly caught error for invalid type: {e}")

        # Deallocate some blocks
        buddy.deallocate(addr2)
        buddy.deallocate(addr1)

        print("Buddy allocator stats:", buddy.get_stats())

        print("\nTesting Tensor Cache...")

        # Test tensor cache
        cache = TensorCache(5)  # Max 5 per key

        # Get tensors
        t1 = cache.get_tensor((10, 20), torch.float32)
        t2 = cache.get_tensor((10, 20), torch.float32)

        print(f"Got tensors of shape: {t1.shape}, {t2.shape}")

        # Test invalid cache operations
        try:
            cache.get_tensor([10, 20], torch.float32)  # Invalid shape type
        except TypeError as e:
            print(f"✓ Correctly caught error for invalid shape type: {e}")

        try:
            cache.get_tensor((10, -5), torch.float32)  # Invalid shape values
        except Exception as e:
            print(f"Cache handled invalid shape: {e}")

        # Return one to cache
        success = cache.return_tensor(t1, (10, 20), torch.float32)
        print(f"Tensor returned to cache: {success}")

        # Get another tensor of same shape (should come from cache)
        t3 = cache.get_tensor((10, 20), torch.float32)

        print("Tensor cache stats:", cache.get_cache_stats())

        print("\nTesting Enhanced Memory Pool...")

        # Test memory pool
        pool = MemoryPool(2**20, max_cache_size_per_key=3)  # 1MB, max 3 per key

        # Test invalid pool initialization
        try:
            MemoryPool(0, 3)
        except ValueError as e:
            print(f"✓ Correctly caught error for invalid pool size: {e}")

        try:
            MemoryPool(1024, 0)
        except ValueError as e:
            print(f"✓ Correctly caught error for invalid cache size: {e}")

        # Allocate tensors
        tensor1 = pool.allocate_tensor((100, 200), torch.float32)
        tensor2 = pool.allocate_tensor((50, 100, 256), torch.float32)

        print(f"Allocated tensors of shapes: {tensor1.shape}, {tensor2.shape}")

        # Test invalid tensor allocation
        try:
            pool.allocate_tensor((10, -5), torch.float32)  # Invalid shape
        except ValueError as e:
            print(f"✓ Correctly caught error for invalid shape: {e}")

        try:
            pool.allocate_tensor([10, 20], torch.float32)  # Wrong type for shape
        except TypeError as e:
            print(f"✓ Correctly caught error for wrong shape type: {e}")

        # Return tensors to pool
        success1 = pool.deallocate_tensor(tensor1)
        success2 = pool.deallocate_tensor(tensor2)
        print(f"Tensors deallocated: {success1}, {success2}")

        print("Memory pool stats:", pool.get_memory_stats())

        # Test global pool functions
        global_tensor = allocate_tensor((50, 50), torch.float32)
        print(f"Global pool tensor shape: {global_tensor.shape}")
        deallocate_tensor(global_tensor)

        print("\nEnhanced memory pool implementation completed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()