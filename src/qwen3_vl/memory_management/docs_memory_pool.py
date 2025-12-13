"""
Enhanced Memory Pool Implementation with Comprehensive Documentation

This module provides a robust memory management system for the Qwen3-VL model with:
- Buddy allocation for efficient memory management
- Tensor caching for frequently used tensor shapes
- Thread-safe operations with proper locking
- Comprehensive error handling and input validation
- Detailed logging and statistics

Used for Phase 2.9: Memory Pooling and Pre-allocation Techniques.

The memory pool system is designed to reduce memory fragmentation, improve allocation
speed, and provide better memory utilization for transformer models, particularly
during inference when many small, frequently-used tensor shapes are allocated.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import threading
import logging


class MemoryPoolError(Exception):
    """
    Custom exception for memory pool related errors.
    
    This exception is raised when operations on the memory pool fail due to
    invalid inputs, resource exhaustion, or other memory management issues.
    
    Attributes:
        message (str): Explanation of the error
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
    
    The algorithm works as follows:
    1. When allocating, find the smallest power-of-2 block that fits the request
    2. Split larger blocks if necessary by recursively dividing the block
    3. When deallocating, check if the buddy block is also free to merge them
    4. Continue merging up the binary tree until no more merges are possible
    
    This approach ensures that allocation and deallocation operations are O(log n),
    where n is the size of the memory pool.
    """

    def __init__(self, initial_size: int = 2**30):
        """
        Initialize buddy allocator with a given initial size.

        The buddy allocator uses a power-of-2 approach where memory blocks are split
        and combined based on binary tree principles. This helps minimize external
        fragmentation while maintaining efficient allocation and deallocation.
        
        The initial memory pool is set up as a single large block. As allocations
        occur, this block is recursively split. When deallocations occur, adjacent
        "buddy" blocks are merged to reduce fragmentation.

        Args:
            initial_size: Initial size of the memory pool in bytes (default: 1GB)
                Must be a positive integer. Will be adjusted to the next power of 2.
                The maximum allocation size will be equal to this initial size.

        Raises:
            ValueError: If initial_size is not positive
            TypeError: If initial_size is not an integer
        """
        if not isinstance(initial_size, int):
            raise TypeError(f"Initial size must be an integer, got {type(initial_size)}")
        if initial_size <= 0:
            raise ValueError(f"Initial size must be positive, got {initial_size}")

        # Ensure initial size is a power of 2
        self.initial_size = self._next_power_of_2(initial_size)
        self.max_order = self._log2(self.initial_size)

        # Initialize free lists for each order (size = 2^order)
        # free_lists[i] contains addresses of all free blocks of size 2^i
        self.free_lists: List[List[int]] = [[] for _ in range(self.max_order + 1)]

        # Keep track of allocated blocks: {start_addr: (size, order)}
        self.allocated_blocks = {}

        # Create initial large block at address 0
        self.free_lists[self.max_order].append(0)

        # Thread lock for thread safety - using RLock allows the same thread to acquire multiple times
        self._lock = threading.RLock()

        # Statistics tracking allocation and deallocation patterns
        self.stats = {
            'allocations': 0,              # Total number of allocation requests
            'deallocations': 0,            # Total number of deallocation requests
            'total_requested': 0,          # Total bytes requested by user
            'total_allocated': 0,          # Total bytes allocated (including internal fragmentation)
            'internal_fragmentation': 0    # Wasted space due to power-of-2 rounding
        }

    def _next_power_of_2(self, x: int) -> int:
        """
        Calculate the next power of 2 that is greater than or equal to x.

        This is used to ensure all memory requests are rounded up to power-of-2
        boundaries, which is required for the buddy allocation algorithm to work.

        Args:
            x: Input value to find the next power of 2 for

        Returns:
            The next power of 2 >= x

        Raises:
            ValueError: If x is not positive
            TypeError: If x is not an integer
        """
        if not isinstance(x, int):
            raise TypeError(f"Input must be an integer, got {type(x)}")
        if x <= 0:
            raise ValueError(f"Input must be positive, got {x}")
        
        if x == 1:
            return 1
        
        # Use bit manipulation to efficiently find next power of 2
        # Example: for x=10 (1010), this produces 10000 (16)
        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        return x + 1

    def _log2(self, x: int) -> int:
        """
        Calculate the base-2 logarithm of x, rounded down.

        This is used to determine the "order" of a memory block, where an order-n
        block has size 2^n bytes.

        Args:
            x: Input value to calculate log2 for (must be a power of 2)

        Returns:
            The base-2 logarithm of x

        Raises:
            ValueError: If x is not positive
            TypeError: If x is not an integer
        """
        if not isinstance(x, int):
            raise TypeError(f"Input must be an integer, got {type(x)}")
        if x <= 0:
            raise ValueError(f"Input must be positive, got {x}")
        
        if x == 1:
            return 0
            
        return int(np.log2(x))

    def allocate(self, size: int) -> Optional[int]:
        """
        Allocate a block of at least 'size' bytes using the buddy allocation algorithm.

        The method finds the smallest block that can accommodate the request, splits
        it if necessary, and returns the starting address of the allocated block.
        The size is rounded up to the next power of 2, so internal fragmentation
        may occur.

        The time complexity is O(log(max_order)) which is typically O(log(pool_size)).

        Args:
            size: Requested size in bytes. Must be positive.

        Returns:
            The address of the allocated block, or None if allocation fails
            (typically due to insufficient memory).

        Raises:
            ValueError: If size is not positive
            TypeError: If size is not an integer
            MemoryPoolError: If an internal error occurs during allocation
        """
        if not isinstance(size, int):
            raise TypeError(f"Size must be an integer, got {type(size)}")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        with self._lock:
            try:
                # Round up to next power of 2 for buddy system requirements
                alloc_size: int = self._next_power_of_2(size)
                order: int = self._log2(alloc_size)

                if order > self.max_order:
                    logging.warning(f"Allocation request of {size} bytes (rounded to {alloc_size}) "
                                  f"exceeds maximum pool size of {self.initial_size} bytes")
                    return None  # Request too large for available memory

                # Find a suitable block by checking increasing orders
                for curr_order in range(order, self.max_order + 1):
                    if self.free_lists[curr_order]:
                        # Found a free block of sufficient size
                        addr: int = self.free_lists[curr_order].pop()

                        # Split the block down to the required size
                        # This creates the exact size needed and puts extra blocks back in free lists
                        while curr_order > order:
                            curr_order -= 1
                            # Split the current block in half
                            buddy_addr: int = addr + (1 << curr_order)
                            # Put the second half back in the free list
                            self.free_lists[curr_order].append(buddy_addr)

                        # Record the allocation with its actual size and order
                        actual_size: int = 1 << order
                        self.allocated_blocks[addr] = (actual_size, order)

                        # Update statistics for monitoring and analysis
                        self.stats['allocations'] += 1
                        self.stats['total_requested'] += size
                        self.stats['total_allocated'] += actual_size
                        self.stats['internal_fragmentation'] += (actual_size - size)

                        logging.debug(f"Allocated block at address {addr} of size {size} "
                                    f"(rounded to {actual_size} at order {order})")
                        return addr

                # No suitable block found - pool is fragmented or exhausted
                logging.warning(f"No suitable block found for allocation of {size} bytes")
                return None
            except Exception as e:
                logging.error(f"Error during allocation of {size} bytes: {e}")
                raise MemoryPoolError(f"Allocation failed: {e}")

    def deallocate(self, addr: int) -> bool:
        """
        Deallocate the block at 'addr' and attempt to merge with free buddy blocks.

        The deallocation process involves returning the block to the free list and
        checking if the "buddy" block at the same level is also free. If so, they
        are merged, and the process continues up the binary tree until no more
        merges are possible.

        This coalescing process helps reduce external fragmentation over time.

        Args:
            addr: Address of the block to deallocate. Must be an address previously
                returned by allocate() and not yet deallocated.

        Returns:
            True if successful, False if the address was not allocated.

        Raises:
            TypeError: If addr is not an integer
            MemoryPoolError: If an internal error occurs during deallocation
        """
        if not isinstance(addr, int):
            raise TypeError(f"Address must be an integer, got {type(addr)}")

        with self._lock:
            try:
                # Verify the address was actually allocated
                if addr not in self.allocated_blocks:
                    logging.warning(f"Attempted to deallocate unallocated address: {addr}")
                    return False

                size, order = self.allocated_blocks[addr]

                # Remove from allocated blocks tracking
                del self.allocated_blocks[addr]

                # Attempt to coalesce with buddy blocks
                # The buddy of a block at address A of size 2^n is at address A XOR 2^n
                curr_addr = addr
                curr_order = order

                while curr_order < self.max_order:
                    buddy_addr = curr_addr ^ (1 << curr_order)

                    # Check if the buddy block is in the same order's free list
                    if buddy_addr in self.free_lists[curr_order]:
                        # Remove buddy from free list to merge
                        self.free_lists[curr_order].remove(buddy_addr)

                        # Keep the lower address for the merged block
                        if buddy_addr < curr_addr:
                            curr_addr = buddy_addr

                        curr_order += 1  # Move to higher order (larger block)
                    else:
                        # Buddy is not free, stop merging
                        break

                # Add the (possibly merged) block to the appropriate free list
                self.free_lists[curr_order].append(curr_addr)

                # Update statistics
                self.stats['deallocations'] += 1

                logging.debug(f"Deallocated block at address {addr} of size {size}")
                return True
            except Exception as e:
                logging.error(f"Error during deallocation of address {addr}: {e}")
                raise MemoryPoolError(f"Deallocation failed: {e}")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive allocation statistics for monitoring and analysis.

        This method provides insights into memory utilization, fragmentation,
        and allocation patterns. These statistics are useful for:
        - Performance optimization
        - Memory leak detection
        - Capacity planning
        - Debugging memory issues

        Returns:
            Dictionary containing various allocation statistics:
            - allocations: Total number of allocation requests
            - deallocations: Total number of deallocation requests
            - total_requested: Total bytes requested by users
            - total_allocated: Total bytes allocated (including fragmentation)
            - internal_fragmentation: Memory wasted due to power-of-2 rounding
            - total_free: Currently free memory in bytes
            - total_memory: Total size of the memory pool
            - utilization: Percentage of pool currently in use
        """
        with self._lock:
            stats = self.stats.copy()

            # Calculate total free memory by summing all free blocks
            total_free = 0
            for order, free_list in enumerate(self.free_lists):
                total_free += len(free_list) * (1 << order)

            # Calculate utilization percentage
            actual_used = self.stats['total_allocated'] - self.stats['internal_fragmentation']
            utilization = actual_used / self.initial_size if self.initial_size > 0 else 0

            stats['total_free'] = total_free
            stats['total_memory'] = self.initial_size
            stats['utilization'] = utilization

            return stats

    def is_valid_address(self, addr: int) -> bool:
        """
        Check if an address is valid (currently allocated).

        This method is useful for debugging and validation to ensure that
        deallocation requests are for actually allocated addresses.

        Args:
            addr: Address to validate

        Returns:
            True if the address is currently allocated, False otherwise
        """
        with self._lock:
            return addr in self.allocated_blocks


class TensorCache:
    """
    Pre-allocated tensor cache for frequently used tensor shapes and dtypes.
    
    The tensor cache stores unused tensors of common shapes for quick reuse,
    avoiding the overhead of tensor creation and destruction. This is particularly
    beneficial for transformer models where many small tensors with fixed shapes
    are created and destroyed frequently during inference.
    
    The cache is thread-safe and includes limits to prevent unlimited memory growth.
    """

    def __init__(self, max_cache_size_per_key: int = 10):
        """
        Initialize tensor cache with size limits.

        The cache is organized by (shape, dtype) tuples, allowing efficient lookup
        of cached tensors that match requested parameters.

        Args:
            max_cache_size_per_key: Maximum number of tensors to cache per
                shape/dtype combination. This prevents unlimited memory growth
                for any single tensor configuration.

        Raises:
            ValueError: If max_cache_size_per_key is not positive
        """
        if not isinstance(max_cache_size_per_key, int) or max_cache_size_per_key <= 0:
            raise ValueError(f"max_cache_size_per_key must be a positive integer, "
                           f"got {max_cache_size_per_key}")
        
        self.max_cache_size_per_key = max_cache_size_per_key
        
        # Cache structure: {(shape, dtype): [tensor_list]}
        # Each key maps to a list of available cached tensors
        self.cache: Dict[Tuple[Tuple[int, ...], torch.dtype], List[torch.Tensor]] = defaultdict(list)
        
        # Statistics for cache performance monitoring
        self.stats = {
            'cache_hits': 0,        # Reused tensors from cache
            'cache_misses': 0,      # New tensors created
            'total_requests': 0     # Total requests
        }
        
        # Thread lock for thread-safe operations
        self._lock = threading.RLock()

    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a tensor of specified shape and dtype, reusing from cache if possible.

        If a cached tensor exists for the requested shape and dtype, it is returned
        from the cache. Otherwise, a new tensor is created. This reduces allocation
        overhead for frequently used tensor shapes.

        Args:
            shape: Shape of the tensor to retrieve/create
            dtype: Data type of the tensor (default: torch.float32)

        Returns:
            A tensor of the specified shape and dtype

        Raises:
            TypeError: If shape is not a tuple or dtype is not a torch.dtype
            MemoryPoolError: If tensor creation fails
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
                    # Cache hit - return tensor from cache
                    tensor = self.cache[key].pop()
                    self.stats['cache_hits'] += 1
                    logging.debug(f"Cache hit for shape {shape}, dtype {dtype}")
                    return tensor
                else:
                    # Cache miss - create new tensor
                    self.stats['cache_misses'] += 1
                    logging.debug(f"Cache miss for shape {shape}, dtype {dtype}, creating new tensor")
                    return torch.empty(shape, dtype=dtype)
            except Exception as e:
                logging.error(f"Error getting tensor from cache: {e}")
                raise MemoryPoolError(f"Failed to get tensor from cache: {e}")

    def return_tensor(self, tensor: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> bool:
        """
        Return a tensor to the cache for future reuse.

        The tensor is added to the cache if space is available. If the cache
        for this shape/dtype is already at its maximum size, the tensor is
        not cached to prevent unlimited memory growth.

        Args:
            tensor: Tensor to return to cache
            shape: Shape of the tensor (should match tensor.shape, but provided separately
                   to avoid depending on the tensor still being valid)
            dtype: Data type of the tensor

        Returns:
            True if successfully cached, False if cache was full

        Raises:
            TypeError: If inputs are of incorrect type
            MemoryPoolError: If cache operation fails
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

                # Only cache if we haven't reached the limit for this key
                if len(self.cache[key]) < self.max_cache_size_per_key:
                    self.cache[key].append(tensor)
                    logging.debug(f"Returned tensor to cache for shape {shape}, dtype {dtype}")
                    return True
                else:
                    # Cache is full for this key, cannot cache
                    logging.debug(f"Cache is full for shape {shape}, dtype {dtype}, not caching tensor")
                    return False
            except Exception as e:
                logging.error(f"Error returning tensor to cache: {e}")
                raise MemoryPoolError(f"Failed to return tensor to cache: {e}")

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics for performance monitoring.

        Returns:
            Dictionary containing cache performance statistics:
            - cache_hits: Number of times a tensor was reused from cache
            - cache_misses: Number of times a new tensor had to be created
            - total_requests: Total number of requests
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - cache_size: Total number of cached tensors
            - max_cache_size_per_key: Maximum cache size per key
        """
        with self._lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            stats['cache_size'] = sum(len(tensors) for tensors in self.cache.values())
            stats['max_cache_size_per_key'] = self.max_cache_size_per_key
            return stats

    def clear_cache(self):
        """
        Clear all cached tensors to free memory.

        This is useful when the cache has grown large or when memory needs to be
        freed for other purposes. Note that this will cause cache misses until
        tensors are cached again.
        """
        with self._lock:
            for key in list(self.cache.keys()):
                del self.cache[key]
            logging.info("Tensor cache cleared")


class MemoryPool:
    """
    Main memory pool class that combines buddy allocation and tensor caching.
    
    This unified memory management system provides both low-level memory allocation
    (using buddy allocation) and high-level tensor management (using tensor caching).
    It's designed to work efficiently with PyTorch tensors and the memory patterns
    typical of transformer models.
    
    The system tracks tensor allocations to enable proper cleanup and implements
    thread-safe operations for concurrent use in multi-threaded environments.
    """

    def __init__(self, initial_size: int = 2**30, max_cache_size_per_key: int = 10):
        """
        Initialize the unified memory pool system.

        The memory pool combines two complementary approaches:
        1. Buddy allocation for efficient large-scale memory management
        2. Tensor caching for efficient reuse of common tensor shapes

        Args:
            initial_size: Initial size of the buddy allocator in bytes (default: 1GB)
            max_cache_size_per_key: Maximum cached tensors per shape/dtype combination

        Raises:
            ValueError: If initial_size or max_cache_size_per_key is not positive
        """
        if not isinstance(initial_size, int) or initial_size <= 0:
            raise ValueError(f"Initial size must be a positive integer, got {initial_size}")
        if not isinstance(max_cache_size_per_key, int) or max_cache_size_per_key <= 0:
            raise ValueError(f"max_cache_size_per_key must be a positive integer, got {max_cache_size_per_key}")

        self.buddy_allocator = BuddyAllocator(initial_size)
        self.tensor_cache = TensorCache(max_cache_size_per_key)
        self._lock = threading.RLock()

        # Track allocated tensors for proper deallocation
        # Structure: {tensor_id: (allocator_address, shape, dtype, device)}
        self.allocated_tensors = {}

        # Common tensor shapes for transformer models to pre-cache
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

        # Pre-allocate common tensors to warm up the cache
        self._preallocate_common_tensors()

    def _preallocate_common_tensors(self):
        """
        Pre-allocate commonly used tensor shapes to warm up the cache.

        This method creates and caches common tensor shapes at initialization
        time, which can improve performance for applications that frequently
        use these shapes.
        """
        for shape, dtype in self.common_shapes:
            # Pre-allocate multiple instances of each common shape
            for _ in range(3):  # Pre-allocate 3 of each common shape
                try:
                    tensor = torch.empty(shape, dtype=dtype)
                    self.tensor_cache.return_tensor(tensor, shape, dtype)
                    logging.debug(f"Pre-allocated tensor of shape {shape}, dtype {dtype}")
                except Exception as e:
                    logging.warning(f"Failed to pre-allocate tensor {shape} with dtype {dtype}: {e}")

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                       device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """
        Allocate a tensor with the specified shape and dtype.

        This method attempts to reuse a cached tensor if available. If not,
        it creates a new tensor. The tensor is tracked for proper deallocation
        and memory management.

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor (default: torch.float32)
            device: Device to allocate the tensor on (default: 'cpu')

        Returns:
            A tensor of the specified shape and dtype on the requested device

        Raises:
            TypeError: If shape is not a tuple, dtype is not torch.dtype, or
                      device is not a valid device specification
            ValueError: If any dimension in shape is not positive
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
            logging.debug(f"Got tensor from cache for shape {shape}, dtype {dtype}")
        except MemoryPoolError:
            # If cache fails, create a new tensor directly
            logging.debug(f"Creating new tensor for shape {shape}, dtype {dtype}")
            tensor = torch.empty(shape, dtype=dtype, device=device)

        # Ensure tensor is on the correct device
        if tensor.device != device:
            tensor = tensor.to(device)

        # Track the allocation for proper cleanup
        tensor_id = id(tensor)
        with self._lock:
            self.allocated_tensors[tensor_id] = (None, shape, dtype, device)

        logging.debug(f"Allocated tensor of shape {shape}, dtype {dtype}, device {device}")
        return tensor

    def deallocate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Deallocate a tensor and return it to the cache if appropriate.

        This method properly cleans up tensor tracking and attempts to cache
        the tensor for future reuse. It's the counterpart to allocate_tensor().

        Args:
            tensor: Tensor to deallocate

        Returns:
            True if the tensor was successfully returned to the cache, False otherwise

        Raises:
            TypeError: If tensor is not a torch.Tensor
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")

        tensor_id = id(tensor)

        with self._lock:
            # Remove from tracking if it was tracked
            if tensor_id in self.allocated_tensors:
                shape, dtype, device = self.allocated_tensors[tensor_id][1], \
                                    self.allocated_tensors[tensor_id][2], \
                                    self.allocated_tensors[tensor_id][3]
                del self.allocated_tensors[tensor_id]
            else:
                # If tensor wasn't tracked, use its actual properties
                shape = tuple(tensor.shape)
                dtype = tensor.dtype
                device = tensor.device

        # Return to cache for potential reuse
        try:
            success = self.tensor_cache.return_tensor(tensor, shape, dtype)
            if success:
                logging.debug(f"Deallocated tensor of shape {shape}, dtype {dtype}, returned to cache")
            else:
                logging.debug(f"Deallocated tensor of shape {shape}, dtype {dtype}, cache was full")
            return success
        except MemoryPoolError:
            logging.warning(f"Failed to return tensor to cache: {shape}, {dtype}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory pool statistics.

        This method provides detailed statistics from both the buddy allocator
        and tensor cache components, giving a complete picture of memory usage.

        Returns:
            Dictionary with statistics from both components:
            - buddy_allocator: Stats from buddy allocation system
            - tensor_cache: Stats from tensor caching system
            - total_allocated_tensors: Count of currently tracked tensors
        """
        buddy_stats = self.buddy_allocator.get_stats()
        cache_stats = self.tensor_cache.get_cache_stats()

        return {
            'buddy_allocator': buddy_stats,
            'tensor_cache': cache_stats,
            'total_allocated_tensors': len(self.allocated_tensors)
        }

    def defragment(self):
        """
        Perform memory defragmentation.

        Currently, this method primarily provides a status report on memory
        fragmentation in the buddy allocator system. In a complete implementation,
        this would include operations to reduce fragmentation.
        """
        with self._lock:
            # In this implementation, we just report fragmentation status
            stats = self.buddy_allocator.get_stats()
            logging.info(f"Current memory utilization: {stats['utilization']:.4f} "
                        f"({stats['total_allocated']} / {stats['total_memory']} bytes)")
            return stats

    def reset(self):
        """
        Reset the memory pool to its initial state.

        This clears all allocated tensor tracking, clears the tensor cache,
        and reinitializes the buddy allocator. Use with caution as this
        will make any currently allocated tensors invalid.
        """
        with self._lock:
            # Clear all tracking
            self.allocated_tensors.clear()
            # Clear tensor cache
            self.tensor_cache.clear_cache()
            # Reinitialize buddy allocator with same size
            self.buddy_allocator = BuddyAllocator(self.buddy_allocator.initial_size)
            # Reinitialize cache and pre-allocate common tensors
            self._preallocate_common_tensors()
            logging.info("Memory pool reset to initial state")


# Global memory pool instance for application-wide use
_global_memory_pool: Optional[MemoryPool] = None
_pool_lock = threading.RLock()


def get_memory_pool() -> MemoryPool:
    """
    Get the global memory pool instance.

    This function implements a thread-safe singleton pattern to provide
    a single memory pool instance for the entire application. This helps
    ensure efficient memory usage and proper resource management.

    Returns:
        The global memory pool instance
    """
    global _global_memory_pool
    if _global_memory_pool is None:
        with _pool_lock:
            if _global_memory_pool is None:
                _global_memory_pool = MemoryPool()
    return _global_memory_pool


def allocate_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
    """
    Allocate a tensor using the global memory pool.

    This is a convenience function that provides access to the global memory
    pool for easy tensor allocation with memory management benefits.

    Args:
        shape: Shape of the tensor to allocate
        dtype: Data type of the tensor (default: torch.float32)
        device: Device to allocate the tensor on (default: 'cpu')

    Returns:
        A tensor of the specified shape and dtype on the requested device
    """
    pool = get_memory_pool()
    return pool.allocate_tensor(shape, dtype, device)


def deallocate_tensor(tensor: torch.Tensor) -> bool:
    """
    Deallocate a tensor using the global memory pool.

    This is a convenience function that provides access to the global memory
    pool for tensor deallocation and potential caching.

    Args:
        tensor: Tensor to deallocate

    Returns:
        True if the tensor was successfully returned to the cache, False otherwise
    """
    pool = get_memory_pool()
    return pool.deallocate_tensor(tensor)


if __name__ == "__main__":
    # Example usage and testing
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

        # Allocate tensors
        tensor1 = pool.allocate_tensor((100, 200), torch.float32)
        tensor2 = pool.allocate_tensor((50, 100, 256), torch.float32)

        print(f"Allocated tensors of shapes: {tensor1.shape}, {tensor2.shape}")

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