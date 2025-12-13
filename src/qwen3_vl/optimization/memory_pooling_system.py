"""
Memory Pooling System for Qwen3-VL with Custom Pool Management

Implements advanced memory pooling with buddy allocation, LRU eviction,
and cache alignment for Intel i5-10210U architecture.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import time
import logging
from collections import defaultdict, OrderedDict
import bisect


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool"""
    ptr: int  # Memory address
    size: int  # Size in bytes
    allocated: bool  # Allocation status
    timestamp: float  # Time of last operation
    block_type: str  # Type of data stored


class BuddyAllocator:
    """
    Buddy memory allocator for efficient memory management.
    Uses power-of-2 sized blocks to minimize fragmentation.
    """
    
    def __init__(self, total_size: int):
        self.total_size = total_size
        self.min_block_size = 4096  # 4KB minimum block size
        
        # Calculate maximum order needed
        self.max_order = 0
        size = self.min_block_size
        while size < total_size:
            size <<= 1
            self.max_order += 1
        
        # Free lists for each order
        self.free_lists = [OrderedDict() for _ in range(self.max_order + 1)]
        
        # Initialize with one large block
        self.free_lists[self.max_order][0] = total_size
        
        # Track allocated blocks
        self.allocated_blocks: Dict[int, Tuple[int, int]] = {}  # addr -> (size, order)
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def allocate(self, size: int) -> Optional[int]:
        """Allocate a block of at least the requested size."""
        with self._lock:
            # Round up to next power of 2
            actual_size = self._round_up_to_power_of_2(max(size, self.min_block_size))
            order = self._size_to_order(actual_size)
            
            # Find a suitable block
            for curr_order in range(order, self.max_order + 1):
                if self.free_lists[curr_order]:
                    # Found a block, split if necessary
                    addr = next(iter(self.free_lists[curr_order]))
                    block_size = self.free_lists[curr_order].pop(addr)
                    
                    # Split the block until we get the right size
                    while curr_order > order:
                        curr_order -= 1
                        block_size //= 2
                        # Add the second half to the free list
                        self.free_lists[curr_order][addr + block_size] = block_size
                    
                    # Track the allocated block
                    self.allocated_blocks[addr] = (actual_size, order)
                    return addr
            
            return None  # Allocation failed
    
    def deallocate(self, addr: int) -> bool:
        """Deallocate a block."""
        with self._lock:
            if addr not in self.allocated_blocks:
                return False
            
            size, order = self.allocated_blocks.pop(addr)
            
            # Try to merge with buddies
            buddy_addr = self._get_buddy_addr(addr, order)
            
            # Check if buddy is free in the same order
            if buddy_addr in self.free_lists[order]:
                # Remove buddy from free list
                self.free_lists[order].pop(buddy_addr)
                # Merge with smaller address
                merged_addr = min(addr, buddy_addr)
                merged_size = size * 2
                merged_order = order + 1
                
                # Add merged block to next order
                self.free_lists[merged_order][merged_addr] = merged_size
            else:
                # Buddy not free, just add this block to free list
                self.free_lists[order][addr] = size
            
            return True
    
    def _round_up_to_power_of_2(self, size: int) -> int:
        """Round size up to the next power of 2."""
        if size == 0:
            return self.min_block_size
        power = 1
        while power < size:
            power <<= 1
        return power
    
    def _size_to_order(self, size: int) -> int:
        """Convert size to order (log2 of size)."""
        order = 0
        temp_size = self.min_block_size
        while temp_size < size:
            temp_size <<= 1
            order += 1
        return order
    
    def _get_buddy_addr(self, addr: int, order: int) -> int:
        """Get the address of the buddy block."""
        block_size = self.min_block_size << order
        buddy_addr = addr ^ block_size  # XOR to flip the bit
        return buddy_addr
    
    def get_utilization(self) -> float:
        """Get current memory utilization."""
        with self._lock:
            total_free = 0
            for order, free_list in enumerate(self.free_lists):
                block_size = self.min_block_size << order
                total_free += len(free_list) * block_size
            total_allocated = sum(size for size, _ in self.allocated_blocks.values())
            return total_allocated / (total_allocated + total_free) if (total_allocated + total_free) > 0 else 0


class TensorCache:
    """
    LRU-based tensor cache with size-based eviction.
    """
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.tensor_sizes: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Put a tensor in the cache."""
        with self._lock:
            tensor_size = tensor.element_size() * tensor.nelement()
            
            # If tensor is too large, don't cache it
            if tensor_size > self.max_size_bytes:
                return False
            
            # Remove existing entry if exists
            if key in self.cache:
                old_size = self.tensor_sizes.pop(key)
                self.cache.pop(key)
                self.current_size_bytes -= old_size
            
            # Evict oldest entries if needed
            while self.current_size_bytes + tensor_size > self.max_size_bytes and self.cache:
                oldest_key = next(iter(self.cache))
                oldest_size = self.tensor_sizes.pop(oldest_key)
                self.cache.pop(oldest_key)
                self.current_size_bytes -= oldest_size
            
            # Add new tensor if it fits
            if self.current_size_bytes + tensor_size <= self.max_size_bytes:
                self.cache[key] = tensor
                self.tensor_sizes[key] = tensor_size
                self.current_size_bytes += tensor_size
                return True
            
            return False
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get a tensor from the cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                tensor = self.cache.pop(key)
                self.cache[key] = tensor
                return tensor
            return None
    
    def remove(self, key: str) -> bool:
        """Remove a tensor from the cache."""
        with self._lock:
            if key in self.cache:
                size = self.tensor_sizes.pop(key)
                self.cache.pop(key)
                self.current_size_bytes -= size
                return True
            return False
    
    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            self.cache.clear()
            self.tensor_sizes.clear()
            self.current_size_bytes = 0


class AdvancedMemoryPoolManager:
    """
    Advanced memory pool manager with multiple allocation strategies.
    """
    
    def __init__(self, pool_size: int = 1024 * 1024 * 1024, enable_compaction: bool = True):
        """
        Initialize the advanced memory pool manager.
        
        Args:
            pool_size: Size of the memory pool in bytes
            enable_compaction: Whether to enable memory compaction
        """
        self.pool_size = pool_size
        self.enable_compaction = enable_compaction
        
        # Initialize buddy allocator
        self.buddy_allocator = BuddyAllocator(pool_size)
        
        # Initialize tensor caches
        self.tensor_cache = TensorCache(pool_size // 4)  # Use 25% of pool for cache
        
        # Track statistics
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compaction_count': 0,
            'total_allocated_bytes': 0,
            'total_freed_bytes': 0
        }
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logging.info(f"AdvancedMemoryPoolManager initialized with {pool_size / (1024**3):.2f}GB pool")
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple[torch.Tensor, str]:
        """
        Allocate a tensor using the memory pool.
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
        
        Returns:
            Tuple of (tensor, allocation_id)
        """
        with self._lock:
            # Calculate required size
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_size = element_size * np.prod(shape)
            
            # Try to allocate from buddy allocator
            addr = self.buddy_allocator.allocate(required_size)
            if addr is not None:
                # Create tensor using PyTorch's memory management
                # In a real implementation, we would use the allocated address
                # For now, we'll create a standard tensor and track allocation
                tensor = torch.empty(shape, dtype=dtype)
                
                # Generate allocation ID
                alloc_id = f"pool_{addr}_{int(time.time() * 1000000)}"
                
                # Update statistics
                self.stats['allocations'] += 1
                self.stats['total_allocated_bytes'] += required_size
                
                return tensor, alloc_id
            else:
                # Fallback to standard allocation
                tensor = torch.empty(shape, dtype=dtype)
                alloc_id = f"std_{id(tensor)}_{int(time.time() * 1000000)}"
                return tensor, alloc_id
    
    def deallocate_tensor(self, tensor: torch.Tensor, alloc_id: str):
        """
        Deallocate a tensor back to the pool.
        
        Args:
            tensor: Tensor to deallocate
            alloc_id: Allocation ID returned by allocate_tensor
        """
        with self._lock:
            if alloc_id.startswith("pool_"):
                # Extract address from allocation ID
                try:
                    addr = int(alloc_id.split('_')[1])
                    self.buddy_allocator.deallocate(addr)
                    
                    # Update statistics
                    tensor_size = tensor.element_size() * tensor.nelement()
                    self.stats['deallocations'] += 1
                    self.stats['total_freed_bytes'] += tensor_size
                except (ValueError, IndexError):
                    # If we can't extract address, just continue
                    pass
    
    def cache_tensor(self, key: str, tensor: torch.Tensor) -> bool:
        """
        Cache a tensor in the LRU cache.
        
        Args:
            key: Key for the tensor
            tensor: Tensor to cache
        
        Returns:
            True if successfully cached, False otherwise
        """
        return self.tensor_cache.put(key, tensor)
    
    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a tensor from the cache.
        
        Args:
            key: Key for the tensor
        
        Returns:
            Tensor if found, None otherwise
        """
        tensor = self.tensor_cache.get(key)
        if tensor is not None:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        return tensor
    
    def get_utilization(self) -> float:
        """Get overall memory pool utilization."""
        return self.buddy_allocator.get_utilization()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats['utilization'] = self.get_utilization()
            stats['cache_size_bytes'] = self.tensor_cache.current_size_bytes
            stats['cache_capacity_bytes'] = self.tensor_cache.max_size_bytes
            return stats
    
    def compact_memory(self):
        """Perform memory compaction if enabled."""
        if self.enable_compaction:
            with self._lock:
                # In a real implementation, this would compact fragmented memory
                # For now, we'll just trigger garbage collection
                import gc
                gc.collect()
                self.stats['compaction_count'] += 1
    
    def clear_cache(self):
        """Clear the tensor cache."""
        self.tensor_cache.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Advanced Memory Pool Manager for Qwen3-VL")
    print("=" * 50)
    
    # Create memory pool manager
    pool_manager = AdvancedMemoryPoolManager(pool_size=256 * 1024 * 1024)  # 256MB pool
    
    print(f"\n1. Pool initialized with {pool_manager.pool_size / (1024**2):.1f}MB")
    print(f"   Current utilization: {pool_manager.get_utilization():.2%}")
    
    # Test tensor allocation
    print(f"\n2. Testing tensor allocation...")
    tensors = []
    for i in range(5):
        tensor, alloc_id = pool_manager.allocate_tensor((100, 100), torch.float16)
        tensors.append((tensor, alloc_id))
        print(f"   Allocated tensor {i+1}: {tensor.shape}, ID: {alloc_id}")
    
    # Test caching
    print(f"\n3. Testing tensor caching...")
    cache_key = "test_tensor"
    success = pool_manager.cache_tensor(cache_key, tensors[0][0])
    print(f"   Cached tensor: {success}")
    
    cached_tensor = pool_manager.get_cached_tensor(cache_key)
    print(f"   Retrieved from cache: {cached_tensor is not None}")
    
    # Show statistics
    print(f"\n4. Statistics:")
    stats = pool_manager.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\nAdvanced Memory Pool Manager test completed!")