"""
Memory allocation system for Qwen3-VL model.

This module implements efficient memory allocation with buddy allocation
and memory pooling for optimal performance.
"""
import torch
from typing import Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class AllocationInfo:
    """Information about a memory allocation."""
    size: int
    device: str
    dtype: torch.dtype
    ptr: int  # Memory pointer address


class MemoryAllocator:
    """
    Memory allocator with buddy allocation system and memory pooling.
    """
    
    def __init__(self, initial_pool_size: int = 256 * 1024 * 1024,  # 256MB
                 max_pool_size: int = 1024 * 1024 * 1024,  # 1GB
                 use_buddy_allocation: bool = True):
        """
        Initialize the memory allocator.
        
        Args:
            initial_pool_size: Initial size of the memory pool
            max_pool_size: Maximum size of the memory pool
            use_buddy_allocation: Whether to use buddy allocation system
        """
        self.initial_pool_size = initial_pool_size
        self.max_pool_size = max_pool_size
        self.use_buddy_allocation = use_buddy_allocation
        
        # Initialize buddy allocation system if enabled
        if use_buddy_allocation:
            self.buddy_allocator = BuddyAllocator(initial_pool_size, max_pool_size)
        else:
            self.buddy_allocator = None
        
        # Track allocations
        self.allocations: dict = {}
        self.tensor_cache = {}  # Cache for frequently used tensor shapes

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype, device: str = "cuda") -> torch.Tensor:
        """
        Allocate memory for a tensor with the specified shape and dtype.

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            device: Device to allocate on ('cuda', 'cpu', etc.)

        Returns:
            Allocated tensor
        """
        # Calculate the size needed
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        size = num_elements * element_size
        
        # Try to use buddy allocator if enabled
        if self.use_buddy_allocation and self.buddy_allocator:
            # Allocate using buddy system
            ptr = self.buddy_allocator.allocate(size)
            if ptr is not None:
                # Create tensor from pre-allocated memory
                tensor = torch.empty(shape, dtype=dtype, device=device)
                self.allocations[id(tensor)] = AllocationInfo(size, device, dtype, ptr)
                return tensor
        
        # Fallback to standard allocation
        tensor = torch.empty(shape, dtype=dtype, device=device)
        self.allocations[id(tensor)] = AllocationInfo(size, device, dtype, id(tensor))
        return tensor

    def free(self, tensor: torch.Tensor):
        """
        Free the memory allocated for a tensor.
        
        Args:
            tensor: Tensor to free
        """
        tensor_id = id(tensor)
        if tensor_id in self.allocations:
            alloc_info = self.allocations[tensor_id]
            
            # Free using buddy allocator if enabled
            if self.use_buddy_allocation and self.buddy_allocator:
                self.buddy_allocator.free(alloc_info.ptr, alloc_info.size)
            
            # Remove from allocations
            del self.allocations[tensor_id]

    def get_stats(self) -> dict:
        """
        Get memory allocation statistics.
        
        Returns:
            Dictionary containing allocation statistics
        """
        total_allocated = sum(alloc.size for alloc in self.allocations.values())
        num_allocations = len(self.allocations)
        
        stats = {
            "total_allocated_bytes": total_allocated,
            "num_active_allocations": num_allocations,
            "use_buddy_allocation": self.use_buddy_allocation
        }
        
        if self.buddy_allocator:
            stats.update(self.buddy_allocator.get_stats())
        
        return stats


class BuddyAllocator:
    """
    Buddy memory allocation system for efficient memory management.
    """
    
    def __init__(self, initial_size: int, max_size: int):
        """
        Initialize the buddy allocator.
        
        Args:
            initial_size: Initial size of the memory pool
            max_size: Maximum size of the memory pool
        """
        self.initial_size = initial_size
        self.max_size = max_size
        self.pool_size = initial_size
        
        # Calculate the order (power of 2) for the pool size
        self.max_order = self._calculate_order(self.pool_size)
        
        # Initialize free lists for each order
        self.free_lists = [[] for _ in range(self.max_order + 1)]
        
        # Start with one block of maximum size
        self.free_lists[self.max_order].append((0, self.pool_size))
        
        # Track allocated blocks
        self.allocated_blocks = {}
    
    def _calculate_order(self, size: int) -> int:
        """Calculate the order (power of 2) for a given size."""
        if size <= 0:
            return 0
        import math
        return int(math.ceil(math.log2(size)))
    
    def _size_from_order(self, order: int) -> int:
        """Calculate the size from an order."""
        return 1 << order
    
    def allocate(self, size: int) -> Optional[int]:
        """
        Allocate a block of memory of at least the specified size.
        
        Args:
            size: Size of memory to allocate
            
        Returns:
            Pointer to the allocated memory block, or None if allocation failed
        """
        # Round up to the next power of 2
        order = self._calculate_order(size)
        if order > self.max_order:
            # Need to expand the pool
            if self.pool_size * 2 <= self.max_size:
                self._expand_pool()
            else:
                return None  # Cannot allocate more memory
        
        # Find a suitable block
        for i in range(order, self.max_order + 1):
            if self.free_lists[i]:
                # Found a block of order i, split if necessary
                addr, block_size = self.free_lists[i].pop()
                
                # Split the block if it's larger than needed
                while i > order:
                    i -= 1
                    block_size //= 2
                    # Put the second half back in the free list
                    self.free_lists[i].append((addr + block_size, block_size))
                
                # Mark as allocated
                self.allocated_blocks[addr] = (size, order)
                return addr
        
        return None  # Allocation failed
    
    def free(self, addr: int, size: int) -> None:
        """
        Free a previously allocated block of memory.

        Args:
            addr: Address of the block to free
            size: Size of the block to free
        """
        if addr not in self.allocated_blocks:
            return  # Block not allocated
        
        # Get the order of the block
        _, order = self.allocated_blocks[addr]
        block_size = self._size_from_order(order)
        
        # Remove from allocated blocks
        del self.allocated_blocks[addr]
        
        # Try to merge with buddies
        current_addr = addr
        current_order = order
        
        while current_order < self.max_order:
            buddy_addr = current_addr ^ self._size_from_order(current_order)
            
            # Check if buddy is free
            buddy_free = False
            buddy_idx = -1
            for i, (free_addr, free_size) in enumerate(self.free_lists[current_order]):
                if free_addr == buddy_addr:
                    buddy_free = True
                    buddy_idx = i
                    break
            
            if buddy_free:
                # Remove buddy from free list
                self.free_lists[current_order].pop(buddy_idx)
                
                # Determine which address to keep (the lower one)
                current_addr = min(current_addr, buddy_addr)
                current_order += 1
            else:
                # Buddy not free, stop merging
                break
        
        # Add the block to the appropriate free list
        self.free_lists[current_order].append((current_addr, self._size_from_order(current_order)))
    
    def _expand_pool(self):
        """Expand the memory pool."""
        new_size = min(self.pool_size * 2, self.max_size)
        if new_size > self.pool_size:
            # Add the new memory to the highest order free list
            self.free_lists[self.max_order].append((self.pool_size, new_size - self.pool_size))
            self.pool_size = new_size
    
    def get_stats(self) -> dict:
        """
        Get buddy allocator statistics.
        
        Returns:
            Dictionary containing allocator statistics
        """
        free_memory = sum(
            size for order_list in self.free_lists 
            for _, size in order_list
        )
        allocated_memory = sum(size for size, _ in self.allocated_blocks.values())
        
        return {
            "pool_size": self.pool_size,
            "allocated_memory": allocated_memory,
            "free_memory": free_memory,
            "max_order": self.max_order
        }