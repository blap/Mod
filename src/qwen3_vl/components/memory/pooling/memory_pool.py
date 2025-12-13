"""
Memory pooling system for Qwen3-VL model.

This module implements efficient memory pooling with pre-allocated caches
for frequently used tensor shapes.
"""
import torch
from typing import Tuple, Dict, Optional, Any
from collections import defaultdict
import gc


class MemoryPool:
    """
    Memory pooling system for efficient tensor allocation and reuse.
    """
    
    def __init__(self, initial_size: int = 128 * 1024 * 1024,  # 128MB
                 max_size: int = 1024 * 1024 * 1024,  # 1GB
                 growth_factor: float = 1.5):
        """
        Initialize the memory pool.
        
        Args:
            initial_size: Initial size of the memory pool
            max_size: Maximum size of the memory pool
            growth_factor: Factor by which to grow the pool when needed
        """
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        
        # Pool of pre-allocated tensors organized by shape and dtype
        self.tensor_pools: Dict[Tuple[Tuple[int, ...], torch.dtype, str], list] = defaultdict(list)
        
        # Track pool statistics
        self.pool_size = 0
        self.max_pool_size = 0
        self.total_allocated = 0
        
        # Initialize with some common tensor shapes
        self._initialize_common_shapes()
    
    def _initialize_common_shapes(self):
        """Initialize the pool with some common tensor shapes."""
        common_shapes = [
            ((512,), torch.float16, "cuda"),
            ((1024,), torch.float16, "cuda"),
            ((2048,), torch.float16, "cuda"),
            ((4096,), torch.float16, "cuda"),
            ((512, 512), torch.float16, "cuda"),
            ((1024, 1024), torch.float16, "cuda"),
            ((2048, 2048), torch.float16, "cuda"),
            ((512,), torch.float32, "cuda"),
            ((1024,), torch.float32, "cuda"),
            ((2048,), torch.float32, "cuda"),
            ((4096,), torch.float32, "cuda"),
            ((512, 512), torch.float32, "cuda"),
            ((1024, 1024), torch.float32, "cuda"),
            ((2048, 2048), torch.float32, "cuda"),
        ]
        
        for shape, dtype, device in common_shapes:
            self._allocate_pool_tensors(shape, dtype, device, count=2)
    
    def _allocate_pool_tensors(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                              device: str, count: int = 1):
        """Allocate a number of tensors of the specified shape to the pool."""
        if self.pool_size >= self.max_size:
            return  # Pool is full
        
        for _ in range(count):
            try:
                tensor = torch.empty(shape, dtype=dtype, device=device)
                self.tensor_pools[(shape, dtype, device)].append(tensor)
                self.pool_size += tensor.element_size() * tensor.nelement()
                self.max_pool_size = max(self.max_pool_size, self.pool_size)
            except RuntimeError:
                # Out of memory, stop allocating
                break
    
    def acquire_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                      device: str = "cuda") -> torch.Tensor:
        """
        Acquire a tensor from the pool or create a new one if none available.
        
        Args:
            shape: Shape of the tensor needed
            dtype: Data type of the tensor
            device: Device for the tensor
            
        Returns:
            Tensor of the requested shape and dtype
        """
        key = (shape, dtype, device)
        
        # Check if we have a tensor of this shape in the pool
        if self.tensor_pools[key]:
            tensor = self.tensor_pools[key].pop()
            # Update pool size
            self.pool_size -= tensor.element_size() * tensor.nelement()
            return tensor
        else:
            # No tensor available, create a new one
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self.total_allocated += tensor.element_size() * tensor.nelement()
            return tensor
    
    def release_tensor(self, tensor: torch.Tensor):
        """
        Release a tensor back to the pool.
        
        Args:
            tensor: Tensor to release back to the pool
        """
        if tensor is None:
            return
            
        # Get tensor properties
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = str(tensor.device)
        
        key = (shape, dtype, device)
        
        # Only add to pool if we're under the size limit
        tensor_size = tensor.element_size() * tensor.nelement()
        if self.pool_size + tensor_size <= self.max_size:
            self.tensor_pools[key].append(tensor)
            self.pool_size += tensor_size
        else:
            # Pool is full, let the tensor be garbage collected
            del tensor
    
    def defragment(self):
        """Perform memory defragmentation by clearing the pool and reinitializing."""
        # Clear all pools
        for tensor_list in self.tensor_pools.values():
            for tensor in tensor_list:
                del tensor
        self.tensor_pools.clear()
        
        # Reset statistics
        self.pool_size = 0
        
        # Reinitialize with common shapes
        self._initialize_common_shapes()
    
    def get_fragmentation_ratio(self) -> float:
        """
        Calculate the fragmentation ratio of the pool.
        
        Returns:
            Ratio of fragmented memory (0.0 to 1.0)
        """
        # Calculate total size of all tensors in pools
        total_tensor_size = 0
        for tensor_list in self.tensor_pools.values():
            for tensor in tensor_list:
                total_tensor_size += tensor.element_size() * tensor.nelement()
        
        if self.pool_size == 0:
            return 0.0
        
        # Fragmentation is the difference between allocated and actual used memory
        fragmentation = (self.pool_size - total_tensor_size) / self.pool_size
        return max(0.0, fragmentation)  # Ensure non-negative
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory pool statistics.
        
        Returns:
            Dictionary containing pool statistics
        """
        total_tensors = sum(len(tensor_list) for tensor_list in self.tensor_pools.values())
        total_shapes = len(self.tensor_pools)
        
        return {
            "pool_size_bytes": self.pool_size,
            "max_pool_size_bytes": self.max_pool_size,
            "total_allocated_bytes": self.total_allocated,
            "total_tensors_in_pool": total_tensors,
            "unique_shapes_in_pool": total_shapes,
            "fragmentation_ratio": self.get_fragmentation_ratio()
        }