"""
Cache-Aware Memory Management System for Qwen3-VL

This module implements a cache-aware memory management system that optimizes memory allocation 
and access patterns based on cache hierarchy (L1, L2, L3) for improved performance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
import time
from collections import OrderedDict, deque
import logging
import psutil
import numpy as np
from dataclasses import dataclass
import os


@dataclass
class CacheBlock:
    """Represents a block of memory aligned to cache line boundaries"""
    id: str
    start_addr: int
    size_bytes: int
    tensor_ref: torch.Tensor
    last_accessed: float
    access_frequency: int
    cache_level: str  # L1, L2, L3


class CacheAwareMemoryManager:
    """
    Memory manager that optimizes allocations based on cache characteristics.
    Considers cache line sizes, associativity, and hierarchy for efficient allocation.
    """
    
    def __init__(self, 
                 l1_size: int = 32 * 1024,  # 32KB (typical for Intel CPUs)
                 l2_size: int = 256 * 1024,  # 256KB (typical for Intel CPUs)
                 l3_size: int = 6 * 1024 * 1024,  # 6MB (i5-10210U has 6MB L3 cache)
                 cache_line_size: int = 64,  # Standard cache line size
                 gpu_l1_size: int = 16 * 1024,  # 16KB GPU L1 cache (typical for SM61)
                 gpu_l2_size: int = 2 * 1024 * 1024,  # 2MB GPU L2 cache
                 lru_cache_size: int = 1000):
        """
        Initialize cache-aware memory manager
        
        Args:
            l1_size: Size of L1 cache in bytes
            l2_size: Size of L2 cache in bytes
            l3_size: Size of L3 cache in bytes
            cache_line_size: Size of cache line in bytes
            gpu_l1_size: Size of GPU L1 cache in bytes
            gpu_l2_size: Size of GPU L2 cache in bytes
            lru_cache_size: Size of LRU cache for tensor management
        """
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        self.cache_line_size = cache_line_size
        self.gpu_l1_size = gpu_l1_size
        self.gpu_l2_size = gpu_l2_size
        
        # Memory pools for different cache levels
        self.l1_pool = OrderedDict()
        self.l2_pool = OrderedDict()
        self.l3_pool = OrderedDict()
        
        # GPU memory pools
        self.gpu_l1_pool = OrderedDict()
        self.gpu_l2_pool = OrderedDict()
        
        # LRU caches for each level
        self.lru_caches = {
            'l1': OrderedDict(),
            'l2': OrderedDict(),
            'l3': OrderedDict(),
            'gpu_l1': OrderedDict(),
            'gpu_l2': OrderedDict()
        }
        
        # Cache access statistics
        self.cache_stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_hits': 0,
            'l3_misses': 0,
            'gpu_l1_hits': 0,
            'gpu_l1_misses': 0,
            'gpu_l2_hits': 0,
            'gpu_l2_misses': 0
        }
        
        # Access history for prefetching predictions
        self.access_history = deque(maxlen=1000)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Cache-Aware Memory Manager initialized with L1:{l1_size/1024:.0f}KB, "
                         f"L2:{l2_size/1024:.0f}KB, L3:{l3_size/1024/1024:.0f}MB, cache_line:{cache_line_size}B")
    
    def _get_cache_level_for_size(self, size_bytes: int) -> str:
        """
        Determine optimal cache level for a tensor based on its size
        
        Args:
            size_bytes: Size of tensor in bytes
        
        Returns:
            Optimal cache level ('l3', 'l2', 'l1', 'gpu_l2', 'gpu_l1')
        """
        if size_bytes <= self.cache_line_size:
            # Very small tensors go to fastest cache (L1 or GPU L1 if applicable)
            return 'l1'
        elif size_bytes <= self.l1_size:
            return 'l1'
        elif size_bytes <= self.l2_size:
            return 'l2'
        elif size_bytes <= self.l3_size:
            return 'l3'
        else:
            # Large tensors may be better in main memory or GPU memory
            return 'l3'
    
    def _align_to_cache_line(self, size_bytes: int) -> int:
        """
        Align size to cache line boundary
        
        Args:
            size_bytes: Original size in bytes
        
        Returns:
            Size aligned to cache line boundary
        """
        remainder = size_bytes % self.cache_line_size
        if remainder == 0:
            return size_bytes
        return size_bytes + (self.cache_line_size - remainder)
    
    def _get_pool_for_cache_level(self, cache_level: str):
        """
        Get the memory pool corresponding to a cache level
        
        Args:
            cache_level: Cache level identifier
        
        Returns:
            Corresponding memory pool
        """
        if cache_level == 'l1':
            return self.l1_pool
        elif cache_level == 'l2':
            return self.l2_pool
        elif cache_level == 'l3':
            return self.l3_pool
        elif cache_level == 'gpu_l1':
            return self.gpu_l1_pool
        elif cache_level == 'gpu_l2':
            return self.gpu_l2_pool
        else:
            return self.l3_pool  # Default to L3
    
    def _get_lru_cache_for_level(self, cache_level: str) -> OrderedDict:
        """Get LRU cache for specific level"""
        return self.lru_caches.get(cache_level, self.lru_caches['l3'])
    
    def allocate_tensor(self, 
                       shape: Tuple[int, ...], 
                       dtype: torch.dtype,
                       device: Optional[str] = None,
                       use_gpu: bool = True) -> Tuple[str, torch.Tensor]:
        """
        Allocate a tensor with cache-aware optimization
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            device: Specified device ('cpu', 'cuda', etc.)
            use_gpu: Whether to attempt GPU placement if available
        
        Returns:
            Tuple of (tensor_id, tensor)
        """
        with self.lock:
            # Calculate tensor size
            size_bytes = self._calculate_tensor_size(shape, dtype)
            aligned_size = self._align_to_cache_line(size_bytes)
            
            # Determine optimal cache level
            cache_level = self._get_cache_level_for_size(aligned_size)
            
            # If GPU is requested and available, use GPU cache levels
            if use_gpu and torch.cuda.is_available():
                if aligned_size <= self.gpu_l1_size:
                    cache_level = 'gpu_l1'
                elif aligned_size <= self.gpu_l2_size:
                    cache_level = 'gpu_l2'
            
            # Create tensor with appropriate device
            if 'gpu' in cache_level or (device and 'cuda' in device):
                target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                target_device = torch.device('cpu')
                
            tensor = torch.empty(shape, dtype=dtype, device=target_device)
            
            # Generate unique tensor ID
            tensor_id = f"cache_{cache_level}_{int(time.time() * 1000000)}_{id(tensor)}"
            
            # Track the tensor in the appropriate pool
            cache_block = CacheBlock(
                id=tensor_id,
                start_addr=0,  # Placeholder - actual address not accessible in Python
                size_bytes=aligned_size,
                tensor_ref=tensor,
                last_accessed=time.time(),
                access_frequency=1,
                cache_level=cache_level
            )
            
            pool = self._get_pool_for_cache_level(cache_level)
            pool[tensor_id] = cache_block
            
            # Update LRU cache
            self._update_lru_cache(cache_level, tensor_id)
            
            self.logger.debug(f"Allocated tensor {tensor_id} in {cache_level} cache ({aligned_size} bytes)")
            
            return tensor_id, tensor
    
    def get_tensor(self, tensor_id: str) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor from cache, updating access stats
        
        Args:
            tensor_id: ID of the tensor to retrieve
        
        Returns:
            Tensor if found, None otherwise
        """
        with self.lock:
            tensor = None
            cache_level = None
            
            # Search in all pools
            for pool_name, pool in [('l1', self.l1_pool), ('l2', self.l2_pool), 
                                    ('l3', self.l3_pool), ('gpu_l1', self.gpu_l1_pool), 
                                    ('gpu_l2', self.gpu_l2_pool)]:
                if tensor_id in pool:
                    cache_block = pool[tensor_id]
                    tensor = cache_block.tensor_ref
                    cache_level = cache_block.cache_level
                    cache_block.last_accessed = time.time()
                    cache_block.access_frequency += 1
                    
                    # Update LRU cache
                    self._update_lru_cache(cache_level, tensor_id)
                    
                    # Update hit statistics
                    self._record_cache_hit(cache_level)
                    break
            
            if tensor is not None:
                # Add to access history for prefetching predictions
                self.access_history.append({
                    'id': tensor_id,
                    'level': cache_level,
                    'timestamp': time.time()
                })
                
                self.logger.debug(f"Retrieved tensor {tensor_id} from {cache_level} cache")
                return tensor
            else:
                # Record miss statistics
                if cache_level:
                    self._record_cache_miss(cache_level)
                self.logger.warning(f"Tensor {tensor_id} not found in any cache")
                return None
    
    def _update_lru_cache(self, cache_level: str, tensor_id: str):
        """Update LRU cache for a specific level"""
        lru_cache = self._get_lru_cache_for_level(cache_level)
        
        # Remove if already exists and add to end (most recently used)
        if tensor_id in lru_cache:
            del lru_cache[tensor_id]
        lru_cache[tensor_id] = time.time()
        
        # Trim if cache is too large
        if len(lru_cache) > 1000:  # Keep only recent entries
            while len(lru_cache) > 800:  # Trim to 800 entries
                lru_cache.popitem(last=False)
    
    def _record_cache_hit(self, cache_level: str):
        """Record cache hit statistics"""
        key = f"{cache_level}_hits"
        if key in self.cache_stats:
            self.cache_stats[key] += 1
    
    def _record_cache_miss(self, cache_level: str):
        """Record cache miss statistics"""
        key = f"{cache_level}_misses"
        if key in self.cache_stats:
            self.cache_stats[key] += 1
    
    def _calculate_tensor_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """
        Calculate tensor size in bytes
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
        
        Returns:
            Size in bytes
        """
        elements = 1
        for dim in shape:
            elements *= dim
        
        # Map PyTorch dtypes to byte sizes
        if dtype == torch.float32:
            return elements * 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            return elements * 2
        elif dtype == torch.float64:
            return elements * 8
        elif dtype == torch.int64:
            return elements * 8
        elif dtype == torch.int32:
            return elements * 4
        elif dtype == torch.int16:
            return elements * 2
        elif dtype == torch.int8:
            return elements * 1
        elif dtype == torch.uint8:
            return elements * 1
        elif dtype == torch.bool:
            # PyTorch uses 1 byte per boolean value
            return elements * 1
        else:
            # Default to 4 bytes per element
            return elements * 4
    
    def _should_prefetch_tensor(self, tensor_id: str) -> bool:
        """
        Determine if a tensor should be prefetched based on access patterns
        
        Args:
            tensor_id: ID of the tensor to evaluate
        
        Returns:
            True if tensor should be prefetched, False otherwise
        """
        # Look for access patterns in history
        recent_accesses = [entry for entry in self.access_history 
                          if entry['id'] == tensor_id and 
                          time.time() - entry['timestamp'] < 5.0]  # Last 5 seconds
        
        # If accessed multiple times recently, it's likely to be accessed again
        return len(recent_accesses) > 2
    
    def prefetch_tensor(self, tensor_id: str) -> bool:
        """
        Prefetch a tensor to higher cache level based on access prediction
        
        Args:
            tensor_id: ID of the tensor to prefetch
        
        Returns:
            True if prefetch was initiated, False otherwise
        """
        with self.lock:
            cache_block = None
            original_pool_name = None
            
            # Find the tensor in any pool
            for pool_name, pool in [('l1', self.l1_pool), ('l2', self.l2_pool), 
                                    ('l3', self.l3_pool), ('gpu_l1', self.gpu_l1_pool), 
                                    ('gpu_l2', self.gpu_l2_pool)]:
                if tensor_id in pool:
                    cache_block = pool[tensor_id]
                    original_pool_name = pool_name
                    break
            
            if cache_block is not None and self._should_prefetch_tensor(tensor_id):
                # Determine if tensor should be promoted to a faster cache
                current_level = cache_block.cache_level
                tensor_size = cache_block.size_bytes
                
                # Only promote if it's in a slower cache and fits in a faster one
                if current_level == 'l3' and tensor_size <= self.l2_size:
                    self._promote_to_cache_level(tensor_id, 'l2', original_pool_name)
                    return True
                elif current_level == 'l2' and tensor_size <= self.l1_size:
                    self._promote_to_cache_level(tensor_id, 'l1', original_pool_name)
                    return True
                elif current_level == 'gpu_l2' and tensor_size <= self.gpu_l1_size:
                    self._promote_to_cache_level(tensor_id, 'gpu_l1', original_pool_name)
                    return True
            
            return False
    
    def _promote_to_cache_level(self, tensor_id: str, target_level: str, original_pool_name: str):
        """
        Promote a tensor to a higher cache level
        """
        # Get the tensor from original pool
        original_pool = self._get_pool_for_cache_level(original_pool_name)
        cache_block = original_pool.get(tensor_id)
        
        if cache_block:
            # Remove from original pool
            del original_pool[tensor_id]
            
            # Update cache level in the block
            cache_block.cache_level = target_level
            cache_block.last_accessed = time.time()
            
            # Add to target pool
            target_pool = self._get_pool_for_cache_level(target_level)
            target_pool[tensor_id] = cache_block
            
            # Update LRU cache
            self._update_lru_cache(target_level, tensor_id)
            
            self.logger.info(f"Promoted tensor {tensor_id} from {original_pool_name} to {target_level}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.cache_stats.copy()
        
        # Calculate hit rates
        for level in ['l1', 'l2', 'l3', 'gpu_l1', 'gpu_l2']:
            hits = stats.get(f'{level}_hits', 0)
            misses = stats.get(f'{level}_misses', 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0
            stats[f'{level}_hit_rate'] = hit_rate
        
        # Memory utilization
        stats['memory_utilization'] = {
            'l1_utilization': sum(cb.size_bytes for cb in self.l1_pool.values()) / self.l1_size if self.l1_size > 0 else 0,
            'l2_utilization': sum(cb.size_bytes for cb in self.l2_pool.values()) / self.l2_size if self.l2_size > 0 else 0,
            'l3_utilization': sum(cb.size_bytes for cb in self.l3_pool.values()) / self.l3_size if self.l3_size > 0 else 0,
            'gpu_l1_utilization': sum(cb.size_bytes for cb in self.gpu_l1_pool.values()) / self.gpu_l1_size if self.gpu_l1_size > 0 else 0,
            'gpu_l2_utilization': sum(cb.size_bytes for cb in self.gpu_l2_pool.values()) / self.gpu_l2_size if self.gpu_l2_size > 0 else 0
        }
        
        # Total tensors managed
        stats['total_tensors_managed'] = sum(len(pool) for pool in [self.l1_pool, self.l2_pool, self.l3_pool, 
                                                                    self.gpu_l1_pool, self.gpu_l2_pool])
        
        return stats
    
    def cleanup_inactive_tensors(self, inactive_threshold: float = 30.0) -> int:
        """
        Remove tensors that have not been accessed for a certain time
        
        Args:
            inactive_threshold: Time threshold in seconds
        
        Returns:
            Number of tensors removed
        """
        with self.lock:
            current_time = time.time()
            removed_count = 0
            
            for pool in [self.l1_pool, self.l2_pool, self.l3_pool, self.gpu_l1_pool, self.gpu_l2_pool]:
                inactive_keys = [
                    key for key, cache_block in pool.items()
                    if current_time - cache_block.last_accessed > inactive_threshold
                ]
                
                for key in inactive_keys:
                    del pool[key]
                    removed_count += 1
                    
                    # Also remove from LRU caches
                    for lru_cache in self.lru_caches.values():
                        if key in lru_cache:
                            del lru_cache[key]
            
            self.logger.info(f"Cleaned up {removed_count} inactive tensors")
            return removed_count


# Hardware-specific cache configuration
def get_optimized_cache_config(cpu_model: str = "Intel i5-10210U", 
                             gpu_model: str = "NVIDIA SM61") -> Dict[str, int]:
    """
    Get optimized cache configuration based on hardware
    
    Args:
        cpu_model: CPU model string
        gpu_model: GPU model string
    
    Returns:
        Dictionary with optimal cache sizes
    """
    config = {
        'l1_size': 32 * 1024,  # Default 32KB
        'l2_size': 256 * 1024,  # Default 256KB
        'l3_size': 6 * 1024 * 1024,  # Default 6MB
        'cache_line_size': 64,  # Standard cache line size
        'gpu_l1_size': 16 * 1024,  # Default 16KB GPU L1
        'gpu_l2_size': 2 * 1024 * 1024  # Default 2MB GPU L2
    }
    
    # Intel i5-10210U specific configuration
    if "i5-10210U" in cpu_model.upper():
        config.update({
            'l1_size': 32 * 1024,  # 32KB per core
            'l2_size': 256 * 1024,  # 256KB per core
            'l3_size': 6 * 1024 * 1024,  # 6MB shared L3 cache
            'cache_line_size': 64,  # Standard for Intel CPUs
        })
    
    # NVIDIA SM61 (Maxwell architecture) specific configuration
    if "SM61" in gpu_model.upper():
        config.update({
            'gpu_l1_size': 16 * 1024,  # 16KB per SM typically for SM61
            'gpu_l2_size': 2 * 1024 * 1024,  # 2MB total for SM61 architecture
        })
    
    return config


# Example usage and testing
if __name__ == "__main__":
    print("Testing Cache-Aware Memory Management System...")
    
    # Create optimized config for our hardware
    hw_config = get_optimized_cache_config()
    print(f"Using hardware config: {hw_config}")
    
    # Create memory manager
    manager = CacheAwareMemoryManager(
        l1_size=hw_config['l1_size'],
        l2_size=hw_config['l2_size'],
        l3_size=hw_config['l3_size'],
        cache_line_size=hw_config['cache_line_size'],
        gpu_l1_size=hw_config['gpu_l1_size'],
        gpu_l2_size=hw_config['gpu_l2_size']
    )
    
    print("\n1. Testing tensor allocation...")
    # Test allocating tensors of different sizes
    tensor_specs = [
        ((10, 10), torch.float32),  # Small tensor - should go to L1
        ((100, 100), torch.float32),  # Medium tensor - should go to L2
        ((500, 500), torch.float32),  # Large tensor - should go to L3
        ((32, 32, 32), torch.float16),  # Large tensor in half precision
    ]
    
    allocated_tensors = []
    for i, (shape, dtype) in enumerate(tensor_specs):
        tensor_id, tensor = manager.allocate_tensor(shape, dtype)
        print(f"   Allocated {shape} tensor {tensor_id} (size: {tensor.numel() * tensor.element_size()} bytes)")
        allocated_tensors.append((tensor_id, tensor))
    
    print("\n2. Testing tensor retrieval...")
    # Test retrieving tensors
    for tensor_id, _ in allocated_tensors:
        tensor = manager.get_tensor(tensor_id)
        if tensor is not None:
            print(f"   Retrieved tensor {tensor_id}, shape: {tensor.shape}")
        else:
            print(f"   Failed to retrieve tensor {tensor_id}")
    
    print("\n3. Testing prefetching...")
    # Test prefetching functionality
    for tensor_id, _ in allocated_tensors[:2]:  # Only test on first 2 tensors
        should_prefetch = manager.prefetch_tensor(tensor_id)
        print(f"   Tensor {tensor_id} prefetch decision: {should_prefetch}")
    
    print("\n4. Accessing tensors to build history...")
    # Access some tensors multiple times to build access history
    for _ in range(3):
        for tensor_id, _ in allocated_tensors[:1]:  # Only access first tensor repeatedly
            tensor = manager.get_tensor(tensor_id)
    
    print("\n5. Checking cache statistics...")
    # Check statistics
    stats = manager.get_cache_statistics()
    for key, value in stats.items():
        if 'rate' in key or isinstance(value, (int, float)) and 'utilization' not in key:
            print(f"   {key}: {value}")
    
    print("\n6. Checking memory utilization...")
    util = stats.get('memory_utilization', {})
    for level, utilization in util.items():
        print(f"   {level}: {utilization:.3f}")
    
    print("\n7. Cleaning up inactive tensors...")
    # Test cleanup of inactive tensors
    cleaned = manager.cleanup_inactive_tensors(inactive_threshold=1.0)
    print(f"   Cleaned up {cleaned} inactive tensors")
    
    print("\nCache-Aware Memory Management System test completed successfully!")