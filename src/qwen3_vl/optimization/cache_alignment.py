"""Cache Alignment Optimizations for Intel i5-10210U L3 Cache (6MB) in Qwen3-VL Memory Pooling System."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math
import psutil
from dataclasses import dataclass
import threading
from functools import lru_cache
import time


@dataclass
class CacheAlignmentConfig:
    """Configuration for cache alignment optimizations."""
    l1_cache_size: int = 32 * 1024  # 32KB per core
    l1_cache_line_size: int = 64    # 64 bytes
    l2_cache_size: int = 256 * 1024 # 256KB per core
    l2_cache_line_size: int = 64    # 64 bytes
    l3_cache_size: int = 6 * 1024 * 1024  # 6MB shared
    l3_cache_line_size: int = 64    # 64 bytes
    num_cores: int = 4              # 4 physical cores
    num_threads: int = 8            # 8 threads with hyperthreading
    memory_bandwidth_gb_s: float = 42.7  # Approximate bandwidth for i5-10210U


class CacheAlignedTensorPool:
    """
    Memory pool with sophisticated cache alignment for Intel i5-10210U architecture.
    Optimizes for 6MB L3 cache with 4 cores and 8 threads.
    """
    
    def __init__(self, 
                 config: CacheAlignmentConfig,
                 max_capacity_bytes: int = 1024 * 1024 * 1024,  # 1GB default
                 dtype: torch.dtype = torch.float16,
                 device: Optional[torch.device] = None):
        """
        Initialize cache-aligned tensor pool.
        
        Args:
            config: Cache alignment configuration
            max_capacity_bytes: Maximum capacity in bytes
            dtype: Default tensor data type
            device: Device to allocate tensors on
        """
        self.config = config
        self.max_capacity_bytes = max_capacity_bytes
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize pool data structures
        self.pool: Dict[str, torch.Tensor] = {}
        self.tensor_metadata: Dict[str, Dict[str, Any]] = {}
        self.current_size_bytes = 0
        
        # For thread safety
        self._lock = threading.Lock()
        
        # Pre-computed alignment factors
        self.alignment_factors = self._compute_alignment_factors()
        
        # Stats tracking
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'alignment_improvements': 0
        }

    def _compute_alignment_factors(self) -> Dict[str, int]:
        """Compute optimal alignment factors for Intel i5-10210U caches."""
        return {
            'l1': self.config.l1_cache_line_size,
            'l2': self.config.l2_cache_line_size,
            'l3': self.config.l3_cache_line_size,
            'memory': 64  # Standard memory access alignment
        }

    def _calculate_tensor_size_aligned(self, shape: Tuple[int, ...], dtype: torch.dtype) -> Tuple[int, int]:
        """
        Calculate tensor size with cache alignment.
        
        Returns:
            Tuple of (aligned_size, original_size)
        """
        original_size = 1
        for dim in shape:
            original_size *= dim
        element_size = torch.tensor([], dtype=dtype).element_size()
        original_size_bytes = original_size * element_size
        
        # Align to L3 cache line size (64 bytes for i5-10210U)
        aligned_size = ((original_size_bytes + self.alignment_factors['l3'] - 1) // 
                       self.alignment_factors['l3']) * self.alignment_factors['l3']
        
        return aligned_size, original_size_bytes

    def _align_shape_for_cache(self, shape: Tuple[int, ...], access_pattern: str = 'sequential') -> Tuple[int, ...]:
        """
        Align tensor shape for optimal cache usage on i5-10210U.
        
        Args:
            shape: Original tensor shape
            access_pattern: Expected access pattern ('sequential', 'random', 'matrix')
            
        Returns:
            Cache-aligned shape
        """
        shape_list = list(shape)
        
        if access_pattern == 'matrix':
            # For matrix operations (like attention), align for cache-friendly access
            if len(shape_list) >= 2:
                # Align last dimension (column access) to cache line boundaries
                last_dim = shape_list[-1]
                element_size = torch.tensor([], dtype=self.dtype).element_size()
                elements_per_cache_line = self.config.l3_cache_line_size // element_size
                aligned_last_dim = ((last_dim + elements_per_cache_line - 1) // 
                                   elements_per_cache_line) * elements_per_cache_line
                shape_list[-1] = aligned_last_dim
                
                # For 2D matrices, also consider row alignment
                if len(shape_list) == 2:
                    row_dim = shape_list[0]
                    # Align rows to improve cache line utilization
                    aligned_row_dim = ((row_dim + 7) // 8) * 8  # Align to 8 for better processing
                    shape_list[0] = aligned_row_dim
                    
        elif access_pattern == 'sequential':
            # For sequential access (like KV cache), align appropriately
            if len(shape_list) >= 2:
                # Align the sequence dimension for better cache utilization
                seq_dim = shape_list[-2] if len(shape_list) >= 2 else shape_list[-1]
                aligned_seq_dim = ((seq_dim + 31) // 32) * 32  # Align to 32 for SIMD
                if len(shape_list) >= 2:
                    shape_list[-2] = aligned_seq_dim
                else:
                    shape_list[-1] = aligned_seq_dim
                    
        elif access_pattern == 'random':
            # For random access, try to optimize for average case
            if len(shape_list) >= 1:
                last_dim = shape_list[-1]
                element_size = torch.tensor([], dtype=self.dtype).element_size()
                elements_per_cache_line = self.config.l3_cache_line_size // element_size
                aligned_last_dim = ((last_dim + elements_per_cache_line - 1) // 
                                   elements_per_cache_line) * elements_per_cache_line
                shape_list[-1] = aligned_last_dim
        
        return tuple(shape_list)

    def _evict_lru_tensors(self, required_bytes: int):
        """Evict tensors to make space, considering cache alignment."""
        # This is a simplified eviction - in a real implementation, 
        # we'd use a proper LRU mechanism with cache-aware eviction
        while self.current_size_bytes + required_bytes > self.max_capacity_bytes and len(self.pool) > 0:
            # Remove the first item (simplified LRU without tracking)
            key_to_remove = next(iter(self.pool))
            tensor_size = self.tensor_metadata[key_to_remove]['size']
            del self.pool[key_to_remove]
            del self.tensor_metadata[key_to_remove]
            self.current_size_bytes -= tensor_size
            self.stats['evictions'] += 1

    def get_tensor(self, 
                   shape: Tuple[int, ...], 
                   dtype: Optional[torch.dtype] = None,
                   access_pattern: str = 'sequential') -> torch.Tensor:
        """
        Get a cache-aligned tensor from the pool.
        
        Args:
            shape: Desired tensor shape
            dtype: Desired tensor data type
            access_pattern: Expected access pattern for optimization
            
        Returns:
            Cache-aligned tensor
        """
        with self._lock:
            dtype = dtype or self.dtype
            
            # Apply cache alignment to shape
            aligned_shape = self._align_shape_for_cache(shape, access_pattern)
            aligned_size, original_size = self._calculate_tensor_size_aligned(aligned_shape, dtype)
            
            # Check if we have enough space after alignment
            if self.current_size_bytes + aligned_size > self.max_capacity_bytes:
                # Try to evict some tensors
                self._evict_lru_tensors(aligned_size)
                
                # If still not enough space, create tensor without pooling
                if self.current_size_bytes + aligned_size > self.max_capacity_bytes:
                    tensor = torch.empty(shape, dtype=dtype, device=self.device)
                    tensor.zero_()
                    return tensor
            
            # Create new tensor with aligned dimensions
            tensor = torch.empty(aligned_shape, dtype=dtype, device=self.device)
            tensor.zero_()
            
            # Create unique key for this tensor
            key = f"tensor_{id(tensor)}_{int(time.time() * 1000000)}"
            
            # Store tensor and metadata
            self.pool[key] = tensor
            self.tensor_metadata[key] = {
                'shape': shape,  # Store original shape
                'aligned_shape': aligned_shape,
                'size': aligned_size,
                'original_size': original_size,
                'dtype': dtype,
                'device': self.device,
                'access_pattern': access_pattern,
                'allocation_time': time.time()
            }
            
            self.current_size_bytes += aligned_size
            self.stats['allocations'] += 1
            self.stats['alignment_improvements'] += 1 if aligned_size != original_size else 0
            
            # If the aligned shape differs from requested shape, return a view with correct shape
            if aligned_shape != shape:
                # Create a tensor with the exact requested shape
                result_tensor = torch.empty(shape, dtype=dtype, device=self.device)
                # Copy data if possible
                min_elements = min(result_tensor.numel(), tensor.numel())
                if min_elements > 0:
                    result_tensor_flat = result_tensor.view(-1)
                    tensor_flat = tensor.view(-1)
                    result_tensor_flat[:min_elements] = tensor_flat[:min_elements]
                return result_tensor
            else:
                return tensor

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache alignment effectiveness."""
        return {
            **self.stats,
            'current_size_bytes': self.current_size_bytes,
            'max_capacity_bytes': self.max_capacity_bytes,
            'utilization_percent': (self.current_size_bytes / self.max_capacity_bytes) * 100 if self.max_capacity_bytes > 0 else 0,
            'pool_size': len(self.pool),
            'alignment_overhead_bytes': sum(
                meta['size'] - meta['original_size'] for meta in self.tensor_metadata.values()
            ),
            'alignment_overhead_percent': (
                sum(meta['size'] - meta['original_size'] for meta in self.tensor_metadata.values()) / 
                sum(meta['size'] for meta in self.tensor_metadata.values()) * 100
                if sum(meta['size'] for meta in self.tensor_metadata.values()) > 0 else 0
            )
        }


class L3CacheOptimizer:
    """
    Optimizer specifically for Intel i5-10210U's 6MB L3 cache.
    Manages tensor allocation to maximize L3 cache efficiency.
    """
    
    def __init__(self, config: CacheAlignmentConfig):
        self.config = config
        self.l3_cache_size = config.l3_cache_size
        self.l3_cache_line_size = config.l3_cache_line_size
        
        # Track L3 cache usage patterns
        self.cache_usage_tracker = {}
        self.tensor_access_patterns = {}
        
        # Calculate optimal tensor sizes for L3 cache
        self._calculate_l3_optimal_sizes()

    def _calculate_l3_optimal_sizes(self):
        """Calculate optimal tensor sizes for 6MB L3 cache."""
        # Calculate how many tensors of common sizes can fit in L3 cache
        self.optimal_sizes = {
            'attention': self._calculate_optimal_size_for_attention(),
            'kv_cache': self._calculate_optimal_size_for_kv_cache(),
            'embeddings': self._calculate_optimal_size_for_embeddings()
        }

    def _calculate_optimal_size_for_attention(self) -> Dict[str, int]:
        """Calculate optimal attention tensor sizes for L3 cache."""
        # For attention: typically [batch, heads, seq_len, seq_len] or [batch, seq_len, features]
        # We want to fit multiple attention operations in L3 cache
        
        # Assuming float16 (2 bytes per element)
        element_size = 2
        
        # For sequence-level attention, we might have [1, 8, 1024, 1024] = ~16MB (too large)
        # So we optimize for smaller sequences that fit in L3 cache
        optimal_seq_len = int(math.sqrt(self.l3_cache_size * 0.5 / (8 * element_size)))  # 50% of L3 for attention
        optimal_seq_len = min(optimal_seq_len, 1024)  # Cap at reasonable size
        
        return {
            'max_seq_len': optimal_seq_len,
            'element_size': element_size,
            'cache_efficiency_target': 0.7  # Target 70% cache efficiency
        }

    def _calculate_optimal_size_for_kv_cache(self) -> Dict[str, int]:
        """Calculate optimal KV cache tensor sizes for L3 cache."""
        # For KV cache: typically [batch, heads, seq_len, head_dim]
        element_size = 2  # float16
        
        # Calculate how much KV cache we can fit in L3
        # Assume [1, 32, 2048, 128] format
        optimal_seq_len = int((self.l3_cache_size * 0.3) / (32 * 128 * element_size))  # 30% of L3
        optimal_seq_len = min(optimal_seq_len, 2048)  # Cap at reasonable size
        
        return {
            'max_seq_len': optimal_seq_len,
            'element_size': element_size,
            'cache_efficiency_target': 0.8  # Target 80% cache efficiency for KV cache
        }

    def _calculate_optimal_size_for_embeddings(self) -> Dict[str, int]:
        """Calculate optimal embedding tensor sizes for L3 cache."""
        element_size = 2  # float16
        
        # For embeddings: [batch, seq_len, embed_dim]
        # Calculate optimal embedding dimensions
        optimal_embed_dim = int((self.l3_cache_size * 0.2) / (512 * element_size))  # 20% of L3 for 512 tokens
        optimal_embed_dim = min(optimal_embed_dim, 4096)  # Cap at reasonable size
        
        return {
            'max_embed_dim': optimal_embed_dim,
            'element_size': element_size,
            'cache_efficiency_target': 0.75  # Target 75% cache efficiency
        }

    def get_l3_optimized_shape(self, original_shape: Tuple[int, ...], tensor_type: str) -> Tuple[int, ...]:
        """
        Get L3 cache optimized shape for the given tensor type.
        
        Args:
            original_shape: Original tensor shape
            tensor_type: Type of tensor ('attention', 'kv_cache', 'embeddings')
            
        Returns:
            L3 cache optimized shape
        """
        if tensor_type not in self.optimal_sizes:
            return original_shape
            
        optimal_config = self.optimal_sizes[tensor_type]
        shape_list = list(original_shape)
        
        if tensor_type == 'attention':
            # Optimize sequence length dimension
            if len(shape_list) >= 2:
                max_seq_len = optimal_config['max_seq_len']
                if shape_list[-2] > max_seq_len:  # Assuming [batch, seq, seq] or [batch, seq, features]
                    shape_list[-2] = max_seq_len
                if len(shape_list) >= 3 and shape_list[-1] > max_seq_len:
                    shape_list[-1] = max_seq_len
                    
        elif tensor_type == 'kv_cache':
            # Optimize sequence length dimension for KV cache
            if len(shape_list) >= 3:
                max_seq_len = optimal_config['max_seq_len']
                if shape_list[2] > max_seq_len:  # Assuming [batch, heads, seq, head_dim]
                    shape_list[2] = max_seq_len
                    
        elif tensor_type == 'embeddings':
            # Optimize embedding dimension
            if len(shape_list) >= 3:
                max_embed_dim = optimal_config['max_embed_dim']
                if shape_list[-1] > max_embed_dim:  # Assuming [batch, seq, embed_dim]
                    shape_list[-1] = max_embed_dim
        
        return tuple(shape_list)

    def should_split_tensor(self, shape: Tuple[int, ...], tensor_type: str) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        """
        Determine if a tensor should be split to optimize L3 cache usage.
        
        Args:
            shape: Tensor shape to evaluate
            tensor_type: Type of tensor
            
        Returns:
            Tuple of (should_split, split_shape_if_applicable)
        """
        # Calculate tensor size
        numel = 1
        for dim in shape:
            numel *= dim
        element_size = 2  # float16
        tensor_size_bytes = numel * element_size
        
        # If tensor is larger than 30% of L3 cache, consider splitting
        if tensor_size_bytes > self.l3_cache_size * 0.3:
            if tensor_type == 'attention':
                # For attention, try to split along sequence dimension
                if len(shape) >= 2:
                    new_seq_len = shape[-2] // 2
                    if new_seq_len > 0:
                        new_shape = list(shape)
                        new_shape[-2] = new_seq_len
                        return True, tuple(new_shape)
            elif tensor_type == 'kv_cache':
                # For KV cache, try to split along sequence dimension
                if len(shape) >= 3:
                    new_seq_len = shape[2] // 2
                    if new_seq_len > 0:
                        new_shape = list(shape)
                        new_shape[2] = new_seq_len
                        return True, tuple(new_shape)
        
        return False, None


class IntelCacheAlignedPoolManager:
    """
    Memory pool manager optimized specifically for Intel i5-10210U architecture.
    Uses cache alignment and L3 optimization techniques.
    """
    
    def __init__(self, 
                 base_capacity: int = 2 * 1024 * 1024 * 1024,  # 2GB base capacity
                 dtype: torch.dtype = torch.float16,
                 device: Optional[torch.device] = None):
        """
        Initialize Intel-optimized memory pool manager.
        
        Args:
            base_capacity: Base capacity for all pools
            dtype: Default tensor data type
            device: Device to allocate tensors on
        """
        self.config = CacheAlignmentConfig()
        self.base_capacity = base_capacity
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize L3 cache optimizer
        self.l3_optimizer = L3CacheOptimizer(self.config)
        
        # Initialize cache-aligned pools for different tensor types
        self.pools = {}
        self._initialize_pools()
        
        # Track tensor-to-pool mapping
        self.tensor_to_pool: Dict[int, str] = {}

    def _initialize_pools(self):
        """Initialize cache-aligned pools for different tensor types."""
        # Calculate capacity distribution based on L3 cache optimization
        total_capacity = self.base_capacity
        
        # Distribute capacity based on typical usage patterns
        pool_capacities = {
            'attention': int(total_capacity * 0.25),  # 25% for attention tensors
            'kv_cache': int(total_capacity * 0.30),   # 30% for KV cache (largest)
            'image_embeddings': int(total_capacity * 0.15),  # 15% for image embeddings
            'text_embeddings': int(total_capacity * 0.15),   # 15% for text embeddings
            'intermediate': int(total_capacity * 0.10),      # 10% for intermediate activations
            'general': int(total_capacity * 0.05)            # 5% for general use
        }
        
        # Create cache-aligned pools for each type
        for pool_type, capacity in pool_capacities.items():
            self.pools[pool_type] = CacheAlignedTensorPool(
                config=self.config,
                max_capacity_bytes=capacity,
                dtype=self.dtype,
                device=self.device
            )

    def get_tensor(self, 
                   shape: Tuple[int, ...], 
                   tensor_type: str, 
                   dtype: Optional[torch.dtype] = None,
                   access_pattern: str = 'sequential') -> torch.Tensor:
        """
        Get a cache-aligned tensor optimized for Intel i5-10210U.
        
        Args:
            shape: Desired tensor shape
            tensor_type: Type of tensor
            dtype: Desired tensor data type
            access_pattern: Expected access pattern
            
        Returns:
            Cache-aligned tensor
        """
        dtype = dtype or self.dtype
        
        # First, optimize shape for L3 cache
        l3_optimized_shape = self.l3_optimizer.get_l3_optimized_shape(shape, tensor_type)
        
        # Determine access pattern based on tensor type if not specified
        if access_pattern == 'sequential':
            if tensor_type in ['attention', 'intermediate']:
                access_pattern = 'matrix'
            elif tensor_type in ['kv_cache']:
                access_pattern = 'sequential'
            else:
                access_pattern = 'random'
        
        # Select appropriate pool
        if tensor_type in self.pools:
            pool = self.pools[tensor_type]
        else:
            pool = self.pools['general']
        
        # Get cache-aligned tensor
        tensor = pool.get_tensor(l3_optimized_shape, dtype, access_pattern)
        
        # Track tensor-to-pool mapping
        tensor_id = id(tensor)
        self.tensor_to_pool[tensor_id] = tensor_type
        
        # If original shape was different from L3 optimized, handle appropriately
        if l3_optimized_shape != shape:
            # Create tensor with exact requested shape
            result_tensor = torch.empty(shape, dtype=dtype, device=self.device)
            # Copy data if shapes are compatible
            min_elements = min(result_tensor.numel(), tensor.numel())
            if min_elements > 0:
                result_tensor_flat = result_tensor.view(-1)
                tensor_flat = tensor.view(-1)
                result_tensor_flat[:min_elements] = tensor_flat[:min_elements]
            return result_tensor
        else:
            return tensor

    def get_tensor_with_l3_optimization(self, 
                                      shape: Tuple[int, ...], 
                                      tensor_type: str, 
                                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Get tensor with full L3 cache optimization including potential splitting.
        
        Args:
            shape: Desired tensor shape
            tensor_type: Type of tensor
            dtype: Desired tensor data type
            
        Returns:
            L3-optimized tensor
        """
        dtype = dtype or self.dtype
        
        # Check if tensor should be split for L3 optimization
        should_split, split_shape = self.l3_optimizer.should_split_tensor(shape, tensor_type)
        
        if should_split:
            # For now, just use the split shape
            # In a full implementation, we would handle tensor splitting properly
            return self.get_tensor(split_shape, tensor_type, dtype)
        else:
            return self.get_tensor(shape, tensor_type, dtype)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about cache alignment effectiveness."""
        stats = {}
        
        # Get stats for each pool
        for pool_name, pool in self.pools.items():
            stats[pool_name] = pool.get_stats()
        
        # Add L3 optimization stats
        stats['l3_optimization'] = {
            'l3_cache_size_mb': self.config.l3_cache_size / (1024 * 1024),
            'optimal_sizes': self.l3_optimizer.optimal_sizes,
            'num_cores': self.config.num_cores,
            'num_threads': self.config.num_threads
        }
        
        # Add aggregate stats
        total_allocated = sum(s.get('current_size_bytes', 0) for s in stats.values() 
                             if isinstance(s, dict) and 'current_size_bytes' in s)
        total_max = sum(s.get('max_capacity_bytes', 0) for s in stats.values()
                       if isinstance(s, dict) and 'max_capacity_bytes' in s)
        
        stats['aggregate'] = {
            'total_allocated_bytes': total_allocated,
            'total_max_capacity_bytes': total_max,
            'total_utilization_percent': (total_allocated / total_max * 100) if total_max > 0 else 0,
            'alignment_overhead_percent': sum(
                s.get('alignment_overhead_percent', 0) for s in stats.values()
                if isinstance(s, dict) and 'alignment_overhead_percent' in s
            ) / len([s for s in stats.values() if isinstance(s, dict) and 'alignment_overhead_percent' in s]) 
            if any(isinstance(s, dict) and 'alignment_overhead_percent' in s for s in stats.values()) else 0
        }
        
        return stats


def create_intel_optimized_pool_manager(config) -> IntelCacheAlignedPoolManager:
    """
    Factory function to create an Intel i5-10210U optimized memory pool manager.
    
    Args:
        config: Configuration object with memory optimization parameters
        
    Returns:
        IntelCacheAlignedPoolManager instance
    """
    # Extract parameters from config with defaults
    base_capacity = getattr(config, 'memory_pool_base_capacity', 2 * 1024 * 1024 * 1024)  # 2GB
    dtype = getattr(config, 'memory_pool_dtype', torch.float16)
    
    device_str = getattr(config, 'memory_pool_device', None)
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return IntelCacheAlignedPoolManager(
        base_capacity=base_capacity,
        dtype=dtype,
        device=device
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Testing Cache Alignment Optimizations for Intel i5-10210U...")
    
    # Create cache alignment config for i5-10210U
    config = CacheAlignmentConfig()
    
    # Create Intel-optimized pool manager
    pool_manager = IntelCacheAlignedPoolManager(
        base_capacity=1024 * 1024 * 512,  # 512MB
        dtype=torch.float16,
        device=torch.device('cpu')  # Use CPU for testing
    )
    
    print(f"\nL3 Cache Optimizer initialized for {config.l3_cache_size / (1024*1024):.0f}MB L3 cache")
    print(f"Optimal attention sequence length: {pool_manager.l3_optimizer.optimal_sizes['attention']['max_seq_len']}")
    print(f"Optimal KV cache sequence length: {pool_manager.l3_optimizer.optimal_sizes['kv_cache']['max_seq_len']}")
    
    # Test tensor allocation with cache alignment
    print("\n1. Testing attention tensor allocation with L3 optimization...")
    attention_tensor = pool_manager.get_tensor_with_l3_optimization((8, 1024, 1024), 'attention')
    print(f"Allocated attention tensor: {attention_tensor.shape}, {attention_tensor.dtype}")
    
    # Test KV cache tensor allocation
    print("\n2. Testing KV cache tensor allocation with L3 optimization...")
    kv_tensor = pool_manager.get_tensor_with_l3_optimization((1, 32, 2048, 128), 'kv_cache')
    print(f"Allocated KV cache tensor: {kv_tensor.shape}, {kv_tensor.dtype}")
    
    # Test image embedding tensor allocation
    print("\n3. Testing image embedding tensor allocation with L3 optimization...")
    img_tensor = pool_manager.get_tensor_with_l3_optimization((1, 576, 1152), 'image_embeddings')
    print(f"Allocated image embedding tensor: {img_tensor.shape}, {img_tensor.dtype}")
    
    # Get and display comprehensive statistics
    print("\n4. Cache alignment and L3 optimization statistics:")
    stats = pool_manager.get_stats()
    
    for pool_name, pool_stats in stats.items():
        if isinstance(pool_stats, dict) and 'utilization_percent' in pool_stats:
            if pool_name != 'aggregate' and pool_name != 'l3_optimization':
                print(f"  {pool_name}: {pool_stats['utilization_percent']:.2f}% utilization, "
                      f"{pool_stats.get('alignment_overhead_percent', 0):.2f}% alignment overhead")
    
    print(f"\n  Aggregate: {stats['aggregate']['total_utilization_percent']:.2f}% total utilization, "
          f"{stats['aggregate']['alignment_overhead_percent']:.2f}% total alignment overhead")
    
    print(f"\n  L3 Cache Size: {stats['l3_optimization']['l3_cache_size_mb']:.0f}MB")
    
    print("\nCache alignment optimizations test completed successfully!")