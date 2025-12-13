"""
Hierarchical Cache Manager for Qwen3-VL: Advanced Caching and Buffering System

Implements a multi-level cache hierarchy optimized for Intel i5-10210U architecture
with L1, L2, and L3 cache levels, each with different characteristics and policies.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging
from collections import OrderedDict, defaultdict
import pickle
import tempfile
from pathlib import Path
import bisect


@dataclass
class CacheConfig:
    """Configuration for hierarchical cache system"""
    l1_size: int = 128 * 1024 * 1024      # 128MB L1 cache (fastest)
    l2_size: int = 512 * 1024 * 1024      # 512MB L2 cache (fast)
    l3_size: int = 1024 * 1024 * 1024     # 1GB L3 cache (medium speed)
    l1_eviction_policy: str = 'lru'        # L1 cache eviction policy
    l2_eviction_policy: str = 'lru'        # L2 cache eviction policy
    l3_eviction_policy: str = 'lru'        # L3 cache eviction policy
    enable_compression: bool = True         # Enable compression in L3
    compression_threshold: float = 0.1      # Compress if >10% savings
    cache_alignment: int = 64               # Cache line alignment (bytes)
    prefetch_enabled: bool = True           # Enable prefetching
    prefetch_distance: int = 2              # Number of items to prefetch


class CacheLevel(Enum):
    """Cache levels in the hierarchy"""
    L1 = "l1"  # CPU L1 cache (fastest, smallest)
    L2 = "l2"  # CPU L2 cache (fast, medium)
    L3 = "l3"  # CPU L3 cache or NVMe SSD (medium speed, large)


@dataclass
class CachedItem:
    """Represents an item in the cache"""
    key: str
    tensor: torch.Tensor
    size_bytes: int
    access_time: float
    access_count: int
    level: CacheLevel
    compressed: bool = False
    compression_ratio: float = 1.0


class BaseCacheLevel:
    """Base class for cache level implementations"""
    
    def __init__(self, level: CacheLevel, max_size_bytes: int, eviction_policy: str = 'lru'):
        self.level = level
        self.max_size_bytes = max_size_bytes
        self.eviction_policy = eviction_policy
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, CachedItem] = OrderedDict()
        self._lock = threading.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get an item from the cache"""
        with self._lock:
            if key in self.cache:
                item = self.cache[key]
                item.access_time = time.time()
                item.access_count += 1
                
                # Move to end for LRU
                if self.eviction_policy == 'lru':
                    self.cache.move_to_end(key)
                
                self.hits += 1
                return item.tensor
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Put an item in the cache"""
        with self._lock:
            tensor_size = tensor.element_size() * tensor.nelement()
            
            # Check if item fits
            if tensor_size > self.max_size_bytes:
                return False
            
            # Make space if needed
            while self.current_size_bytes + tensor_size > self.max_size_bytes and len(self.cache) > 0:
                # Evict based on policy
                if self.eviction_policy == 'lru':
                    old_key, old_item = self.cache.popitem(last=False)
                else:  # Default to LRU
                    old_key, old_item = self.cache.popitem(last=False)
                
                self.current_size_bytes -= old_item.size_bytes
                self.evictions += 1
            
            # Add new item
            item = CachedItem(
                key=key,
                tensor=tensor,
                size_bytes=tensor_size,
                access_time=time.time(),
                access_count=1,
                level=self.level
            )
            
            self.cache[key] = item
            self.current_size_bytes += tensor_size
            return True
    
    def remove(self, key: str) -> bool:
        """Remove an item from the cache"""
        with self._lock:
            if key in self.cache:
                item = self.cache.pop(key)
                self.current_size_bytes -= item.size_bytes
                return True
            return False
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
    
    def get_utilization(self) -> float:
        """Get cache utilization"""
        return self.current_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'current_size_bytes': self.current_size_bytes,
            'max_size_bytes': self.max_size_bytes,
            'utilization': self.get_utilization(),
            'item_count': len(self.cache)
        }


class L1CacheLevel(BaseCacheLevel):
    """L1 cache level - fastest but smallest"""
    
    def __init__(self, max_size_bytes: int, eviction_policy: str = 'lru'):
        super().__init__(CacheLevel.L1, max_size_bytes, eviction_policy)
        # L1 cache is in CPU memory, optimized for speed


class L2CacheLevel(BaseCacheLevel):
    """L2 cache level - fast but medium size"""
    
    def __init__(self, max_size_bytes: int, eviction_policy: str = 'lru'):
        super().__init__(CacheLevel.L2, max_size_bytes, eviction_policy)
        # L2 cache is in CPU memory


class L3CacheLevel(BaseCacheLevel):
    """L3 cache level - medium speed but large, may use disk"""
    
    def __init__(self, max_size_bytes: int, eviction_policy: str = 'lru', 
                 enable_compression: bool = True, compression_threshold: float = 0.1):
        super().__init__(CacheLevel.L3, max_size_bytes, eviction_policy)
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.compressed_cache = {}  # For compressed items
        self.cache_dir = Path(tempfile.gettempdir()) / "qwen3vl_l3_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def put(self, key: str, tensor: torch.Tensor) -> bool:
        """Put an item in the L3 cache with optional compression"""
        with self._lock:
            tensor_size = tensor.element_size() * tensor.nelement()
            
            # Check if item fits
            if tensor_size > self.max_size_bytes:
                return False
            
            # Try compression if enabled
            compressed_tensor = None
            compressed_size = 0
            compression_ratio = 1.0
            
            if self.enable_compression:
                # Simple compression check - in reality, this would use more sophisticated methods
                # For now, we'll just check if tensor has many zeros or repeated values
                if tensor.numel() > 1000:  # Only compress larger tensors
                    unique_values = torch.unique(tensor).numel()
                    total_values = tensor.numel()
                    
                    # If tensor has many repeated values, compression might be effective
                    if unique_values / total_values < 0.5:  # Less than 50% unique values
                        # Simulate compression by reducing effective size
                        compressed_size = int(tensor_size * 0.5)  # 2:1 compression ratio
                        compression_ratio = 0.5
                        compressed_tensor = tensor  # In real implementation, this would be the compressed data
            
            effective_size = compressed_size if compressed_tensor is not None else tensor_size
            
            # Make space if needed
            while self.current_size_bytes + effective_size > self.max_size_bytes and len(self.cache) > 0:
                old_key, old_item = self.cache.popitem(last=False)
                old_effective_size = (
                    int(old_item.size_bytes * old_item.compression_ratio)
                    if old_item.compressed else old_item.size_bytes
                )
                self.current_size_bytes -= old_effective_size
                self.evictions += 1
            
            # Add new item
            item = CachedItem(
                key=key,
                tensor=tensor,
                size_bytes=tensor_size,
                access_time=time.time(),
                access_count=1,
                level=self.level,
                compressed=compressed_tensor is not None,
                compression_ratio=compression_ratio
            )
            
            self.cache[key] = item
            self.current_size_bytes += effective_size
            return True


class HierarchicalCacheManager:
    """
    Main hierarchical cache manager that coordinates L1, L2, and L3 caches.
    Implements cache coherency and data movement between levels.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Initialize cache levels
        self.l1_cache = L1CacheLevel(config.l1_size, config.l1_eviction_policy)
        self.l2_cache = L2CacheLevel(config.l2_size, config.l2_eviction_policy)
        self.l3_cache = L3CacheLevel(
            config.l3_size,
            config.l3_eviction_policy,
            config.enable_compression,
            config.compression_threshold
        )
        
        # Cache hierarchy: L1 -> L2 -> L3
        self.cache_levels = [self.l1_cache, self.l2_cache, self.l3_cache]
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0,
            'promotions': 0,  # From lower to higher levels
            'demotions': 0,   # From higher to lower levels
        }
        
        # Prefetching
        self.prefetch_enabled = config.prefetch_enabled
        self.prefetch_distance = config.prefetch_distance
        self.access_history = OrderedDict()
        self.access_history_size = 1000
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logging.info("HierarchicalCacheManager initialized with L1, L2, and L3 caches")
    
    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a tensor from the cache hierarchy.
        Implements cache coherency by checking all levels.
        """
        with self._lock:
            self.stats['total_requests'] += 1
            
            # Check L1 cache first
            tensor = self.l1_cache.get(key)
            if tensor is not None:
                self.stats['l1_hits'] += 1
                self._record_access(key)
                return tensor
            
            # Check L2 cache
            tensor = self.l2_cache.get(key)
            if tensor is not None:
                self.stats['l2_hits'] += 1
                # Promote to L1 if space allows
                if self.l1_cache.put(key, tensor):
                    # Remove from L2 to maintain coherency
                    self.l2_cache.remove(key)
                self._record_access(key)
                return tensor
            
            # Check L3 cache
            tensor = self.l3_cache.get(key)
            if tensor is not None:
                self.stats['l3_hits'] += 1
                # Promote to L2 if space allows, then to L1
                if self.l2_cache.put(key, tensor):
                    # Remove from L3 to maintain coherency
                    self.l3_cache.remove(key)
                    # Try to promote to L1 as well
                    l2_tensor = self.l2_cache.get(key)
                    if l2_tensor is not None and self.l1_cache.put(key, l2_tensor):
                        self.l2_cache.remove(key)
                self._record_access(key)
                return tensor
            
            # Miss in all levels
            self.stats['misses'] += 1
            self._record_access(key, miss=True)
            return None
    
    def put_tensor(self, tensor: torch.Tensor, key: str = None) -> bool:
        """
        Put a tensor in the cache hierarchy.
        
        Args:
            tensor: Tensor to cache
            key: Key for the tensor (if None, will generate one)
        
        Returns:
            True if successfully cached, False otherwise
        """
        if key is None:
            key = f"tensor_{id(tensor)}_{int(time.time() * 1000000)}"
        
        with self._lock:
            # Try to put in L1 first (fastest)
            success = self.l1_cache.put(key, tensor)
            
            # If L1 fails, try L2, then L3
            if not success:
                success = self.l2_cache.put(key, tensor)
            if not success:
                success = self.l3_cache.put(key, tensor)
            
            if success:
                self._record_access(key)
            
            return success
    
    def remove_tensor(self, key: str) -> bool:
        """Remove a tensor from all cache levels."""
        with self._lock:
            # Remove from all levels
            l1_removed = self.l1_cache.remove(key)
            l2_removed = self.l2_cache.remove(key)
            l3_removed = self.l3_cache.remove(key)
            
            # Return True if removed from any level
            return l1_removed or l2_removed or l3_removed
    
    def _record_access(self, key: str, miss: bool = False):
        """Record access for prefetching and statistics."""
        current_time = time.time()
        
        # Maintain access history for prefetching
        self.access_history[key] = current_time
        if len(self.access_history) > self.access_history_size:
            self.access_history.popitem(last=False)
    
    def get_tensor_multi_level(self, keys: List[str]) -> List[Optional[torch.Tensor]]:
        """
        Get multiple tensors, optimizing for cache performance.
        
        Args:
            keys: List of tensor keys to retrieve
        
        Returns:
            List of tensors (None for misses)
        """
        results = []
        for key in keys:
            tensor = self.get_tensor(key)
            results.append(tensor)
        
        return results
    
    def promote_tensor(self, key: str, target_level: CacheLevel) -> bool:
        """
        Explicitly promote a tensor to a higher cache level.
        
        Args:
            key: Key of tensor to promote
            target_level: Target cache level
        
        Returns:
            True if promotion was successful
        """
        with self._lock:
            tensor = None
            
            # Find tensor in any level
            if key in self.l3_cache.cache:
                tensor = self.l3_cache.cache[key].tensor
                self.l3_cache.remove(key)
            elif key in self.l2_cache.cache:
                tensor = self.l2_cache.cache[key].tensor
                self.l2_cache.remove(key)
            elif key in self.l1_cache.cache:
                tensor = self.l1_cache.cache[key].tensor
                # Already at highest level
                return True
            
            if tensor is None:
                return False
            
            # Put in target level and higher levels
            if target_level == CacheLevel.L3:
                return self.l3_cache.put(key, tensor)
            elif target_level == CacheLevel.L2:
                success = self.l2_cache.put(key, tensor)
                if success:
                    # Also try to put in L1
                    self.l1_cache.put(key, tensor)
                return success
            elif target_level == CacheLevel.L1:
                success = self.l1_cache.put(key, tensor)
                if success:
                    # Also put in L2 and L3 for coherency
                    self.l2_cache.put(key, tensor)
                    self.l3_cache.put(key, tensor)
                return success
            
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            l1_stats = self.l1_cache.get_stats()
            l2_stats = self.l2_cache.get_stats()
            l3_stats = self.l3_cache.get_stats()
            
            total_hits = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']
            total_requests = self.stats['total_requests']
            global_hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                'global_hit_rate': global_hit_rate,
                'total_requests': total_requests,
                'total_hits': total_hits,
                'total_misses': self.stats['misses'],
                'promotions': self.stats['promotions'],
                'demotions': self.stats['demotions'],
                'l1_stats': l1_stats,
                'l2_stats': l2_stats,
                'l3_stats': l3_stats,
                'total_cache_size_bytes': (
                    l1_stats['current_size_bytes'] +
                    l2_stats['current_size_bytes'] +
                    l3_stats['current_size_bytes']
                ),
                'total_max_size_bytes': (
                    l1_stats['max_size_bytes'] +
                    l2_stats['max_size_bytes'] +
                    l3_stats['max_size_bytes']
                )
            }
    
    def clear_cache(self, level: Optional[CacheLevel] = None):
        """Clear cache levels."""
        with self._lock:
            if level is None:
                # Clear all levels
                self.l1_cache.clear()
                self.l2_cache.clear()
                self.l3_cache.clear()
                self.access_history.clear()
            elif level == CacheLevel.L1:
                self.l1_cache.clear()
            elif level == CacheLevel.L2:
                self.l2_cache.clear()
            elif level == CacheLevel.L3:
                self.l3_cache.clear()
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Get memory footprint of each cache level."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        
        return {
            'l1_memory_bytes': l1_stats['current_size_bytes'],
            'l2_memory_bytes': l2_stats['current_size_bytes'],
            'l3_memory_bytes': l3_stats['current_size_bytes'],
            'total_memory_bytes': (
                l1_stats['current_size_bytes'] +
                l2_stats['current_size_bytes'] +
                l3_stats['current_size_bytes']
            )
        }


def create_hierarchical_cache_manager(config: Optional[CacheConfig] = None) -> HierarchicalCacheManager:
    """
    Factory function to create a hierarchical cache manager with hardware-specific settings.
    
    Args:
        config: Optional cache configuration
    
    Returns:
        HierarchicalCacheManager instance
    """
    if config is None:
        # Default configuration optimized for Intel i5-10210U
        config = CacheConfig(
            l1_size=64 * 1024 * 1024,      # 64MB for L1
            l2_size=256 * 1024 * 1024,     # 256MB for L2
            l3_size=512 * 1024 * 1024,     # 512MB for L3
            enable_compression=True,
            compression_threshold=0.1
        )
    
    return HierarchicalCacheManager(config)


# Example usage and testing
if __name__ == "__main__":
    print("Hierarchical Cache Manager for Qwen3-VL")
    print("=" * 60)
    
    # Create cache manager with default configuration
    config = CacheConfig()
    cache_manager = create_hierarchical_cache_manager(config)
    
    print(f"\n1. Created cache manager with:")
    print(f"   L1 cache: {config.l1_size / (1024**2):.1f}MB")
    print(f"   L2 cache: {config.l2_size / (1024**2):.1f}MB") 
    print(f"   L3 cache: {config.l3_size / (1024**2):.1f}MB")
    print(f"   Compression enabled: {config.enable_compression}")
    
    # Test tensor operations
    print(f"\n2. Testing tensor operations...")
    
    # Create and cache tensors
    tensor1 = torch.randn(100, 100, dtype=torch.float16)
    key1 = "tensor_1"
    success1 = cache_manager.put_tensor(tensor1, key1)
    print(f"   Cached tensor1: {success1}")
    
    tensor2 = torch.randn(200, 200, dtype=torch.float16)
    key2 = "tensor_2"
    success2 = cache_manager.put_tensor(tensor2, key2)
    print(f"   Cached tensor2: {success2}")
    
    # Retrieve tensors
    retrieved1 = cache_manager.get_tensor(key1)
    print(f"   Retrieved tensor1: {retrieved1 is not None}")
    
    retrieved2 = cache_manager.get_tensor(key2)
    print(f"   Retrieved tensor2: {retrieved2 is not None}")
    
    # Test multiple tensor retrieval
    keys = [key1, key2, "nonexistent_tensor"]
    results = cache_manager.get_tensor_multi_level(keys)
    print(f"   Multi-level retrieval: {sum(1 for r in results if r is not None)}/{len(keys)} hits")
    
    # Show cache statistics
    print(f"\n3. Cache statistics:")
    stats = cache_manager.get_cache_stats()
    print(f"   Global hit rate: {stats['global_hit_rate']:.2%}")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total hits: {stats['total_hits']}")
    print(f"   Total misses: {stats['total_misses']}")
    
    print(f"\n   L1 cache - Hit rate: {stats['l1_stats']['hits']/(stats['l1_stats']['hits']+stats['l1_stats']['misses']) if (stats['l1_stats']['hits']+stats['l1_stats']['misses']) > 0 else 0:.2%}, "
          f"Utilization: {stats['l1_stats']['utilization']:.2%}")
    print(f"   L2 cache - Hit rate: {stats['l2_stats']['hits']/(stats['l2_stats']['hits']+stats['l2_stats']['misses']) if (stats['l2_stats']['hits']+stats['l2_stats']['misses']) > 0 else 0:.2%}, "
          f"Utilization: {stats['l2_stats']['utilization']:.2%}")
    print(f"   L3 cache - Hit rate: {stats['l3_stats']['hits']/(stats['l3_stats']['hits']+stats['l3_stats']['misses']) if (stats['l3_stats']['hits']+stats['l3_stats']['misses']) > 0 else 0:.2%}, "
          f"Utilization: {stats['l3_stats']['utilization']:.2%}")
    
    # Show memory footprint
    footprint = cache_manager.get_memory_footprint()
    print(f"\n4. Memory footprint:")
    print(f"   L1: {footprint['l1_memory_bytes'] / (1024**2):.2f}MB")
    print(f"   L2: {footprint['l2_memory_bytes'] / (1024**2):.2f}MB")
    print(f"   L3: {footprint['l3_memory_bytes'] / (1024**2):.2f}MB")
    print(f"   Total: {footprint['total_memory_bytes'] / (1024**2):.2f}MB")
    
    print(f"\nHierarchical Cache Manager test completed!")