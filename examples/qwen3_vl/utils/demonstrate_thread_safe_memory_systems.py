"""Demonstration of thread-safe memory prefetching and caching systems for Qwen3-VL model."""

import torch
import threading
import time
from typing import Dict, Optional, Any, List, Tuple
import queue
import numpy as np
import math


# Import the thread-safe memory systems
from src.qwen3_vl.utils.thread_safe_memory_systems import (
    TensorType,
    ThreadSafeTensorCache,
    ThreadSafeBuddyAllocator,
    ThreadSafePrefetchingManager,
    ThreadSafeKVCacheManager,
    ThreadSafeMemoryPool,
    KVCacheOptimizedAttention,
    BlockSparseAttention
)


def demonstrate_tensor_cache():
    """Demonstrate thread-safe tensor cache functionality."""
    print("=== Thread-Safe Tensor Cache Demonstration ===")

    cache = ThreadSafeTensorCache(max_cache_size=100, max_cache_size_per_key=5)

    # Test cache miss (first allocation)
    print("\n1. Testing cache miss (first allocation)...")
    tensor1 = cache.get_tensor((10, 20), torch.float32)
    print(f"   Allocated tensor: {tensor1.shape}, dtype: {tensor1.dtype}")

    # Return tensor to cache
    cache.return_tensor(tensor1)
    print(f"   Returned tensor to cache")

    # Test cache hit (second allocation of same shape)
    print("\n2. Testing cache hit (second allocation of same shape)...")
    tensor2 = cache.get_tensor((10, 20), torch.float32)
    print(f"   Retrieved tensor from cache: {tensor2.shape}, dtype: {tensor2.dtype}")
    print(f"   Same tensor object? {tensor1.data_ptr() == tensor2.data_ptr()}")

    # Check cache statistics
    stats = cache.get_cache_stats()
    print(f"\n3. Cache statistics:")
    print(f"   - Cache hits: {stats['cache_hits']}")
    print(f"   - Cache misses: {stats['cache_misses']}")
    print(f"   - Total requests: {stats['total_requests']}")
    print(f"   - Hit rate: {stats['hit_rate']:.2f}")
    print(f"   - Cached tensors: {stats['cached_tensors']}")

    print("\n[SUCCESS] Tensor cache demonstration completed")


def demonstrate_buddy_allocator():
    """Demonstrate thread-safe buddy allocator functionality."""
    print("\n=== Thread-Safe Buddy Allocator Demonstration ===")

    allocator = ThreadSafeBuddyAllocator(total_size=1024*1024, min_block_size=256)  # 1MB pool

    # Allocate some blocks
    print("\n1. Allocating memory blocks...")
    addr1, size1 = allocator.allocate(512)
    addr2, size2 = allocator.allocate(1024)
    addr3, size3 = allocator.allocate(256)

    print(f"   Allocated block 1: addr={addr1}, size={size1}")
    print(f"   Allocated block 2: addr={addr2}, size={size2}")
    print(f"   Allocated block 3: addr={addr3}, size={size3}")

    # Deallocate some blocks
    print("\n2. Deallocating memory blocks...")
    success1 = allocator.deallocate(addr1, size1)
    success2 = allocator.deallocate(addr2, size2)
    print(f"   Deallocation 1: {success1}")
    print(f"   Deallocation 2: {success2}")

    # Allocate a larger block (should reuse space)
    print("\n3. Allocating larger block (should reuse space)...")
    addr4, size4 = allocator.allocate(2048)
    print(f"   Allocated block 4: addr={addr4}, size={size4}")

    # Get allocator statistics
    stats = allocator.get_stats()
    print(f"\n4. Allocator statistics:")
    print(f"   - Total free bytes: {stats['total_free_bytes']:,}")
    print(f"   - Largest free block: {stats['largest_free_block']:,}")
    print(f"   - Fragmentation: {stats['fragmentation']:.2f}")
    print(f"   - Allocated blocks: {stats['allocated_blocks']}")
    print(f"   - Num free blocks: {stats['num_free_blocks']}")
    print(f"   - Allocation attempts: {stats['allocation_stats']['allocations']}")
    print(f"   - Deallocation attempts: {stats['allocation_stats']['deallocations']}")

    print("\n[SUCCESS] Buddy allocator demonstration completed")


def demonstrate_prefetching_manager():
    """Demonstrate thread-safe prefetching manager functionality."""
    print("\n=== Thread-Safe Prefetching Manager Demonstration ===")
    
    prefetcher = ThreadSafePrefetchingManager(prefetch_buffer_size=10, num_prefetch_workers=2)
    prefetcher.start_prefetching()
    
    # Create some tensors and prefetch them
    print("\n1. Prefetching tensors...")
    tensors = []
    for i in range(5):
        tensor = torch.randn(100, 100)
        success = prefetcher.prefetch_tensor(tensor, torch.device('cpu'))
        tensors.append(tensor)
        print(f"   Prefetch tensor {i+1}: {success}")
    
    # Wait a bit for prefetching to complete
    time.sleep(0.1)
    
    # Check prefetch status
    status = prefetcher.get_prefetch_status()
    print(f"\n2. Prefetch status:")
    print(f"   - Queue size: {status['queue_size']}")
    print(f"   - History length: {status['history_length']}")
    print(f"   - Active: {status['active']}")
    print(f"   - Prefetch attempts: {status['stats']['prefetch_attempts']}")
    print(f"   - Successful prefetches: {status['stats']['successful_prefetches']}")
    
    # Stop prefetching
    prefetcher.stop_prefetching()
    
    print("\n[SUCCESS] Prefetching manager demonstration completed")


def demonstrate_kv_cache_manager():
    """Demonstrate thread-safe KV cache manager functionality."""
    print("\n=== Thread-Safe KV Cache Manager Demonstration ===")
    
    kv_manager = ThreadSafeKVCacheManager(cache_size=1024*1024*10)  # 10MB cache
    
    # Create test tensors
    key_tensor = torch.randn(1, 32, 128, 64)  # [batch, heads, seq_len, head_dim]
    value_tensor = torch.randn(1, 32, 128, 64)
    
    # Update cache
    print("\n1. Updating KV cache...")
    kv_manager.update_cache(0, key_tensor, value_tensor)
    print(f"   Updated cache for layer 0 with tensors of shape {key_tensor.shape}")
    
    # Get from cache
    print("\n2. Retrieving from KV cache...")
    cached_kv = kv_manager.get_cache(0)
    if cached_kv is not None:
        cached_k, cached_v = cached_kv
        print(f"   Retrieved K tensor: {cached_k.shape}")
        print(f"   Retrieved V tensor: {cached_v.shape}")
        print(f"   Values match: {torch.allclose(key_tensor, cached_k) and torch.allclose(value_tensor, cached_v)}")
    
    # Check cache statistics
    stats = kv_manager.get_cache_stats()
    print(f"\n3. KV Cache statistics:")
    print(f"   - Cache hits: {stats['cache_hits']}")
    print(f"   - Cache misses: {stats['cache_misses']}")
    print(f"   - Total requests: {stats['total_requests']}")
    print(f"   - Hit rate: {stats['hit_rate']:.2f}")
    print(f"   - Current size: {stats['current_size']:,} bytes")
    print(f"   - Cache utilization: {stats['cache_utilization']:.2f}")
    
    print("\n[SUCCESS] KV cache manager demonstration completed")


def demonstrate_memory_pool():
    """Demonstrate thread-safe memory pool functionality."""
    print("\n=== Thread-Safe Memory Pool Demonstration ===")
    
    memory_pool = ThreadSafeMemoryPool(pool_size=1024*1024*32, min_block_size=256)  # 32MB pool
    
    # Allocate tensors through the pool
    print("\n1. Allocating tensors through memory pool...")
    tensor1 = memory_pool.allocate_tensor((100, 200), torch.float32)
    tensor2 = memory_pool.allocate_tensor((50, 50, 128), torch.float16)
    print(f"   Allocated tensor 1: {tensor1.shape}, {tensor1.dtype}")
    print(f"   Allocated tensor 2: {tensor2.shape}, {tensor2.dtype}")
    
    # Return tensors to cache
    print("\n2. Returning tensors to pool...")
    memory_pool.deallocate_tensor(tensor1)
    memory_pool.deallocate_tensor(tensor2)
    print(f"   Returned 2 tensors to pool")
    
    # Allocate memory blocks
    print("\n3. Allocating memory blocks...")
    addr1, size1 = memory_pool.allocate_memory_block(1024)
    addr2, size2 = memory_pool.allocate_memory_block(2048)
    print(f"   Allocated block 1: addr={addr1}, size={size1}")
    print(f"   Allocated block 2: addr={addr2}, size={size2}")
    
    # Deallocate memory blocks
    print("\n4. Deallocating memory blocks...")
    memory_pool.deallocate_memory_block(addr1, size1)
    memory_pool.deallocate_memory_block(addr2, size2)
    print(f"   Deallocated 2 memory blocks")
    
    # Get comprehensive pool statistics
    stats = memory_pool.get_pool_stats()
    print(f"\n5. Memory pool statistics:")
    print(f"   - Buddy allocator stats: {stats['buddy_allocator']['total_free_bytes']:,} free bytes")
    print(f"   - Tensor cache stats: {stats['tensor_cache']['cached_tensors']} cached tensors")
    print(f"   - Prefetch manager stats: {stats['prefetch_manager']['history_length']} prefetches")
    
    # Clean up
    memory_pool.shutdown()
    
    print("\n[SUCCESS] Memory pool demonstration completed")


def demonstrate_kv_cache_optimized_attention():
    """Demonstrate KV cache optimized attention functionality."""
    print("\n=== KV Cache Optimized Attention Demonstration ===")
    
    # Create a simple config for testing
    class TestConfig:
        hidden_size = 128
        num_attention_heads = 8
        head_dim = 16
        kv_cache_size = 1024*1024  # 1MB cache
    
    config = TestConfig()
    attention = KVCacheOptimizedAttention(config, layer_idx=0, cache_strategy="standard")
    
    # Create input tensors
    hidden_states = torch.randn(2, 10, 128)  # [batch, seq_len, hidden_size]
    
    print("\n1. Running KV cache optimized attention...")
    output, weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=True,
        cache_position=None
    )
    
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Past key value returned: {past_key_value is not None}")

    # Get cache statistics
    cache_stats = attention.get_cache_stats()
    print(f"\n2. Attention cache statistics:")
    print(f"   - Cache hits: {cache_stats['cache_hits']}")
    print(f"   - Cache misses: {cache_stats['cache_misses']}")
    print(f"   - Hit rate: {cache_stats['hit_rate']:.2f}")

    print("\n[SUCCESS] KV cache optimized attention demonstration completed")


def demonstrate_block_sparse_attention():
    """Demonstrate block sparse attention functionality."""
    print("\n=== Block Sparse Attention Demonstration ===")
    
    # Create a simple config for testing
    class TestConfig:
        hidden_size = 128
        num_attention_heads = 8
        head_dim = 16
        attention_implementation = "block_sparse"
    
    config = TestConfig()
    attention = BlockSparseAttention(config, layer_idx=0, block_size=32, sparsity_ratio=0.5)
    
    # Create input tensors
    hidden_states = torch.randn(2, 16, 128)  # [batch, seq_len, hidden_size]
    
    print("\n1. Running block sparse attention...")
    output, weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None
    )
    
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Sparsity ratio: {attention.sparsity_ratio}")
    print(f"   Block size: {attention.block_size}")
    
    print("\n[SUCCESS] Block sparse attention demonstration completed")


def demonstrate_concurrent_operations():
    """Demonstrate concurrent operations on thread-safe memory systems."""
    print("\n=== Concurrent Operations Demonstration ===")
    
    memory_pool = ThreadSafeMemoryPool(pool_size=1024*1024*16, min_block_size=256)  # 16MB pool
    
    results = []
    
    def worker(worker_id):
        """Worker function for concurrent operations."""
        for i in range(3):
            # Allocate tensor
            tensor = memory_pool.allocate_tensor((worker_id + 1, 10), torch.float32)
            
            # Simulate some work
            time.sleep(0.001)
            
            # Deallocate tensor
            memory_pool.deallocate_tensor(tensor)
            
            # Allocate memory block
            addr, size = memory_pool.allocate_memory_block(512)
            if addr is not None:
                # Simulate work with memory block
                time.sleep(0.001)
                # Deallocate memory block
                memory_pool.deallocate_memory_block(addr, size)
            
            results.append(f"worker_{worker_id}_iter_{i}_completed")
    
    # Start multiple threads
    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print(f"\n1. Concurrent operations completed:")
    print(f"   - Total operations: {len(results)}")
    print(f"   - Operations per worker: {len([r for r in results if 'worker_0' in r])}")
    
    # Check final pool statistics
    stats = memory_pool.get_pool_stats()
    print(f"\n2. Final memory pool statistics:")
    print(f"   - Buddy allocator free bytes: {stats['buddy_allocator']['total_free_bytes']:,}")
    print(f"   - Tensor cache size: {stats['tensor_cache']['cached_tensors']}")
    print(f"   - Cache hit rate: {stats['tensor_cache']['hit_rate']:.2f}")
    
    # Clean up
    memory_pool.shutdown()
    
    print("\n[SUCCESS] Concurrent operations demonstration completed")


def main():
    """Main demonstration function."""
    print("Qwen3-VL Thread-Safe Memory Prefetching and Caching Systems")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_tensor_cache()
    demonstrate_buddy_allocator()
    demonstrate_prefetching_manager()
    demonstrate_kv_cache_manager()
    demonstrate_memory_pool()
    demonstrate_kv_cache_optimized_attention()
    demonstrate_block_sparse_attention()
    demonstrate_concurrent_operations()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("\nKey Features Demonstrated:")
    print("1. Thread-safe tensor caching with LRU eviction")
    print("2. Buddy memory allocation with proper synchronization")
    print("3. Prefetching manager with concurrent workers")
    print("4. KV cache management with multiple strategies")
    print("5. Integrated memory pool with all optimizations")
    print("6. KV cache optimized attention mechanism")
    print("7. Block sparse attention with learned patterns")
    print("8. Safe concurrent operations across all components")


if __name__ == "__main__":
    main()