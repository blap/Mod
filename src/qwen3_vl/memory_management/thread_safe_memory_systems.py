"""Thread-safe implementations of memory prefetching and caching systems with proper synchronization mechanisms."""

import threading
import time
import queue
import torch
import numpy as np
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Any, List, Tuple, Union
from enum import Enum
import logging
import os
import tempfile
import math
import psutil


class TensorType(Enum):
    """Types of tensors in the system"""
    GENERAL = "general"
    KV_CACHE = "kv_cache"
    IMAGE_FEATURES = "image_features"
    TEXT_EMBEDDINGS = "text_embeddings"
    GRADIENTS = "gradients"
    ACTIVATIONS = "activations"


class ThreadSafeTensorCache:
    """Thread-safe tensor cache with proper synchronization for memory prefetching systems."""
    
    def __init__(self, max_cache_size: int = 100, max_cache_size_per_key: int = 5):
        self.cache = defaultdict(list)  # {(shape, dtype, device): [tensor, ...]}
        self.max_cache_size = max_cache_size
        self.max_cache_size_per_key = max_cache_size_per_key
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Track cache statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'cached_tensors': 0,
            'evictions': 0
        }
        
    def get_tensor(self, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device('cpu')):
        """Get a tensor from cache if available, thread-safe."""
        with self._lock:
            key = (tuple(shape), dtype, device)
            self.stats['total_requests'] += 1
            
            if self.cache[key] and len(self.cache[key]) > 0:
                tensor = self.cache[key].pop()
                self.stats['cached_tensors'] -= 1
                self.stats['cache_hits'] += 1
                return tensor
            else:
                self.stats['cache_misses'] += 1
                # Create new tensor if not in cache
                return torch.empty(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the cache for reuse, thread-safe."""
        with self._lock:
            key = (tensor.shape, tensor.dtype, tensor.device)
            
            # Only cache if not part of computation graph and cache isn't too large
            if (tensor.requires_grad or tensor.grad_fn is not None or 
                len(self.cache[key]) >= self.max_cache_size_per_key or
                self.stats['cached_tensors'] >= self.max_cache_size):
                return False  # Don't cache tensors that are part of the computation graph or if cache is full
            
            self.cache[key].append(tensor)
            self.stats['cached_tensors'] += 1
            return True
    
    def clear_cache(self):
        """Clear the entire cache, thread-safe."""
        with self._lock:
            self.cache.clear()
            self.stats['cached_tensors'] = 0
            self.stats['evictions'] = 0
    
    def get_cache_stats(self):
        """Get cache statistics, thread-safe."""
        with self._lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            stats['cache_size'] = sum(len(tensors) for tensors in self.cache.values())
            return stats


class ThreadSafeBuddyAllocator:
    """Thread-safe buddy allocator with proper synchronization for memory management systems."""
    
    def __init__(self, total_size: int = 1024*1024*128, min_block_size: int = 256):
        self.total_size = total_size
        self.min_block_size = min_block_size
        self.max_order = int(math.log2(total_size // min_block_size))
        
        # Free blocks organized by order (size = min_block_size * 2^order)
        self.free_blocks = defaultdict(set)  # {order: {start_address}}
        
        # Initially, the entire pool is one free block
        self.free_blocks[self.max_order].add(0)
        
        # Keep track of allocated blocks
        self.allocated_blocks = {}  # {start_addr: (size, order)}
        
        # Thread safety
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)  # For signaling changes
        
        # Track allocation statistics
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'failed_allocations': 0,
            'fragmentation_events': 0
        }
        
    def allocate(self, size: int):
        """Allocate a block of at least 'size' bytes, thread-safe."""
        with self._condition:
            actual_size = self._round_up_to_power_of_2(max(size, self.min_block_size))
            req_order = int(math.log2(actual_size // self.min_block_size))
            
            # Find a free block of at least the required order
            for order in range(req_order, self.max_order + 1):
                if self.free_blocks[order]:
                    # Found a block, allocate it
                    addr = self.free_blocks[order].pop()
                    
                    # Split if the block is larger than needed
                    for current_order in range(order, req_order, -1):
                        # Split the block in half - put the second half in the lower order list
                        buddy_addr = addr + (self.min_block_size << (current_order - 1))
                        self.free_blocks[current_order - 1].add(buddy_addr)
                        
                        # Update the size of the current block
                        actual_size = self.min_block_size << (current_order - 1)
                    
                    self.allocated_blocks[addr] = (actual_size, req_order)
                    self.stats['allocations'] += 1
                    self._condition.notify_all()  # Notify any waiting threads
                    return addr, actual_size
            
            self.stats['failed_allocations'] += 1
            return None, 0  # No suitable block found
    
    def deallocate(self, addr: int, size: int):
        """Deallocate the block at 'addr', thread-safe."""
        with self._condition:
            if addr not in self.allocated_blocks:
                return False  # Block not allocated by this allocator
            
            actual_size, order = self.allocated_blocks[addr]
            if actual_size != size:
                raise ValueError(f"Deallocating wrong size: expected {actual_size}, got {size}")
            
            # Remove from allocated blocks
            del self.allocated_blocks[addr]
            
            # Add to free list
            current_addr = addr
            current_order = order
            
            # Try to merge with buddies
            while current_order < self.max_order:
                buddy_addr = self._get_buddy_addr(current_addr, current_order)
                
                # Check if buddy is free
                if buddy_addr in self.free_blocks[current_order]:
                    # Remove buddy from free list
                    self.free_blocks[current_order].remove(buddy_addr)
                    
                    # Merge with current block (the lower address becomes the new block)
                    current_addr = min(current_addr, buddy_addr)
                    current_order += 1
                else:
                    # Buddy not free, add current block to free list and stop
                    self.free_blocks[current_order].add(current_addr)
                    break
            else:
                # Reached max order, add to free list
                self.free_blocks[current_order].add(current_addr)
            
            self.stats['deallocations'] += 1
            self._condition.notify_all()  # Notify any waiting threads
            return True
    
    def _round_up_to_power_of_2(self, size: int) -> int:
        """Round size up to the next power of 2."""
        power = 1
        while power < size:
            power <<= 1
        return power
    
    def _get_buddy_addr(self, addr: int, order: int) -> int:
        """Get the address of the buddy block."""
        block_size = self.min_block_size << order
        return addr ^ block_size  # XOR to flip the block bit
    
    def get_stats(self):
        """Get memory pool statistics, thread-safe."""
        with self._lock:
            total_free = sum((1 << order) * len(blocks) for order, blocks in self.free_blocks.items())
            largest_free_block_size = 0
            for order, blocks in self.free_blocks.items():
                if blocks:
                    block_size = 1 << order
                    largest_free_block_size = max(largest_free_block_size, block_size)
            
            fragmentation = 1.0 - (largest_free_block_size / total_free) if total_free > 0 else 0.0
            
            return {
                'total_free_bytes': total_free,
                'largest_free_block': largest_free_block_size,
                'fragmentation': fragmentation,
                'allocated_blocks': len(self.allocated_blocks),
                'num_free_blocks': sum(len(blocks) for blocks in self.free_blocks.values()),
                'allocation_stats': self.stats.copy()
            }
    
    def wait_for_memory(self, min_free_size: int, timeout: float = 1.0) -> bool:
        """Wait for memory to become available, thread-safe."""
        with self._condition:
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if we have a block of at least the required size
                required_order = int(math.ceil(math.log2(min_free_size / self.min_block_size)))
                
                for order in range(required_order, self.max_order + 1):
                    if self.free_blocks[order]:
                        return True  # Memory is available
                
                # Wait for notification about memory availability
                self._condition.wait(timeout=0.1)
            
            return False  # Timeout reached
    
    def clear_cache(self):
        """Clear cache (for compatibility with existing interfaces), thread-safe."""
        with self._lock:
            pass  # In buddy allocator, we don't have a cache to clear


class ThreadSafePrefetchingManager:
    """Thread-safe prefetching manager with proper synchronization mechanisms."""
    
    def __init__(self, prefetch_buffer_size: int = 10, num_prefetch_workers: int = 2):
        self.prefetch_queue = queue.Queue(maxsize=prefetch_buffer_size)
        self.prefetch_threads = []
        self.prefetch_active = False
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # For tracking
        self.prefetch_history = []
        self._shutdown_event = threading.Event()
        self.num_prefetch_workers = num_prefetch_workers
        
        # Prefetch statistics
        self.stats = {
            'prefetch_attempts': 0,
            'successful_prefetches': 0,
            'failed_prefetches': 0,
            'prefetch_queue_full': 0
        }
    
    def start_prefetching(self):
        """Start the prefetching threads."""
        if not self.prefetch_active:
            self.prefetch_active = True
            for i in range(self.num_prefetch_workers):
                thread = threading.Thread(target=self._prefetch_worker, daemon=True)
                thread.start()
                self.prefetch_threads.append(thread)
    
    def stop_prefetching(self):
        """Stop the prefetching threads."""
        self.prefetch_active = False
        self._shutdown_event.set()
        
        # Send sentinel to stop threads
        for _ in range(self.num_prefetch_workers):
            try:
                self.prefetch_queue.put(None, timeout=1.0)
            except queue.Full:
                pass  # Queue is full but shutdown event is set
        
        # Join all threads
        for thread in self.prefetch_threads:
            thread.join(timeout=2.0)
        
        self.prefetch_threads.clear()
    
    def _prefetch_worker(self):
        """Background worker for prefetching tensors."""
        while self.prefetch_active and not self._shutdown_event.is_set():
            try:
                item = self.prefetch_queue.get(timeout=1.0)
                if item is None:  # Sentinel to stop thread
                    break
                
                tensor, target_device = item
                # Simulate prefetching operation
                with self._condition:
                    try:
                        _ = tensor.to(target_device, non_blocking=True)
                        self.prefetch_history.append((time.time(), target_device))
                        self.stats['successful_prefetches'] += 1
                    except Exception as e:
                        print(f"Prefetch error: {e}")
                        self.stats['failed_prefetches'] += 1
                
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch worker error: {e}")
    
    def prefetch_tensor(self, tensor: torch.Tensor, target_device: torch.device, timeout: float = 0.1) -> bool:
        """Prefetch a tensor to the target device, thread-safe."""
        try:
            self.prefetch_queue.put((tensor, target_device), block=True, timeout=timeout)
            self.stats['prefetch_attempts'] += 1
            return True
        except queue.Full:
            self.stats['prefetch_queue_full'] += 1
            return False  # Buffer is full, can't prefetch
    
    def get_prefetch_status(self):
        """Get prefetching status, thread-safe."""
        with self._lock:
            return {
                'queue_size': self.prefetch_queue.qsize() if not self.prefetch_active else 0,
                'history_length': len(self.prefetch_history),
                'active': self.prefetch_active,
                'stats': self.stats.copy()
            }


class ThreadSafeKVCacheManager:
    """Thread-safe KV cache manager with proper synchronization for multi-threaded access."""
    
    def __init__(self, cache_size: int = 1024*1024*1024):  # 1GB default
        self.cache_size = cache_size
        self.kv_cache = OrderedDict()  # {layer_idx: {'k': tensor, 'v': tensor}}
        self.cache_timestamps = {}  # {layer_idx: timestamp}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'evictions': 0,
            'current_size': 0
        }
    
    def update_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
        """Update the KV cache for a specific layer, thread-safe."""
        with self._condition:
            # Calculate tensor size in bytes
            tensor_size = (key_states.numel() * key_states.element_size() + 
                          value_states.numel() * value_states.element_size())
            
            # Check if we need to evict old entries to make space
            while (self.stats['current_size'] + tensor_size > self.cache_size and 
                   len(self.kv_cache) > 0):
                oldest_layer_idx, _ = self.kv_cache.popitem(last=False)
                if oldest_layer_idx in self.cache_timestamps:
                    del self.cache_timestamps[oldest_layer_idx]
                self.stats['evictions'] += 1
                self.stats['current_size'] -= self._get_tensor_size(oldest_layer_idx)
            
            # Store the new tensors
            self.kv_cache[layer_idx] = {'k': key_states, 'v': value_states}
            self.cache_timestamps[layer_idx] = time.time()
            self.stats['current_size'] += tensor_size
            self._condition.notify_all()  # Notify any waiting threads
    
    def get_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the KV cache for a specific layer, thread-safe."""
        with self._lock:
            self.stats['total_requests'] += 1
            
            if layer_idx in self.kv_cache:
                cache_entry = self.kv_cache[layer_idx]
                self.stats['cache_hits'] += 1
                
                # Move to end (LRU)
                self.kv_cache.move_to_end(layer_idx)
                self.cache_timestamps[layer_idx] = time.time()
                
                return cache_entry['k'], cache_entry['v']
            else:
                self.stats['cache_misses'] += 1
                return None
    
    def _get_tensor_size(self, layer_idx: int) -> int:
        """Get the size of tensors in the cache for a specific layer."""
        if layer_idx in self.kv_cache:
            entry = self.kv_cache[layer_idx]
            k_tensor = entry['k']
            v_tensor = entry['v']
            return (k_tensor.numel() * k_tensor.element_size() + 
                   v_tensor.numel() * v_tensor.element_size())
        return 0
    
    def clear_cache(self):
        """Clear the KV cache, thread-safe."""
        with self._condition:
            self.kv_cache.clear()
            self.cache_timestamps.clear()
            self.stats['current_size'] = 0
            self.stats['evictions'] += len(self.kv_cache)  # Count cleared entries as evictions
    
    def get_cache_stats(self):
        """Get KV cache statistics, thread-safe."""
        with self._lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            stats['cache_utilization'] = stats['current_size'] / self.cache_size if self.cache_size > 0 else 0
            return stats


class ThreadSafeMemoryPool:
    """Thread-safe memory pool with proper synchronization for all operations."""
    
    def __init__(self, pool_size: int = 1024*1024*128, min_block_size: int = 256):
        # Initialize buddy allocator
        self.buddy_allocator = ThreadSafeBuddyAllocator(pool_size, min_block_size)
        
        # Initialize tensor cache
        self.tensor_cache = ThreadSafeTensorCache()
        
        # Initialize prefetching manager
        self.prefetch_manager = ThreadSafePrefetchingManager()
        
        # Thread safety for high-level operations
        self._lock = threading.RLock()
        
        # Initialize prefetching
        self.prefetch_manager.start_prefetching()
    
    def allocate_tensor(self, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device('cpu'), 
                       tensor_type: TensorType = TensorType.GENERAL):
        """Allocate a tensor using the memory pool, thread-safe."""
        with self._lock:
            # First, try to get from tensor cache
            cached_tensor = self.tensor_cache.get_tensor(shape, dtype, device)
            if cached_tensor is not None:
                return cached_tensor
            
            # If not in cache, allocate new tensor
            return torch.empty(shape, dtype=dtype, device=device)
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Deallocate a tensor and return it to the cache if appropriate, thread-safe."""
        with self._lock:
            # Return to tensor cache
            return self.tensor_cache.return_tensor(tensor)
    
    def allocate_memory_block(self, size: int):
        """Allocate a memory block using the buddy allocator, thread-safe."""
        return self.buddy_allocator.allocate(size)
    
    def deallocate_memory_block(self, addr: int, size: int):
        """Deallocate a memory block, thread-safe."""
        return self.buddy_allocator.deallocate(addr, size)
    
    def prefetch_tensor_to_device(self, tensor: torch.Tensor, target_device: torch.device):
        """Prefetch a tensor to a target device, thread-safe."""
        return self.prefetch_manager.prefetch_tensor(tensor, target_device)
    
    def get_pool_stats(self):
        """Get comprehensive pool statistics, thread-safe."""
        with self._lock:
            buddy_stats = self.buddy_allocator.get_stats()
            cache_stats = self.tensor_cache.get_cache_stats()
            prefetch_stats = self.prefetch_manager.get_prefetch_status()
            
            return {
                'buddy_allocator': buddy_stats,
                'tensor_cache': cache_stats,
                'prefetch_manager': prefetch_stats
            }
    
    def clear_all_caches(self):
        """Clear all caches in the memory pool, thread-safe."""
        with self._lock:
            self.tensor_cache.clear_cache()
            self.buddy_allocator.clear_cache()
    
    def shutdown(self):
        """Shut down the memory pool and clean up resources."""
        self.prefetch_manager.stop_prefetching()
        self.clear_all_caches()


# KV Cache Optimized Attention with thread-safe memory management
class KVCacheOptimizedAttention(torch.nn.Module):
    """KV cache optimized attention with thread-safe memory management."""
    
    def __init__(self, config, layer_idx: int, cache_strategy: str = "standard"):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', 64)
        self.num_heads = getattr(config, 'num_attention_heads', 32)
        self.hidden_size = getattr(config, 'hidden_size', 2048)
        self.scale = self.head_dim ** -0.5
        
        # KV cache manager with thread-safe operations
        self.kv_cache_manager = ThreadSafeKVCacheManager(
            cache_size=getattr(config, 'kv_cache_size', 1024*1024*512)  # 512MB default
        )
        
        # Initialize projections
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Cache strategy
        self.cache_strategy = cache_strategy
        self.kv_cache = None
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with thread-safe KV cache operations."""
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if use_cache:
            # Thread-safe cache update
            self.kv_cache_manager.update_cache(self.layer_idx, key_states, value_states)
            
            # Retrieve from cache if available
            cached_kv = self.kv_cache_manager.get_cache(self.layer_idx)
            if cached_kv is not None:
                key_states, value_states = cached_kv
            else:
                # If not in cache, use the newly computed states
                pass
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Return output, attention weights, and past key value
        return attn_output, attn_weights, (key_states, value_states) if use_cache else None

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.kv_cache_manager.get_cache_stats()


# Block Sparse Attention with thread-safe operations
class BlockSparseAttention(torch.nn.Module):
    """Block sparse attention with thread-safe operations."""
    
    def __init__(self, config, layer_idx: int, block_size: int = 64, sparsity_ratio: float = 0.5):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', 64)
        self.num_heads = getattr(config, 'num_attention_heads', 32)
        self.hidden_size = getattr(config, 'hidden_size', 2048)
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        
        # Thread-safe memory pool
        self.memory_pool = ThreadSafeMemoryPool()
        
        # Initialize projections
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Block sparse pattern
        num_blocks = math.ceil(self.head_dim / self.block_size)
        self.register_buffer('block_mask', torch.ones(
            self.num_heads, 
            num_blocks, 
            num_blocks
        ).bool())
        
        # Apply sparsity to the mask
        self._apply_sparsity()
    
    def _apply_sparsity(self):
        """Apply sparsity pattern to the block mask."""
        with torch.no_grad():
            # Apply sparsity by randomly masking out blocks
            num_blocks = self.block_mask.shape[1] * self.block_mask.shape[2]
            num_masked = int(num_blocks * (1 - self.sparsity_ratio))
            
            # Flatten mask for random selection
            flat_mask = self.block_mask.view(self.num_heads, -1)
            
            # Create random indices to mask
            for head_idx in range(self.num_heads):
                indices = torch.randperm(flat_mask.shape[1])[:num_masked]
                flat_mask[head_idx, indices] = False
            
            # Reshape back
            self.block_mask = flat_mask.view_as(self.block_mask)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with block sparse attention and thread-safe operations."""
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if use_cache and past_key_value is not None:
            # Thread-safe cache update
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # Calculate number of blocks
        num_q_blocks = math.ceil(q_len / self.block_size)
        num_kv_blocks = math.ceil(key_states.size(2) / self.block_size)
        
        # Pad sequences to be divisible by block size
        padded_q_len = num_q_blocks * self.block_size
        padded_kv_len = num_kv_blocks * self.block_size
        
        # Pad query and key/value tensors
        if query_states.size(2) < padded_q_len:
            pad_size = padded_q_len - query_states.size(2)
            query_states = torch.cat([query_states, torch.zeros(
                bsz, self.num_heads, pad_size, self.head_dim, 
                dtype=query_states.dtype, device=query_states.device)], dim=2)
        
        if key_states.size(2) < padded_kv_len:
            pad_size = padded_kv_len - key_states.size(2)
            key_states = torch.cat([key_states, torch.zeros(
                bsz, self.num_heads, pad_size, self.head_dim, 
                dtype=key_states.dtype, device=key_states.device)], dim=2)
            value_states = torch.cat([value_states, torch.zeros(
                bsz, self.num_heads, pad_size, self.head_dim, 
                dtype=value_states.dtype, device=value_states.device)], dim=2)
        
        # Initialize output tensor
        output = torch.zeros_like(query_states)
        
        # Process each query block with corresponding key-value blocks based on the sparse pattern
        for q_block_idx in range(num_q_blocks):
            for kv_block_idx in range(num_kv_blocks):
                # Check if this block should be computed based on the sparse pattern
                if self.block_mask[0, q_block_idx, kv_block_idx]:  # Using first head's pattern for all heads
                    # Calculate block boundaries
                    q_start = q_block_idx * self.block_size
                    q_end = min(q_start + self.block_size, padded_q_len)
                    k_start = kv_block_idx * self.block_size
                    k_end = min(k_start + self.block_size, padded_kv_len)
                    
                    # Extract block tensors
                    q_block = query_states[:, :, q_start:q_end, :]  # [bsz, num_heads, block_size, head_dim]
                    k_block = key_states[:, :, k_start:k_end, :]   # [bsz, num_heads, block_size, head_dim]
                    v_block = value_states[:, :, k_start:k_end, :] # [bsz, num_heads, block_size, head_dim]
                    
                    # Compute attention scores: [bsz, num_heads, block_size, block_size]
                    attn_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                    
                    # Apply attention mask if provided
                    if attention_mask is not None:
                        # Need to slice the mask appropriately for this block
                        mask_start = q_start
                        mask_end = min(q_end, q_len)
                        block_mask = attention_mask[:, :, mask_start:mask_end, k_start:k_end]
                        attn_scores = attn_scores + block_mask
                    
                    # Apply softmax
                    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    
                    # Compute output: [bsz, num_heads, block_size, head_dim]
                    block_output = torch.matmul(attn_weights, v_block)
                    
                    # Store in output tensor
                    output_start = q_start
                    output_end = min(q_end, q_len)
                    output[:, :, output_start:output_end, :] = block_output[:, :, :output_end-output_start, :]
        
        # Remove padding if necessary
        output = output[:, :, :q_len, :].contiguous()
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, q_len, self.hidden_size)
        output = self.o_proj(output)
        
        # Return output, attention weights, and past key value
        return output, None, (key_states, value_states) if use_cache else None

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.memory_pool.get_pool_stats()


def create_thread_safe_memory_manager():
    """Factory function to create a thread-safe memory manager with all optimizations."""
    return ThreadSafeMemoryPool()


def create_thread_safe_kv_cache_manager():
    """Factory function to create a thread-safe KV cache manager."""
    return ThreadSafeKVCacheManager()


def create_thread_safe_block_sparse_attention(config, layer_idx: int):
    """Factory function to create a thread-safe block sparse attention module."""
    return BlockSparseAttention(config, layer_idx)


def create_thread_safe_kv_cache_optimized_attention(config, layer_idx: int):
    """Factory function to create a thread-safe KV cache optimized attention module."""
    return KVCacheOptimizedAttention(config, layer_idx)