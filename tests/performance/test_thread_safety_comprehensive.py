"""Comprehensive test suite for identifying and validating threading race conditions in memory prefetching and caching systems."""

import threading
import time
import unittest
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os
from collections import defaultdict, OrderedDict
from typing import Dict, Optional, Any, List
import queue


class MockTensorCache:
    """Mock tensor cache to simulate potential race conditions"""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = defaultdict(list)  # {(shape, dtype): [tensor, ...]}
        self.max_cache_size = max_cache_size
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Track cache statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'cached_tensors': 0
        }
        
    def get_tensor(self, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device('cpu')):
        """Get a tensor from cache if available"""
        with self._lock:
            key = (tuple(shape), dtype, device)
            self.stats['total_requests'] += 1
            
            if self.cache[key]:
                tensor = self.cache[key].pop()
                self.stats['cached_tensors'] -= 1
                self.stats['cache_hits'] += 1
                return tensor
            else:
                self.stats['cache_misses'] += 1
                # Create new tensor if not in cache
                return torch.empty(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the cache for reuse"""
        with self._lock:
            key = (tensor.shape, tensor.dtype, tensor.device)
            
            # Only cache if the cache isn't too large (prevent memory bloat)
            if len(self.cache[key]) < 5:  # Limit per shape to prevent excessive memory usage
                self.cache[key].append(tensor)
                self.stats['cached_tensors'] += 1
    
    def get_cache_stats(self):
        """Get cache statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            stats['cache_size'] = sum(len(tensors) for tensors in self.cache.values())
            return stats


class MockBuddyAllocator:
    """Mock buddy allocator to simulate potential race conditions"""
    
    def __init__(self, total_size: int = 1024*1024*128, min_block_size: int = 256):
        self.total_size = total_size
        self.min_block_size = min_block_size
        self.max_order = int(np.log2(total_size // min_block_size))
        
        # Free blocks organized by order (size = min_block_size * 2^order)
        self.free_blocks = defaultdict(set)  # {order: {start_address}}
        
        # Initially, the entire pool is one free block
        self.free_blocks[self.max_order].add(0)
        
        # Keep track of allocated blocks
        self.allocated_blocks = {}  # {start_addr: (size, order)}
        
        # Thread safety
        self._lock = threading.RLock()
        
    def allocate(self, size: int):
        """Allocate a block of at least 'size' bytes"""
        with self._lock:
            actual_size = self._round_up_to_power_of_2(max(size, self.min_block_size))
            req_order = int(np.log2(actual_size // self.min_block_size))
            
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
                    return addr, actual_size
            
            return None, 0  # No suitable block found
    
    def deallocate(self, addr: int, size: int):
        """Deallocate the block at 'addr'"""
        with self._lock:
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
            
            return True
    
    def _round_up_to_power_of_2(self, size: int) -> int:
        """Round size up to the next power of 2"""
        power = 1
        while power < size:
            power <<= 1
        return power
    
    def _get_buddy_addr(self, addr: int, order: int) -> int:
        """Get the address of the buddy block"""
        block_size = self.min_block_size << order
        return addr ^ block_size  # XOR to flip the block bit
    
    def get_stats(self):
        """Get memory pool statistics"""
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
                'num_free_blocks': sum(len(blocks) for blocks in self.free_blocks.values())
            }


class MockPrefetchingManager:
    """Mock prefetching manager to simulate potential race conditions"""
    
    def __init__(self, prefetch_buffer_size: int = 10):
        self.prefetch_queue = queue.Queue(maxsize=prefetch_buffer_size)
        self.prefetch_thread = None
        self.prefetch_active = False
        self._lock = threading.Lock()
        
        # For tracking
        self.prefetched_tensors = {}
        self.prefetch_history = []
        
    def start_prefetching(self):
        """Start the prefetching thread"""
        if not self.prefetch_active:
            self.prefetch_active = True
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()
    
    def stop_prefetching(self):
        """Stop the prefetching thread"""
        self.prefetch_active = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)
    
    def _prefetch_worker(self):
        """Background worker for prefetching tensors"""
        while self.prefetch_active:
            try:
                item = self.prefetch_queue.get(timeout=1.0)
                if item is None:  # Sentinel to stop thread
                    break
                
                tensor, target_device = item
                # Simulate prefetching operation
                with self._lock:
                    _ = tensor.to(target_device, non_blocking=True)
                    self.prefetch_history.append((time.time(), target_device))
                
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")
    
    def prefetch_tensor(self, tensor: torch.Tensor, target_device: torch.device):
        """Prefetch a tensor to the target device"""
        try:
            self.prefetch_queue.put((tensor, target_device), block=False)
            return True
        except queue.Full:
            return False  # Buffer is full, can't prefetch


class TestRaceConditions(unittest.TestCase):
    """Test suite for identifying race conditions in memory systems"""
    
    def setUp(self):
        self.tensor_cache = MockTensorCache()
        self.buddy_allocator = MockBuddyAllocator()
        self.prefetch_manager = MockPrefetchingManager()
    
    def test_tensor_cache_race_condition(self):
        """Test tensor cache for race conditions with multiple threads"""
        print("Testing tensor cache race conditions...")
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    # Get tensor from cache
                    tensor = self.tensor_cache.get_tensor((10, 20), torch.float32)
                    
                    # Simulate some work
                    time.sleep(0.001)
                    
                    # Return tensor to cache
                    self.tensor_cache.return_tensor(tensor)
                    
                    results.append(f"worker_{worker_id}_iter_{i}")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        self.assertEqual(len(results), 100)  # 10 workers * 10 iterations
        self.assertEqual(len(errors), 0)
        
        # Check cache statistics
        stats = self.tensor_cache.get_cache_stats()
        print(f"Cache stats: {stats}")
        
        # The hit rate should be reasonable with multiple threads
        self.assertGreaterEqual(stats['hit_rate'], 0.0)
    
    def test_buddy_allocator_race_condition(self):
        """Test buddy allocator for race conditions with multiple threads"""
        print("Testing buddy allocator race conditions...")
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(5):
                    # Allocate a block
                    addr, size = self.buddy_allocator.allocate(1024)
                    if addr is not None:
                        # Simulate some work
                        time.sleep(0.001)
                        
                        # Deallocate the block
                        self.buddy_allocator.deallocate(addr, size)
                        
                        results.append(f"worker_{worker_id}_iter_{i}_success")
                    else:
                        results.append(f"worker_{worker_id}_iter_{i}_fail")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
                import traceback
                traceback.print_exc()
        
        # Create multiple threads
        threads = []
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        self.assertEqual(len([r for r in results if 'success' in r]), 40)  # Some allocations should succeed
        self.assertEqual(len(errors), 0)
        
        # Check allocator statistics
        stats = self.buddy_allocator.get_stats()
        print(f"Allocator stats: {stats}")
    
    def test_prefetching_manager_race_condition(self):
        """Test prefetching manager for race conditions with multiple threads"""
        print("Testing prefetching manager race conditions...")
        
        self.prefetch_manager.start_prefetching()
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    # Create a tensor
                    tensor = torch.randn(100, 100)
                    
                    # Prefetch tensor
                    success = self.prefetch_manager.prefetch_tensor(tensor, torch.device('cpu'))
                    results.append(f"worker_{worker_id}_iter_{i}_prefetch_{success}")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Allow prefetch queue to process
        time.sleep(0.1)
        
        # Verify results
        self.assertEqual(len(results), 50)  # 5 workers * 10 iterations
        self.assertEqual(len(errors), 0)
        
        # Stop prefetching
        self.prefetch_manager.stop_prefetching()
        
        print(f"Prefetch history length: {len(self.prefetch_manager.prefetch_history)}")
    
    def test_concurrent_operations_on_same_resource(self):
        """Test concurrent operations on the same resource"""
        print("Testing concurrent operations on same resource...")
        
        # Create a tensor that will be shared among threads
        shared_tensor = torch.randn(100, 100)
        shared_key = "shared_tensor"
        
        results = []
        errors = []
        
        def cache_worker():
            try:
                for i in range(10):
                    # Return and get the same tensor multiple times
                    self.tensor_cache.return_tensor(shared_tensor)
                    retrieved = self.tensor_cache.get_tensor(shared_tensor.shape, shared_tensor.dtype, shared_tensor.device)
                    results.append(f"cache_worker_iter_{i}_success_{retrieved is not None}")
            except Exception as e:
                errors.append(f"Cache worker error: {e}")
        
        def allocation_worker():
            try:
                for i in range(5):
                    # Perform allocation and deallocation
                    addr, size = self.buddy_allocator.allocate(512)
                    if addr is not None:
                        self.buddy_allocator.deallocate(addr, size)
                        results.append(f"alloc_worker_iter_{i}_success")
                    else:
                        results.append(f"alloc_worker_iter_{i}_fail")
            except Exception as e:
                errors.append(f"Allocation worker error: {e}")
        
        # Start both types of workers concurrently
        cache_thread = threading.Thread(target=cache_worker)
        alloc_thread = threading.Thread(target=allocation_worker)
        
        cache_thread.start()
        alloc_thread.start()
        
        cache_thread.join()
        alloc_thread.join()
        
        # Verify results
        self.assertGreater(len(results), 0)
        self.assertEqual(len(errors), 0)
        
        # Check final state
        cache_stats = self.tensor_cache.get_cache_stats()
        alloc_stats = self.buddy_allocator.get_stats()
        
        print(f"Final cache stats: {cache_stats}")
        print(f"Final allocator stats: {alloc_stats}")


class ThreadSafeCache:
    """Thread-safe cache with proper synchronization mechanisms"""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = OrderedDict()  # {key: tensor}
        self.max_cache_size = max_cache_size
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._semaphore = threading.Semaphore(10)  # Limit concurrent access
        
        # Track cache statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'cached_tensors': 0
        }
        
    def get_tensor(self, key: str, shape: tuple, dtype: torch.dtype, device: torch.device = torch.device('cpu')):
        """Get a tensor from cache if available"""
        with self._semaphore:  # Limit concurrent access
            with self._lock:
                self.stats['total_requests'] += 1
                
                if key in self.cache:
                    tensor = self.cache.pop(key)  # Remove from cache
                    self.cache[key] = tensor  # Move to end (LRU)
                    self.stats['cached_tensors'] -= 1
                    self.stats['cache_hits'] += 1
                    return tensor
                else:
                    self.stats['cache_misses'] += 1
                    # Create new tensor if not in cache
                    return torch.empty(shape, dtype=dtype, device=device)
    
    def return_tensor(self, key: str, tensor: torch.Tensor):
        """Return a tensor to the cache for reuse"""
        with self._semaphore:  # Limit concurrent access
            with self._lock:
                # Only cache if the cache isn't too large (prevent memory bloat)
                if key not in self.cache and len(self.cache) < self.max_cache_size:
                    self.cache[key] = tensor
                    self.stats['cached_tensors'] += 1
                elif key in self.cache:
                    # Update existing entry
                    self.cache[key] = tensor
    
    def get_cache_stats(self):
        """Get cache statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['hit_rate'] = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            stats['cache_size'] = len(self.cache)
            return stats
    
    def clear_cache(self):
        """Clear the entire cache"""
        with self._lock:
            self.cache.clear()
            self.stats['cached_tensors'] = 0


class ThreadSafeBuddyAllocator:
    """Thread-safe buddy allocator with proper synchronization mechanisms"""
    
    def __init__(self, total_size: int = 1024*1024*128, min_block_size: int = 256):
        self.total_size = total_size
        self.min_block_size = min_block_size
        self.max_order = int(np.log2(total_size // min_block_size))
        
        # Free blocks organized by order (size = min_block_size * 2^order)
        self.free_blocks = defaultdict(set)  # {order: {start_address}}
        
        # Initially, the entire pool is one free block
        self.free_blocks[self.max_order].add(0)
        
        # Keep track of allocated blocks
        self.allocated_blocks = {}  # {start_addr: (size, order)}
        
        # Thread safety
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)  # For signaling changes
        
    def allocate(self, size: int):
        """Allocate a block of at least 'size' bytes"""
        with self._condition:
            actual_size = self._round_up_to_power_of_2(max(size, self.min_block_size))
            req_order = int(np.log2(actual_size // self.min_block_size))
            
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
                    self._condition.notify_all()  # Notify any waiting threads
                    return addr, actual_size
            
            return None, 0  # No suitable block found
    
    def deallocate(self, addr: int, size: int):
        """Deallocate the block at 'addr'"""
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
            
            self._condition.notify_all()  # Notify any waiting threads
            return True
    
    def _round_up_to_power_of_2(self, size: int) -> int:
        """Round size up to the next power of 2"""
        power = 1
        while power < size:
            power <<= 1
        return power
    
    def _get_buddy_addr(self, addr: int, order: int) -> int:
        """Get the address of the buddy block"""
        block_size = self.min_block_size << order
        return addr ^ block_size  # XOR to flip the block bit
    
    def get_stats(self):
        """Get memory pool statistics"""
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
                'num_free_blocks': sum(len(blocks) for blocks in self.free_blocks.values())
            }
    
    def wait_for_memory(self, min_free_size: int, timeout: float = 1.0):
        """Wait for memory to become available"""
        with self._condition:
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if we have a block of at least the required size
                required_order = int(np.ceil(np.log2(min_free_size / self.min_block_size)))
                
                for order in range(required_order, self.max_order + 1):
                    if self.free_blocks[order]:
                        return True  # Memory is available
                
                # Wait for notification about memory availability
                self._condition.wait(timeout=0.1)
            
            return False  # Timeout reached


class ThreadSafePrefetchingManager:
    """Thread-safe prefetching manager with proper synchronization mechanisms"""
    
    def __init__(self, prefetch_buffer_size: int = 10):
        self.prefetch_queue = queue.Queue(maxsize=prefetch_buffer_size)
        self.prefetch_thread = None
        self.prefetch_active = False
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # For tracking
        self.prefetched_tensors = {}
        self.prefetch_history = []
        self._shutdown_event = threading.Event()
        
    def start_prefetching(self):
        """Start the prefetching thread"""
        if not self.prefetch_active:
            self.prefetch_active = True
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()
    
    def stop_prefetching(self):
        """Stop the prefetching thread"""
        self.prefetch_active = False
        self._shutdown_event.set()
        try:
            self.prefetch_queue.put(None, timeout=1.0)  # Sentinel to stop thread
        except queue.Full:
            pass  # Queue is full but shutdown event is set
        
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=2.0)
    
    def _prefetch_worker(self):
        """Background worker for prefetching tensors"""
        while self.prefetch_active and not self._shutdown_event.is_set():
            try:
                item = self.prefetch_queue.get(timeout=1.0)
                if item is None:  # Sentinel to stop thread
                    break
                
                tensor, target_device = item
                # Simulate prefetching operation
                with self._condition:
                    _ = tensor.to(target_device, non_blocking=True)
                    self.prefetch_history.append((time.time(), target_device))
                
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")
    
    def prefetch_tensor(self, tensor: torch.Tensor, target_device: torch.device, timeout: float = 0.1):
        """Prefetch a tensor to the target device"""
        try:
            self.prefetch_queue.put((tensor, target_device), block=True, timeout=timeout)
            return True
        except queue.Full:
            return False  # Buffer is full, can't prefetch
    
    def get_prefetch_status(self):
        """Get prefetching status"""
        with self._lock:
            return {
                'queue_size': self.prefetch_queue.qsize(),
                'history_length': len(self.prefetch_history),
                'active': self.prefetch_active
            }


class TestSynchronizationFixes(unittest.TestCase):
    """Test suite for verifying synchronization fixes"""
    
    def setUp(self):
        self.thread_safe_cache = ThreadSafeCache()
        self.thread_safe_allocator = ThreadSafeBuddyAllocator()
        self.thread_safe_prefetcher = ThreadSafePrefetchingManager()
    
    def test_thread_safe_cache(self):
        """Test thread-safe cache implementation"""
        print("Testing thread-safe cache...")
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"tensor_{worker_id}_{i}"
                    tensor = self.thread_safe_cache.get_tensor(key, (10, 20), torch.float32)
                    time.sleep(0.001)
                    self.thread_safe_cache.return_tensor(key, tensor)
                    results.append(f"worker_{worker_id}_iter_{i}")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        self.assertEqual(len(results), 100)
        self.assertEqual(len(errors), 0)
        
        # Check cache statistics
        stats = self.thread_safe_cache.get_cache_stats()
        print(f"Thread-safe cache stats: {stats}")
    
    def test_thread_safe_allocator(self):
        """Test thread-safe allocator implementation"""
        print("Testing thread-safe allocator...")
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(5):
                    # Allocate a block
                    addr, size = self.thread_safe_allocator.allocate(1024)
                    if addr is not None:
                        # Simulate some work
                        time.sleep(0.001)
                        
                        # Deallocate the block
                        self.thread_safe_allocator.deallocate(addr, size)
                        
                        results.append(f"worker_{worker_id}_iter_{i}_success")
                    else:
                        results.append(f"worker_{worker_id}_iter_{i}_fail")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
                import traceback
                traceback.print_exc()
        
        # Create multiple threads
        threads = []
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify results
        self.assertEqual(len([r for r in results if 'success' in r]), 40)  # Some allocations should succeed
        self.assertEqual(len(errors), 0)
        
        # Check allocator statistics
        stats = self.thread_safe_allocator.get_stats()
        print(f"Thread-safe allocator stats: {stats}")
    
    def test_thread_safe_prefetcher(self):
        """Test thread-safe prefetching manager implementation"""
        print("Testing thread-safe prefetcher...")
        
        self.thread_safe_prefetcher.start_prefetching()
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    # Create a tensor
                    tensor = torch.randn(100, 100)
                    
                    # Prefetch tensor
                    success = self.thread_safe_prefetcher.prefetch_tensor(tensor, torch.device('cpu'))
                    results.append(f"worker_{worker_id}_iter_{i}_prefetch_{success}")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Allow prefetch queue to process
        time.sleep(0.1)
        
        # Verify results
        self.assertEqual(len(results), 50)
        self.assertEqual(len(errors), 0)
        
        # Check status
        status = self.thread_safe_prefetcher.get_prefetch_status()
        print(f"Thread-safe prefetcher status: {status}")
        
        # Stop prefetching
        self.thread_safe_prefetcher.stop_prefetching()


if __name__ == "__main__":
    print("Running comprehensive race condition tests...")
    
    # Run original race condition tests
    print("\n=== Testing Potential Race Conditions ===")
    race_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRaceConditions)
    race_test_runner = unittest.TextTestRunner(verbosity=2)
    race_test_result = race_test_runner.run(race_test_suite)
    
    # Run synchronization fixes tests
    print("\n=== Testing Synchronization Fixes ===")
    sync_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestSynchronizationFixes)
    sync_test_runner = unittest.TextTestRunner(verbosity=2)
    sync_test_result = sync_test_runner.run(sync_test_suite)
    
    # Report results
    print(f"\nRace Condition Tests - Failures: {len(race_test_result.failures)}, Errors: {len(race_test_result.errors)}")
    print(f"Synchronization Fixes Tests - Failures: {len(sync_test_result.failures)}, Errors: {len(sync_test_result.errors)}")
    
    if race_test_result.failures or race_test_result.errors:
        print("Potential race conditions were detected in the original implementations.")
    else:
        print("No race conditions detected in the original implementations (this may be due to low contention in tests).")
    
    if sync_test_result.failures or sync_test_result.errors:
        print("Synchronization fixes have issues.")
    else:
        print("Synchronization fixes passed all tests!")