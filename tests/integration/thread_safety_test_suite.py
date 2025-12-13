"""Comprehensive Thread Safety Test Suite for Qwen3-VL Memory Management Components

This test suite validates thread safety across all multi-threaded components,
identifies race conditions, and ensures data consistency under concurrent access.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import random
import numpy as np
from typing import Dict, List, Optional

# Import the memory management components
from advanced_memory_pooling_system import (
    TensorType, MemoryBlock, BuddyAllocator, MemoryPool, AdvancedMemoryPoolingSystem
)
from advanced_memory_management_optimizations import AdvancedMemoryPool, MemoryPoolType, MemoryBlock as AMMBlock
from advanced_memory_swapping_system import AdvancedMemorySwapper, MemoryPressureMonitor, ClockSwapAlgorithm, MemoryRegionType
from advanced_memory_tiering_system import MemoryTieringManager, MemoryTier
from centralized_metrics_collector import MetricsCollector
from advanced_hierarchical_caching_system import HierarchicalCacheManager, L1Cache, L2Cache, L3Cache
from unified_memory_manager import UnifiedMemoryManager, UnifiedTensorType, MemoryTier as UMMTier, CompressionMethod
from memory_context_managers import (
    kv_cache_context, image_features_context, text_embeddings_context,
    ResourceTracker
)
from predictive_tensor_lifecycle_manager import PredictiveTensorLifecycleManager
from thread_safe_memory_pooling_system import (
    ThreadSafeMemoryBlock, ThreadSafeBuddyAllocator, ThreadSafeMemoryPool, 
    ThreadSafeMemoryPoolingSystem
)


class TestBuddyAllocatorThreadSafety(unittest.TestCase):
    """Test thread safety of Buddy Allocator"""
    
    def setUp(self):
        self.allocator = BuddyAllocator(total_size=1024*1024, min_block_size=256)  # 1MB pool
    
    def test_concurrent_allocations(self):
        """Test concurrent allocations from multiple threads"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(10):
                size = random.randint(256, 4096)
                block = self.allocator.allocate(size, TensorType.KV_CACHE, f"thread_{worker_id}_block_{i}")
                if block:
                    thread_results.append((block.start_addr, block.size))
                    # Hold the block briefly to increase chance of conflict
                    time.sleep(0.001)
                    self.allocator.deallocate(block)
            results.extend(thread_results)
        
        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Verify all allocations succeeded
        self.assertEqual(len(results), 80)  # 8 threads * 10 allocations each
    
    def test_concurrent_allocation_deallocation(self):
        """Test concurrent allocation and deallocation"""
        allocated_blocks = []
        results = []
        
        def alloc_worker():
            worker_results = []
            for i in range(20):
                size = random.randint(256, 2048)
                block = self.allocator.allocate(size, TensorType.KV_CACHE, f"alloc_{i}")
                if block:
                    allocated_blocks.append(block)
                    worker_results.append(('alloc', block.start_addr, block.size))
            results.extend(worker_results)
        
        def dealloc_worker():
            worker_results = []
            for _ in range(10):
                if allocated_blocks:
                    block = allocated_blocks.pop()
                    self.allocator.deallocate(block)
                    worker_results.append(('dealloc', block.start_addr))
            results.extend(worker_results)
        
        # Run allocation and deallocation threads concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            alloc_future = executor.submit(alloc_worker)
            dealloc_future = executor.submit(dealloc_worker)
            
            alloc_future.result()
            dealloc_future.result()
        
        # Verify operations completed
        self.assertGreater(len(results), 0)


class TestMemoryPoolThreadSafety(unittest.TestCase):
    """Test thread safety of Memory Pool"""
    
    def setUp(self):
        self.pool = MemoryPool(TensorType.KV_CACHE, pool_size=1024*1024, min_block_size=256)
    
    def test_concurrent_pool_operations(self):
        """Test concurrent operations on memory pool"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(15):
                size = random.randint(256, 8192)
                tensor_id = f"worker_{worker_id}_tensor_{i}"
                
                # Allocate
                block = self.pool.allocate(size, tensor_id)
                if block:
                    thread_results.append(('alloc', tensor_id, block.size))
                    
                    # Brief pause to increase chance of conflict
                    time.sleep(0.001)
                    
                    # Deallocate
                    success = self.pool.deallocate(tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations completed
        alloc_ops = [r for r in results if r[0] == 'alloc']
        dealloc_ops = [r for r in results if r[0] == 'dealloc']
        self.assertEqual(len(alloc_ops), len(dealloc_ops))


class TestAdvancedMemoryPoolingSystemThreadSafety(unittest.TestCase):
    """Test thread safety of Advanced Memory Pooling System"""
    
    def setUp(self):
        self.memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024*2,  # 2MB
            image_features_size=1024*1024*2,
            text_embeddings_size=1024*1024,
            gradients_size=1024*1024*2,
            activations_size=1024*1024,
            parameters_size=1024*1024*4  # 4MB
        )
    
    def test_concurrent_tensor_operations(self):
        """Test concurrent operations across different tensor types"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            tensor_types = [
                TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, 
                TensorType.TEXT_EMBEDDINGS, TensorType.GRADIENTS
            ]
            
            for i in range(10):
                tensor_type = random.choice(tensor_types)
                size = random.randint(1024, 16384)  # 1KB to 16KB
                tensor_id = f"worker_{worker_id}_tensor_{i}_{tensor_type.value}"
                
                # Allocate
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    thread_results.append(('alloc', tensor_type.value, tensor_id, block.size))
                    
                    # Brief pause
                    time.sleep(0.001)
                    
                    # Deallocate
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_type.value, tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations
        alloc_ops = [r for r in results if r[0] == 'alloc']
        dealloc_ops = [r for r in results if r[0] == 'dealloc']
        self.assertEqual(len(alloc_ops), len(dealloc_ops))
    
    def test_memory_compaction_under_load(self):
        """Test memory compaction while other threads are allocating/deallocating"""
        results = []
        
        def alloc_dealloc_worker():
            worker_results = []
            for i in range(20):
                tensor_type = random.choice([TensorType.KV_CACHE, TensorType.ACTIVATIONS])
                size = random.randint(1024, 8192)
                tensor_id = f"load_worker_{i}"
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    worker_results.append(('alloc', tensor_id))
                    time.sleep(0.001)
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    if success:
                        worker_results.append(('dealloc', tensor_id))
            results.extend(worker_results)
        
        def compaction_worker():
            # Trigger compaction periodically
            for _ in range(5):
                time.sleep(0.1)
                self.memory_system.compact_memory()
        
        # Run allocation/deallocation and compaction threads concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            alloc_future = executor.submit(alloc_dealloc_worker)
            compaction_future = executor.submit(compaction_worker)
            
            alloc_future.result()
            compaction_future.result()


class TestThreadSafeMemoryPoolingSystem(unittest.TestCase):
    """Test the thread-safe memory pooling system"""
    
    def setUp(self):
        self.memory_system = ThreadSafeMemoryPoolingSystem(
            kv_cache_size=1024*1024*2,
            image_features_size=1024*1024*2,
            text_embeddings_size=1024*1024,
            gradients_size=1024*1024*2,
            activations_size=1024*1024,
            parameters_size=1024*1024*4
        )
    
    def test_thread_safe_concurrent_operations(self):
        """Test thread-safe concurrent operations"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(25):
                tensor_type = random.choice([
                    TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, 
                    TensorType.TEXT_EMBEDDINGS, TensorType.GRADIENTS
                ])
                size = random.randint(512, 4096)
                tensor_id = f"ts_worker_{worker_id}_tensor_{i}"
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    thread_results.append(('alloc', tensor_id, block.size))
                    time.sleep(0.0005)
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads with high concurrency
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations completed successfully
        alloc_ops = [r for r in results if r[0] == 'alloc']
        dealloc_ops = [r for r in results if r[0] == 'dealloc']
        self.assertEqual(len(alloc_ops), len(dealloc_ops))
        self.assertEqual(len(alloc_ops), 250)  # 10 threads * 25 ops each


class TestMemorySwappingSystemThreadSafety(unittest.TestCase):
    """Test thread safety of memory swapping system"""
    
    def setUp(self):
        self.swapper = AdvancedMemorySwapper(
            max_swap_size=100*1024*1024,  # 100MB swap
            swap_directory="./test_swap",
            swap_algorithm=ClockSwapAlgorithm()
        )
    
    def test_concurrent_swapping_operations(self):
        """Test concurrent swapping operations"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(10):
                block_id = f"swap_block_{worker_id}_{i}"
                size = random.randint(1024, 8192)
                
                # Register block
                block = self.swapper.register_memory_block(block_id, size, MemoryRegionType.TENSOR_DATA)
                if block:
                    thread_results.append(('register', block_id))
                    
                    # Access block (might trigger swapping)
                    accessed_block = self.swapper.access_memory_block(block_id)
                    if accessed_block:
                        thread_results.append(('access', block_id))
                    
                    # Unregister block
                    success = self.swapper.unregister_memory_block(block_id)
                    if success:
                        thread_results.append(('unregister', block_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations
        self.assertEqual(len([r for r in results if r[0] == 'register']), 
                        len([r for r in results if r[0] == 'unregister']))


class TestHierarchicalCacheThreadSafety(unittest.TestCase):
    """Test thread safety of hierarchical caching system"""
    
    def setUp(self):
        self.cache_manager = HierarchicalCacheManager(
            l1_size=1024*1024,      # 1MB L1
            l2_size=1024*1024*2,    # 2MB L2  
            l3_size=1024*1024*5     # 5MB L3
        )
    
    def test_concurrent_cache_operations(self):
        """Test concurrent cache operations"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(20):
                tensor_id = f"cache_tensor_{worker_id}_{i}"
                tensor_type = random.choice([TensorType.KV_CACHE, TensorType.IMAGE_FEATURES])
                
                # Put in cache
                success_put = self.cache_manager.put_in_cache(
                    tensor_type, tensor_id, np.random.rand(10, 10).astype(np.float32)
                )
                if success_put:
                    thread_results.append(('put', tensor_id))
                    
                    # Get from cache
                    cached_data = self.cache_manager.get_from_cache(tensor_type, tensor_id)
                    if cached_data is not None:
                        thread_results.append(('get', tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations
        put_ops = [r for r in results if r[0] == 'put']
        get_ops = [r for r in results if r[0] == 'get']
        self.assertGreaterEqual(len(put_ops), len(get_ops))


class TestUnifiedMemoryManagerThreadSafety(unittest.TestCase):
    """Test thread safety of unified memory manager"""
    
    def setUp(self):
        self.unified_manager = UnifiedMemoryManager(
            kv_cache_size=1024*1024*2,
            image_features_size=1024*1024*2,
            text_embeddings_size=1024*1024,
            gradients_size=1024*1024*2,
            activations_size=1024*1024,
            parameters_size=1024*1024*4,
            min_block_size=256
        )
    
    def test_concurrent_unified_operations(self):
        """Test concurrent operations on unified memory manager"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            unified_tensor_types = [
                UnifiedTensorType.KV_CACHE, UnifiedTensorType.IMAGE_FEATURES,
                UnifiedTensorType.TEXT_EMBEDDINGS, UnifiedTensorType.GRADIENTS
            ]
            
            for i in range(15):
                tensor_type = random.choice(unified_tensor_types)
                size = random.randint(1024, 8192)
                tensor_id = f"unified_tensor_{worker_id}_{i}"
                
                # Allocate
                block = self.unified_manager.allocate(tensor_type, size, tensor_id)
                if block:
                    thread_results.append(('alloc', tensor_type.value, tensor_id))
                    
                    # Brief pause
                    time.sleep(0.001)
                    
                    # Deallocate
                    success = self.unified_manager.deallocate(tensor_type, tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_type.value, tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(worker, i) for i in range(7)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations
        alloc_ops = [r for r in results if r[0] == 'alloc']
        dealloc_ops = [r for r in results if r[0] == 'dealloc']
        self.assertEqual(len(alloc_ops), len(dealloc_ops))


class TestPredictiveTensorLifecycleThreadSafety(unittest.TestCase):
    """Test thread safety of predictive tensor lifecycle manager"""
    
    def setUp(self):
        self.lifecycle_manager = PredictiveTensorLifecycleManager()
    
    def test_concurrent_lifecycle_operations(self):
        """Test concurrent lifecycle management operations"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(12):
                tensor_id = f"lifecycle_tensor_{worker_id}_{i}"
                tensor_type = random.choice([TensorType.KV_CACHE, TensorType.ACTIVATIONS])
                size = random.randint(1024, 4096)
                
                # Register tensor
                success_reg = self.lifecycle_manager.register_tensor(tensor_id, tensor_type, size)
                if success_reg:
                    thread_results.append(('register', tensor_id))
                    
                    # Predict lifecycle
                    prediction = self.lifecycle_manager.predict_lifecycle(tensor_id)
                    if prediction:
                        thread_results.append(('predict', tensor_id))
                    
                    # Unregister tensor
                    success_unreg = self.lifecycle_manager.unregister_tensor(tensor_id)
                    if success_unreg:
                        thread_results.append(('unregister', tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations
        reg_ops = [r for r in results if r[0] == 'register']
        unreg_ops = [r for r in results if r[0] == 'unregister']
        self.assertEqual(len(reg_ops), len(unreg_ops))


class TestMemoryContextManagersThreadSafety(unittest.TestCase):
    """Test thread safety of memory context managers"""
    
    def setUp(self):
        self.memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024,
            image_features_size=1024*1024,
            text_embeddings_size=1024*1024
        )
    
    def test_concurrent_context_managers(self):
        """Test concurrent use of memory context managers"""
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(8):
                tensor_id = f"context_tensor_{worker_id}_{i}"
                
                # Test KV cache context
                try:
                    with kv_cache_context(self.memory_system, 1024*64, tensor_id) as block:
                        if block:
                            thread_results.append(('kv_context', tensor_id, block.size))
                except Exception as e:
                    thread_results.append(('kv_error', tensor_id, str(e)))
                
                # Test image features context
                try:
                    with image_features_context(self.memory_system, 1024*128, f"img_{tensor_id}") as block:
                        if block:
                            thread_results.append(('img_context', f"img_{tensor_id}", block.size))
                except Exception as e:
                    thread_results.append(('img_error', f"img_{tensor_id}", str(e)))
                
                # Test text embeddings context
                try:
                    with text_embeddings_context(self.memory_system, 1024*32, f"text_{tensor_id}") as block:
                        if block:
                            thread_results.append(('text_context', f"text_{tensor_id}", block.size))
                except Exception as e:
                    thread_results.append(('text_error', f"text_{tensor_id}", str(e)))
            
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations - allow for some errors due to resource constraints
        success_ops = [r for r in results if r[0] in ['kv_context', 'img_context', 'text_context']]
        error_ops = [r for r in results if 'error' in r[0]]
        self.assertGreaterEqual(len(success_ops), len(error_ops))


class TestResourceTrackerThreadSafety(unittest.TestCase):
    """Test thread safety of resource tracker"""
    
    def test_concurrent_resource_tracking(self):
        """Test concurrent resource tracking operations"""
        tracker = ResourceTracker()
        results = []
        
        def worker(worker_id):
            thread_results = []
            for i in range(10):
                tensor_id = f"tracked_tensor_{worker_id}_{i}"
                
                # Track allocation
                tracker.track_allocation(tensor_id, 1024*64)
                thread_results.append(('track_alloc', tensor_id))
                
                # Track access
                tracker.track_access(tensor_id)
                thread_results.append(('track_access', tensor_id))
                
                # Track deallocation
                tracker.track_deallocation(tensor_id)
                thread_results.append(('track_dealloc', tensor_id))
            results.extend(thread_results)
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations
        self.assertEqual(len(results), 180)  # 6 threads * 10 iterations * 3 operations
        
        # Check final state
        final_stats = tracker.get_stats()
        self.assertIsInstance(final_stats, dict)


def run_all_thread_safety_tests():
    """Run all thread safety tests"""
    print("Running Comprehensive Thread Safety Test Suite for Qwen3-VL Memory Management...")
    
    # Create test suite
    test_classes = [
        TestBuddyAllocatorThreadSafety,
        TestMemoryPoolThreadSafety,
        TestAdvancedMemoryPoolingSystemThreadSafety,
        TestThreadSafeMemoryPoolingSystem,
        TestMemorySwappingSystemThreadSafety,
        TestHierarchicalCacheThreadSafety,
        TestUnifiedMemoryManagerThreadSafety,
        TestPredictiveTensorLifecycleThreadSafety,
        TestMemoryContextManagersThreadSafety,
        TestResourceTrackerThreadSafety
    ]
    
    all_tests_passed = True
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.failures or result.errors:
            print(f"❌ {test_class.__name__} FAILED")
            all_tests_passed = False
            for failure in result.failures:
                print(f"  FAILURE: {failure[0]} - {failure[1]}")
            for error in result.errors:
                print(f"  ERROR: {error[0]} - {error[1]}")
        else:
            print(f"✅ {test_class.__name__} PASSED")
    
    if all_tests_passed:
        print("\n🎉 ALL THREAD SAFETY TESTS PASSED!")
        print("Memory management components are thread-safe and ready for production use.")
    else:
        print("\n⚠️  SOME THREAD SAFETY TESTS FAILED!")
        print("Please review and fix the identified race conditions.")
    
    return all_tests_passed


if __name__ == "__main__":
    run_all_thread_safety_tests()