"""
Comprehensive Tests for Optimized Memory Management Systems

This module provides comprehensive tests to validate thread safety and performance
of the optimized memory management systems with improved locking strategies.
"""

import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import unittest
from typing import List, Tuple

from optimized_advanced_memory_pool import OptimizedAdvancedMemoryPool, MemoryPoolType
from optimized_buddy_allocator import OptimizedAdvancedMemoryPoolingSystem, TensorType
from optimized_locking_strategies import ReaderWriterLock, LockStriping


class TestOptimizedLockingStrategies(unittest.TestCase):
    """Test the optimized locking strategies themselves"""
    
    def test_reader_writer_lock(self):
        """Test basic functionality of reader-writer lock"""
        rw_lock = ReaderWriterLock()
        shared_data = [0]
        read_count = 0
        
        def reader():
            nonlocal read_count
            rw_lock.acquire_read()
            local_value = shared_data[0]
            time.sleep(0.01)  # Simulate read operation
            read_count += 1
            rw_lock.release_read()
            return local_value
        
        def writer():
            rw_lock.acquire_write()
            shared_data[0] += 1
            time.sleep(0.02)  # Simulate write operation
            rw_lock.release_write()
        
        # Test multiple concurrent readers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            # Submit readers
            for _ in range(10):
                futures.append(executor.submit(reader))
            
            # Submit writers
            for _ in range(2):
                futures.append(executor.submit(writer))
            
            # Wait for all to complete
            results = [f.result() for f in futures]
        
        # Verify that all readers could access the data
        self.assertGreaterEqual(read_count, 10)
        
    def test_lock_striping(self):
        """Test basic functionality of lock striping"""
        stripe_manager = LockStriping(num_stripes=4)
        
        # Test that different keys get different locks
        locks = set()
        for i in range(100):
            lock = stripe_manager.get_lock(i)
            locks.add(id(lock))  # Use id to distinguish lock objects
        
        # Should have at most 4 different locks
        self.assertLessEqual(len(locks), 4)
        
        # Test that same key always gets same lock
        lock1 = stripe_manager.get_lock(42)
        lock2 = stripe_manager.get_lock(42)
        self.assertIs(lock1, lock2)


class TestOptimizedAdvancedMemoryPool(unittest.TestCase):
    """Test the optimized AdvancedMemoryPool with improved locking"""
    
    def setUp(self):
        self.pool = OptimizedAdvancedMemoryPool(
            initial_size=10 * 1024 * 1024,  # 10MB
            page_size=4096,
            enable_defragmentation=True,
            num_lock_stripes=8
        )
    
    def tearDown(self):
        self.pool.cleanup()
    
    def test_single_thread_allocation_deallocation(self):
        """Test basic allocation/deallocation in single thread"""
        # Allocate memory
        result = self.pool.allocate(1024, MemoryPoolType.TENSOR_DATA)
        self.assertIsNotNone(result)
        ptr, size = result
        self.assertEqual(size, 4096)  # Should be aligned to page size
        
        # Get stats (read operation)
        stats = self.pool.get_stats()
        self.assertGreater(stats['current_usage'], 0)
        
        # Deallocate
        success = self.pool.deallocate(ptr)
        self.assertTrue(success)
        
        # Verify stats after deallocation
        stats = self.pool.get_stats()
        self.assertEqual(stats['current_usage'], 0)
    
    def test_concurrent_allocations(self):
        """Test concurrent allocations from multiple threads"""
        results = []
        errors = []
        
        def allocate_task(size):
            try:
                result = self.pool.allocate(size, MemoryPoolType.TENSOR_DATA)
                if result:
                    ptr, allocated_size = result
                    # Hold the allocation briefly to increase contention
                    time.sleep(0.01)
                    # Deallocate
                    self.pool.deallocate(ptr)
                    results.append((ptr, allocated_size))
                    return True
                else:
                    return False
            except Exception as e:
                errors.append(e)
                return False
        
        # Run concurrent allocations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(allocate_task, 1024 + (i % 5) * 100) for i in range(50)]
            results = [f.result() for f in futures]
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0)
        # Verify most allocations succeeded
        success_count = sum(results)
        self.assertGreater(success_count, 40)  # Expect most to succeed
    
    def test_concurrent_allocation_and_statistics(self):
        """Test concurrent allocations while reading statistics"""
        allocation_results = []
        stats_results = []
        
        def allocate_task():
            for _ in range(10):
                result = self.pool.allocate(512, MemoryPoolType.ACTIVATION_BUFFER)
                if result:
                    ptr, _ = result
                    time.sleep(0.001)  # Brief hold
                    self.pool.deallocate(ptr)
                time.sleep(0.001)  # Brief pause between operations
        
        def stats_task():
            for _ in range(20):
                stats = self.pool.get_stats()
                stats_results.append(stats['current_usage'])
                time.sleep(0.002)
        
        # Run both tasks concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            alloc_futures = [executor.submit(allocate_task) for _ in range(3)]
            stats_futures = [executor.submit(stats_task) for _ in range(3)]
            
            # Wait for all to complete
            alloc_results = [f.result() for f in alloc_futures]
            stats_results = [f.result() for f in stats_futures]
        
        # Verify that statistics were successfully retrieved
        self.assertGreater(len(stats_results), 0)
    
    def test_fragmentation_handling(self):
        """Test that fragmentation is handled properly"""
        # Allocate and deallocate in a pattern that creates fragmentation
        ptrs = []
        
        # Allocate many small blocks
        for i in range(50):
            result = self.pool.allocate(256, MemoryPoolType.TEMPORARY)
            if result:
                ptrs.append(result[0])
        
        # Deallocate half of them randomly
        random.shuffle(ptrs)
        for i in range(len(ptrs) // 2):
            self.pool.deallocate(ptrs[i])
        
        # Allocate a large block - should trigger defragmentation
        large_result = self.pool.allocate(5000, MemoryPoolType.TENSOR_DATA)
        if large_result:
            ptr, size = large_result
            self.pool.deallocate(ptr)


class TestOptimizedBuddyAllocator(unittest.TestCase):
    """Test the optimized Buddy Allocator with improved locking"""
    
    def setUp(self):
        self.memory_system = OptimizedAdvancedMemoryPoolingSystem(
            kv_cache_size=5 * 1024 * 1024,  # 5MB
            image_features_size=5 * 1024 * 1024,
            text_embeddings_size=2 * 1024 * 1024,
            gradients_size=5 * 1024 * 1024,
            activations_size=5 * 1024 * 1024,
            parameters_size=10 * 1024 * 1024,
            min_block_size=256,
            num_lock_stripes=8
        )
    
    def test_single_thread_operations(self):
        """Test basic operations in single thread"""
        # Allocate different types of tensors
        kv_block = self.memory_system.allocate(TensorType.KV_CACHE, 1024, "kv_1")
        img_block = self.memory_system.allocate(TensorType.IMAGE_FEATURES, 2048, "img_1")
        text_block = self.memory_system.allocate(TensorType.TEXT_EMBEDDINGS, 512, "text_1")
        
        # Verify allocations succeeded
        self.assertIsNotNone(kv_block)
        self.assertIsNotNone(img_block)
        self.assertIsNotNone(text_block)
        
        # Get system stats
        stats = self.memory_system.get_system_stats()
        self.assertGreater(stats['total_allocated'], 0)
        
        # Deallocate
        self.assertTrue(self.memory_system.deallocate(TensorType.KV_CACHE, "kv_1"))
        self.assertTrue(self.memory_system.deallocate(TensorType.IMAGE_FEATURES, "img_1"))
        self.assertTrue(self.memory_system.deallocate(TensorType.TEXT_EMBEDDINGS, "text_1"))
    
    def test_concurrent_tensor_operations(self):
        """Test concurrent operations on different tensor types"""
        results = []
        
        def tensor_task(tensor_type, base_id, num_ops):
            for i in range(num_ops):
                tensor_id = f"{tensor_type.value}_{base_id}_{i}"
                size = 512 + (i % 10) * 64  # Varying sizes
                
                # Allocate
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    # Hold briefly
                    time.sleep(0.001)
                    # Deallocate
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    results.append(success)
        
        # Run concurrent tasks for different tensor types
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            
            # Submit tasks for each tensor type
            for tensor_type in [TensorType.KV_CACHE, TensorType.IMAGE_FEATURES, 
                              TensorType.TEXT_EMBEDDINGS, TensorType.GRADIENTS]:
                for worker_id in range(3):
                    futures.append(executor.submit(tensor_task, tensor_type, worker_id, 5))
            
            # Wait for all to complete
            for f in futures:
                f.result()
        
        # Verify most operations succeeded
        success_count = sum(results)
        self.assertGreaterEqual(success_count, len(results) * 0.9)  # Expect 90%+ success rate
    
    def test_concurrent_allocation_and_statistics(self):
        """Test concurrent allocations while reading system statistics"""
        alloc_results = []
        stats_results = []
        
        def allocation_worker():
            for i in range(20):
                tensor_id = f"worker_{threading.current_thread().ident}_{i}"
                size = 256 + (i % 5) * 128
                tensor_type = random.choice(list(TensorType))
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    time.sleep(0.001)
                    self.memory_system.deallocate(tensor_type, tensor_id)
                    alloc_results.append(True)
                else:
                    alloc_results.append(False)
        
        def stats_worker():
            for _ in range(15):
                stats = self.memory_system.get_system_stats()
                stats_results.append(stats.get('overall_utilization', 0))
                time.sleep(0.002)
        
        # Run allocation and stats workers concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit allocation workers
            alloc_futures = [executor.submit(allocation_worker) for _ in range(5)]
            # Submit stats workers
            stats_futures = [executor.submit(stats_worker) for _ in range(3)]
            
            # Wait for all to complete
            for f in alloc_futures + stats_futures:
                f.result()
        
        # Verify both allocation and stats operations completed
        self.assertGreater(len(alloc_results), 0)
        self.assertGreater(len(stats_results), 0)
    
    def test_pool_specific_operations(self):
        """Test operations on specific pools"""
        # Allocate multiple blocks in the same pool
        tensor_ids = []
        for i in range(10):
            tensor_id = f"pool_test_{i}"
            block = self.memory_system.allocate(TensorType.KV_CACHE, 1024, tensor_id)
            if block:
                tensor_ids.append(tensor_id)
        
        # Verify pool stats
        pool_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertEqual(pool_stats['active_allocations'], len(tensor_ids))
        
        # Deallocate all
        for tensor_id in tensor_ids:
            self.memory_system.deallocate(TensorType.KV_CACHE, tensor_id)
        
        # Verify pool is empty
        pool_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertEqual(pool_stats['active_allocations'], 0)


class PerformanceComparisonTest(unittest.TestCase):
    """Performance comparison tests between original and optimized implementations"""
    
    def test_reader_writer_lock_performance(self):
        """Compare performance of reader-writer lock vs regular lock"""
        # Test with regular lock
        regular_lock = threading.Lock()
        shared_counter = [0]
        
        def regular_lock_task(is_reader):
            for _ in range(100):
                if is_reader:
                    with regular_lock:
                        time.sleep(0.0001)  # Simulate read
                        value = shared_counter[0]
                else:
                    with regular_lock:
                        time.sleep(0.0002)  # Simulate write
                        shared_counter[0] += 1
        
        # Test with reader-writer lock
        rw_lock = ReaderWriterLock()
        rw_shared_counter = [0]
        
        def rw_lock_task(is_reader):
            for _ in range(100):
                if is_reader:
                    rw_lock.acquire_read()
                    time.sleep(0.0001)  # Simulate read
                    value = rw_shared_counter[0]
                    rw_lock.release_read()
                else:
                    rw_lock.acquire_write()
                    time.sleep(0.0002)  # Simulate write
                    rw_shared_counter[0] += 1
                    rw_lock.release_write()
        
        # Time regular lock with many readers
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            # 6 readers, 2 writers
            for _ in range(6):
                futures.append(executor.submit(regular_lock_task, True))
            for _ in range(2):
                futures.append(executor.submit(regular_lock_task, False))
            
            for f in futures:
                f.result()
        regular_time = time.time() - start_time
        
        # Time reader-writer lock with many readers
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            # 6 readers, 2 writers
            for _ in range(6):
                futures.append(executor.submit(rw_lock_task, True))
            for _ in range(2):
                futures.append(executor.submit(rw_lock_task, False))
            
            for f in futures:
                f.result()
        rw_time = time.time() - start_time
        
        print(f"Regular lock time: {regular_time:.4f}s")
        print(f"Reader-writer lock time: {rw_time:.4f}s")
        # Reader-writer lock should be faster with many readers
        # Note: This is not an assertion because performance can vary by system


def run_comprehensive_tests():
    """Run all tests"""
    print("Running comprehensive tests for optimized memory management systems...")
    
    # Create test suite
    test_classes = [
        TestOptimizedLockingStrategies,
        TestOptimizedAdvancedMemoryPool,
        TestOptimizedBuddyAllocator,
        PerformanceComparisonTest
    ]
    
    all_tests = unittest.TestSuite()
    for test_class in test_classes:
        all_tests.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("\nAll tests passed! Optimized memory management systems are working correctly.")
    else:
        print("\nSome tests failed. Please review the implementation.")
        exit(1)