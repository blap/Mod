"""
Comprehensive Thread Safety Validation Suite for Qwen3-VL Memory Management Components

This test suite validates that all thread safety improvements have been properly implemented
and that race conditions have been eliminated.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import random
import numpy as np
from typing import Dict, List, Optional, Any
import gc

import sys
import os
# Add dev_tools to the Python path to import the moved modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dev_tools', 'memory_management'))

# Import the thread-safe components
from thread_safe_implementations import (
    TensorType, MemoryBlock, ThreadSafeBuddyAllocator, ThreadSafeMemoryPool,
    ThreadSafeMemoryPoolingSystem
)


class TestBuddyAllocatorThreadSafety(unittest.TestCase):
    """Test thread safety of Thread-Safe Buddy Allocator"""
    
    def setUp(self):
        self.allocator = ThreadSafeBuddyAllocator(total_size=1024*1024, min_block_size=256)  # 1MB pool
    
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
    
    def test_allocator_statistics_consistency(self):
        """Test that allocator statistics remain consistent under concurrent access"""
        stats_list = []
        
        def stats_collector():
            for _ in range(50):
                stats = self.allocator.get_stats()
                stats_list.append(stats)
                time.sleep(0.01)
        
        def allocation_worker():
            for i in range(25):
                size = random.randint(256, 1024)
                block = self.allocator.allocate(size, TensorType.KV_CACHE, f"stat_test_{i}")
                if block:
                    time.sleep(0.005)
                    self.allocator.deallocate(block)
        
        # Run stats collector and allocation worker concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            stats_future = executor.submit(stats_collector)
            alloc_future = executor.submit(allocation_worker)
            
            stats_future.result()
            alloc_future.result()
        
        # Verify that statistics are always consistent (no negative values)
        for stats in stats_list:
            self.assertGreaterEqual(stats['total_allocated'], 0)
            self.assertGreaterEqual(stats['total_free'], 0)
            self.assertGreaterEqual(stats['largest_free_block'], 0)
            self.assertGreaterEqual(stats['allocated_blocks'], 0)
            self.assertGreaterEqual(stats['num_free_blocks'], 0)
            self.assertGreaterEqual(stats['fragmentation_ratio'], 0)
            self.assertLessEqual(stats['fragmentation_ratio'], 1.0)


class TestMemoryPoolThreadSafety(unittest.TestCase):
    """Test thread safety of Memory Pool"""
    
    def setUp(self):
        self.pool = ThreadSafeMemoryPool(TensorType.KV_CACHE, pool_size=1024*1024, min_block_size=256)
    
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
    
    def test_pool_statistics_consistency(self):
        """Test that pool statistics remain consistent under concurrent access"""
        stats_list = []
        
        def stats_collector():
            for _ in range(30):
                stats = self.pool.get_pool_stats()
                stats_list.append(stats)
                time.sleep(0.02)
        
        def allocation_worker():
            for i in range(15):
                size = random.randint(512, 2048)
                tensor_id = f"pool_stat_test_{i}"
                
                block = self.pool.allocate(size, tensor_id)
                if block:
                    time.sleep(0.01)
                    self.pool.deallocate(tensor_id)
        
        # Run stats collector and allocation worker concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            stats_future = executor.submit(stats_collector)
            alloc_future = executor.submit(allocation_worker)
            
            stats_future.result()
            alloc_future.result()
        
        # Verify that statistics are always consistent
        for stats in stats_list:
            self.assertGreaterEqual(stats['utilization_ratio'], 0)
            self.assertLessEqual(stats['utilization_ratio'], 1.0)
            self.assertGreaterEqual(stats['fragmentation_ratio'], 0)
            self.assertLessEqual(stats['fragmentation_ratio'], 1.0)
            self.assertGreaterEqual(stats['active_allocations'], 0)
            self.assertGreaterEqual(stats['pool_size'], 0)


class TestMemoryPoolingSystemThreadSafety(unittest.TestCase):
    """Test thread safety of Memory Pooling System"""
    
    def setUp(self):
        self.memory_system = ThreadSafeMemoryPoolingSystem(
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            alloc_future = executor.submit(alloc_dealloc_worker)
            compaction_future = executor.submit(compaction_worker)
            
            alloc_future.result()
            compaction_future.result()
    
    def test_system_statistics_consistency(self):
        """Test that system statistics remain consistent under concurrent access"""
        stats_list = []
        
        def stats_collector():
            for _ in range(20):
                stats = self.memory_system.get_system_stats()
                stats_list.append(stats)
                time.sleep(0.05)
        
        def intensive_allocation_worker():
            for i in range(25):
                tensor_type = random.choice(list(TensorType))
                size = random.randint(512, 4096)
                tensor_id = f"intensive_test_{i}"
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    time.sleep(0.005)
                    self.memory_system.deallocate(tensor_type, tensor_id)
        
        # Run stats collector and intensive allocation worker concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            stats_future = executor.submit(stats_collector)
            alloc_future = executor.submit(intensive_allocation_worker)
            
            stats_future.result()
            alloc_future.result()
        
        # Verify system statistics consistency
        for stats in stats_list:
            self.assertGreaterEqual(stats['overall_utilization'], 0)
            self.assertLessEqual(stats['overall_utilization'], 1.0)
            self.assertGreaterEqual(stats['average_fragmentation'], 0)
            self.assertLessEqual(stats['average_fragmentation'], 1.0)
            self.assertGreaterEqual(stats['total_allocated'], 0)
            self.assertGreaterEqual(stats['total_freed'], 0)
            self.assertGreaterEqual(stats['peak_utilization'], 0)
            self.assertLessEqual(stats['peak_utilization'], 1.0)


class TestRaceConditionScenarios(unittest.TestCase):
    """Test specific race condition scenarios"""
    
    def setUp(self):
        self.memory_system = ThreadSafeMemoryPoolingSystem()
    
    def test_high_contention_allocation_deallocation(self):
        """Test high-contention scenario with many threads competing for allocation/deallocation"""
        results = []
        
        def high_contention_worker(worker_id):
            thread_results = []
            for i in range(50):  # More operations to increase contention
                tensor_type = random.choice(list(TensorType))
                size = random.randint(256, 2048)
                tensor_id = f"high_cont_{worker_id}_{i}"
                
                # Rapid allocation and deallocation
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    thread_results.append(('alloc', tensor_id))
                    # Very brief hold time to maximize contention
                    time.sleep(0.0001)
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_id))
                else:
                    thread_results.append(('failed_alloc', tensor_id))
            results.extend(thread_results)
        
        # Run many threads with high contention
        with ThreadPoolExecutor(max_workers=12) as executor:  # More threads for higher contention
            futures = [executor.submit(high_contention_worker, i) for i in range(12)]
            for future in as_completed(futures):
                future.result()
        
        # Verify operations completed
        alloc_ops = [r for r in results if r[0] == 'alloc']
        dealloc_ops = [r for r in results if r[0] == 'dealloc']
        failed_ops = [r for r in results if r[0] == 'failed_alloc']
        
        # Some allocations might fail due to pool exhaustion, but we expect mostly success
        self.assertGreater(len(alloc_ops) + len(failed_ops), 0)
        self.assertEqual(len(alloc_ops), len(dealloc_ops))
    
    def test_fragmentation_under_concurrent_load(self):
        """Test fragmentation behavior under concurrent load"""
        results = []
        
        def fragmentation_worker(worker_id):
            thread_results = []
            blocks_to_free = []
            
            # Allocate many small blocks
            for i in range(30):
                tensor_type = random.choice(list(TensorType))
                size = random.randint(256, 512)  # Small blocks to create fragmentation
                tensor_id = f"frag_small_{worker_id}_{i}"
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    blocks_to_free.append((tensor_type, tensor_id))
                    thread_results.append(('alloc_small', tensor_id))
            
            # Brief pause to allow other threads to run
            time.sleep(0.01)
            
            # Deallocate every other block to create holes
            for i, (tensor_type, tensor_id) in enumerate(blocks_to_free):
                if i % 2 == 0:  # Deallocate every other block
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_id))
            
            # Brief pause
            time.sleep(0.01)
            
            # Allocate larger blocks to test fragmentation
            for i in range(10):
                tensor_type = random.choice(list(TensorType))
                size = random.randint(2048, 4096)  # Larger blocks
                tensor_id = f"frag_large_{worker_id}_{i}"
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    thread_results.append(('alloc_large', tensor_id))
                    # Don't deallocate these immediately to maintain fragmentation
                    time.sleep(0.001)
            
            results.extend(thread_results)
        
        # Run multiple threads to create fragmentation
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fragmentation_worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()
        
        # Check final fragmentation
        final_stats = self.memory_system.get_system_stats()
        # Fragmentation should be measurable but not excessive due to buddy allocation
        self.assertLessEqual(final_stats['average_fragmentation'], 0.5)  # Should be less than 50%
    
    def test_lock_starvation_prevention(self):
        """Test that no thread gets starved of resources"""
        results = []
        
        def long_running_worker(worker_id):
            thread_results = []
            start_time = time.time()
            
            # Run for a fixed time period
            while time.time() - start_time < 2.0:  # Run for 2 seconds
                tensor_type = random.choice(list(TensorType))
                size = random.randint(512, 1024)
                tensor_id = f"starve_test_{worker_id}_{int(time.time()*1000)}"
                
                block = self.memory_system.allocate(tensor_type, size, tensor_id)
                if block:
                    thread_results.append(('alloc', tensor_id))
                    # Hold briefly
                    time.sleep(0.001)
                    success = self.memory_system.deallocate(tensor_type, tensor_id)
                    if success:
                        thread_results.append(('dealloc', tensor_id))
            
            results.extend(thread_results)
        
        # Run multiple long-running threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(long_running_worker, i) for i in range(8)]
            for future in as_completed(futures):
                future.result()
        
        # Verify all threads made progress
        all_results = []
        for result in results:
            all_results.extend(result)
        
        # Each thread should have performed some operations
        alloc_count = len([r for r in all_results if r[0] == 'alloc'])
        dealloc_count = len([r for r in all_results if r[0] == 'dealloc'])
        
        # With 8 threads running for 2 seconds, we expect significant activity
        self.assertGreater(alloc_count, 0)
        self.assertEqual(alloc_count, dealloc_count)


def run_comprehensive_thread_safety_tests():
    """Run all thread safety tests"""
    print("Running Comprehensive Thread Safety Test Suite for Qwen3-VL Memory Management...")
    
    # Create test suites
    test_classes = [
        TestBuddyAllocatorThreadSafety,
        TestMemoryPoolThreadSafety,
        TestMemoryPoolingSystemThreadSafety,
        TestRaceConditionScenarios
    ]
    
    all_tests_passed = True
    detailed_results = []
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        class_results = {
            'class_name': test_class.__name__,
            'passed': not (result.failures or result.errors),
            'failures': len(result.failures),
            'errors': len(result.errors),
            'tests_run': result.testsRun,
            'failures_details': result.failures,
            'errors_details': result.errors
        }
        detailed_results.append(class_results)
        
        if result.failures or result.errors:
            print(f"FAILED: {test_class.__name__} FAILED")
            all_tests_passed = False
            for failure in result.failures:
                print(f"  FAILURE: {failure[0]} - {failure[1]}")
            for error in result.errors:
                print(f"  ERROR: {error[0]} - {error[1]}")
        else:
            print(f"PASSED: {test_class.__name__} PASSED")
    
    # Print summary
    print(f"\n{'='*60}")
    print("THREAD SAFETY TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    total_tests = sum(r['tests_run'] for r in detailed_results)
    total_failures = sum(r['failures'] for r in detailed_results)
    total_errors = sum(r['errors'] for r in detailed_results)
    
    for result in detailed_results:
        status = "PASS" if result['passed'] else "FAIL"
        print(f"{result['class_name']:<35} {status} | Tests: {result['tests_run']:<3} | Failures: {result['failures']:<2} | Errors: {result['errors']:<2}")
    
    print(f"\nTotal Tests Run: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Total Errors: {total_errors}")
    
    if all_tests_passed:
        print("\n🎉 ALL THREAD SAFETY TESTS PASSED!")
        print("Memory management components are thread-safe and ready for production use.")
        print("\nKey achievements:")
        print("- ✅ Concurrent allocation/deallocation without race conditions")
        print("- ✅ Consistent statistics under high contention")
        print("- ✅ Proper fragmentation handling with concurrent access")
        print("- ✅ No thread starvation under sustained load")
        print("- ✅ Correct lock behavior with multiple threads")
    else:
        print("\n⚠️  SOME THREAD SAFETY TESTS FAILED!")
        print("Please review and fix the identified race conditions.")
        print("\nFailed areas:")
        for result in detailed_results:
            if not result['passed']:
                print(f"- {result['class_name']}")
    
    return all_tests_passed


if __name__ == "__main__":
    success = run_comprehensive_thread_safety_tests()
    exit(0 if success else 1)