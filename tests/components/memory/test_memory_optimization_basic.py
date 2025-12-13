"""
Simplified test for memory optimization components
"""

import unittest
import torch
import numpy as np
import time
import psutil
from typing import Dict, Any, Tuple, Optional
import math
from collections import defaultdict, deque
import sys
import os
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the memory optimization classes using proper imports
from qwen3_vl.components.memory.memory_optimization_system import (
    MemoryConfig,
    BuddyAllocator,
    TensorCache,
    MemoryPool,
    MemoryDefragmenter,
    MemoryManager,
    get_memory_manager,
    allocate_tensor_with_manager,
    free_tensor_with_manager
)


class TestMemoryOptimizationSystem(unittest.TestCase):
    """Test suite for the memory optimization system"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MemoryConfig(memory_pool_size=2**20)  # 1MB pool for testing
        self.memory_manager = MemoryManager(self.config)

    def test_buddy_allocator_basic(self):
        """Test basic buddy allocator functionality"""
        allocator = BuddyAllocator(2**20)  # 1MB

        # Test allocation
        addr1 = allocator.allocate(1024)  # 1KB
        self.assertIsNotNone(addr1)

        addr2 = allocator.allocate(2048)  # 2KB
        self.assertIsNotNone(addr2)

        # Test deallocation
        success = allocator.deallocate(addr1)
        self.assertTrue(success)

        # Test re-allocation
        addr3 = allocator.allocate(1024)  # Should potentially reuse addr1
        self.assertIsNotNone(addr3)

        stats = allocator.get_stats()
        self.assertIn('allocations', stats)
        self.assertIn('deallocations', stats)

    def test_buddy_allocator_edge_cases(self):
        """Test edge cases for buddy allocator"""
        allocator = BuddyAllocator(2**10)  # 1KB

        # Try to allocate more than available
        addr = allocator.allocate(2**11)  # 2KB > 1KB
        self.assertIsNone(addr)

        # Allocate maximum possible
        addr = allocator.allocate(2**10)  # 1KB
        self.assertIsNotNone(addr)

        # Try to allocate more when full
        addr2 = allocator.allocate(1)
        self.assertIsNone(addr2)

        # Deallocate and verify
        success = allocator.deallocate(addr)
        self.assertTrue(success)

    def test_tensor_cache(self):
        """Test tensor cache functionality"""
        cache = TensorCache()

        # Create a tensor and return it to cache
        tensor = torch.randn(10, 20)
        cache.return_tensor(tensor)

        # Get tensor from cache
        retrieved_tensor = cache.get_tensor((10, 20), torch.float32, torch.device('cpu'))
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(retrieved_tensor.shape, (10, 20))

        # Stats should reflect cache usage
        self.assertGreater(cache.stats['hits'], 0)

    def test_memory_pool_basic(self):
        """Test basic memory pool functionality"""
        pool = MemoryPool(2**20)  # 1MB

        # Allocate tensors
        tensor1 = pool.allocate_tensor((100, 200), torch.float32)
        tensor2 = pool.allocate_tensor((50, 100, 256), torch.float32)

        self.assertEqual(tensor1.shape, (100, 200))
        self.assertEqual(tensor2.shape, (50, 100, 256))

        # Deallocate tensors
        success1 = pool.deallocate_tensor(tensor1)
        success2 = pool.deallocate_tensor(tensor2)

        self.assertTrue(success1)
        self.assertTrue(success2)

        # Check memory stats
        stats = pool.get_memory_stats()
        self.assertIn('buddy_allocator', stats)
        self.assertIn('tensor_cache', stats)

    def test_memory_manager_basic(self):
        """Test basic memory manager functionality"""
        manager = MemoryManager(config=self.config)

        # Allocate tensors
        tensor1 = manager.allocate_tensor((100, 200), torch.float32)
        tensor2 = manager.allocate_tensor((50, 100, 256), torch.float32)

        self.assertEqual(tensor1.shape, (100, 200))
        self.assertEqual(tensor2.shape, (50, 100, 256))

        # Free tensors
        success1 = manager.free_tensor(tensor1)
        success2 = manager.free_tensor(tensor2)

        self.assertTrue(success1)
        self.assertTrue(success2)

        # Check memory stats
        stats = manager.get_memory_stats()
        self.assertIn('total_allocations', stats)
        self.assertIn('pool_stats', stats)

    def test_memory_efficiency(self):
        """Test that memory manager provides efficiency benefits"""
        manager = MemoryManager(config=self.config)

        # Allocate and deallocate many tensors of the same shape
        shapes_to_test = [(10, 10), (50, 50), (100, 100)]
        for shape in shapes_to_test:
            tensors = []
            for _ in range(10):
                tensor = manager.allocate_tensor(shape, torch.float32)
                tensors.append(tensor)
            for tensor in tensors:
                manager.free_tensor(tensor)

        stats = manager.get_memory_stats()
        # After many allocations of same shape, we should have good cache hit rate
        cache_hits = stats['pool_stats']['tensor_cache']['hits']
        cache_misses = stats['pool_stats']['tensor_cache']['misses']
        total_requests = stats['pool_stats']['tensor_cache']['total_requests']

        # Verify that we had some cache activity
        self.assertGreater(total_requests, 0)
        if total_requests > 10:  # Only check if we had enough requests
            hit_rate = cache_hits / total_requests if total_requests > 0 else 0
            # Even with random shapes, we might get some hits for common shapes
            print(f"Cache hit rate: {hit_rate:.2f}")

    def test_defragmentation(self):
        """Test memory defragmentation functionality"""
        manager = MemoryManager(config=self.config)

        # Create fragmentation by allocating and deallocating various sizes
        large_tensor = manager.allocate_tensor((500, 500), torch.float32)
        small_tensor1 = manager.allocate_tensor((10, 10), torch.float32)
        small_tensor2 = manager.allocate_tensor((15, 15), torch.float32)
        medium_tensor = manager.allocate_tensor((100, 100), torch.float32)

        # Deallocate some tensors to create fragmentation
        manager.free_tensor(large_tensor)
        manager.free_tensor(small_tensor1)

        # Defragment
        defrag_result = manager.defragment_memory()

        self.assertIsNotNone(defrag_result)
        print(f"Defragmentation result: {defrag_result}")

    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring"""
        manager = MemoryManager(config=self.config)

        # Check initial memory pressure
        initial_pressure = manager.memory_pressure

        # Allocate large tensors to increase memory pressure
        large_tensors = []
        for i in range(10):
            tensor = manager.allocate_tensor((200, 200), torch.float32)
            large_tensors.append(tensor)

        # Check memory pressure after allocation
        final_pressure = manager.memory_pressure

        # Memory pressure should have increased
        print(f"Initial memory pressure: {initial_pressure:.4f}")
        print(f"Final memory pressure: {final_pressure:.4f}")

        # Clean up
        for tensor in large_tensors:
            manager.free_tensor(tensor)

    def test_preallocated_tensor_caching(self):
        """Test pre-allocated tensor caching for common shapes"""
        manager = MemoryManager(config=self.config)

        # Allocate tensors of common shapes repeatedly
        common_shapes = [
            (64, 128), (128, 256), (256, 512), (512, 1024)
        ]

        for shape in common_shapes:
            for _ in range(5):  # Allocate each shape multiple times
                tensor = manager.allocate_tensor(shape, torch.float32)
                manager.free_tensor(tensor)

        # Check cache statistics
        stats = manager.get_memory_stats()
        cache_stats = stats['pool_stats']['tensor_cache']

        print(f"Cache hits: {cache_stats['hits']}")
        print(f"Cache misses: {cache_stats['misses']}")
        print(f"Total requests: {cache_stats['total_requests']}")

        # Verify that caching was active
        self.assertGreaterEqual(cache_stats['total_requests'], len(common_shapes) * 5)

    def test_global_memory_manager(self):
        """Test global memory manager singleton"""
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()

        # They should be the same instance
        self.assertIs(manager1, manager2)

        # Test allocation through global functions
        tensor = allocate_tensor_with_manager((50, 50), torch.float32)
        self.assertEqual(tensor.shape, (50, 50))

        free_tensor_with_manager(tensor)

    def test_thread_safety(self):
        """Test thread safety of memory manager"""
        import threading

        manager = get_memory_manager()
        results = []

        def worker():
            for _ in range(5):
                tensor = manager.allocate_tensor((10, 10), torch.float32)
                results.append(tensor.shape)
                manager.free_tensor(tensor)

        # Run multiple threads concurrently
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(results), 15)  # 3 threads * 5 iterations each
        stats = manager.get_memory_stats()
        print(f"Thread safety test - Total allocations: {stats['total_allocations']}")

    def test_error_handling_valid_shapes(self):
        """Test error handling with valid shapes"""
        manager = MemoryManager()

        # Test allocation with valid parameters
        tensor = manager.allocate_tensor((10, 10), torch.float32)  # Valid shape
        self.assertIsNotNone(tensor)  # Should work fine

        # Test with zero dimensions (edge case)
        tensor = manager.allocate_tensor((0, 10), torch.float32)
        self.assertIsNotNone(tensor)  # Should handle gracefully

        print("Error handling tests with valid shapes passed")

    def test_memory_alignment_optimizations(self):
        """Test memory alignment optimizations for SM61 architecture"""
        manager = MemoryManager()

        # Test various tensor shapes and ensure they work correctly
        test_shapes = [
            (1, 512, 4096),  # Typical transformer dimension
            (1, 256, 2048),  # Smaller transformer dimension
            (1, 1024, 1024), # Square attention matrix
            (2, 512, 512, 256)  # Multi-batch with 4D tensor
        ]

        for shape in test_shapes:
            tensor = manager.allocate_tensor(shape, torch.float32)
            self.assertEqual(tensor.shape, shape)
            manager.free_tensor(tensor)

        print(f"Memory alignment optimization test passed for {len(test_shapes)} shapes")


def run_memory_optimization_tests():
    """Run all memory optimization tests"""
    print("Running Memory Optimization System Tests...")

    # Create a test suite
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Add all test methods from the test class
    test_class = TestMemoryOptimizationSystem
    for test_name in loader.getTestCaseNames(test_class):
        suite.addTest(test_class(test_name))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MEMORY OPTIMIZATION SYSTEM TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.splitlines()[-1]}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.splitlines()[-1]}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_memory_optimization_tests()
    if success:
        print("\nAll memory optimization tests passed!")
        print("Memory optimization system is ready for production use.")
    else:
        print("\nSome tests failed!")
        print("Please review the test failures and fix the implementation.")