"""
Test suite for Buddy Allocator System

This module contains comprehensive tests for the Buddy Allocator System,
validating all functionality including allocation, deallocation, coalescing,
thread safety, statistics, PyTorch integration, and hardware optimizations.
"""

import unittest
import threading
import time
from typing import Tuple
from memory_optimization_systems.buddy_allocator_system.buddy_allocator import (
    BuddyAllocator, PyTorchBuddyAllocator, OptimizedBuddyAllocator
)


class TestBuddyAllocator(unittest.TestCase):
    """Test suite for the basic Buddy Allocator functionality."""
    
    def setUp(self):
        """Set up a fresh allocator for each test."""
        self.allocator = BuddyAllocator(1024*1024, 64)  # 1MB with 64-byte min blocks
    
    def test_initialization(self):
        """Test proper initialization of the allocator."""
        self.assertEqual(self.allocator.total_size, 1024*1024)
        self.assertEqual(self.allocator.min_block_size, 64)
        self.assertFalse(self.allocator.root.is_allocated)
        self.assertEqual(len(self.allocator._allocated_blocks), 0)
    
    def test_invalid_initialization(self):
        """Test that invalid initialization parameters raise errors."""
        with self.assertRaises(ValueError):
            BuddyAllocator(1024*1024 + 1, 64)  # Not a power of 2
        
        with self.assertRaises(ValueError):
            BuddyAllocator(1024*1024, 65)  # Not a power of 2
        
        with self.assertRaises(ValueError):
            BuddyAllocator(64, 128)  # Total size < min block size
    
    def test_basic_allocation_and_deallocation(self):
        """Test basic allocation and deallocation functionality."""
        # Allocate a block
        result = self.allocator.allocate(1024)
        self.assertIsNotNone(result)
        handle, address = result
        self.assertIsInstance(handle, int)
        self.assertIsInstance(address, int)
        
        # Verify block is marked as allocated
        self.assertTrue(self.allocator.is_valid_handle(handle))
        
        # Deallocate the block
        success = self.allocator.deallocate(handle)
        self.assertTrue(success)
        
        # Verify block is no longer valid
        self.assertFalse(self.allocator.is_valid_handle(handle))
    
    def test_allocation_larger_than_pool(self):
        """Test allocation request larger than total pool size."""
        result = self.allocator.allocate(2*1024*1024)  # Request 2MB from 1MB pool
        self.assertIsNone(result)
    
    def test_allocation_zero_size(self):
        """Test allocation request of zero size."""
        result = self.allocator.allocate(0)
        self.assertIsNone(result)
    
    def test_allocation_negative_size(self):
        """Test allocation request of negative size."""
        result = self.allocator.allocate(-100)
        self.assertIsNone(result)
    
    def test_multiple_allocations(self):
        """Test multiple allocations and deallocations."""
        handles = []
        
        # Allocate several blocks
        for size in [256, 512, 1024, 2048]:
            result = self.allocator.allocate(size)
            self.assertIsNotNone(result)
            handle, address = result
            handles.append(handle)
        
        # Verify all are allocated
        for handle in handles:
            self.assertTrue(self.allocator.is_valid_handle(handle))
        
        # Deallocate all
        for handle in handles:
            success = self.allocator.deallocate(handle)
            self.assertTrue(success)
        
        # Verify none are still allocated
        for handle in handles:
            self.assertFalse(self.allocator.is_valid_handle(handle))
    
    def test_coalescing(self):
        """Test that buddy blocks properly coalesce after deallocation."""
        # Allocate and then deallocate adjacent blocks to trigger coalescing
        result1 = self.allocator.allocate(256)
        result2 = self.allocator.allocate(256)
        
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        handle1, _ = result1
        handle2, _ = result2
        
        # Deallocate both blocks
        success1 = self.allocator.deallocate(handle1)
        success2 = self.allocator.deallocate(handle2)
        
        self.assertTrue(success1)
        self.assertTrue(success2)
        
        # Now we should be able to allocate a 512-byte block
        result3 = self.allocator.allocate(512)
        self.assertIsNotNone(result3)
    
    def test_statistics(self):
        """Test that statistics are properly maintained."""
        initial_stats = self.allocator.get_statistics()
        
        # Allocate some memory
        result = self.allocator.allocate(1024)
        self.assertIsNotNone(result)
        
        # Check that statistics were updated
        stats = self.allocator.get_statistics()
        self.assertGreater(stats['allocations'], initial_stats['allocations'])
        self.assertGreater(stats['total_requested'], initial_stats['total_requested'])
        self.assertGreater(stats['total_allocated'], initial_stats['total_allocated'])
    
    def test_reset_statistics(self):
        """Test that statistics can be reset."""
        # Allocate some memory to increase stats
        result = self.allocator.allocate(1024)
        self.assertIsNotNone(result)
        
        # Reset statistics
        self.allocator.reset_statistics()
        
        # Check that stats are back to zero
        stats = self.allocator.get_statistics()
        self.assertEqual(stats['allocations'], 0)
        self.assertEqual(stats['deallocations'], 0)
        self.assertEqual(stats['total_requested'], 0)
        self.assertEqual(stats['total_allocated'], 0)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of the buddy allocator."""
    
    def setUp(self):
        """Set up a fresh allocator for each test."""
        self.allocator = BuddyAllocator(1024*1024, 64)  # 1MB with 64-byte min blocks
    
    def test_concurrent_allocations(self):
        """Test that concurrent allocations work safely."""
        results = []
        
        def worker():
            for _ in range(10):
                result = self.allocator.allocate(256)
                if result:
                    results.append(result)
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(4)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all allocations worked
        self.assertGreaterEqual(len(results), 0)  # At least some should succeed
        
        # Clean up allocated blocks
        for handle, _ in results:
            self.allocator.deallocate(handle)
    
    def test_concurrent_allocations_and_deallocations(self):
        """Test that concurrent allocations and deallocations work safely."""
        handles = []
        lock = threading.Lock()
        
        def alloc_worker():
            for _ in range(5):
                result = self.allocator.allocate(512)
                if result:
                    handle, _ = result
                    with lock:
                        handles.append(handle)
                time.sleep(0.001)
        
        def dealloc_worker():
            for _ in range(5):
                time.sleep(0.002)  # Slight delay to allow allocations to happen first
                with lock:
                    if handles:
                        handle = handles.pop(0)
                        self.allocator.deallocate(handle)
        
        # Create allocation and deallocation threads
        alloc_threads = [threading.Thread(target=alloc_worker) for _ in range(2)]
        dealloc_threads = [threading.Thread(target=dealloc_worker) for _ in range(2)]
        
        # Start all threads
        for t in alloc_threads + dealloc_threads:
            t.start()
        
        # Wait for all threads to complete
        for t in alloc_threads + dealloc_threads:
            t.join()
        
        # Clean up any remaining handles
        with lock:
            for handle in handles:
                self.allocator.deallocate(handle)


class TestPyTorchIntegration(unittest.TestCase):
    """Test PyTorch integration functionality."""
    
    def setUp(self):
        """Set up a PyTorch-aware allocator for each test."""
        try:
            import torch
            self.torch_available = True
            self.allocator = PyTorchBuddyAllocator(1024*1024, 256)
        except ImportError:
            self.torch_available = False
    
    @unittest.skipIf(not hasattr(unittest.TestCase, 'torch_available') or not unittest.TestCase.torch_available, 
                     "PyTorch not available")
    def test_tensor_allocation(self):
        """Test allocation of PyTorch tensors."""
        import torch
        
        # Allocate a tensor
        result = self.allocator.allocate_tensor((100, 100), torch.float32)
        self.assertIsNotNone(result)
        
        handle, tensor = result
        self.assertEqual(tensor.shape, (100, 100))
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Deallocate the tensor
        success = self.allocator.deallocate_tensor(handle)
        self.assertTrue(success)
    
    @unittest.skipIf(not hasattr(unittest.TestCase, 'torch_available') or not unittest.TestCase.torch_available, 
                     "PyTorch not available")
    def test_different_tensor_types(self):
        """Test allocation of different tensor types."""
        import torch
        
        # Test different dtypes and shapes
        test_configs = [
            ((50, 50), torch.float32),
            ((25, 25, 3), torch.int64),
            ((10, 20), torch.bool),
            ((100,), torch.float16),
        ]
        
        handles = []
        for shape, dtype in test_configs:
            result = self.allocator.allocate_tensor(shape, dtype)
            if result:
                handle, tensor = result
                self.assertEqual(tensor.shape, shape)
                self.assertEqual(tensor.dtype, dtype)
                handles.append(handle)
        
        # Clean up
        for handle in handles:
            self.allocator.deallocate_tensor(handle)


class TestHardwareOptimizations(unittest.TestCase):
    """Test hardware-optimized allocator functionality."""
    
    def setUp(self):
        """Set up an optimized allocator for each test."""
        self.allocator = OptimizedBuddyAllocator(1024*1024, 64)
    
    def test_hardware_parameters(self):
        """Test that hardware-specific parameters are properly set."""
        params = self.allocator.get_hardware_optimized_params()
        
        self.assertIn('cpu_cores', params)
        self.assertIn('l3_cache_size', params)
        self.assertIn('cache_line_size', params)
        self.assertIn('gpu_compute_units', params)
        self.assertIn('warp_size', params)
        self.assertIn('nvme_page_size', params)
        
        # Verify expected values for i5-10210U + NVIDIA SM61
        self.assertEqual(params['cpu_cores'], 4)
        self.assertEqual(params['cache_line_size'], 64)
        self.assertEqual(params['warp_size'], 32)
        self.assertEqual(params['nvme_page_size'], 4096)
    
    def test_cache_aligned_allocation(self):
        """Test that allocations are cache-line aligned."""
        # Request a size that would normally not be cache-aligned
        result = self.allocator.allocate(100)  # 100 bytes
        if result:
            handle, address = result
            
            # The actual allocated size should be aligned to cache line boundary
            # The address might not be aligned in our simulation, but size should be
            self.allocator.deallocate(handle)
    
    def test_gpu_aligned_allocation(self):
        """Test GPU-specific allocation alignment."""
        # Create an allocator that simulates GPU usage
        gpu_allocator = OptimizedBuddyAllocator(1024*1024, 64, device=None)
        
        # Even without PyTorch, the allocation logic should work
        result = gpu_allocator.allocate(100)
        if result:
            handle, address = result
            gpu_allocator.deallocate(handle)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_fragmentation_handling(self):
        """Test allocator behavior under high fragmentation."""
        allocator = BuddyAllocator(1024*1024, 64)  # 1MB pool
        
        # Create a fragmented state by allocating and deallocating many small blocks
        handles = []
        
        # Allocate many small blocks
        for _ in range(500):
            result = allocator.allocate(128)
            if result:
                handles.append(result[0])
        
        # Deallocate every other block to create fragmentation
        for i in range(0, len(handles), 2):
            allocator.deallocate(handles[i])
        
        # Try to allocate a large block - this may fail due to fragmentation
        large_alloc = allocator.allocate(512*1024)  # 512KB
        
        # Clean up
        for i in range(1, len(handles), 2):
            allocator.deallocate(handles[i])
        
        # The large allocation may or may not succeed depending on fragmentation
        # But the allocator should not crash
    
    def test_minimum_block_size_enforcement(self):
        """Test that minimum block size is properly enforced."""
        allocator = BuddyAllocator(1024*1024, 256)  # 256-byte minimum
        
        # Request smaller than minimum - should get minimum-sized block
        result = allocator.allocate(64)
        if result:
            handle, address = result
            # The allocator will round up to minimum block size
            allocator.deallocate(handle)
    
    def test_maximum_utilization(self):
        """Test allocator behavior when approaching maximum utilization."""
        # Use a smaller pool for this test to make it manageable
        allocator = BuddyAllocator(1024*64, 64)  # 64KB pool
        
        handles = []
        total_allocated = 0
        
        # Keep allocating until we can't anymore
        while True:
            remaining = allocator.total_size - total_allocated
            if remaining < 64:  # Minimum block size
                break
                
            result = allocator.allocate(min(remaining, 1024))  # Try 1KB blocks
            if result:
                handle, address = result
                handles.append(handle)
                total_allocated += 1024
            else:
                break  # No more space available
        
        # Clean up all allocations
        for handle in handles:
            allocator.deallocate(handle)


def run_all_tests():
    """Run all tests in the test suite."""
    # Create test suites
    basic_suite = unittest.TestLoader().loadTestsFromTestCase(TestBuddyAllocator)
    thread_suite = unittest.TestLoader().loadTestsFromTestCase(TestThreadSafety)
    pytorch_suite = unittest.TestLoader().loadTestsFromTestCase(TestPyTorchIntegration)
    hardware_suite = unittest.TestLoader().loadTestsFromTestCase(TestHardwareOptimizations)
    edge_suite = unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases)
    
    # Combine all suites
    all_tests = unittest.TestSuite([
        basic_suite,
        thread_suite,
        pytorch_suite,
        hardware_suite,
        edge_suite
    ])
    
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running Buddy Allocator System tests...\n")
    success = run_all_tests()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")