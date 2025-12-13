import threading
import time
import unittest
from advanced_memory_pooling_system import BuddyAllocator, MemoryBlock, TensorType


class TestBuddyAllocatorThreadSafety(unittest.TestCase):
    """Test suite for thread safety of Buddy Allocator"""

    def test_concurrent_allocation_and_deallocation(self):
        """Test concurrent allocation and deallocation operations"""
        allocator = BuddyAllocator(1024 * 1024)  # 1MB
        threads = []
        results = []
        errors = []

        def worker(worker_id):
            try:
                # Allocate multiple blocks
                allocated_blocks = []
                for i in range(10):
                    block = allocator.allocate(1024, TensorType.ACTIVATIONS, f"worker_{worker_id}_block_{i}")
                    if block:
                        allocated_blocks.append(block)
                        # Sleep briefly to allow other threads to run
                        time.sleep(0.001)
                
                # Deallocate some blocks
                for i, block in enumerate(allocated_blocks):
                    if i % 2 == 0:  # Deallocate every other block
                        allocator.deallocate(block)
                
                # Allocate more blocks after deallocation
                for i in range(5):
                    block = allocator.allocate(2048, TensorType.ACTIVATIONS, f"worker_{worker_id}_block_after_{i}")
                    if block:
                        allocator.deallocate(block)
                
                results.append(f"Worker {worker_id} completed successfully")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {str(e)}")

        # Create and start multiple threads
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")
        self.assertEqual(len(results), 5, "All workers should have completed")

    def test_concurrent_merge_operations(self):
        """Test concurrent merge operations that could cause race conditions"""
        allocator = BuddyAllocator(1024 * 1024)  # 1MB
        threads = []
        results = []
        errors = []

        def worker(worker_id):
            try:
                # Allocate and deallocate blocks to trigger merge operations
                blocks = []
                
                # Allocate several blocks
                for i in range(20):
                    block = allocator.allocate(1024, TensorType.ACTIVATIONS, f"worker_{worker_id}_block_{i}")
                    if block:
                        blocks.append(block)
                
                # Deallocate all blocks in reverse order to maximize merge opportunities
                for block in reversed(blocks):
                    allocator.deallocate(block)
                
                results.append(f"Worker {worker_id} completed merge test")
            except Exception as e:
                errors.append(f"Worker {worker_id} merge error: {str(e)}")

        # Create and start multiple threads that will trigger many merge operations
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Merge thread errors occurred: {errors}")
        self.assertEqual(len(results), 3, "All merge workers should have completed")

    def test_high_concurrency_stress_test(self):
        """Stress test with high concurrency to expose race conditions"""
        allocator = BuddyAllocator(2 * 1024 * 1024)  # 2MB
        threads = []
        results = []
        errors = []

        def worker(worker_id):
            try:
                local_results = []
                
                # Perform many allocation/deallocation cycles
                for cycle in range(50):
                    # Allocate multiple blocks
                    blocks = []
                    for i in range(5):
                        size = 512 + (i * 256)  # Varying sizes
                        block = allocator.allocate(size, TensorType.ACTIVATIONS, 
                                                 f"worker_{worker_id}_cycle_{cycle}_block_{i}")
                        if block:
                            blocks.append(block)
                    
                    # Randomly deallocate some blocks
                    for i, block in enumerate(blocks):
                        if i % 3 == 0:  # Deallocate every third block
                            allocator.deallocate(block)
                    
                    # Sleep briefly to increase chance of race conditions
                    time.sleep(0.0001)
                
                results.append(f"Worker {worker_id} completed stress test")
            except Exception as e:
                errors.append(f"Worker {worker_id} stress error: {str(e)}")

        # Create many threads for high concurrency
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Stress test errors occurred: {errors}")
        self.assertEqual(len(results), 10, "All stress workers should have completed")

    def test_memory_consistency_after_concurrent_operations(self):
        """Test that memory structure remains consistent after concurrent operations"""
        allocator = BuddyAllocator(1024 * 1024)  # 1MB
        threads = []
        results = []
        errors = []

        def worker(worker_id):
            try:
                # Perform allocation/deallocation pattern
                blocks = []
                
                # Phase 1: Allocate blocks
                for i in range(10):
                    block = allocator.allocate(1024, TensorType.ACTIVATIONS, f"test_block_{worker_id}_{i}")
                    if block:
                        blocks.append(block)
                
                # Phase 2: Deallocate half of them
                for i, block in enumerate(blocks):
                    if i % 2 == 0:
                        allocator.deallocate(block)
                
                # Phase 3: Allocate more blocks (should potentially trigger merges)
                more_blocks = []
                for i in range(5):
                    block = allocator.allocate(2048, TensorType.ACTIVATIONS, f"more_blocks_{worker_id}_{i}")
                    if block:
                        more_blocks.append(block)
                
                # Phase 4: Clean up
                for block in more_blocks:
                    allocator.deallocate(block)
                
                results.append(f"Worker {worker_id} completed consistency test")
            except Exception as e:
                errors.append(f"Worker {worker_id} consistency error: {str(e)}")

        # Run multiple workers
        for i in range(8):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check for errors and validate memory consistency
        self.assertEqual(len(errors), 0, f"Consistency test errors: {errors}")
        
        # Verify that the allocator is still in a valid state
        with allocator.lock:  # Use the allocator's lock to safely check internal state
            total_free_size = 0
            for level_blocks in allocator.free_blocks.values():
                for block in level_blocks:
                    total_free_size += block.size
            
            total_allocated_size = sum(block.size for block in allocator.allocated_blocks.values())
            
            # Total should equal the original size
            self.assertEqual(total_free_size + total_allocated_size, allocator.total_size,
                           "Memory accounting is inconsistent after concurrent operations")


if __name__ == '__main__':
    unittest.main()