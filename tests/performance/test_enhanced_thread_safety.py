import threading
import time
import unittest
from advanced_memory_pooling_system import BuddyAllocator, MemoryBlock, TensorType


class TestBuddyAllocatorEnhancedThreadSafety(unittest.TestCase):
    """Enhanced test suite for thread safety of Buddy Allocator with specific focus on merge operations"""

    def test_concurrent_merge_stress(self):
        """Test concurrent merge operations that specifically target the same buddy blocks"""
        allocator = BuddyAllocator(1024 * 1024)  # 1MB
        threads = []
        results = []
        errors = []

        def worker(worker_id):
            try:
                # Create a scenario where multiple threads will try to merge the same buddies
                blocks = []
                
                # First, allocate and deallocate blocks to create fragmented memory
                # This will create many small free blocks that can potentially be merged
                for i in range(50):
                    block = allocator.allocate(1024, TensorType.ACTIVATIONS, f"init_block_{i}")
                    if block:
                        blocks.append(block)
                
                # Deallocate all blocks to create many free blocks at the same level
                for block in blocks:
                    allocator.deallocate(block)
                
                # Now run multiple threads that will try to allocate and trigger merges
                for cycle in range(10):
                    # Allocate some blocks
                    cycle_blocks = []
                    for i in range(5):
                        block = allocator.allocate(2048, TensorType.ACTIVATIONS, f"cycle_{cycle}_worker_{worker_id}_block_{i}")
                        if block:
                            cycle_blocks.append(block)
                    
                    # Deallocate them to trigger merge operations
                    for block in cycle_blocks:
                        allocator.deallocate(block)
                
                results.append(f"Worker {worker_id} completed merge stress test")
            except Exception as e:
                errors.append(f"Worker {worker_id} merge stress error: {str(e)}")

        # Run multiple threads that will trigger many merge operations
        for i in range(8):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Merge stress test errors occurred: {errors}")
        self.assertEqual(len(results), 8, "All merge stress workers should have completed")

    def test_buddy_collision_scenario(self):
        """Test the specific scenario where two threads try to merge the same buddy blocks"""
        allocator = BuddyAllocator(256 * 1024)  # 256KB to make it easier to create specific scenarios
        threads = []
        results = []
        errors = []

        def create_collision_scenario():
            """Create a scenario where two adjacent blocks can be merged together"""
            # Allocate all memory in 4KB chunks
            blocks = []
            for i in range(64):  # 64 * 4KB = 256KB
                block = allocator.allocate(4096, TensorType.ACTIVATIONS, f"filler_{i}")
                if block:
                    blocks.append(block)
            
            # Deallocate alternating blocks to create potential for merging
            for i, block in enumerate(blocks):
                if i % 2 == 0:
                    allocator.deallocate(block)
            
            return blocks

        def worker(worker_id):
            try:
                # Create the collision scenario
                if worker_id == 0:
                    blocks = create_collision_scenario()
                
                # Wait for all threads to start to maximize concurrency
                time.sleep(0.01)
                
                # Now try to allocate blocks which will trigger merge operations
                for i in range(5):
                    block = allocator.allocate(8192, TensorType.ACTIVATIONS, f"collision_worker_{worker_id}_block_{i}")
                    if block:
                        allocator.deallocate(block)
                
                results.append(f"Worker {worker_id} completed collision test")
            except Exception as e:
                errors.append(f"Worker {worker_id} collision error: {str(e)}")

        # Create multiple threads that will potentially trigger the same merges
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Collision test errors occurred: {errors}")
        self.assertEqual(len(results), 5, "All collision workers should have completed")

    def test_memory_state_consistency(self):
        """Test that memory state remains consistent during high-concurrency operations"""
        allocator = BuddyAllocator(512 * 1024)  # 512KB
        threads = []
        results = []
        errors = []

        def worker(worker_id):
            try:
                local_blocks = []
                
                # Perform operations in phases to maximize the chance of race conditions
                for phase in range(5):
                    # Phase 1: Allocate many small blocks
                    for i in range(10):
                        size = 512 + (i % 3) * 512  # 512, 1024, or 1536 bytes
                        block = allocator.allocate(size, TensorType.ACTIVATIONS, 
                                                 f"phase_{phase}_worker_{worker_id}_block_{i}")
                        if block:
                            local_blocks.append(block)
                    
                    # Phase 2: Randomly deallocate some blocks to trigger merges
                    for i, block in enumerate(local_blocks):
                        if (i + worker_id) % 3 == 0:  # Different pattern per worker
                            allocator.deallocate(block)
                            local_blocks.remove(block)  # Remove from local tracking
                    
                    # Phase 3: Allocate larger blocks to force more merges
                    for i in range(3):
                        large_block = allocator.allocate(4096, TensorType.ACTIVATIONS,
                                                       f"large_phase_{phase}_worker_{worker_id}_block_{i}")
                        if large_block:
                            # Keep this block for later deallocation
                            local_blocks.append(large_block)
                
                # At the end, deallocate all remaining blocks
                for block in local_blocks:
                    allocator.deallocate(block)
                
                results.append(f"Worker {worker_id} completed consistency test")
            except Exception as e:
                errors.append(f"Worker {worker_id} consistency error: {str(e)}")

        # Run many concurrent workers to stress the system
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify memory consistency
        with allocator.lock:  # Use allocator's lock to safely check internal state
            # Calculate total free space
            total_free = 0
            for level_blocks in allocator.free_blocks.values():
                for block in level_blocks:
                    total_free += block.size
            
            # Calculate total allocated space
            total_allocated = sum(block.size for block in allocator.allocated_blocks.values())
            
            # Total should equal the original size
            expected_total = allocator.total_size
            actual_total = total_free + total_allocated
            
            self.assertEqual(actual_total, expected_total,
                           f"Memory accounting inconsistency: expected {expected_total}, got {actual_total}")
            self.assertEqual(len(allocator.allocated_blocks), 0,
                           "All blocks should be deallocated at the end")
        
        self.assertEqual(len(errors), 0, f"Consistency test errors occurred: {errors}")
        self.assertEqual(len(results), 10, "All consistency workers should have completed")

    def test_performance_under_concurrency(self):
        """Test that the thread safety mechanisms don't severely impact performance"""
        allocator = BuddyAllocator(1024 * 1024)  # 1MB
        start_time = time.time()
        errors = []  # Define errors here to fix scoping issue

        def worker(worker_id):
            successful_ops = 0
            try:
                for i in range(50):  # Reduced to avoid excessive runtime in tests
                    # Allocate a block
                    block = allocator.allocate(1024, TensorType.ACTIVATIONS, f"perf_{worker_id}_{i}")
                    if block:
                        successful_ops += 1
                        # Immediately deallocate to trigger merge logic
                        allocator.deallocate(block)
                return successful_ops
            except Exception as e:
                errors.append(f"Worker {worker_id} performance error: {str(e)}")
                return 0

        # Run performance test with multiple threads
        threads = []
        results = []

        for i in range(4):  # 4 threads for reasonable parallelism
            thread = threading.Thread(target=lambda i=i: results.append(worker(i)))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        total_ops = sum(results)

        # Basic sanity checks
        self.assertGreater(total_ops, 0, "Should have performed some operations")
        self.assertEqual(len(errors), 0, f"Performance test errors: {errors}")

        # Log performance metrics
        duration = end_time - start_time
        ops_per_second = total_ops / duration if duration > 0 else float('inf')
        print(f"Performance test: {total_ops} operations in {duration:.3f}s ({ops_per_second:.1f} ops/sec)")


if __name__ == '__main__':
    unittest.main()