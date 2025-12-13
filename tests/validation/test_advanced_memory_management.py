"""
Comprehensive tests for the Advanced Memory Management System
Testing the fixes for cross-platform compatibility, memory leaks, and cleanup issues
"""

import unittest
import numpy as np
import os
import tempfile
import gc
from unittest.mock import patch, MagicMock
from advanced_memory_management_vl_fixed import (
    AdvancedMemoryPool, MemoryPoolType, MemoryDefragmenter,
    VisionLanguageMemoryOptimizer, CacheAwareMemoryManager, GPUCPUMemoryOptimizer
)


class TestAdvancedMemoryPool(unittest.TestCase):
    """Test the AdvancedMemoryPool class for cross-platform compatibility and memory management"""

    def test_initialization(self):
        """Test basic initialization of the memory pool"""
        pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB
        self.assertIsNotNone(pool.pool_ptr)
        self.assertIsNotNone(pool.pool_base)
        self.assertEqual(len(pool.blocks), 1)
        self.assertFalse(pool.blocks[0].allocated)
        pool.cleanup()

    def test_allocation_and_deallocation(self):
        """Test basic allocation and deallocation functionality"""
        pool = AdvancedMemoryPool(initial_size=2*1024*1024)  # 2MB
        size = 1024  # 1KB
        
        # Test allocation
        result = pool.allocate(size)
        self.assertIsNotNone(result)
        ptr, allocated_size = result
        self.assertGreater(ptr, 0)
        self.assertGreater(allocated_size, 0)
        
        # Test deallocation
        success = pool.deallocate(ptr)
        self.assertTrue(success)
        
        pool.cleanup()

    def test_cross_platform_compatibility(self):
        """Test that the implementation works across platforms"""
        pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB
        self.assertTrue(os.path.exists(pool.temp_file_path))
        pool.cleanup()
        # After cleanup, the temp file should be deleted
        self.assertFalse(os.path.exists(pool.temp_file_path))

    def test_memory_pool_expansion(self):
        """Test that the memory pool expands when needed"""
        pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB
        
        # Allocate a large block that should trigger expansion
        large_size = 2*1024*1024  # 2MB, larger than initial size
        result = pool.allocate(large_size)
        self.assertIsNotNone(result)
        
        ptr, allocated_size = result
        self.assertGreater(ptr, 0)
        self.assertGreater(allocated_size, 0)
        
        pool.cleanup()

    def test_memory_leak_prevention(self):
        """Test that memory is properly managed to prevent leaks"""
        initial_temp_files = set(os.listdir(tempfile.gettempdir()))
        
        # Create and destroy multiple pools
        for i in range(5):
            pool = AdvancedMemoryPool(initial_size=1024*1024)
            pool.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Check that no temporary files remain
        final_temp_files = set(os.listdir(tempfile.gettempdir()))
        remaining_files = final_temp_files - initial_temp_files
        
        # Filter out files that might be created by other processes
        temp_files_from_our_pools = [f for f in remaining_files if 'tmp' in f and f not in initial_temp_files]
        self.assertEqual(len(temp_files_from_our_pools), 0, 
                         f"Memory leak detected: temporary files not cleaned up: {temp_files_from_our_pools}")

    def test_cleanup_method(self):
        """Test that the cleanup method properly releases resources"""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        temp_file_path = pool.temp_file_path
        
        # Verify temp file exists
        self.assertTrue(os.path.exists(temp_file_path))
        
        # Perform cleanup
        pool.cleanup()
        
        # Verify temp file is deleted
        self.assertFalse(os.path.exists(temp_file_path))
        
        # Verify mmap is closed
        self.assertIsNone(pool.pool_ptr)

    def test_proper_temp_file_deletion_on_windows(self):
        """Test that temporary files are properly deleted on Windows"""
        pool = AdvancedMemoryPool(initial_size=1024*1024)
        temp_file_path = pool.temp_file_path
        
        # Verify temp file exists
        self.assertTrue(os.path.exists(temp_file_path))
        
        # Perform cleanup
        pool.cleanup()
        
        # Verify temp file is deleted
        self.assertFalse(os.path.exists(temp_file_path))

    def test_reference_counting(self):
        """Test reference counting functionality"""
        pool = AdvancedMemoryPool(initial_size=2*1024*1024)
        size = 1024
        
        # Allocate a block
        result = pool.allocate(size)
        self.assertIsNotNone(result)
        ptr, _ = result
        
        # Get the block to check reference count
        block = pool.block_map[ptr]
        self.assertEqual(block.ref_count, 1)
        
        # Simulate multiple references by manually increasing count
        block.ref_count = 2
        self.assertEqual(block.ref_count, 2)
        
        # Deallocate once - should decrease ref count but not free
        success = pool.deallocate(ptr)
        self.assertTrue(success)
        self.assertEqual(block.ref_count, 1)
        self.assertTrue(block.allocated)  # Still allocated due to ref count
        
        # Deallocate again - should now free the block
        success = pool.deallocate(ptr)
        self.assertTrue(success)
        self.assertEqual(block.ref_count, 0)
        self.assertFalse(block.allocated)
        
        pool.cleanup()

    def test_fragmentation_calculation(self):
        """Test fragmentation calculation"""
        pool = AdvancedMemoryPool(initial_size=4*1024*1024)  # 4MB
        
        # Allocate and deallocate blocks to create fragmentation
        blocks = []
        for i in range(5):
            result = pool.allocate(512*1024)  # 512KB each
            if result:
                blocks.append(result[0])
        
        # Deallocate alternating blocks to create fragmentation
        for i in range(0, len(blocks), 2):
            pool.deallocate(blocks[i])
        
        # Calculate fragmentation
        fragmentation = pool.defragmenter.calculate_fragmentation()
        self.assertGreaterEqual(fragmentation, 0.0)
        self.assertLessEqual(fragmentation, 1.0)
        
        pool.cleanup()


class TestMemoryDefragmenter(unittest.TestCase):
    """Test the MemoryDefragmenter class"""

    def test_compact_memory(self):
        """Test memory compaction functionality"""
        pool = AdvancedMemoryPool(initial_size=4*1024*1024)  # 4MB
        
        # Allocate and deallocate blocks to create fragmented state
        blocks = []
        for i in range(5):
            result = pool.allocate(512*1024)  # 512KB each
            if result:
                blocks.append(result[0])
        
        # Deallocate alternating blocks to create fragmentation
        for i in range(0, len(blocks), 2):
            pool.deallocate(blocks[i])
        
        # Compact the memory
        pool.defragmenter.compact_memory()
        
        # Check that fragmentation is reduced
        new_fragmentation = pool.defragmenter.calculate_fragmentation()
        self.assertLessEqual(new_fragmentation, 0.5)  # Should be reduced
        
        pool.cleanup()

    def test_should_defragment(self):
        """Test the should_defragment method"""
        pool = AdvancedMemoryPool(initial_size=4*1024*1024)  # 4MB
        
        # Initially, fragmentation should be low
        self.assertFalse(pool.defragmenter.should_defragment())
        
        # Create fragmentation
        blocks = []
        for i in range(5):
            result = pool.allocate(512*1024)  # 512KB each
            if result:
                blocks.append(result[0])
        
        # Deallocate alternating blocks to create high fragmentation
        for i in range(0, len(blocks), 2):
            pool.deallocate(blocks[i])
        
        # Now defragmentation should be needed
        self.assertTrue(pool.defragmenter.should_defragment())
        
        pool.cleanup()


class TestVisionLanguageMemoryOptimizer(unittest.TestCase):
    """Test the VisionLanguageMemoryOptimizer class"""

    def test_tensor_allocation_and_cleanup(self):
        """Test tensor allocation and cleanup"""
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4*1024*1024,  # 4MB
            enable_memory_pool=True
        )
        
        # Allocate a tensor
        tensor = optimizer.allocate_tensor_memory((100, 100), dtype=np.float32, tensor_type="general")
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (100, 100))
        
        # Free the tensor
        optimizer.free_tensor_memory(tensor)
        
        optimizer.cleanup()

    def test_different_tensor_types(self):
        """Test allocation for different tensor types"""
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,  # 8MB
            enable_memory_pool=True
        )
        
        # Test different tensor types
        shapes = {
            "kv_cache": (10, 100, 512),
            "image_features": (5, 196, 768),
            "text_embeddings": (8, 128, 512),
            "general": (20, 256)
        }
        
        tensors = {}
        for tensor_type, shape in shapes.items():
            tensor = optimizer.allocate_tensor_memory(shape, dtype=np.float32, tensor_type=tensor_type)
            self.assertIsNotNone(tensor)
            tensors[tensor_type] = tensor
        
        # Free all tensors
        for tensor_type, tensor in tensors.items():
            optimizer.free_tensor_memory(tensor, tensor_type)
        
        optimizer.cleanup()

    def test_attention_memory_optimization(self):
        """Test attention memory optimization"""
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=16*1024*1024,  # 16MB
            enable_memory_pool=True
        )
        
        # Test attention memory optimization
        components = optimizer.optimize_attention_memory(
            batch_size=2,
            seq_len=512,
            hidden_dim=768,
            num_heads=12
        )
        
        # Check that all components are created
        expected_keys = ['query', 'key', 'value', 'attention_scores', 'head_dim']
        for key in expected_keys:
            self.assertIn(key, components)
        
        # Check that tensor shapes are correct
        self.assertEqual(components['query'].shape, (2, 512, 768))
        self.assertEqual(components['key'].shape, (2, 512, 768))
        self.assertEqual(components['value'].shape, (2, 512, 768))
        self.assertEqual(components['attention_scores'].shape, (2, 12, 512, 512))
        self.assertEqual(components['head_dim'], 768 // 12)  # 64
        
        optimizer.cleanup()

    def test_memory_stats(self):
        """Test memory statistics"""
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4*1024*1024,  # 4MB
            enable_memory_pool=True
        )
        
        # Get initial stats
        initial_stats = optimizer.get_memory_stats()
        
        # Allocate some memory
        tensor = optimizer.allocate_tensor_memory((100, 100), dtype=np.float32, tensor_type="general")
        
        # Get stats after allocation
        final_stats = optimizer.get_memory_stats()
        
        # Check that stats reflect the allocation
        if 'general_pool' in final_stats:
            self.assertGreater(final_stats['general_pool']['current_usage'], 
                             initial_stats.get('general_pool', {}).get('current_usage', 0))
        
        optimizer.cleanup()


class TestCacheAwareMemoryManager(unittest.TestCase):
    """Test the CacheAwareMemoryManager class"""

    def test_memory_layout_optimization(self):
        """Test memory layout optimization"""
        manager = CacheAwareMemoryManager()
        
        # Test cache-friendly layout
        data = np.random.random((100, 100)).astype(np.float32)
        optimized = manager.optimize_memory_layout(data, "cache_friendly")
        
        # Should be contiguous
        self.assertTrue(optimized.flags.c_contiguous)
        
        # Test blocked layout
        blocked = manager.optimize_memory_layout(data, "blocked")
        self.assertEqual(blocked.shape, data.shape)


class TestGPUCPUMemoryOptimizer(unittest.TestCase):
    """Test the GPUCPUMemoryOptimizer class"""

    def test_initialization(self):
        """Test GPU-CPU memory optimizer initialization"""
        optimizer = GPUCPUMemoryOptimizer(device_memory_limit=1024*1024*1024)  # 1GB
        self.assertEqual(optimizer.device_memory_limit, 1024*1024*1024)
        optimizer = None  # This should not cause issues


class TestIntegration(unittest.TestCase):
    """Test integration between different components"""

    def test_full_pipeline(self):
        """Test the full memory optimization pipeline"""
        # Create optimizer
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=8*1024*1024,  # 8MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True
        )
        
        # Create sample image batch
        image_batch = np.random.random((4, 224, 224, 3)).astype(np.float32)
        
        # Optimize image processing memory
        optimized_batch = optimizer.optimize_image_processing_memory(image_batch)
        self.assertEqual(optimized_batch.shape, image_batch.shape)
        
        # Allocate tensor memory
        tensor = optimizer.allocate_tensor_memory((100, 256), dtype=np.float32, tensor_type="general")
        self.assertIsNotNone(tensor)
        
        # Get memory stats
        stats = optimizer.get_memory_stats()
        self.assertIsInstance(stats, dict)
        
        # Cleanup
        optimizer.cleanup()


if __name__ == '__main__':
    print("Running Advanced Memory Management System Tests...")
    print("=" * 60)
    
    # Run all tests
    unittest.main(verbosity=2)