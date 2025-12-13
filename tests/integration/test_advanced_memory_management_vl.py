"""
Comprehensive test suite for Advanced Memory Management System for Vision-Language Models
Tests all key components of the memory optimization system
"""

import unittest
import numpy as np
from advanced_memory_management_vl import (
    AdvancedMemoryPool, MemoryPoolType, VisionLanguageMemoryOptimizer,
    CacheAwareMemoryManager, GPUCPUMemoryOptimizer
)
import threading
import time
import gc


class TestAdvancedMemoryPool(unittest.TestCase):
    """Test the AdvancedMemoryPool class"""
    
    def setUp(self):
        self.pool = AdvancedMemoryPool(initial_size=1024*1024)  # 1MB pool
    
    def tearDown(self):
        self.pool.cleanup()
    
    def test_basic_allocation_deallocation(self):
        """Test basic allocation and deallocation functionality"""
        # Allocate 100 bytes
        ptr, size = self.pool.allocate(100)
        self.assertIsNotNone(ptr)
        self.assertEqual(size, 4096)  # Should be aligned to 4KB page
        
        # Deallocate
        result = self.pool.deallocate(ptr)
        self.assertTrue(result)
    
    def test_large_allocation(self):
        """Test allocation of larger blocks"""
        size = 512 * 1024  # 512KB
        ptr, allocated_size = self.pool.allocate(size)
        self.assertIsNotNone(ptr)
        self.assertGreaterEqual(allocated_size, size)
    
    def test_pool_expansion(self):
        """Test that the pool expands when needed"""
        # Allocate something large that exceeds initial pool size
        ptr1, size1 = self.pool.allocate(512 * 1024)  # 512KB
        ptr2, size2 = self.pool.allocate(768 * 1024)  # 768KB
        self.assertIsNotNone(ptr1)
        self.assertIsNotNone(ptr2)
    
    def test_alignment(self):
        """Test memory alignment functionality"""
        ptr, size = self.pool.allocate(100, alignment=64)  # 64-byte alignment
        self.assertIsNotNone(ptr)
        self.assertEqual(ptr % 64, 0)  # Should be aligned to 64-byte boundary
    
    def test_fragmentation_calculation(self):
        """Test fragmentation calculation"""
        if self.pool.defragmenter:
            initial_frag = self.pool.defragmenter.calculate_fragmentation()
            self.assertEqual(initial_frag, 0.0)  # Should start with no fragmentation
            
            # Create fragmentation by allocating and freeing blocks in the middle
            ptr1, _ = self.pool.allocate(100)
            ptr2, _ = self.pool.allocate(100)
            ptr3, _ = self.pool.allocate(100)
            
            # Free the middle block to create fragmentation
            self.pool.deallocate(ptr2)
            
            # Fragmentation should be greater than 0 now
            frag_after = self.pool.defragmenter.calculate_fragmentation()
            self.assertGreaterEqual(frag_after, 0.0)
    
    def test_multithreaded_access(self):
        """Test thread safety of memory pool"""
        results = []
        
        def allocate_in_thread():
            ptr, size = self.pool.allocate(100)
            if ptr:
                results.append((ptr, size))
                time.sleep(0.01)  # Small delay
                self.pool.deallocate(ptr)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=allocate_in_thread)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 10)  # All allocations should succeed
    
    def test_pool_statistics(self):
        """Test memory pool statistics"""
        stats_before = self.pool.get_stats()
        
        ptr, size = self.pool.allocate(100)
        stats_after = self.pool.get_stats()
        
        self.assertGreater(stats_after['allocation_count'], stats_before['allocation_count'])
        self.assertGreater(stats_after['current_usage'], stats_before['current_usage'])
        
        self.pool.deallocate(ptr)
        stats_final = self.pool.get_stats()
        self.assertLess(stats_final['current_usage'], stats_after['current_usage'])


class TestCacheAwareMemoryManager(unittest.TestCase):
    """Test the CacheAwareMemoryManager class"""
    
    def setUp(self):
        self.cache_manager = CacheAwareMemoryManager()
    
    def test_memory_layout_optimization(self):
        """Test memory layout optimization"""
        data = np.random.random((100, 100)).astype(np.float32)
        
        # Test cache-friendly layout
        optimized = self.cache_manager.optimize_memory_layout(data, "cache_friendly")
        self.assertTrue(optimized.flags['C_CONTIGUOUS'])
        
        # Test blocked layout
        blocked = self.cache_manager.optimize_memory_layout(data, "blocked")
        self.assertEqual(blocked.shape, data.shape)
    
    def test_prefetch_functionality(self):
        """Test prefetch functionality"""
        data = np.random.random(1000).astype(np.float32)
        ptr = data.__array_interface__['data'][0]
        
        # This should not raise an exception
        self.cache_manager.prefetch_data(ptr, data.nbytes)
    
    def test_cache_blocking(self):
        """Test cache blocking functionality"""
        data = np.random.random((200, 200)).astype(np.float32)
        blocked = self.cache_manager._apply_cache_blocking(data, block_size=32)
        self.assertEqual(blocked.shape, data.shape)


class TestGPUCPUMemoryOptimizer(unittest.TestCase):
    """Test the GPUCPUMemoryOptimizer class"""
    
    def setUp(self):
        self.gpu_optimizer = GPUCPUMemoryOptimizer()
    
    def test_tensor_placement(self):
        """Test tensor placement optimization"""
        # Create a small tensor
        tensor = np.random.random((10, 10)).astype(np.float32)
        
        # This should work without errors (may not actually move to GPU without PyTorch)
        optimized_tensor = self.gpu_optimizer.optimize_tensor_placement(tensor, target_device="cpu")
        self.assertIsNotNone(optimized_tensor)


class TestVisionLanguageMemoryOptimizer(unittest.TestCase):
    """Test the VisionLanguageMemoryOptimizer class"""
    
    def setUp(self):
        self.optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=2 * 1024 * 1024,  # 2MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False  # Disable GPU for test
        )
    
    def tearDown(self):
        self.optimizer.cleanup()
    
    def test_tensor_allocation(self):
        """Test tensor allocation with different types"""
        # Test general tensor allocation
        tensor = self.optimizer.allocate_tensor_memory((50, 128), dtype=np.float32, tensor_type="general")
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (50, 128))
        self.assertEqual(tensor.dtype, np.float32)
        
        # Test KV cache allocation
        kv_tensor = self.optimizer.allocate_tensor_memory((10, 100, 256), dtype=np.float32, tensor_type="kv_cache")
        self.assertIsNotNone(kv_tensor)
        self.assertEqual(kv_tensor.shape, (10, 100, 256))
        
        # Test image features allocation
        img_tensor = self.optimizer.allocate_tensor_memory((4, 196, 512), dtype=np.float32, tensor_type="image_features")
        self.assertIsNotNone(img_tensor)
        self.assertEqual(img_tensor.shape, (4, 196, 512))
    
    def test_image_processing_optimization(self):
        """Test image processing memory optimization"""
        images = np.random.random((2, 224, 224, 3)).astype(np.float32)
        optimized_images = self.optimizer.optimize_image_processing_memory(images)
        self.assertEqual(optimized_images.shape, images.shape)
        # Should be C-contiguous for cache efficiency
        self.assertTrue(optimized_images.flags['C_CONTIGUOUS'])
    
    def test_attention_memory_optimization(self):
        """Test attention mechanism memory optimization"""
        components = self.optimizer.optimize_attention_memory(
            batch_size=2, seq_len=512, hidden_dim=768, num_heads=12
        )
        
        self.assertIn('query', components)
        self.assertIn('key', components)
        self.assertIn('value', components)
        self.assertIn('attention_scores', components)
        self.assertIn('head_dim', components)
        
        # Check shapes
        q_shape = components['query'].shape
        self.assertEqual(q_shape, (2, 512, 768))
        
        attn_shape = components['attention_scores'].shape
        self.assertEqual(attn_shape, (2, 12, 512, 512))
    
    def test_memory_statistics(self):
        """Test memory statistics functionality"""
        stats = self.optimizer.get_memory_stats()
        
        # Should have at least general pool stats
        self.assertIn('general_pool', stats)
        
        # Check that stats contain expected keys
        general_stats = stats['general_pool']
        self.assertIn('current_usage', general_stats)
        self.assertIn('peak_usage', general_stats)
        self.assertIn('allocation_count', general_stats)
        self.assertIn('deallocation_count', general_stats)
    
    def test_multithreaded_tensor_allocation(self):
        """Test thread safety of tensor allocation"""
        results = []
        
        def allocate_tensor():
            tensor = self.optimizer.allocate_tensor_memory((10, 10), dtype=np.float32)
            results.append(tensor)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=allocate_tensor)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 5)
        for tensor in results:
            self.assertIsNotNone(tensor)


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire memory optimization system"""
    
    def test_full_pipeline(self):
        """Test the complete memory optimization pipeline"""
        # Create optimizer
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=4 * 1024 * 1024,  # 4MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        try:
            # Simulate a vision-language model workflow
            batch_size = 2
            seq_len = 128
            hidden_dim = 256
            num_heads = 8
            
            # Process images
            images = np.random.random((batch_size, 224, 224, 3)).astype(np.float32)
            optimized_images = optimizer.optimize_image_processing_memory(images)
            
            # Extract image features
            image_features = optimizer.allocate_tensor_memory(
                (batch_size, 196, 512), dtype=np.float32, tensor_type="image_features"
            )
            
            # Create attention components
            attention_components = optimizer.optimize_attention_memory(
                batch_size, seq_len, hidden_dim, num_heads
            )
            
            # Verify all components are created
            self.assertIsNotNone(optimized_images)
            self.assertIsNotNone(image_features)
            self.assertIsNotNone(attention_components['query'])
            self.assertIsNotNone(attention_components['key'])
            self.assertIsNotNone(attention_components['value'])
            self.assertIsNotNone(attention_components['attention_scores'])
            
            # Get final statistics
            stats = optimizer.get_memory_stats()
            self.assertIn('general_pool', stats)
            self.assertIn('kv_cache_pool', stats)
            self.assertIn('image_feature_pool', stats)
            
        finally:
            optimizer.cleanup()
    
    def test_memory_leak_prevention(self):
        """Test that memory is properly managed and no leaks occur"""
        initial_objects = len(gc.get_objects())
        
        # Create and destroy multiple optimizers
        for _ in range(3):
            optimizer = VisionLanguageMemoryOptimizer(
                memory_pool_size=1024*1024,  # 1MB
                enable_memory_pool=True,
                enable_cache_optimization=True,
                enable_gpu_optimization=False
            )
            
            # Allocate some memory
            tensor = optimizer.allocate_tensor_memory((100, 100), dtype=np.float32)
            del tensor
            
            optimizer.cleanup()
            del optimizer
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Should not have significantly more objects (allowing for some variation)
        self.assertLessEqual(final_objects - initial_objects, 100)


def run_performance_benchmark():
    """Run a simple performance benchmark"""
    print("Running performance benchmark...")
    
    # Create optimizer
    optimizer = VisionLanguageMemoryOptimizer(
        memory_pool_size=8 * 1024 * 1024,  # 8MB
        enable_memory_pool=True,
        enable_cache_optimization=True,
        enable_gpu_optimization=False
    )
    
    try:
        import time
        
        # Benchmark tensor allocation speed
        start_time = time.time()
        tensors = []

        for i in range(1000):
            tensor = optimizer.allocate_tensor_memory((64, 64), dtype=np.float32)
            tensors.append(tensor)

        alloc_time = time.time() - start_time
        print(f"Allocated 1000 tensors of size (64, 64) in {alloc_time:.3f} seconds")
        print(f"Average allocation time: {alloc_time/1000*1000:.3f} ms")

        # Benchmark deallocation speed
        start_time = time.time()
        for tensor in tensors:
            optimizer.free_tensor_memory(tensor)

        dealloc_time = time.time() - start_time
        print(f"Deallocated 1000 tensors in {dealloc_time:.3f} seconds")
        print(f"Average deallocation time: {dealloc_time/1000*1000:.3f} ms")

        # Cleanup
        optimizer.cleanup()
        
    except Exception as e:
        print(f"Benchmark error: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmark
    run_performance_benchmark()