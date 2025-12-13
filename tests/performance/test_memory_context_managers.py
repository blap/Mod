"""
Comprehensive tests for the Memory Context Manager System
"""

import unittest
import threading
import time
from unittest.mock import Mock, patch
import numpy as np

# Import our context managers
from memory_context_managers import (
    MemoryResourceContextManager,
    KVCacheContextManager,
    ImageFeaturesContextManager,
    TextEmbeddingsContextManager,
    AdvancedMemoryPoolContextManager,
    VisionLanguageMemoryOptimizerContextManager,
    memory_resource_context,
    kv_cache_context,
    image_features_context,
    text_embeddings_context,
    memory_pool_context,
    vl_tensor_context,
    cleanup_all_resources,
    ResourceState,
    resource_tracker
)

# Import from existing modules
from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType
from advanced_memory_management_vl import (
    AdvancedMemoryPool,
    MemoryPoolType,
    VisionLanguageMemoryOptimizer
)


class TestMemoryResourceContextManager(unittest.TestCase):
    """Test the general-purpose MemoryResourceContextManager"""
    
    def setUp(self):
        self.resource_id = "test_resource_1"
        self.allocated_value = "allocated_resource"
        
        def mock_allocator():
            return self.allocated_value
        
        def mock_deallocator(value):
            self.deallocated_value = value
            return True
        
        self.allocator = mock_allocator
        self.deallocator = mock_deallocator
        self.ctx = MemoryResourceContextManager(
            self.allocator,
            self.deallocator,
            self.resource_id
        )
    
    def test_basic_allocation_and_deallocation(self):
        """Test basic allocation and deallocation"""
        with self.ctx as resource:
            self.assertEqual(resource, self.allocated_value)
            self.assertEqual(self.ctx.allocated_resource, self.allocated_value)
        
        # After exit, resource should be deallocated
        self.assertEqual(self.deallocated_value, self.allocated_value)
    
    def test_exception_handling_in_context(self):
        """Test that exceptions in context are properly handled"""
        def raising_allocator():
            return "allocated"
        
        def simple_deallocator(value):
            self.deallocated_value = value
            return True
        
        ctx = MemoryResourceContextManager(raising_allocator, simple_deallocator, "test_exc")
        
        with self.assertRaises(ValueError):
            with ctx as resource:
                raise ValueError("Test exception")
        
        # Resource should still be deallocated despite exception
        self.assertEqual(self.deallocated_value, "allocated")
    
    def test_double_entry_prevention(self):
        """Test that context manager prevents double entry"""
        with self.ctx:
            with self.assertRaises(RuntimeError):
                with self.ctx:
                    pass  # This should raise an error
    
    def test_context_manager_factory(self):
        """Test the context manager factory function"""
        def mock_alloc():
            return "factory_resource"
        
        def mock_dealloc(value):
            self.factory_deallocated = value
            return True
        
        with memory_resource_context(mock_alloc, mock_dealloc, "factory_test") as resource:
            self.assertEqual(resource, "factory_resource")
        
        self.assertEqual(self.factory_deallocated, "factory_resource")


class TestKVCacheContextManager(unittest.TestCase):
    """Test the KVCacheContextManager"""
    
    def setUp(self):
        self.memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024*10,  # 10MB
            min_block_size=256
        )
        self.size = 1024*1024  # 1MB
        self.tensor_id = "test_kv_tensor"
    
    def test_kv_cache_allocation_and_deallocation(self):
        """Test basic KV cache allocation and deallocation"""
        with kv_cache_context(self.memory_system, self.size, self.tensor_id) as block:
            self.assertIsNotNone(block)
            self.assertEqual(block.size, 1024*1024)  # Size should be rounded up to power of 2
        
        # After exit, block should be deallocated
        pool_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertEqual(pool_stats['active_allocations'], 0)
    
    def test_kv_cache_exception_handling(self):
        """Test KV cache context manager handles exceptions properly"""
        try:
            with kv_cache_context(self.memory_system, self.size, self.tensor_id) as block:
                self.assertIsNotNone(block)
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected
        
        # Block should still be deallocated despite exception
        pool_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertEqual(pool_stats['active_allocations'], 0)
    
    def test_direct_context_manager_usage(self):
        """Test using the KVCacheContextManager directly"""
        ctx = KVCacheContextManager(self.memory_system, self.size, self.tensor_id)
        
        with ctx as block:
            self.assertIsNotNone(block)
            pool_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
            self.assertEqual(pool_stats['active_allocations'], 1)
        
        # After exit, block should be deallocated
        pool_stats = self.memory_system.get_pool_stats(TensorType.KV_CACHE)
        self.assertEqual(pool_stats['active_allocations'], 0)


class TestImageFeaturesContextManager(unittest.TestCase):
    """Test the ImageFeaturesContextManager"""
    
    def setUp(self):
        self.memory_system = AdvancedMemoryPoolingSystem(
            image_features_size=1024*1024*20,  # 20MB
            min_block_size=256
        )
        self.size = 2 * 1024*1024  # 2MB
        self.tensor_id = "test_img_tensor"
    
    def test_image_features_allocation_and_deallocation(self):
        """Test basic image features allocation and deallocation"""
        with image_features_context(self.memory_system, self.size, self.tensor_id) as block:
            self.assertIsNotNone(block)
            self.assertEqual(block.size, 2 * 1024*1024)  # Size should be rounded up to power of 2
        
        # After exit, block should be deallocated
        pool_stats = self.memory_system.get_pool_stats(TensorType.IMAGE_FEATURES)
        self.assertEqual(pool_stats['active_allocations'], 0)
    
    def test_image_features_exception_handling(self):
        """Test image features context manager handles exceptions properly"""
        try:
            with image_features_context(self.memory_system, self.size, self.tensor_id) as block:
                self.assertIsNotNone(block)
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected
        
        # Block should still be deallocated despite exception
        pool_stats = self.memory_system.get_pool_stats(TensorType.IMAGE_FEATURES)
        self.assertEqual(pool_stats['active_allocations'], 0)


class TestTextEmbeddingsContextManager(unittest.TestCase):
    """Test the TextEmbeddingsContextManager"""
    
    def setUp(self):
        self.memory_system = AdvancedMemoryPoolingSystem(
            text_embeddings_size=1024*1024*5,  # 5MB
            min_block_size=256
        )
        self.size = 512*1024  # 512KB
        self.tensor_id = "test_text_tensor"
    
    def test_text_embeddings_allocation_and_deallocation(self):
        """Test basic text embeddings allocation and deallocation"""
        with text_embeddings_context(self.memory_system, self.size, self.tensor_id) as block:
            self.assertIsNotNone(block)
            self.assertEqual(block.size, 512*1024)  # Size should be rounded up to power of 2
        
        # After exit, block should be deallocated
        pool_stats = self.memory_system.get_pool_stats(TensorType.TEXT_EMBEDDINGS)
        self.assertEqual(pool_stats['active_allocations'], 0)
    
    def test_text_embeddings_exception_handling(self):
        """Test text embeddings context manager handles exceptions properly"""
        try:
            with text_embeddings_context(self.memory_system, self.size, self.tensor_id) as block:
                self.assertIsNotNone(block)
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected
        
        # Block should still be deallocated despite exception
        pool_stats = self.memory_system.get_pool_stats(TensorType.TEXT_EMBEDDINGS)
        self.assertEqual(pool_stats['active_allocations'], 0)


class TestAdvancedMemoryPoolContextManager(unittest.TestCase):
    """Test the AdvancedMemoryPoolContextManager"""
    
    def setUp(self):
        self.memory_pool = AdvancedMemoryPool(
            initial_size=1024*1024*10,  # 10MB
            page_size=256
        )
        self.size = 1024*512  # 512KB
    
    def test_memory_pool_allocation_and_deallocation(self):
        """Test basic memory pool allocation and deallocation"""
        with memory_pool_context(self.memory_pool, self.size) as (ptr, actual_size):
            self.assertIsNotNone(ptr)
            self.assertGreater(actual_size, 0)
        
        # The deallocation happens automatically, so we can't directly verify
        # but we can check that the context manager completed without error
    
    def test_memory_pool_exception_handling(self):
        """Test memory pool context manager handles exceptions properly"""
        try:
            with memory_pool_context(self.memory_pool, self.size) as (ptr, actual_size):
                self.assertIsNotNone(ptr)
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected
    
    def test_direct_memory_pool_context_manager_usage(self):
        """Test using the AdvancedMemoryPoolContextManager directly"""
        ctx = AdvancedMemoryPoolContextManager(self.memory_pool, self.size)
        
        with ctx as (ptr, actual_size):
            self.assertIsNotNone(ptr)
            self.assertGreater(actual_size, 0)
        
        # Context manager should have handled deallocation


class TestVisionLanguageMemoryOptimizerContextManager(unittest.TestCase):
    """Test the VisionLanguageMemoryOptimizerContextManager"""
    
    def setUp(self):
        self.optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=1024*1024*20,  # 20MB
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False  # Disable GPU to avoid torch dependency
        )
        self.shape = (100, 256)
        self.dtype = np.float32
    
    def test_vl_tensor_allocation_and_deallocation(self):
        """Test basic VL tensor allocation and deallocation"""
        with vl_tensor_context(self.optimizer, self.shape, self.dtype) as tensor:
            self.assertIsNotNone(tensor)
            self.assertEqual(tensor.shape, self.shape)
            self.assertEqual(tensor.dtype, self.dtype)
        
        # Context manager should handle cleanup automatically
    
    def test_vl_tensor_exception_handling(self):
        """Test VL tensor context manager handles exceptions properly"""
        try:
            with vl_tensor_context(self.optimizer, self.shape, self.dtype) as tensor:
                self.assertIsNotNone(tensor)
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected
    
    def test_different_tensor_types(self):
        """Test VL tensor context with different tensor types"""
        tensor_types = ["general", "kv_cache", "image_features", "text_embeddings"]
        
        for tensor_type in tensor_types:
            with vl_tensor_context(self.optimizer, self.shape, self.dtype, tensor_type) as tensor:
                self.assertIsNotNone(tensor)
                self.assertEqual(tensor.shape, self.shape)
                self.assertEqual(tensor.dtype, self.dtype)


class TestResourceTracker(unittest.TestCase):
    """Test the ResourceTracker functionality"""
    
    def setUp(self):
        # Clear all resources before each test
        cleanup_all_resources()
    
    def tearDown(self):
        # Clear all resources after each test
        cleanup_all_resources()
    
    def test_resource_tracking(self):
        """Test that resources are properly tracked"""
        # Initially no resources
        self.assertEqual(len(resource_tracker.get_active_resources()), 0)
        
        # Create and enter a context manager
        memory_system = AdvancedMemoryPoolingSystem(kv_cache_size=1024*1024*5)
        with kv_cache_context(memory_system, 1024*512, "tracked_tensor") as block:
            active_resources = resource_tracker.get_active_resources()
            self.assertEqual(len(active_resources), 1)
            resource_id = f"kv_cache_tracked_tensor"
            self.assertIn(resource_id, active_resources)
            self.assertEqual(active_resources[resource_id].state, ResourceState.ALLOCATED)
        
        # After exit, no active resources
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 0)
    
    def test_forced_cleanup(self):
        """Test that forced cleanup works properly"""
        # Create a context manager but don't exit it properly (simulate crash)
        memory_system = AdvancedMemoryPoolingSystem(kv_cache_size=1024*1024*5)
        ctx = KVCacheContextManager(memory_system, 1024*512, "cleanup_tensor")
        
        # Enter the context
        block = ctx.__enter__()
        self.assertIsNotNone(block)
        
        # Verify resource is tracked
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 1)
        
        # Force cleanup
        cleaned_count = cleanup_all_resources()
        self.assertEqual(cleaned_count, 1)
        
        # No resources should remain
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 0)
    
    def test_multiple_resources_tracking(self):
        """Test tracking of multiple resources simultaneously"""
        memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024*5,
            image_features_size=1024*1024*5
        )
        
        # Allocate multiple resources
        with kv_cache_context(memory_system, 1024*256, "tensor1") as block1:
            with image_features_context(memory_system, 1024*256, "tensor2") as block2:
                active_resources = resource_tracker.get_active_resources()
                self.assertEqual(len(active_resources), 2)
                self.assertIn("kv_cache_tensor1", active_resources)
                self.assertIn("image_features_tensor2", active_resources)
        
        # After all contexts exit, no active resources
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 0)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of context managers and resource tracker"""
    
    def setUp(self):
        # Clear all resources before each test
        cleanup_all_resources()
    
    def tearDown(self):
        # Clear all resources after each test
        cleanup_all_resources()
    
    def test_concurrent_context_managers(self):
        """Test that multiple threads can use context managers safely"""
        results = []
        
        def worker(worker_id):
            memory_system = AdvancedMemoryPoolingSystem(kv_cache_size=1024*1024*10)
            try:
                with kv_cache_context(memory_system, 1024*256, f"thread_{worker_id}") as block:
                    time.sleep(0.01)  # Simulate some work
                    results.append(f"worker_{worker_id}_success")
            except Exception as e:
                results.append(f"worker_{worker_id}_error: {str(e)}")
        
        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all workers succeeded
        success_count = sum(1 for r in results if "success" in r)
        self.assertEqual(success_count, 5)
    
    def test_resource_tracker_thread_safety(self):
        """Test that resource tracker is thread-safe"""
        def track_resources(worker_id):
            memory_system = AdvancedMemoryPoolingSystem(kv_cache_size=1024*1024*5)
            with kv_cache_context(memory_system, 1024*128, f"tracked_{worker_id}") as block:
                time.sleep(0.01)  # Simulate work
                return f"tracked_{worker_id}"
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=track_resources, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all resources were properly tracked and cleaned up
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple context managers"""
    
    def setUp(self):
        # Clear all resources before each test
        cleanup_all_resources()
    
    def tearDown(self):
        # Clear all resources after each test
        cleanup_all_resources()
    
    def test_complex_memory_workflow(self):
        """Test a complex workflow with multiple context managers"""
        memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024*10,
            image_features_size=1024*1024*10,
            text_embeddings_size=1024*1024*5
        )
        
        optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=1024*1024*20,
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=False
        )
        
        # Simulate a complex memory workflow
        with kv_cache_context(memory_system, 1024*512, "query_tensor") as query_block:
            with image_features_context(memory_system, 2*1024*512, "img_features") as img_block:
                with text_embeddings_context(memory_system, 1024*256, "text_emb") as text_block:
                    # Allocate some optimized tensors
                    with vl_tensor_context(optimizer, (50, 128), np.float32, "general") as tensor1:
                        with vl_tensor_context(optimizer, (25, 64), np.float32, "kv_cache") as tensor2:
                            # Do some work
                            self.assertIsNotNone(query_block)
                            self.assertIsNotNone(img_block)
                            self.assertIsNotNone(text_block)
                            self.assertIsNotNone(tensor1)
                            self.assertIsNotNone(tensor2)
        
        # All resources should be cleaned up
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 0)
        
        # Verify pools are empty
        self.assertEqual(memory_system.get_pool_stats(TensorType.KV_CACHE)['active_allocations'], 0)
        self.assertEqual(memory_system.get_pool_stats(TensorType.IMAGE_FEATURES)['active_allocations'], 0)
        self.assertEqual(memory_system.get_pool_stats(TensorType.TEXT_EMBEDDINGS)['active_allocations'], 0)
    
    def test_exception_in_complex_workflow(self):
        """Test that exceptions in complex workflow are handled properly"""
        memory_system = AdvancedMemoryPoolingSystem(
            kv_cache_size=1024*1024*10,
            image_features_size=1024*1024*10,
            text_embeddings_size=1024*1024*5
        )
        
        try:
            with kv_cache_context(memory_system, 1024*512, "query_tensor") as query_block:
                with image_features_context(memory_system, 2*1024*512, "img_features") as img_block:
                    with text_embeddings_context(memory_system, 1024*256, "text_emb") as text_block:
                        raise RuntimeError("Simulated error in workflow")
        except RuntimeError:
            pass  # Expected
        
        # All resources should still be cleaned up despite exception
        active_resources = resource_tracker.get_active_resources()
        self.assertEqual(len(active_resources), 0)
        
        # Verify pools are empty
        self.assertEqual(memory_system.get_pool_stats(TensorType.KV_CACHE)['active_allocations'], 0)
        self.assertEqual(memory_system.get_pool_stats(TensorType.IMAGE_FEATURES)['active_allocations'], 0)
        self.assertEqual(memory_system.get_pool_stats(TensorType.TEXT_EMBEDDINGS)['active_allocations'], 0)


if __name__ == '__main__':
    unittest.main()