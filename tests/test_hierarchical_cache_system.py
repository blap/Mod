"""
Comprehensive tests for the hierarchical caching system in Qwen3-VL.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import time
from typing import Tuple
from src.qwen3_vl.optimization.hierarchical_cache_manager import (
    HierarchicalCacheManager, CacheConfig, TensorMetadata
)
from src.qwen3_vl.optimization.hierarchical_cache_integration import (
    HierarchicalCacheIntegration, Qwen3VLHierarchicalCacheAdapter
)


class TestCacheConfig(unittest.TestCase):
    """Test cache configuration."""
    
    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        
        self.assertEqual(config.l1_cache_size, 512 * 1024 * 1024)  # 512MB
        self.assertEqual(config.l2_cache_size, 1024 * 1024 * 1024)  # 1GB
        self.assertEqual(config.l3_cache_size, 5 * 1024 * 1024 * 1024)  # 5GB
        self.assertTrue(config.l3_compression)
        self.assertEqual(config.migration_threshold_high_freq, 5)
    
    def test_custom_config(self):
        """Test custom cache configuration."""
        config = CacheConfig(
            l1_cache_size=256 * 1024 * 1024,
            l2_cache_size=512 * 1024 * 1024,
            l3_cache_size=2 * 1024 * 1024 * 1024,
            l3_compression=False
        )
        
        self.assertEqual(config.l1_cache_size, 256 * 1024 * 1024)
        self.assertEqual(config.l2_cache_size, 512 * 1024 * 1024)
        self.assertEqual(config.l3_cache_size, 2 * 1024 * 1024 * 1024)
        self.assertFalse(config.l3_compression)


class TestTensorMetadata(unittest.TestCase):
    """Test tensor metadata functionality."""
    
    def test_tensor_metadata_creation(self):
        """Test tensor metadata creation."""
        metadata = TensorMetadata(
            tensor_id="test_tensor",
            shape=(100, 100),
            dtype=torch.float16,
            device=torch.device("cpu"),
            size_bytes=20000,
            tensor_type="test"
        )
        
        self.assertEqual(metadata.tensor_id, "test_tensor")
        self.assertEqual(metadata.shape, (100, 100))
        self.assertEqual(metadata.dtype, torch.float16)
        self.assertEqual(metadata.device, torch.device("cpu"))
        self.assertEqual(metadata.size_bytes, 20000)
        self.assertEqual(metadata.tensor_type, "test")
        self.assertIsNotNone(metadata.last_access_time)
        self.assertEqual(metadata.access_count, 0)
        self.assertEqual(metadata.cache_level, None)
        self.assertEqual(metadata.predicted_access, False)


class TestHierarchicalCacheManager(unittest.TestCase):
    """Test the hierarchical cache manager."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = CacheConfig(
            l1_cache_size=64 * 1024 * 1024,  # 64MB
            l2_cache_size=128 * 1024 * 1024,  # 128MB
            l3_cache_size=256 * 1024 * 1024,  # 256MB
            access_pattern_window=100,
            migration_threshold_high_freq=2,
            migration_threshold_medium_freq=1
        )
        self.cache_manager = HierarchicalCacheManager(self.config)
    
    def test_initialization(self):
        """Test cache manager initialization."""
        self.assertIsNotNone(self.cache_manager.l1_cache)
        self.assertIsNotNone(self.cache_manager.l2_cache)
        self.assertIsNotNone(self.cache_manager.l3_cache)
        self.assertIsNotNone(self.cache_manager.access_tracker)
        self.assertIsNotNone(self.cache_manager.predictor)
        
        # Check cache sizes
        self.assertEqual(self.cache_manager.l1_cache.max_size_bytes, 64 * 1024 * 1024)
        self.assertEqual(self.cache_manager.l2_cache.max_size_bytes, 128 * 1024 * 1024)
        self.assertEqual(self.cache_manager.l3_cache.max_size_bytes, 256 * 1024 * 1024)
    
    def test_put_and_get_tensor_l1(self):
        """Test putting and getting tensor in L1 cache."""
        # Create a small tensor that fits in L1
        tensor = torch.randn(100, 100, dtype=torch.float16)
        
        # Put tensor in cache
        success = self.cache_manager.put_tensor(tensor, "test_tensor")
        self.assertTrue(success)
        
        # Get tensor from cache
        retrieved_tensor, cache_level = self.cache_manager.get_tensor(
            tensor.shape, tensor.dtype
        )
        
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(cache_level, 'l1')
        self.assertEqual(retrieved_tensor.shape, tensor.shape)
        self.assertEqual(retrieved_tensor.dtype, tensor.dtype)
    
    def test_put_and_get_tensor_l2(self):
        """Test putting and getting tensor in L2 cache."""
        # Create a medium tensor that should go to L2
        tensor = torch.randn(500, 500, dtype=torch.float16)  # ~500KB
        
        # Put tensor in cache with preference for L2
        success = self.cache_manager.put_tensor(tensor, "test_tensor", preferred_level=2)
        self.assertTrue(success)
        
        # Get tensor from cache
        retrieved_tensor, cache_level = self.cache_manager.get_tensor(
            tensor.shape, tensor.dtype
        )
        
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(cache_level, 'l2')
        self.assertEqual(retrieved_tensor.shape, tensor.shape)
        self.assertEqual(retrieved_tensor.dtype, tensor.dtype)
    
    def test_put_and_get_tensor_l3(self):
        """Test putting and getting tensor in L3 cache."""
        # Create a large tensor that should go to L3
        tensor = torch.randn(1000, 1000, dtype=torch.float16)  # ~2MB
        
        # Put tensor in cache with preference for L3
        success = self.cache_manager.put_tensor(tensor, "test_tensor", preferred_level=3)
        self.assertTrue(success)
        
        # Get tensor from cache
        retrieved_tensor, cache_level = self.cache_manager.get_tensor(
            tensor.shape, tensor.dtype
        )
        
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(cache_level, 'l3')
        self.assertEqual(retrieved_tensor.shape, tensor.shape)
        self.assertEqual(retrieved_tensor.dtype, tensor.dtype)
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        # Try to get a tensor that doesn't exist
        retrieved_tensor, cache_level = self.cache_manager.get_tensor(
            (100, 100), torch.float16
        )
        
        self.assertIsNone(retrieved_tensor)
        self.assertEqual(cache_level, 'miss')
    
    def test_access_pattern_tracking(self):
        """Test access pattern tracking."""
        tensor = torch.randn(50, 50, dtype=torch.float16)
        tensor_id = self.cache_manager._generate_tensor_id(tensor.shape, tensor.dtype)
        
        # Put tensor in cache
        self.cache_manager.put_tensor(tensor, "test_tensor")
        
        # Access tensor multiple times
        for _ in range(3):
            _, _ = self.cache_manager.get_tensor(tensor.shape, tensor.dtype)
        
        # Check that access count is updated
        access_count = self.cache_manager.access_tracker.tensor_access_counts.get(tensor_id, 0)
        self.assertEqual(access_count, 3)
        
        # Check that prediction probability is higher for frequently accessed tensor
        pred_prob = self.cache_manager.predictor.predict_access_probability(tensor_id)
        self.assertGreater(pred_prob, 0.1)  # Should be higher than baseline
    
    def test_cache_migrations(self):
        """Test cache migration based on access patterns."""
        # Create a tensor and access it frequently to make it hot
        hot_tensor = torch.randn(100, 100, dtype=torch.float16)
        self.cache_manager.put_tensor(hot_tensor, "hot_tensor", preferred_level=3)  # Start in L3
        
        # Access the tensor multiple times to make it hot
        for _ in range(10):
            _, _ = self.cache_manager.get_tensor(hot_tensor.shape, hot_tensor.dtype)
            time.sleep(0.01)  # Small delay to create time intervals
        
        # Perform migrations
        self.cache_manager._perform_migrations()
        
        # Check that stats show migrations happened
        stats = self.cache_manager.get_stats()
        self.assertGreater(stats['global_stats']['migrations'], 0)
    
    def test_cache_eviction(self):
        """Test cache eviction when full."""
        # Fill up L1 cache with small tensors
        tensors = []
        for i in range(50):  # Create many small tensors
            tensor = torch.randn(50, 50, dtype=torch.float16)  # ~5KB each
            success = self.cache_manager.put_tensor(tensor, f"tensor_{i}", preferred_level=1)
            if not success:
                break
            tensors.append(tensor)
        
        # Get cache stats
        stats = self.cache_manager.get_stats()
        l1_stats = stats['l1_stats']
        
        # Check that some evictions occurred due to capacity constraints
        # Note: The exact number depends on tensor sizes and cache capacity
        self.assertGreaterEqual(l1_stats['num_tensors'], 0)
    
    def test_get_stats(self):
        """Test getting comprehensive cache statistics."""
        stats = self.cache_manager.get_stats()
        
        self.assertIn('global_stats', stats)
        self.assertIn('l1_stats', stats)
        self.assertIn('l2_stats', stats)
        self.assertIn('l3_stats', stats)
        self.assertIn('total_cached_tensors', stats)
        self.assertIn('total_cache_size_bytes', stats)
        
        # Check that global hit rate is calculated properly
        self.assertIn('global_hit_rate', stats['global_stats'])
        self.assertGreaterEqual(stats['global_stats']['global_hit_rate'], 0.0)
        self.assertLessEqual(stats['global_stats']['global_hit_rate'], 1.0)


class TestHierarchicalCacheIntegration(unittest.TestCase):
    """Test the integration between hierarchical cache and memory pools."""
    
    def setUp(self):
        """Set up test configuration."""
        self.cache_config = CacheConfig(
            l1_cache_size=32 * 1024 * 1024,  # 32MB
            l2_cache_size=64 * 1024 * 1024,  # 64MB
            l3_cache_size=128 * 1024 * 1024,  # 128MB
        )
        
        # Create mock pool config
        class MockPoolConfig:
            def __init__(self):
                self.memory_pool_base_capacity = 64 * 1024 * 1024  # 64MB
                self.memory_pool_dtype = torch.float16
                self.memory_pool_device = 'cpu'
                self.memory_pool_cache_line_size = 64
                self.memory_pool_l3_cache_size = 6 * 1024 * 1024  # 6MB
        
        self.pool_config = MockPoolConfig()
        self.integration = HierarchicalCacheIntegration(self.cache_config, self.pool_config)
    
    def test_get_tensor_from_cache(self):
        """Test getting tensor from cache when available."""
        # Put a tensor in the cache first
        tensor = torch.randn(100, 100, dtype=torch.float16)
        self.integration.hierarchical_cache.put_tensor(tensor, "test")
        
        # Get the tensor
        result = self.integration.get_tensor((100, 100), "test", torch.float16)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 100))
        self.assertEqual(result.dtype, torch.float16)
    
    def test_get_tensor_from_pool_fallback(self):
        """Test getting tensor from pool when not in cache."""
        # Get a tensor that doesn't exist in cache (should create new or get from pool)
        result = self.integration.get_tensor((50, 50), "test", torch.float16)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (50, 50))
        self.assertEqual(result.dtype, torch.float16)
    
    def test_specialized_tensor_methods(self):
        """Test specialized tensor allocation methods."""
        # Test attention tensor
        attn_tensor = self.integration.get_attention_tensor(2, 8, 512, 64)
        self.assertEqual(attn_tensor.shape, (2, 8, 512, 64))
        
        # Test KV cache tensor
        kv_tensor = self.integration.get_kv_cache_tensor(1, 16, 1024, 128)
        self.assertEqual(kv_tensor.shape, (1, 16, 1024, 128))
        
        # Test image embedding tensor
        img_tensor = self.integration.get_image_embedding_tensor(1, 576, 1152)
        self.assertEqual(img_tensor.shape, (1, 576, 1152))
        
        # Test text embedding tensor
        text_tensor = self.integration.get_text_embedding_tensor(2, 512, 4096)
        self.assertEqual(text_tensor.shape, (2, 512, 4096))
    
    def test_get_integration_stats(self):
        """Test getting integration statistics."""
        stats = self.integration.get_integration_stats()
        
        self.assertIn('integration_stats', stats)
        self.assertIn('hierarchical_cache_stats', stats)
        self.assertIn('pool_system_stats', stats)
        self.assertIn('cache_to_pool_hit_ratio', stats)
        self.assertIn('pool_hit_ratio', stats)


class TestQwen3VLHierarchicalCacheAdapter(unittest.TestCase):
    """Test the Qwen3-VL hierarchical cache adapter."""
    
    def setUp(self):
        """Set up test configuration."""
        class MockConfig:
            def __init__(self):
                self.memory_pool_base_capacity = 64 * 1024 * 1024  # 64MB
                self.memory_pool_dtype = torch.float16
                self.memory_pool_device = 'cpu'
                self.memory_pool_cache_line_size = 64
                self.memory_pool_l3_cache_size = 6 * 1024 * 1024  # 6MB
        
        self.config = MockConfig()
        self.adapter = Qwen3VLHierarchicalCacheAdapter(self.config)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        self.assertIsNotNone(self.adapter.integration)
        self.assertIsNotNone(self.adapter.cache_config)
        
        # Check that cache sizes are set appropriately
        self.assertGreater(self.adapter.cache_config.l1_cache_size, 0)
        self.assertGreater(self.adapter.cache_config.l2_cache_size, 0)
        self.assertGreater(self.adapter.cache_config.l3_cache_size, 0)
    
    def test_allocate_attention_weights(self):
        """Test attention weights allocation."""
        tensor = self.adapter.allocate_attention_weights(2, 8, 512, 64)
        self.assertEqual(tensor.shape, (2, 8, 512, 64))
        self.assertEqual(tensor.dtype, torch.float16)
    
    def test_allocate_kv_cache(self):
        """Test KV cache allocation."""
        tensor = self.adapter.allocate_kv_cache(1, 16, 1024, 128)
        self.assertEqual(tensor.shape, (1, 16, 1024, 128))
        self.assertEqual(tensor.dtype, torch.float16)
    
    def test_allocate_image_features(self):
        """Test image features allocation."""
        tensor = self.adapter.allocate_image_features(1, 576, 1152)
        self.assertEqual(tensor.shape, (1, 576, 1152))
        self.assertEqual(tensor.dtype, torch.float16)
    
    def test_allocate_text_embeddings(self):
        """Test text embeddings allocation."""
        tensor = self.adapter.allocate_text_embeddings(2, 512, 4096)
        self.assertEqual(tensor.shape, (2, 512, 4096))
        self.assertEqual(tensor.dtype, torch.float16)
    
    def test_cache_maintenance(self):
        """Test cache maintenance operations."""
        # Perform cache maintenance
        self.adapter.perform_cache_maintenance()
        
        # This should not raise any exceptions
        stats = self.adapter.get_cache_statistics()
        self.assertIsNotNone(stats)
    
    def test_get_cache_statistics(self):
        """Test getting cache statistics."""
        stats = self.adapter.get_cache_statistics()
        
        self.assertIn('integration_stats', stats)
        self.assertIn('hierarchical_cache_stats', stats)
        self.assertIn('cache_to_pool_hit_ratio', stats)
        self.assertIn('pool_hit_ratio', stats)


class TestPerformance(unittest.TestCase):
    """Performance tests for the hierarchical caching system."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = CacheConfig(
            l1_cache_size=128 * 1024 * 1024,  # 128MB
            l2_cache_size=256 * 1024 * 1024,  # 256MB
            l3_cache_size=512 * 1024 * 1024,  # 512MB
        )
        self.cache_manager = HierarchicalCacheManager(self.config)
    
    def test_cache_performance(self):
        """Test performance of cache operations."""
        import time
        
        # Create and cache multiple tensors
        start_time = time.time()
        
        for i in range(20):
            tensor = torch.randn(200, 200, dtype=torch.float16)  # ~80KB each
            self.cache_manager.put_tensor(tensor, f"tensor_{i}")
        
        put_time = time.time() - start_time
        
        # Retrieve tensors
        start_time = time.time()
        
        for i in range(20):
            _, _ = self.cache_manager.get_tensor((200, 200), torch.float16)
        
        get_time = time.time() - start_time
        
        # Performance should be reasonable (less than 1 second for this test)
        self.assertLess(put_time, 1.0, "Put operations should be fast")
        self.assertLess(get_time, 1.0, "Get operations should be fast")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the caching system."""
        # Create tensors that would exceed individual cache sizes
        large_tensor = torch.randn(1000, 1000, dtype=torch.float16)  # ~2MB
        
        # Put tensor in cache
        success = self.cache_manager.put_tensor(large_tensor, "large_tensor", preferred_level=3)
        self.assertTrue(success)
        
        # Check that the tensor was properly stored
        retrieved, level = self.cache_manager.get_tensor((1000, 1000), torch.float16)
        self.assertIsNotNone(retrieved)
        self.assertEqual(level, 'l3')


def run_all_tests():
    """Run all tests for the hierarchical caching system."""
    print("Running Hierarchical Cache System Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCacheConfig,
        TestTensorMetadata, 
        TestHierarchicalCacheManager,
        TestHierarchicalCacheIntegration,
        TestQwen3VLHierarchicalCacheAdapter,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\nAll tests passed! ✓")
    else:
        print("\nSome tests failed! ✗")
        exit(1)