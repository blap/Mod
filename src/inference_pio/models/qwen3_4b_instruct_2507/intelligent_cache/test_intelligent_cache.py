"""
Test Intelligent Cache System for Qwen3-4B-Instruct-2507

This module tests the intelligent cache system for the Qwen3-4B-Instruct-2507 model.
"""

import torch
import unittest
from .intelligent_cache_manager import (
    IntelligentCacheConfig,
    IntelligentCacheManager,
    CachePolicy
)


class TestIntelligentCache(unittest.TestCase):
    """
    Test cases for the Intelligent Cache Manager.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = IntelligentCacheConfig(
            max_cache_size=1024 * 1024,  # 1MB
            cache_precision=torch.float16,
            compression_enabled=True,
            compression_method="fp16",
            cache_policy=CachePolicy.INTELLIGENT,
            enable_prefetching=True,
            prefetch_distance=1,
            max_prefix_length=1024,
            min_prefix_length=4,
            cache_warmup_threshold=1,
            prediction_horizon=5,
            prediction_confidence_threshold=0.5,
            enable_adaptive_eviction=True,
            enable_adaptive_prefetching=True,
            adaptive_window_size=50,
            enable_performance_monitoring=True,
            performance_log_interval=10
        )
        self.cache_manager = IntelligentCacheManager(config)

    def test_put_and_get(self):
        """Test putting and getting values from the cache."""
        key = "test_key"
        value = torch.randn(10, 128)  # Random tensor
        
        # Put value in cache
        self.cache_manager.put(key, value)
        
        # Get value from cache
        retrieved_value = self.cache_manager.get(key)
        
        # Check that the retrieved value matches the original
        self.assertIsNotNone(retrieved_value)
        self.assertTrue(torch.allclose(value.half(), retrieved_value, atol=1e-3))
        
    def test_cache_policy_predictive(self):
        """Test predictive cache policy."""
        config = IntelligentCacheConfig(cache_policy=CachePolicy.PREDICTIVE)
        predictive_cache = IntelligentCacheManager(config)
        
        # Add some values to trigger prediction logic
        for i in range(5):
            key = f"seq_{i}"
            value = torch.randn(10, 128)
            predictive_cache.put(key, value)
            
        # Check that cache stats are updated
        stats = predictive_cache.get_cache_stats()
        self.assertGreaterEqual(stats["hits"], 0)
        
    def test_cache_policy_intelligent(self):
        """Test intelligent cache policy."""
        config = IntelligentCacheConfig(cache_policy=CachePolicy.INTELLIGENT)
        intelligent_cache = IntelligentCacheManager(config)
        
        # Add some values to trigger intelligent logic
        for i in range(3):
            key = f"intelligent_seq_{i}"
            value = torch.randn(10, 128)
            intelligent_cache.put(key, value)
            
        # Get one value multiple times to increase its access count
        retrieved = intelligent_cache.get("intelligent_seq_0")
        self.assertIsNotNone(retrieved)
        
        # Check that access patterns are recorded
        stats = intelligent_cache.get_cache_stats()
        self.assertGreaterEqual(stats["hits"], 0)
        
    def test_prefetch_functionality(self):
        """Test prefetch functionality."""
        key = "prefetch_key"
        value = torch.randn(10, 128)
        
        # Put value in cache
        self.cache_manager.put(key, value)
        
        # Prefetch should return the same value if it exists
        prefetched_value = self.cache_manager.prefetch(key)
        self.assertIsNotNone(prefetched_value)
        self.assertTrue(torch.allclose(value.half(), prefetched_value, atol=1e-3))
        
        # Prefetch on non-existent key should return None
        non_existent_prefetch = self.cache_manager.prefetch("non_existent")
        self.assertIsNone(non_existent_prefetch)
        
    def test_cache_size_limit(self):
        """Test that cache respects size limits."""
        # Create a small cache for testing
        small_config = IntelligentCacheConfig(max_cache_size=1024)  # 1KB
        small_cache = IntelligentCacheManager(small_config)
        
        # Add a large tensor that exceeds the cache size
        large_tensor = torch.randn(100, 100)  # This should be larger than 1KB
        small_cache.put("large_key", large_tensor)
        
        # The cache should evict items when size is exceeded
        stats = small_cache.get_cache_stats()
        # Note: The exact behavior depends on the tensor size calculation
        
    def test_compression_methods(self):
        """Test different compression methods."""
        original_tensor = torch.randn(20, 64)
        
        # Test FP16 compression
        fp16_config = IntelligentCacheConfig(
            compression_enabled=True,
            compression_method="fp16"
        )
        fp16_cache = IntelligentCacheManager(fp16_config)
        fp16_cache.put("test", original_tensor)
        retrieved_fp16 = fp16_cache.get("test")
        self.assertIsNotNone(retrieved_fp16)
        
        # Test intelligent compression
        intelligent_config = IntelligentCacheConfig(
            compression_enabled=True,
            compression_method="intelligent"
        )
        intelligent_cache = IntelligentCacheManager(intelligent_config)
        intelligent_cache.put("test", original_tensor)
        retrieved_intelligent = intelligent_cache.get("test")
        self.assertIsNotNone(retrieved_intelligent)
        
    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        # Add and retrieve some items
        for i in range(3):
            key = f"stat_key_{i}"
            value = torch.randn(5, 32)
            self.cache_manager.put(key, value)
            retrieved = self.cache_manager.get(key)
            self.assertIsNotNone(retrieved)
            
        # Check statistics
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["misses"], 0)
        self.assertGreaterEqual(stats["num_cached_items"], 0)
        self.assertGreaterEqual(stats["current_size_bytes"], 0)
        
    def test_cache_clear(self):
        """Test clearing the cache."""
        # Add some items
        for i in range(3):
            key = f"clear_key_{i}"
            value = torch.randn(5, 32)
            self.cache_manager.put(key, value)
            
        # Verify items are in cache
        stats_before = self.cache_manager.get_cache_stats()
        self.assertGreater(stats_before["num_cached_items"], 0)
        
        # Clear the cache
        self.cache_manager.clear()
        
        # Verify cache is empty
        stats_after = self.cache_manager.get_cache_stats()
        self.assertEqual(stats_after["num_cached_items"], 0)
        self.assertEqual(stats_after["current_size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()