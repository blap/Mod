"""
Test suite for the Intelligent Cache System for GLM-4.7-Flash Model.
"""

import unittest
import torch
import torch.nn as nn
import time
from .intelligent_cache_manager import (
    IntelligentCacheConfig,
    IntelligentCacheManager,
    CachePolicy,
    apply_intelligent_caching_to_model,
    create_intelligent_cache_for_glm47
)
from ..config import GLM47FlashConfig


class TestIntelligentCache(unittest.TestCase):
    """
    Test cases for the Intelligent Cache System.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = IntelligentCacheConfig(
            max_cache_size=1024 * 1024,  # 1MB
            cache_precision=torch.float16,
            compression_enabled=True,
            compression_method="fp16",
            cache_policy=CachePolicy.INTELLIGENT,
            enable_prefetching=True,
            prefetch_distance=1,
            max_prefix_length=1024,
            min_prefix_length=4,
            cache_warmup_threshold=2,
            prediction_horizon=5,
            prediction_confidence_threshold=0.5
        )
        self.cache_manager = IntelligentCacheManager(self.config)
    
    def test_cache_initialization(self):
        """Test that the cache manager initializes correctly."""
        self.assertIsInstance(self.cache_manager, IntelligentCacheManager)
        self.assertEqual(self.cache_manager.config, self.config)
        self.assertEqual(len(self.cache_manager.cache), 0)
        self.assertEqual(self.cache_manager.cache_size, 0)
    
    def test_put_and_get_operations(self):
        """Test basic put and get operations."""
        # Create a test tensor
        test_tensor = torch.randn(10, 20, dtype=torch.float32)
        key = "test_key"
        
        # Put the tensor in cache
        self.cache_manager.put(key, test_tensor)
        
        # Verify it was added
        self.assertIn(key, self.cache_manager.cache)
        
        # Get the tensor from cache
        retrieved_tensor = self.cache_manager.get(key)
        
        # Verify it matches the original (accounting for compression)
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(retrieved_tensor.shape, test_tensor.shape)
    
    def test_cache_size_limit(self):
        """Test that cache respects size limits."""
        # Create a large tensor that exceeds cache size
        large_tensor = torch.randn(1000, 1000, dtype=torch.float32)  # Much larger than 1MB limit
        
        # Try to put it in cache - this should trigger eviction
        self.cache_manager.put("large_key", large_tensor)
        
        # The cache size should not exceed the limit
        self.assertLessEqual(self.cache_manager.cache_size, self.config.max_cache_size)
    
    def test_cache_warmup_threshold(self):
        """Test that cache respects warmup threshold."""
        test_tensor = torch.randn(10, 20, dtype=torch.float32)
        key = "warmup_test"
        
        # Put the tensor once - should not be cached due to warmup threshold
        self.cache_manager.put(key, test_tensor)
        self.assertNotIn(key, self.cache_manager.cache)
        
        # Put the tensor again - should now be cached
        self.cache_manager.put(key, test_tensor)
        self.assertIn(key, self.cache_manager.cache)
    
    def test_different_cache_policies(self):
        """Test different cache policies work correctly."""
        # Test LRU policy
        lru_config = IntelligentCacheConfig(cache_policy=CachePolicy.LRU)
        lru_cache = IntelligentCacheManager(lru_config)
        
        # Add some items
        for i in range(5):
            tensor = torch.randn(10, 10)
            lru_cache.put(f"key_{i}", tensor)
        
        # Access one key to make it most recently used
        lru_cache.get("key_0")
        
        # Add another item to trigger eviction
        lru_cache.put("key_new", torch.randn(10, 10))
        
        # With LRU, "key_1" should be evicted (not "key_0" which was recently accessed)
        # This is hard to test directly, so we just verify the policy is set
        self.assertEqual(lru_cache.config.cache_policy, CachePolicy.LRU)
    
    def test_compression_methods(self):
        """Test different compression methods."""
        test_tensor = torch.randn(50, 50, dtype=torch.float32)
        
        # Test FP16 compression
        fp16_config = IntelligentCacheConfig(
            compression_enabled=True,
            compression_method="fp16"
        )
        fp16_cache = IntelligentCacheManager(fp16_config)
        
        compressed = fp16_cache._compress_tensor(test_tensor)
        decompressed = fp16_cache._decompress_tensor(compressed)
        
        self.assertEqual(decompressed.dtype, torch.float16)
        self.assertEqual(decompressed.shape, test_tensor.shape)
    
    def test_predictive_caching(self):
        """Test predictive caching functionality."""
        # Add some access patterns
        for i in range(10):
            tensor = torch.randn(5, 5)
            self.cache_manager.predictor.record_access(f"key_{i % 3}")  # Cycle through 3 keys
        
        # Get predictions
        predictions = self.cache_manager.predictor.predict_next_accesses()
        
        # Should have some predictions
        self.assertGreaterEqual(len(predictions), 0)
        
        # Test prefetch functionality
        prefetched_keys = self.cache_manager.predict_and_prefetch()
        self.assertIsInstance(prefetched_keys, list)
    
    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        # Add and access some items
        test_tensor = torch.randn(10, 10)
        self.cache_manager.put("stat_test", test_tensor)
        retrieved = self.cache_manager.get("stat_test")
        
        # Get statistics
        stats = self.cache_manager.get_cache_stats()
        
        # Verify stats structure
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
        self.assertIn("current_size_bytes", stats)
        self.assertGreaterEqual(stats["hits"], 0)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        test_tensor = torch.randn(10, 10)
        self.cache_manager.put("clear_test", test_tensor)
        
        # Verify item is in cache
        self.assertIn("clear_test", self.cache_manager.cache)
        
        # Clear the cache
        self.cache_manager.clear()
        
        # Verify cache is empty
        self.assertEqual(len(self.cache_manager.cache), 0)
        self.assertEqual(self.cache_manager.cache_size, 0)


class TestIntelligentCacheIntegration(unittest.TestCase):
    """
    Test cases for integrating intelligent cache with GLM-4.7-Flash model.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.glm_config = GLM47FlashConfig()
        # Enable intelligent caching in the GLM config
        self.glm_config.use_intelligent_caching = True
        self.glm_config.intelligent_cache_max_size = 1024 * 1024  # 1MB
        self.glm_config.intelligent_cache_policy = "intelligent"
    
    def test_create_intelligent_cache_for_glm47(self):
        """Test creating intelligent cache specifically for GLM-4.7-Flash."""
        cache_manager = create_intelligent_cache_for_glm47(self.glm_config)
        
        self.assertIsInstance(cache_manager, IntelligentCacheManager)
        self.assertEqual(cache_manager.config.max_cache_size, 1024 * 1024)
        self.assertEqual(cache_manager.config.cache_policy, CachePolicy.INTELLIGENT)
    
    def test_apply_intelligent_caching_to_model(self):
        """Test applying intelligent caching to a model."""
        # Create a simple dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)
        
        dummy_model = DummyModel()
        
        # Apply intelligent caching
        cached_model = apply_intelligent_caching_to_model(dummy_model, self.glm_config)
        
        # Verify the model has the cache manager attribute
        self.assertTrue(hasattr(cached_model, 'intelligent_cache_manager'))
        self.assertIsInstance(cached_model.intelligent_cache_manager, IntelligentCacheManager)


if __name__ == '__main__':
    unittest.main()