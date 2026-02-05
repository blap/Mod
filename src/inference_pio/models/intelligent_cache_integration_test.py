"""
Integration Test for Intelligent Cache System in Qwen3 Models

This module tests the integration of the intelligent cache system with the Qwen3 models.
"""

import torch
import unittest
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.intelligent_cache.intelligent_cache_manager import (
    IntelligentCacheConfig,
    IntelligentCacheManager,
    CachePolicy,
    create_intelligent_cache_for_qwen3_4b
)
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.intelligent_cache.intelligent_cache_manager import (
    create_intelligent_cache_for_qwen3_coder
)
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
from src.inference_pio.models.qwen3_0_6b.intelligent_cache.intelligent_cache_manager import (
    create_intelligent_cache_for_qwen3_0_6b
)
from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
from src.inference_pio.models.qwen3_coder_next.intelligent_cache.intelligent_cache_manager import (
    create_intelligent_cache_for_qwen3_coder_next
)


class TestIntelligentCacheIntegration(unittest.TestCase):
    """
    Integration tests for the Intelligent Cache System with Qwen3 models.
    """
    
    def test_qwen3_4b_instruct_intelligent_cache_creation(self):
        """Test creating intelligent cache for Qwen3-4B-Instruct-2507."""
        config = Qwen34BInstruct2507Config()
        cache_manager = create_intelligent_cache_for_qwen3_4b(config)
        
        self.assertIsInstance(cache_manager, IntelligentCacheManager)
        self.assertEqual(cache_manager.config.max_cache_size, config.intelligent_cache_max_size)
        self.assertEqual(cache_manager.config.cache_policy, CachePolicy.INTELLIGENT)
        
    def test_qwen3_coder_30b_intelligent_cache_creation(self):
        """Test creating intelligent cache for Qwen3-Coder-30B."""
        config = Qwen3Coder30BConfig()
        cache_manager = create_intelligent_cache_for_qwen3_coder(config)
        
        self.assertIsInstance(cache_manager, IntelligentCacheManager)
        self.assertEqual(cache_manager.config.max_cache_size, config.intelligent_cache_max_size)
        self.assertEqual(cache_manager.config.cache_policy, CachePolicy.INTELLIGENT)
        
    def test_qwen3_0_6b_intelligent_cache_creation(self):
        """Test creating intelligent cache for Qwen3-0.6B."""
        config = Qwen3_0_6B_Config()
        cache_manager = create_intelligent_cache_for_qwen3_0_6b(config)
        
        self.assertIsInstance(cache_manager, IntelligentCacheManager)
        self.assertEqual(cache_manager.config.max_cache_size, config.intelligent_cache_max_size)
        self.assertEqual(cache_manager.config.cache_policy, CachePolicy.INTELLIGENT)
        
    def test_qwen3_coder_next_intelligent_cache_creation(self):
        """Test creating intelligent cache for Qwen3-Coder-Next."""
        config = Qwen3CoderNextConfig()
        cache_manager = create_intelligent_cache_for_qwen3_coder_next(config)
        
        self.assertIsInstance(cache_manager, IntelligentCacheManager)
        self.assertEqual(cache_manager.config.max_cache_size, config.intelligent_cache_max_size)
        self.assertEqual(cache_manager.config.cache_policy, CachePolicy.INTELLIGENT)
        
    def test_intelligent_cache_put_get_operations(self):
        """Test put/get operations on intelligent cache."""
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
        
        cache_manager = IntelligentCacheManager(config)
        
        # Test putting and getting a tensor
        key = "test_tensor"
        original_tensor = torch.randn(10, 128)
        
        cache_manager.put(key, original_tensor)
        retrieved_tensor = cache_manager.get(key)
        
        self.assertIsNotNone(retrieved_tensor)
        # Check that the retrieved tensor is approximately equal to the original (within tolerance due to compression)
        self.assertTrue(torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2))
        
    def test_intelligent_cache_predictive_policy(self):
        """Test predictive cache policy."""
        config = IntelligentCacheConfig(
            cache_policy=CachePolicy.PREDICTIVE,
            prediction_horizon=3,
            prediction_confidence_threshold=0.3
        )
        cache_manager = IntelligentCacheManager(config)
        
        # Add some items to build history
        for i in range(5):
            key = f"item_{i}"
            tensor = torch.randn(5, 64)
            cache_manager.put(key, tensor)
            # Access some items multiple times to establish patterns
            if i < 3:
                cache_manager.get(key)  # Access again to record pattern
        
        # Test prediction functionality
        predicted_items = cache_manager.predict_and_prefetch()
        self.assertIsInstance(predicted_items, list)
        
    def test_intelligent_cache_intelligent_policy(self):
        """Test intelligent cache policy."""
        config = IntelligentCacheConfig(
            cache_policy=CachePolicy.INTELLIGENT,
            prediction_horizon=3,
            prediction_confidence_threshold=0.3
        )
        cache_manager = IntelligentCacheManager(config)
        
        # Add some items
        for i in range(3):
            key = f"intelligent_item_{i}"
            tensor = torch.randn(5, 64)
            cache_manager.put(key, tensor)
        
        # Access one item multiple times to make it more valuable
        retrieved = cache_manager.get("intelligent_item_0")
        self.assertIsNotNone(retrieved)
        
        # Get cache statistics
        stats = cache_manager.get_cache_stats()
        self.assertGreaterEqual(stats["hits"], 0)
        self.assertIn("hit_rate", stats)
        
    def test_cache_compression_methods(self):
        """Test different compression methods."""
        original_tensor = torch.randn(20, 64)
        
        # Test intelligent compression
        intelligent_config = IntelligentCacheConfig(
            compression_enabled=True,
            compression_method="intelligent"
        )
        intelligent_cache = IntelligentCacheManager(intelligent_config)
        intelligent_cache.put("test", original_tensor)
        retrieved_intelligent = intelligent_cache.get("test")
        self.assertIsNotNone(retrieved_intelligent)
        
        # Test sparse compression
        sparse_config = IntelligentCacheConfig(
            compression_enabled=True,
            compression_method="sparse"
        )
        sparse_cache = IntelligentCacheManager(sparse_config)
        sparse_cache.put("test", original_tensor)
        retrieved_sparse = sparse_cache.get("test")
        self.assertIsNotNone(retrieved_sparse)
        
    def test_cache_size_management(self):
        """Test cache size management and eviction."""
        small_config = IntelligentCacheConfig(
            max_cache_size=1024,  # Small cache size
            cache_policy=CachePolicy.LRU
        )
        small_cache = IntelligentCacheManager(small_config)
        
        # Add tensors until cache is full
        for i in range(10):
            key = f"size_test_{i}"
            tensor = torch.randn(10, 10)  # Each tensor is ~400 bytes in float32
            small_cache.put(key, tensor)
        
        # Check that cache size is managed
        stats = small_cache.get_cache_stats()
        self.assertLessEqual(stats["current_size_bytes"], small_config.max_cache_size)
        
    def test_cache_prefetching(self):
        """Test cache prefetching functionality."""
        config = IntelligentCacheConfig(
            enable_prefetching=True,
            prediction_horizon=2,
            prediction_confidence_threshold=0.1  # Low threshold to ensure predictions
        )
        cache_manager = IntelligentCacheManager(config)
        
        # Add some items to enable prediction
        for i in range(3):
            key = f"prefetch_item_{i}"
            tensor = torch.randn(5, 32)
            cache_manager.put(key, tensor)
            cache_manager.get(key)  # Access to record pattern
        
        # Test prefetch functionality
        prefetched = cache_manager.prefetch("prefetch_item_0")
        self.assertIsNotNone(prefetched)
        
        # Test predict and prefetch
        predicted_and_prefetched = cache_manager.predict_and_prefetch()
        self.assertIsInstance(predicted_and_prefetched, list)


if __name__ == "__main__":
    unittest.main()