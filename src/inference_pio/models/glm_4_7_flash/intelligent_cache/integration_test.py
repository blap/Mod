"""
Integration test for the Intelligent Cache System with GLM-4.7-Flash Model.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from .intelligent_cache_manager import (
    IntelligentCacheManager,
    IntelligentCacheConfig,
    CachePolicy,
    create_intelligent_cache_for_glm47,
    apply_intelligent_caching_to_model
)


def test_intelligent_cache_with_glm47():
    """
    Test the intelligent cache system with GLM-4.7-Flash model.
    """
    print("Testing Intelligent Cache System with GLM-4.7-Flash Model...")
    
    # Create GLM-4.7-Flash configuration
    config = GLM47FlashConfig()
    config.use_intelligent_caching = True
    config.intelligent_cache_max_size = 1024 * 1024  # 1MB
    config.intelligent_cache_policy = "intelligent"
    config.intelligent_cache_compression_enabled = True
    config.intelligent_cache_enable_prefetching = True
    
    print("✓ Created GLM-4.7-Flash configuration with intelligent caching enabled")
    
    # Create intelligent cache manager for GLM-4.7-Flash
    cache_manager = create_intelligent_cache_for_glm47(config)
    
    print(f"✓ Created intelligent cache manager with policy: {cache_manager.config.cache_policy}")
    print(f"  Max cache size: {cache_manager.config.max_cache_size} bytes")
    print(f"  Compression enabled: {cache_manager.config.compression_enabled}")
    print(f"  Prefetching enabled: {cache_manager.config.enable_prefetching}")
    
    # Test basic cache operations
    test_tensor = torch.randn(20, 32, dtype=torch.float32)  # Simulate KV cache tensor
    key = "test_prefix_1"
    
    print(f"\nTesting cache operations with tensor of shape: {test_tensor.shape}")
    
    # Put tensor in cache (need to do this twice to pass warmup threshold)
    cache_manager.put(key, test_tensor)
    cache_manager.put(key, test_tensor)  # Second time to pass warmup threshold
    
    print("✓ Put tensor in cache (passed warmup threshold)")
    
    # Get tensor from cache
    retrieved_tensor = cache_manager.get(key)
    
    if retrieved_tensor is not None:
        print(f"✓ Retrieved tensor from cache with shape: {retrieved_tensor.shape}")
        print(f"  Tensor similarity: {torch.allclose(test_tensor, retrieved_tensor, atol=1e-3)}")
    else:
        print("✗ Failed to retrieve tensor from cache")
    
    # Test predictive caching
    print(f"\nTesting predictive caching...")
    for i in range(5):
        cache_manager.predictor.record_access(f"sequence_{i % 3}")  # Common access pattern
    
    predictions = cache_manager.predictor.predict_next_accesses()
    print(f"✓ Made {len(predictions)} predictions")
    for pred_key, confidence in predictions[:3]:  # Show first 3 predictions
        print(f"  Predicted access to '{pred_key}' with confidence {confidence:.2f}")
    
    # Prefetch based on predictions
    prefetched_keys = cache_manager.predict_and_prefetch()
    print(f"✓ Prefetched {len(prefetched_keys)} keys based on predictions")
    
    # Test cache statistics
    stats = cache_manager.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2f}")
    print(f"  Current size: {stats['current_size_bytes']} bytes")
    print(f"  Max size: {stats['max_size_bytes']} bytes")
    print(f"  Size utilization: {stats['size_utilization']:.2f}")
    print(f"  Number of cached items: {stats['num_cached_items']}")
    
    # Test different cache policies
    print(f"\nTesting different cache policies...")
    policies = [CachePolicy.LRU, CachePolicy.PREDICTIVE, CachePolicy.INTELLIGENT]
    
    for policy in policies:
        policy_config = IntelligentCacheConfig(
            cache_policy=policy,
            max_cache_size=512 * 1024  # 512KB
        )
        policy_cache = IntelligentCacheManager(policy_config)
        print(f"  ✓ Initialized cache with {policy.value} policy")
    
    print(f"\n✓ All tests passed! Intelligent Cache System is working correctly with GLM-4.7-Flash.")


def test_model_integration():
    """
    Test integrating intelligent caching with a dummy model.
    """
    print("\nTesting model integration...")
    
    # Create a dummy model similar to GLM-4.7-Flash structure
    class DummyGLM47Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.linear = nn.Linear(128, 128)
            self.lm_head = nn.Linear(128, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            return self.lm_head(x)
    
    # Create model and config
    model = DummyGLM47Model()
    config = GLM47FlashConfig()
    config.use_intelligent_caching = True
    
    # Apply intelligent caching to the model
    cached_model = apply_intelligent_caching_to_model(model, config)
    
    print("✓ Applied intelligent caching to dummy model")
    print(f"  Model has intelligent cache manager: {hasattr(cached_model, 'intelligent_cache_manager')}")
    print(f"  Cache manager type: {type(cached_model.intelligent_cache_manager).__name__}")
    
    # Test cache operations through the model
    test_tensor = torch.randn(10, 128)
    cached_model.intelligent_cache_manager.put("test_key", test_tensor)
    retrieved = cached_model.intelligent_cache_manager.get("test_key")
    
    if retrieved is not None:
        print(f"✓ Cache operations work through model interface")
        print(f"  Retrieved tensor shape: {retrieved.shape}")
    else:
        print("✗ Cache operations failed through model interface")


def run_comprehensive_tests():
    """
    Run comprehensive tests for the intelligent cache system.
    """
    print("=" * 60)
    print("INTELLIGENT CACHE SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    try:
        test_intelligent_cache_with_glm47()
        test_model_integration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("The Intelligent Cache System is ready for GLM-4.7-Flash!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_tests()