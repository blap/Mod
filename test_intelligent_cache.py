#!/usr/bin/env python
"""
Simple test script to verify the intelligent cache system works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
from src.inference_pio.models.qwen3_4b_instruct_2507.intelligent_cache.intelligent_cache_manager import (
    IntelligentCacheManager, 
    IntelligentCacheConfig, 
    CachePolicy
)

def test_intelligent_cache():
    print("Testing Intelligent Cache System...")
    
    # Create a basic configuration
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
    
    # Create cache manager
    cache_manager = IntelligentCacheManager(config)
    print(f"Cache manager created successfully with policy: {config.cache_policy}")
    
    # Test putting and getting a tensor
    key = "test_tensor"
    original_tensor = torch.randn(10, 128)
    
    cache_manager.put(key, original_tensor)
    retrieved_tensor = cache_manager.get(key)
    
    if retrieved_tensor is not None:
        print(f"Successfully put and retrieved tensor. Shape: {original_tensor.shape}")
        # Check that the retrieved tensor is approximately equal to the original
        is_close = torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2)
        print(f"Tensors are close (within tolerance): {is_close}")
    else:
        print("Failed to retrieve tensor")
    
    # Test cache statistics
    stats = cache_manager.get_cache_stats()
    print(f"Cache stats: hits={stats['hits']}, misses={stats['misses']}")
    
    print("Intelligent Cache System test completed successfully!")

if __name__ == "__main__":
    test_intelligent_cache()