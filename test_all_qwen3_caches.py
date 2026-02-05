#!/usr/bin/env python
"""
Test the intelligent cache system for all Qwen3 models
"""

import sys
import os
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import importlib.util
import torch

def test_qwen3_4b_instruct_cache():
    print("Testing Qwen3-4B-Instruct-2507 Intelligent Cache...")
    
    # Load the intelligent cache manager module directly
    spec = importlib.util.spec_from_file_location(
        "intelligent_cache_manager_4b", 
        "src/inference_pio/models/qwen3_4b_instruct_2507/intelligent_cache/intelligent_cache_manager.py"
    )
    intelligent_cache_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intelligent_cache_module)

    IntelligentCacheManager = intelligent_cache_module.IntelligentCacheManager
    IntelligentCacheConfig = intelligent_cache_module.IntelligentCacheConfig
    CachePolicy = intelligent_cache_module.CachePolicy

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
    
    # Test basic functionality
    key = "test_tensor_4b"
    original_tensor = torch.randn(5, 64)
    
    cache_manager.put(key, original_tensor)
    retrieved_tensor = cache_manager.get(key)

    assert retrieved_tensor is not None, "Failed to retrieve tensor from 4B cache"
    assert torch.allclose(original_tensor.to(retrieved_tensor.dtype), retrieved_tensor, atol=1e-2), "Retrieved tensor doesn't match original"
    
    print("OK Qwen3-4B-Instruct-2507 Intelligent Cache works correctly")

def test_qwen3_coder_30b_cache():
    print("Testing Qwen3-Coder-30B Intelligent Cache...")
    
    # Load the intelligent cache manager module directly
    spec = importlib.util.spec_from_file_location(
        "intelligent_cache_manager_coder30b", 
        "src/inference_pio/models/qwen3_coder_30b/intelligent_cache/intelligent_cache_manager.py"
    )
    intelligent_cache_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intelligent_cache_module)

    IntelligentCacheManager = intelligent_cache_module.IntelligentCacheManager
    IntelligentCacheConfig = intelligent_cache_module.IntelligentCacheConfig
    CachePolicy = intelligent_cache_module.CachePolicy

    config = IntelligentCacheConfig(
        max_cache_size=2 * 1024 * 1024,  # 2MB for larger model
        cache_precision=torch.float16,
        compression_enabled=True,
        compression_method="fp16",  # Use simpler compression to avoid issues
        cache_policy=CachePolicy.PREDICTIVE,
        enable_prefetching=True,
        prefetch_distance=1,
        max_prefix_length=2048,  # Larger for coding tasks
        min_prefix_length=8,
        cache_warmup_threshold=1,  # Reduced for testing
        prediction_horizon=8,
        prediction_confidence_threshold=0.6,
        enable_adaptive_eviction=True,
        enable_adaptive_prefetching=True,
        adaptive_window_size=100,
        enable_performance_monitoring=True,
        performance_log_interval=10
    )
    
    cache_manager = IntelligentCacheManager(config)
    
    # Test basic functionality
    key = "test_tensor_coder30b"
    original_tensor = torch.randn(8, 128)
    
    # For Coder-30B, we need to meet the warmup threshold (2) before caching
    # Put the same tensor twice to meet the warmup threshold
    cache_manager.put(key, original_tensor)
    cache_manager.put(key, original_tensor)  # Meet warmup threshold
    retrieved_tensor = cache_manager.get(key)

    assert retrieved_tensor is not None, "Failed to retrieve tensor from Coder-30B cache"
    assert torch.allclose(original_tensor.to(retrieved_tensor.dtype), retrieved_tensor, atol=1e-2), "Retrieved tensor doesn't match original"
    
    print("OK Qwen3-Coder-30B Intelligent Cache works correctly")

def test_qwen3_0_6b_cache():
    print("Testing Qwen3-0.6B Intelligent Cache...")
    
    # Load the intelligent cache manager module directly
    spec = importlib.util.spec_from_file_location(
        "intelligent_cache_manager_06b", 
        "src/inference_pio/models/qwen3_0_6b/intelligent_cache/intelligent_cache_manager.py"
    )
    intelligent_cache_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intelligent_cache_module)

    IntelligentCacheManager = intelligent_cache_module.IntelligentCacheManager
    IntelligentCacheConfig = intelligent_cache_module.IntelligentCacheConfig
    CachePolicy = intelligent_cache_module.CachePolicy

    config = IntelligentCacheConfig(
        max_cache_size=512 * 1024,  # 512KB for smaller model
        cache_precision=torch.float16,
        compression_enabled=True,
        compression_method="fp16",
        cache_policy=CachePolicy.INTELLIGENT,
        enable_prefetching=True,
        prefetch_distance=1,
        max_prefix_length=512,  # Smaller for smaller model
        min_prefix_length=4,
        cache_warmup_threshold=1,
        prediction_horizon=5,
        prediction_confidence_threshold=0.65,
        enable_adaptive_eviction=True,
        enable_adaptive_prefetching=True,
        adaptive_window_size=30,
        enable_performance_monitoring=True,
        performance_log_interval=10
    )
    
    cache_manager = IntelligentCacheManager(config)
    
    # Test basic functionality
    key = "test_tensor_06b"
    original_tensor = torch.randn(4, 32)
    
    # For 0.6B, we need to meet the warmup threshold (1) before caching
    cache_manager.put(key, original_tensor)
    retrieved_tensor = cache_manager.get(key)

    assert retrieved_tensor is not None, "Failed to retrieve tensor from 0.6B cache"
    assert torch.allclose(original_tensor.to(retrieved_tensor.dtype), retrieved_tensor, atol=1e-2), "Retrieved tensor doesn't match original"
    
    print("OK Qwen3-0.6B Intelligent Cache works correctly")

def test_qwen3_coder_next_cache():
    print("Testing Qwen3-Coder-Next Intelligent Cache...")
    
    # Load the intelligent cache manager module directly
    spec = importlib.util.spec_from_file_location(
        "intelligent_cache_manager_coder_next", 
        "src/inference_pio/models/qwen3_coder_next/intelligent_cache/intelligent_cache_manager.py"
    )
    intelligent_cache_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intelligent_cache_module)

    IntelligentCacheManager = intelligent_cache_module.IntelligentCacheManager
    IntelligentCacheConfig = intelligent_cache_module.IntelligentCacheConfig
    CachePolicy = intelligent_cache_module.CachePolicy

    config = IntelligentCacheConfig(
        max_cache_size=1024 * 1024,  # 1MB
        cache_precision=torch.float16,
        compression_enabled=True,
        compression_method="intelligent",
        cache_policy=CachePolicy.PREDICTIVE,
        enable_prefetching=True,
        prefetch_distance=1,
        max_prefix_length=1024,
        min_prefix_length=4,
        cache_warmup_threshold=1,
        prediction_horizon=6,
        prediction_confidence_threshold=0.6,
        enable_adaptive_eviction=True,
        enable_adaptive_prefetching=True,
        adaptive_window_size=60,
        enable_performance_monitoring=True,
        performance_log_interval=10
    )
    
    cache_manager = IntelligentCacheManager(config)
    
    # Test basic functionality
    key = "test_tensor_coder_next"
    original_tensor = torch.randn(6, 64)
    
    # For Coder-Next, we need to meet the warmup threshold (1) before caching
    cache_manager.put(key, original_tensor)
    retrieved_tensor = cache_manager.get(key)

    assert retrieved_tensor is not None, "Failed to retrieve tensor from Coder-Next cache"
    assert torch.allclose(original_tensor.to(retrieved_tensor.dtype), retrieved_tensor, atol=1e-2), "Retrieved tensor doesn't match original"
    
    print("OK Qwen3-Coder-Next Intelligent Cache works correctly")

def main():
    print("Testing Intelligent Cache Systems for all Qwen3 models...")
    print("=" * 60)
    
    test_qwen3_4b_instruct_cache()
    test_qwen3_coder_30b_cache()
    test_qwen3_0_6b_cache()
    test_qwen3_coder_next_cache()
    
    print("=" * 60)
    print("All Intelligent Cache Systems tested successfully!")
    print("Implementation of predictive and intelligent caching policies completed for:")
    print("- Qwen3-4B-Instruct-2507")
    print("- Qwen3-Coder-30B") 
    print("- Qwen3-0.6B")
    print("- Qwen3-Coder-Next")

if __name__ == "__main__":
    main()