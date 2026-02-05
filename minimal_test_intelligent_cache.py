#!/usr/bin/env python
"""
Minimal test of the intelligent cache system components
"""

import sys
import os
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the intelligent cache components directly by reading the file content
import importlib.util

# Load the intelligent cache manager module directly
spec = importlib.util.spec_from_file_location(
    "intelligent_cache_manager", 
    "src/inference_pio/models/qwen3_4b_instruct_2507/intelligent_cache/intelligent_cache_manager.py"
)
intelligent_cache_module = importlib.util.module_from_spec(spec)

# Execute the module to load its contents
spec.loader.exec_module(intelligent_cache_module)

# Now we can access the classes and functions
IntelligentCacheManager = intelligent_cache_module.IntelligentCacheManager
IntelligentCacheConfig = intelligent_cache_module.IntelligentCacheConfig
CachePolicy = intelligent_cache_module.CachePolicy
AccessPatternPredictor = intelligent_cache_module.AccessPatternPredictor
PerformanceMonitor = intelligent_cache_module.PerformanceMonitor

import torch

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
    print(f"Cache stats: hits={stats['hits']}, misses={stats['misses']}, hit_rate={stats['hit_rate']:.2f}")
    
    # Test predictor
    predictor = AccessPatternPredictor(config)
    predictor.record_access("test_key", "access")
    predictions = predictor.predict_next_accesses()
    print(f"Predictions: {len(predictions)} predictions made")
    
    # Test performance monitor
    perf_monitor = PerformanceMonitor(config)
    perf_monitor.record_hit()
    perf_monitor.record_miss()
    metrics = perf_monitor.get_metrics()
    print(f"Performance metrics: hit_rate={metrics['hit_rate']:.2f}")
    
    print("Intelligent Cache System test completed successfully!")

if __name__ == "__main__":
    test_intelligent_cache()