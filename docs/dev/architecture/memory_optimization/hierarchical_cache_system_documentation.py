"""
Hierarchical Cache System for Qwen3-VL - Documentation and Usage Examples

This document provides comprehensive documentation and usage examples for the 
advanced hierarchical caching and buffering system implemented for Qwen3-VL.
The system includes L1 (GPU), L2 (CPU pinned), and L3 (NVMe SSD) caches with
intelligent migration policies and hardware-specific optimizations.

Table of Contents:
1. Overview
2. Architecture
3. Components
4. Usage Examples
5. Configuration
6. Performance Considerations
7. Hardware Optimizations
8. Integration with Existing Systems
"""

# 1. OVERVIEW
"""
The Hierarchical Cache System for Qwen3-VL is designed to optimize tensor memory
management across three levels:

- L1 Cache (GPU Memory): For tensors accessed with high frequency
- L2 Cache (CPU Pinned Memory): For tensors accessed with medium frequency  
- L3 Cache (NVMe SSD): For tensors accessed with low frequency

The system includes:
- Intelligent access pattern tracking
- ML-based access prediction
- Dynamic migration policies
- Hardware-specific optimizations for Intel i5-10210U + NVIDIA SM61 + NVMe SSD
- Integration with existing memory pool systems
"""

# 2. ARCHITECTURE
"""
The architecture consists of several key components:

Cache Levels:
- L1: Fast GPU memory cache using torch.Tensor storage
- L2: CPU pinned memory cache for efficient GPU transfers
- L3: NVMe SSD cache with optional compression

Management Components:
- HierarchicalCacheManager: Coordinates all cache levels
- AccessPatternTracker: Tracks tensor access patterns
- SimpleMLPredictor: Predicts future access based on patterns
- MigrationManager: Handles tensor movement between levels

Integration Components:
- HierarchicalCacheIntegration: Bridges with existing pools
- Qwen3VLHierarchicalCacheAdapter: Main interface for Qwen3-VL
"""

# 3. COMPONENTS
"""
Key Classes and Functions:

HierarchicalCacheManager:
- Main cache coordinator
- Handles get/put operations across all levels
- Manages migration policies
- Provides comprehensive statistics

CacheConfig:
- Configuration class for cache parameters
- Defines sizes, policies, and thresholds
- Supports hardware-specific optimization

HardwareOptimizer:
- Optimizes cache config for specific hardware
- Calculates optimal sizes based on available resources
- Provides tensor-specific optimization suggestions

TensorMetadata:
- Stores metadata for cached tensors
- Tracks access patterns, frequency, timing
- Supports prediction algorithms
"""

# 4. USAGE EXAMPLES

def example_basic_usage():
    """
    Example 1: Basic usage of the hierarchical cache system
    """
    from src.qwen3_vl.optimization.hierarchical_cache_manager import (
        HierarchicalCacheManager, CacheConfig
    )
    
    # Create cache configuration
    config = CacheConfig(
        l1_cache_size=512 * 1024 * 1024,  # 512MB GPU cache
        l2_cache_size=1024 * 1024 * 1024, # 1GB CPU cache
        l3_cache_size=4 * 1024 * 1024 * 1024, # 4GB SSD cache
    )
    
    # Create cache manager
    cache_manager = HierarchicalCacheManager(config)
    
    # Create and cache a tensor
    tensor = torch.randn(100, 100, dtype=torch.float16)
    success = cache_manager.put_tensor(tensor, "example_tensor")
    print(f"Tensor cached successfully: {success}")
    
    # Retrieve tensor from cache
    retrieved_tensor, cache_level = cache_manager.get_tensor(
        tensor.shape, tensor.dtype
    )
    print(f"Retrieved from {cache_level}: {retrieved_tensor is not None}")
    
    # Get cache statistics
    stats = cache_manager.get_stats()
    print(f"Global hit rate: {stats['global_stats']['global_hit_rate']:.2%}")


def example_specialized_tensors():
    """
    Example 2: Using specialized tensor methods with the adapter
    """
    from src.qwen3_vl.optimization.hierarchical_cache_integration import (
        Qwen3VLHierarchicalCacheAdapter
    )
    
    # Create adapter (uses hardware-optimized defaults)
    adapter = Qwen3VLHierarchicalCacheAdapter()
    
    # Allocate different types of tensors
    attention_tensor = adapter.allocate_attention_weights(2, 8, 512, 64)
    print(f"Attention tensor: {attention_tensor.shape}")
    
    kv_cache_tensor = adapter.allocate_kv_cache(1, 16, 1024, 128)
    print(f"KV cache tensor: {kv_cache_tensor.shape}")
    
    image_tensor = adapter.allocate_image_features(1, 576, 1152)
    print(f"Image features tensor: {image_tensor.shape}")
    
    text_tensor = adapter.allocate_text_embeddings(2, 512, 4096)
    print(f"Text embeddings tensor: {text_tensor.shape}")
    
    # Perform cache maintenance
    adapter.perform_cache_maintenance()
    
    # Get statistics
    stats = adapter.get_cache_statistics()
    print(f"Cache hit rate: {stats['hierarchical_cache_stats']['global_stats']['global_hit_rate']:.2%}")


def example_hardware_optimized():
    """
    Example 3: Using hardware-optimized cache manager
    """
    from src.qwen3_vl.optimization.hardware_specific_optimizations import (
        create_hardware_optimized_cache_manager
    )
    
    # Create hardware-optimized cache manager
    hw_cache_manager = create_hardware_optimized_cache_manager()
    
    # Get optimization report
    report = hw_cache_manager.get_hardware_optimization_report()
    print("Hardware optimization applied:")
    print(f"  L1 size: {report['calculated_cache_config'].l1_cache_size / (1024**2):.0f}MB")
    print(f"  L2 size: {report['calculated_cache_config'].l2_cache_size / (1024**2):.0f}MB")
    print(f"  L3 size: {report['calculated_cache_config'].l3_cache_size / (1024**3):.1f}GB")
    
    # Use the optimized cache
    tensor = hw_cache_manager.get_tensor((512, 512), torch.float16, "general")
    print(f"Created tensor with hardware optimization: {tensor.shape}")


def example_integration_with_existing_pools():
    """
    Example 4: Integration with existing memory pool systems
    """
    from src.qwen3_vl.optimization.hierarchical_cache_integration import (
        HierarchicalCacheIntegration
    )
    
    # Create mock pool configuration
    class MockPoolConfig:
        def __init__(self):
            self.memory_pool_base_capacity = 512 * 1024 * 1024
            self.memory_pool_dtype = torch.float16
            self.memory_pool_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pool_config = MockPoolConfig()
    
    # Create integrated cache system
    integration = HierarchicalCacheIntegration(pool_config=pool_config)
    
    # Use the integrated system - will try hierarchical cache first, then fall back to pools
    tensor = integration.get_tensor((256, 256), "test", torch.float16)
    print(f"Got tensor from integrated system: {tensor.shape}")
    
    # Get integration statistics
    stats = integration.get_integration_stats()
    print(f"Cache hit ratio: {stats['cache_to_pool_hit_ratio']:.2%}")
    print(f"Pool hit ratio: {stats['pool_hit_ratio']:.2%}")


# 5. CONFIGURATION

def example_custom_configuration():
    """
    Example 5: Custom cache configuration
    """
    from src.qwen3_vl.optimization.hierarchical_cache_manager import CacheConfig
    
    # Custom configuration with specific parameters
    custom_config = CacheConfig(
        # Cache sizes
        l1_cache_size=1024 * 1024 * 1024,      # 1GB
        l2_cache_size=2 * 1024 * 1024 * 1024, # 2GB
        l3_cache_size=8 * 1024 * 1024 * 1024, # 8GB
        
        # Eviction policies
        l1_eviction_policy="lru",
        l2_eviction_policy="lru", 
        l3_eviction_policy="fifo",
        
        # Features
        l2_pin_memory=True,      # Use pinned memory for CPU cache
        l3_compression=True,     # Compress tensors in SSD cache
        
        # Migration thresholds
        migration_threshold_high_freq=4,      # Access count for L1 promotion
        migration_threshold_medium_freq=2,    # Access count for L2 promotion
        migration_time_threshold=600.0,       # 10 minutes for L3 migration
        
        # Prediction parameters
        prediction_window=200,                # Look at last 200 accesses
        prediction_threshold=0.7              # 70% confidence for prediction
    )
    
    print("Custom configuration created with:")
    print(f"  L1: {custom_config.l1_cache_size / (1024**3):.1f}GB")
    print(f"  L2: {custom_config.l2_cache_size / (1024**3):.1f}GB") 
    print(f"  L3: {custom_config.l3_cache_size / (1024**3):.1f}GB")
    print(f"  High freq threshold: {custom_config.migration_threshold_high_freq}")


# 6. PERFORMANCE CONSIDERATIONS

"""
Performance Best Practices:

1. Tensor Size Optimization:
   - Small tensors (< 1MB) work best in L1
   - Medium tensors (1-100MB) work best in L2
   - Large tensors (>100MB) work best in L3

2. Access Pattern Optimization:
   - Frequently accessed tensors will automatically migrate to L1
   - Infrequently accessed tensors will migrate to L3
   - Access patterns are tracked to predict future needs

3. Memory Management:
   - Use pinned memory (l2_pin_memory=True) for faster GPU transfers
   - Enable compression (l3_compression=True) for efficient SSD usage
   - Monitor cache utilization to avoid excessive evictions

4. Threading Considerations:
   - The system is thread-safe with internal locking
   - For high-throughput applications, consider batch operations
   - Monitor migration overhead during peak usage times
"""


# 7. HARDWARE OPTIMIZATIONS

"""
Hardware-Specific Optimizations for Intel i5-10210U + NVIDIA SM61 + NVMe SSD:

1. Cache Size Calculations:
   - L1: Up to 25% of GPU memory (max 1GB)
   - L2: Up to 20% of system RAM (max 2GB)
   - L3: Up to 50% of available RAM as SSD cache (max 10GB)

2. Memory Bandwidth Optimization:
   - Optimized for CPU memory bandwidth ~42.7 GB/s
   - Optimized for GPU memory bandwidth ~128 GB/s
   - Uses pinned memory for efficient CPU-GPU transfers

3. Threading Optimization:
   - Limits cache threads to avoid contention on 4-core CPU
   - Dedicated threads for prefetching and migrations
   - Optimized I/O thread configuration for NVMe SSD

4. Tensor-Specific Optimizations:
   - Attention tensors: Optimized for matrix operations
   - KV cache tensors: Optimized for sequential access
   - Image embeddings: Optimized for vision processing
   - Text embeddings: Optimized for language processing
"""


# 8. INTEGRATION WITH EXISTING SYSTEMS

"""
Integration with Existing Memory Pool Systems:

The hierarchical cache system seamlessly integrates with existing Qwen3-VL 
memory pool systems:

1. Hierarchical First Approach:
   - Tries hierarchical cache first
   - Falls back to existing pools if not found
   - Automatically feeds frequently accessed tensors back to cache

2. Backward Compatibility:
   - All existing memory pool APIs continue to work
   - New hierarchical features are opt-in
   - Gradual migration path from old to new system

3. Performance Benefits:
   - Reduces pressure on existing pools
   - Improves overall memory access patterns
   - Provides better tensor reuse across the system
"""


# MAIN EXAMPLE COMBINING ALL FEATURES
def main_example():
    """
    Complete example combining all features of the hierarchical cache system
    """
    print("=== Qwen3-VL Hierarchical Cache System Example ===\n")
    
    # 1. Use hardware-optimized cache manager
    from src.qwen3_vl.optimization.hardware_specific_optimizations import (
        create_hardware_optimized_cache_manager
    )
    
    print("1. Creating hardware-optimized cache manager...")
    hw_cache = create_hardware_optimized_cache_manager()
    
    report = hw_cache.get_hardware_optimization_report()
    print(f"   Optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD")
    print(f"   L1: {report['calculated_cache_config'].l1_cache_size / (1024**2):.0f}MB, "
          f"L2: {report['calculated_cache_config'].l2_cache_size / (1024**2):.0f}MB, "
          f"L3: {report['calculated_cache_config'].l3_cache_size / (1024**3):.1f}GB")
    
    # 2. Create various tensors
    print("\n2. Creating and caching different tensor types...")
    
    # Attention tensor (high frequency)
    attn_tensor = hw_cache.get_tensor((4, 8, 256, 256), torch.float16, "attention")
    print(f"   Attention tensor: {attn_tensor.shape}")
    
    # KV cache tensor (medium frequency)
    kv_tensor = hw_cache.get_tensor((1, 16, 1024, 64), torch.float16, "kv_cache")
    print(f"   KV cache tensor: {kv_tensor.shape}")
    
    # Image embeddings tensor (medium frequency)
    img_tensor = hw_cache.get_tensor((1, 576, 1152), torch.float16, "image_embeddings")
    print(f"   Image embeddings: {img_tensor.shape}")
    
    # 3. Simulate access patterns
    print("\n3. Simulating access patterns to trigger migrations...")
    
    # Access attention tensor multiple times (should migrate to L1)
    for i in range(5):
        _, _ = hw_cache.get_tensor((4, 8, 256, 256), torch.float16, "attention")
    
    # Perform cache maintenance
    from src.qwen3_vl.optimization.hierarchical_cache_manager import HierarchicalCacheManager
    if hasattr(hw_cache, 'cache_manager') and isinstance(hw_cache.cache_manager, HierarchicalCacheManager):
        hw_cache.cache_manager._perform_migrations()
    
    # 4. Get statistics
    print("\n4. Cache statistics:")
    stats = hw_cache.get_stats()
    
    print(f"   Global hit rate: {stats['global_stats']['global_hit_rate']:.2%}")
    print(f"   Total migrations: {stats['global_stats']['migrations']}")
    print(f"   L1 hit rate: {stats['l1_stats']['hit_rate']:.2%} (util: {stats['l1_stats']['utilization']:.1%})")
    print(f"   L2 hit rate: {stats['l2_stats']['hit_rate']:.2%} (util: {stats['l2_stats']['utilization']:.1%})")
    print(f"   L3 hit rate: {stats['l3_stats']['hit_rate']:.2%} (util: {stats['l3_stats']['utilization']:.1%})")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    import torch
    main_example()