# Hierarchical Cache System for Qwen3-VL

This project implements an advanced hierarchical caching and buffering system for the Qwen3-VL model, optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD hardware.

## Overview

The system implements a three-level hierarchical cache:
- **L1**: GPU memory cache for high-frequency accessed tensors
- **L2**: CPU pinned memory cache for medium-frequency accessed tensors  
- **L3**: NVMe SSD cache for low-frequency accessed tensors

The system includes intelligent access pattern tracking, ML-based access prediction, and dynamic migration policies between cache levels.

## Features

- **Three-level cache hierarchy** (L1/L2/L3) with automatic tensor migration
- **Access pattern tracking** to identify hot/cold tensors
- **ML-based prediction** for future access patterns
- **Hardware-specific optimizations** for Intel i5-10210U + NVIDIA SM61 + NVMe SSD
- **Seamless integration** with existing memory pool systems
- **Thread-safe implementation** with internal locking
- **Comprehensive statistics** and monitoring

## Architecture

### Core Components
- `HierarchicalCacheManager`: Main coordinator for all cache levels
- `L1GPUCache`: GPU memory cache with LRU eviction
- `L2CPUCache`: CPU pinned memory cache for efficient GPU transfers
- `L3SSDCache`: NVMe SSD cache with optional compression
- `AccessPatternTracker`: Tracks tensor access patterns
- `SimpleMLPredictor`: Predicts future tensor access

### Integration Components
- `HierarchicalCacheIntegration`: Bridges with existing memory pools
- `Qwen3VLHierarchicalCacheAdapter`: Main interface for Qwen3-VL

### Hardware Optimizations
- `HardwareOptimizer`: Calculates optimal cache sizes for specific hardware
- `IntelNvidiaCacheManager`: Hardware-optimized cache manager

## Usage Examples

### Basic Usage
```python
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

# Retrieve tensor from cache
retrieved_tensor, cache_level = cache_manager.get_tensor(
    tensor.shape, tensor.dtype
)
```

### Hardware-Optimized Usage
```python
from src.qwen3_vl.optimization.hardware_specific_optimizations import (
    create_hardware_optimized_cache_manager
)

# Create hardware-optimized cache manager
hw_cache_manager = create_hardware_optimized_cache_manager()

# Use the optimized cache
tensor = hw_cache_manager.get_tensor((512, 512), torch.float16, "general")
```

### Integration with Existing Systems
```python
from src.qwen3_vl.optimization.hierarchical_cache_integration import (
    Qwen3VLHierarchicalCacheAdapter
)

# Create adapter
adapter = Qwen3VLHierarchicalCacheAdapter()

# Allocate specialized tensors
attention_tensor = adapter.allocate_attention_weights(2, 8, 512, 64)
kv_cache_tensor = adapter.allocate_kv_cache(1, 16, 1024, 128)
```

## Configuration

The system supports extensive configuration through the `CacheConfig` class:

```python
config = CacheConfig(
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
    migration_time_threshold=600.0,       # Time for L3 migration
    
    # Prediction parameters
    prediction_window=200,                # Look at last 200 accesses
    prediction_threshold=0.7              # Confidence threshold
)
```

## Performance Considerations

- Small tensors (< 1MB) work best in L1
- Medium tensors (1-100MB) work best in L2  
- Large tensors (>100MB) work best in L3
- Use pinned memory for faster GPU transfers
- Enable compression for efficient SSD usage
- Monitor cache utilization to avoid excessive evictions

## Hardware Optimizations

The system includes specific optimizations for Intel i5-10210U + NVIDIA SM61 + NVMe SSD:
- Cache size calculations based on available hardware resources
- Memory bandwidth optimization for CPU-GPU transfers
- Threading optimization for 4-core CPU
- Tensor-specific optimizations for different tensor types

## Testing

Run the comprehensive test suite:
```bash
python -m tests.test_hierarchical_cache_system
```

## Files

- `src/qwen3_vl/optimization/hierarchical_cache_manager.py` - Core hierarchical cache implementation
- `src/qwen3_vl/optimization/hierarchical_cache_integration.py` - Integration with existing systems
- `src/qwen3_vl/optimization/hardware_specific_optimizations.py` - Hardware-specific optimizations
- `tests/test_hierarchical_cache_system.py` - Comprehensive test suite
- `docs/hierarchical_cache_system_documentation.py` - Documentation and examples

## Performance Benefits

- Reduces memory allocation overhead
- Improves cache hit rates through intelligent migration
- Optimizes memory access patterns based on tensor usage
- Reduces pressure on existing memory pool systems
- Provides better tensor reuse across the system