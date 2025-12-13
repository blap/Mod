# Comprehensive Unified Configuration System for Qwen3-VL

## Overview

The unified configuration system provides a centralized, production-ready approach to managing all configuration aspects of the Qwen3-VL model. It consolidates all scattered configuration approaches into a single, maintainable system with support for multiple sources, validation, type checking, and runtime updates.

## Architecture

### Core Components

1. **BaseConfig**: Base configuration class with validation and serialization capabilities
2. **MemoryConfig**: Memory optimization configuration
3. **CPUConfig**: CPU optimization configuration  
4. **GPUConfig**: GPU optimization configuration
5. **PowerManagementConfig**: Power management configuration
6. **OptimizationConfig**: Configuration for all optimization techniques
7. **UnifiedConfig**: Main configuration that combines all specialized configurations
8. **ConfigValidator**: Validates configuration parameters and compatibility
9. **ConfigSourceManager**: Manages configuration from multiple sources
10. **UnifiedConfigManager**: Centralized configuration manager

## Usage Guide

### Basic Usage

```python
from src.qwen3_vl.core.config import UnifiedConfig, UnifiedConfigManager

# Get the default configuration
config = get_default_config()

# Or create a manager to get configurations with different optimization levels
config_manager = create_unified_config_manager()
balanced_config = config_manager.get_config("balanced")
minimal_config = config_manager.get_config("minimal")
aggressive_config = config_manager.get_config("aggressive")
```

### Loading from Different Sources

#### From File
```python
# Load from JSON or YAML file
config = UnifiedConfig.from_file("path/to/config.json")
config = UnifiedConfig.from_file("path/to/config.yaml")

# Or load through manager
config_manager.load_config_from_file("path/to/config.json", "my_config")
```

#### From Environment Variables
```python
# Load from environment variables with prefix (default: QWEN3_)
config_manager.load_config_from_env("env_config", "QWEN3_")
```

### Updating Configuration at Runtime

```python
# Update configuration with new values
updates = {
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "optimization_config": {
        "use_sparsity": True,
        "sparsity_ratio": 0.4
    }
}

updated_config = config_manager.update_config(current_config, updates)
```

### Hardware-Specific Configuration

```python
# Get configuration optimized for specific hardware
hardware_specs = {
    "gpu_memory": 6 * 1024 * 1024 * 1024,  # 6GB
    "cpu_cores": 4,
    "memory_gb": 8,
    "storage_type": "nvme"
}

hw_optimized_config = config_manager.get_hardware_optimized_config(hardware_specs)
```

## Configuration Parameters

### Core Model Configuration
- `num_hidden_layers`: Number of hidden layers (default: 32, preserves full capacity)
- `num_attention_heads`: Number of attention heads (default: 32, preserves full capacity)
- `hidden_size`: Size of hidden layers (default: 4096)
- `intermediate_size`: Size of intermediate layers (default: 11008)
- `vocab_size`: Vocabulary size (default: 152064)
- `max_position_embeddings`: Maximum position embeddings (default: 32768)

### Vision Model Configuration
- `vision_num_hidden_layers`: Number of vision transformer layers (default: 24)
- `vision_num_attention_heads`: Number of vision attention heads (default: 16)
- `vision_hidden_size`: Size of vision hidden layers (default: 1152)
- `vision_image_size`: Size of vision input images (default: 448)
- `vision_patch_size`: Patch size for vision processing (default: 14)

### Memory Configuration (`MemoryConfig`)
- `memory_pool_size`: Size of memory pool (default: 2GB)
- `enable_memory_tiering`: Enable memory tiering (default: True)
- `gpu_memory_size`: GPU memory size (default: 6GB)
- `cpu_memory_size`: CPU memory size (default: 8GB)
- `ssd_memory_size`: SSD memory size (default: 50GB)
- `enable_memory_compression`: Enable memory compression (default: True)
- `compression_level`: Compression level ("low", "medium", "high") (default: "medium")
- `enable_memory_swapping`: Enable memory swapping (default: True)
- `swap_threshold`: Memory usage threshold for swapping (default: 0.8)
- `enable_memory_defragmentation`: Enable memory defragmentation (default: True)

### CPU Configuration (`CPUConfig`)
- `num_threads`: Number of CPU threads (default: 4)
- `num_workers`: Number of CPU workers (default: 4)
- `l1_cache_size`: L1 cache size (default: 32KB)
- `l2_cache_size`: L2 cache size (default: 256KB)
- `l3_cache_size`: L3 cache size (default: 6MB)
- `enable_cpu_optimizations`: Enable CPU optimizations (default: True)
- `use_hyperthreading`: Use hyperthreading (default: True)
- `enable_simd_optimizations`: Enable SIMD optimizations (default: True)
- `simd_instruction_set`: SIMD instruction set ("avx2", "sse", "scalar") (default: "avx2")

### GPU Configuration (`GPUConfig`)
- `gpu_compute_capability`: GPU compute capability (default: (6, 1) for SM61)
- `max_threads_per_block`: Max threads per CUDA block (default: 1024)
- `shared_memory_per_block`: Shared memory per CUDA block (default: 48KB)
- `memory_bandwidth_gbps`: Memory bandwidth in GB/s (default: 320.0)
- `enable_gpu_optimizations`: Enable GPU optimizations (default: True)
- `use_tensor_cores`: Use tensor cores (default: True)
- `use_mixed_precision`: Use mixed precision (default: True)
- `attention_implementation`: Attention implementation ("flash_attention_2", etc.) (default: "flash_attention_2")

### Power Management Configuration (`PowerManagementConfig`)
- `enable_power_optimization`: Enable power optimization (default: True)
- `power_constraint`: Power usage constraint (default: 0.8)
- `thermal_constraint`: Thermal constraint in Celsius (default: 75.0)
- `performance_target`: Performance target (default: 0.9)
- `adaptation_frequency`: Adaptation frequency in seconds (default: 1.0)

### Optimization Configuration (`OptimizationConfig`)
- `use_memory_pooling`: Enable memory pooling (default: True)
- `use_hierarchical_memory_compression`: Enable hierarchical memory compression (default: True)
- `use_memory_efficient_attention`: Enable memory efficient attention (default: True)
- `use_sparsity`: Enable activation sparsity (default: True)
- `sparsity_ratio`: Sparsity ratio (default: 0.5)
- `use_dynamic_sparse_attention`: Enable dynamic sparse attention (default: True)
- `use_adaptive_precision`: Enable adaptive precision (default: True)
- `use_moe`: Enable mixture of experts (default: True)
- `moe_num_experts`: Number of MoE experts (default: 4)
- `moe_top_k`: Top-k experts to use (default: 2)
- `use_flash_attention_2`: Enable Flash Attention 2 (default: True)
- `use_adaptive_depth`: Enable adaptive depth (default: True)
- `use_gradient_checkpointing`: Enable gradient checkpointing (default: True)
- `use_context_adaptive_positional_encoding`: Enable context adaptive positional encoding (default: True)
- `use_conditional_feature_extraction`: Enable conditional feature extraction (default: True)
- `use_cross_modal_compression`: Enable cross-modal compression (default: True)
- `use_cross_layer_memory_sharing`: Enable cross-layer memory sharing (default: True)
- `use_hierarchical_vision`: Enable hierarchical vision (default: True)
- `use_learned_activation_routing`: Enable learned activation routing (default: True)
- `use_adaptive_batch_processing`: Enable adaptive batch processing (default: True)
- `use_adaptive_sequence_packing`: Enable adaptive sequence packing (default: True)
- `use_memory_efficient_grad_accumulation`: Enable memory efficient grad accumulation (default: True)
- `use_faster_rotary_embeddings`: Enable faster rotary embeddings (default: True)
- `use_hardware_specific_kernels`: Enable hardware specific kernels (default: True)

## Validation

The configuration system includes comprehensive validation to ensure:
- Core model parameters maintain full capacity (32 layers, 32 attention heads)
- Hidden size is divisible by number of attention heads
- Sparsity ratios are between 0 and 1
- MoE parameters are valid (num_experts ≥ 2, 1 ≤ top_k ≤ num_experts)
- Memory configuration parameters are positive
- CPU/GPU configuration parameters are valid

## Backward Compatibility

The system maintains backward compatibility with legacy configuration approaches:
- `get_legacy_config()` returns a dictionary-compatible configuration
- `update_legacy_config()` updates legacy configurations with new values
- All parameters are preserved and accessible through the unified system

## Best Practices

1. **Always validate configurations before use**:
   ```python
   is_valid = config_manager.validate_config(config)
   if not is_valid:
       # Handle invalid configuration
       pass
   ```

2. **Use appropriate optimization levels**:
   - "minimal": For basic functionality with minimal optimizations
   - "balanced": For good performance with reasonable resource usage
   - "aggressive": For maximum performance with higher resource usage

3. **Consider hardware specifications when creating configurations**:
   ```python
   hw_config = config_manager.get_hardware_optimized_config(hardware_specs)
   ```

4. **Use the manager for configuration lifecycle**:
   ```python
   manager = UnifiedConfigManager()
   config = manager.get_config("balanced")
   updated_config = manager.update_config(config, updates)
   ```

## Implementation Details

The unified configuration system is implemented with:
- Dataclass-based configuration classes for type safety
- Automatic validation in `__post_init__` methods
- Serialization/deserialization support for JSON/YAML
- Multiple inheritance levels for specialized configurations
- Hardware-aware optimization configuration
- Runtime configuration updates with validation
- Source management for configuration from multiple origins

## Performance Considerations

1. **Memory Pooling**: Reduces memory allocation overhead
2. **Tiered Memory**: Optimizes memory access patterns across GPU/CPU/SSD
3. **Hardware-Specific Configs**: Optimizes for target hardware capabilities
4. **Adaptive Optimizations**: Dynamically adjusts based on input complexity
5. **Compressed Representations**: Reduces memory footprint while maintaining capacity