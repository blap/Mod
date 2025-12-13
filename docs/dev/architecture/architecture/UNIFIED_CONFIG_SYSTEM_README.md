# Unified Configuration System for Qwen3-VL Model

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
7. **Qwen3VLConfig**: Main configuration that combines all specialized configurations

## Key Features

- **Full Capacity Preservation**: Maintains 32 transformer layers and 32 attention heads as required
- **Hardware-Aware Configuration**: Optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD configuration
- **Multi-Source Support**: Loads configuration from files, environment variables, and programmatic sources
- **Validation & Type Checking**: Comprehensive validation of configuration parameters
- **Runtime Updates**: Ability to update configurations at runtime
- **Backward Compatibility**: Maintains compatibility with existing configuration patterns
- **Specialized Sections**: Dedicated configuration sections for different components (memory, CPU, GPU, power management)

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

## Usage Examples

### Basic Usage
```python
from src.qwen3_vl.core.config import Qwen3VLConfig

# Get the default configuration
config = Qwen3VLConfig()
print(f"Config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
```

### Loading from File
```python
# Load from JSON or YAML file
config = Qwen3VLConfig.from_file("path/to/config.json")
config = Qwen3VLConfig.from_file("path/to/config.yaml")
```

### Creating Configurations with Different Optimization Levels
```python
# Minimal optimization config
minimal_config = Qwen3VLConfig(
    optimization_config=OptimizationConfig(
        use_memory_pooling=False,
        use_sparsity=False,
        use_moe=False,
        # ... other optimizations disabled
    ),
    optimization_level="minimal"
)

# Aggressive optimization config
aggressive_config = Qwen3VLConfig(
    optimization_config=OptimizationConfig(
        use_memory_pooling=True,
        use_sparsity=True,
        sparsity_ratio=0.6,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        # ... other optimizations enabled
    ),
    optimization_level="aggressive"
)
```

### Hardware-Specific Configuration
```python
# Create configuration optimized for specific hardware
config = Qwen3VLConfig(
    gpu_config=GPUConfig(
        gpu_compute_capability=(6, 1),  # SM61
        memory_bandwidth_gbps=320.0,
        # ... other GPU-specific optimizations
    ),
    cpu_config=CPUConfig(
        num_threads=4,
        l3_cache_size=6 * 1024 * 1024,  # 6MB
        # ... other CPU-specific optimizations
    ),
    hardware_target="intel_i5_10210u_nvidia_sm61_nvme"
)
```

### Serialization and Deserialization
```python
# Save configuration to file
config.save_to_file("config.json")
config.save_to_file("config.yaml")

# Load configuration from dictionary
config_dict = config.to_dict()
restored_config = Qwen3VLConfig.from_dict(config_dict)
```

### Runtime Configuration Updates
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

# Apply updates (you would typically create a new config from the updated dict)
updated_config_dict = {**config.to_dict(), **updates}
updated_config = Qwen3VLConfig.from_dict(updated_config_dict)
```

## Validation

The configuration system includes comprehensive validation to ensure:
- Core model parameters maintain full capacity (32 layers, 32 attention heads)
- Hidden size is divisible by number of attention heads
- Sparsity ratios are between 0 and 1
- MoE parameters are valid (num_experts ≥ 2, 1 ≤ top_k ≤ num_experts)
- Memory configuration parameters are positive
- CPU/GPU configuration parameters are valid

## Integration with Model Components

The unified configuration system seamlessly integrates with all model components:
- Attention mechanisms (standard, flash, sparse, etc.)
- MLP implementations
- Transformer layers
- Vision encoders
- Multimodal projectors
- Memory managers

## Performance Considerations

1. **Memory Pooling**: Reduces memory allocation overhead
2. **Tiered Memory**: Optimizes memory access patterns across GPU/CPU/SSD
3. **Hardware-Specific Configs**: Optimizes for target hardware capabilities
4. **Adaptive Optimizations**: Dynamically adjusts based on input complexity
5. **Compressed Representations**: Reduces memory footprint while maintaining capacity

## Best Practices

1. Always validate configurations before using them in production
2. Use appropriate optimization levels based on your hardware constraints:
   - "minimal": For basic functionality with minimal optimizations
   - "balanced": For good performance with reasonable resource usage
   - "aggressive": For maximum performance with higher resource usage
3. Consider hardware specifications when creating configurations
4. Use the `from_pretrained` method to load configurations from model repositories
5. Take advantage of the hardware-aware configuration system for optimal performance

## Backward Compatibility

The system maintains backward compatibility with legacy configuration approaches:
- All original configuration parameters are preserved
- New optimization parameters are optional additions
- Existing code continues to work without modifications
- Enhanced functionality is available through new APIs