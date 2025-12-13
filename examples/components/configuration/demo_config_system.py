"""Demo script for the unified configuration system."""

import torch
import time
from pathlib import Path
import tempfile
import os
import json
from typing import Dict, Any


from src.qwen3_vl.core.config import Qwen3VLConfig, OptimizationConfig, GPUConfig, CPUConfig


def demo_basic_configuration():
    """Demo basic configuration usage."""
    print("=== Basic Configuration Demo ===")
    
    # Create default configuration
    config = Qwen3VLConfig()
    print(f"Default config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Torch dtype: {config.torch_dtype}")
    
    # Show sub-configurations
    print(f"  - Memory tiering enabled: {config.memory_config.enable_memory_tiering}")
    print(f"  - CPU config threads: {config.cpu_config.num_threads}")
    print(f"  - GPU config compute capability: {config.gpu_config.gpu_compute_capability}")
    print(f"  - Optimization config sparsity: {config.optimization_config.use_sparsity}")


def demo_configuration_serialization():
    """Demo configuration serialization."""
    print("\n=== Configuration Serialization Demo ===")
    
    config = Qwen3VLConfig(
        num_hidden_layers=8,
        num_attention_heads=16,
        hidden_size=1024
    )
    
    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_file = f.name
    
    try:
        # Save to JSON
        config.save_to_file(json_file)
        print(f"Saved config to JSON: {json_file}")
        
        # Save to YAML
        config.save_to_file(yaml_file)
        print(f"Saved config to YAML: {yaml_file}")
        
        # Load from JSON
        loaded_json_config = Qwen3VLConfig.from_file(json_file)
        print(f"Loaded from JSON: {loaded_json_config.num_hidden_layers} layers, {loaded_json_config.num_attention_heads} heads")
        
        # Load from YAML
        loaded_yaml_config = Qwen3VLConfig.from_file(yaml_file)
        print(f"Loaded from YAML: {loaded_yaml_config.num_hidden_layers} layers, {loaded_yaml_config.num_attention_heads} heads")
    finally:
        # Clean up temporary files
        os.unlink(json_file)
        os.unlink(yaml_file)


def demo_different_optimization_levels():
    """Demo different optimization levels."""
    print("\n=== Different Optimization Levels Demo ===")
    
    # Create configurations with different optimization settings
    configs = {
        "minimal": Qwen3VLConfig(
            optimization_config=OptimizationConfig(
                use_memory_pooling=False,
                use_sparsity=False,
                use_moe=False,
                use_flash_attention_2=False,
                performance_improvement_threshold=0.0,
                accuracy_preservation_threshold=0.9
            ),
            optimization_level="minimal"
        ),
        "balanced": Qwen3VLConfig(
            optimization_config=OptimizationConfig(
                use_memory_pooling=True,
                use_sparsity=True,
                sparsity_ratio=0.3,
                use_moe=True,
                moe_num_experts=2,
                moe_top_k=1,
                use_flash_attention_2=True,
                performance_improvement_threshold=0.05,
                accuracy_preservation_threshold=0.95
            ),
            optimization_level="balanced"
        ),
        "aggressive": Qwen3VLConfig(
            optimization_config=OptimizationConfig(
                use_memory_pooling=True,
                use_sparsity=True,
                sparsity_ratio=0.6,
                use_moe=True,
                moe_num_experts=4,
                moe_top_k=2,
                use_flash_attention_2=True,
                use_dynamic_sparse_attention=True,
                use_adaptive_depth=True,
                performance_improvement_threshold=0.1,
                accuracy_preservation_threshold=0.9
            ),
            optimization_level="aggressive"
        )
    }
    
    for name, config in configs.items():
        print(f"{name.capitalize()} config:")
        print(f"  - Memory pooling: {config.optimization_config.use_memory_pooling}")
        print(f"  - Sparsity: {config.optimization_config.use_sparsity}")
        print(f"  - Sparsity ratio: {config.optimization_config.sparsity_ratio}")
        print(f"  - MoE: {config.optimization_config.use_moe}")
        print(f"  - MoE experts: {config.optimization_config.moe_num_experts}")
        print(f"  - MoE top_k: {config.optimization_config.moe_top_k}")
        print(f"  - Flash attention: {config.optimization_config.use_flash_attention_2}")


def demo_hardware_optimized_configuration():
    """Demo hardware-optimized configuration."""
    print("\n=== Hardware-Optimized Configuration Demo ===")
    
    # Create a config with hardware-specific optimizations
    hardware_config = Qwen3VLConfig(
        gpu_config=GPUConfig(
            gpu_compute_capability=(6, 1),  # SM61
            max_threads_per_block=1024,
            shared_memory_per_block=48 * 1024,  # 48KB
            memory_bandwidth_gbps=320.0
        ),
        cpu_config=CPUConfig(
            num_threads=4,
            l3_cache_size=6 * 1024 * 1024,  # 6MB
            simd_instruction_set="avx2"
        ),
        hardware_target="intel_i5_10210u_nvidia_sm61_nvme",
        target_hardware="nvidia_sm61"
    )
    
    print(f"Hardware-optimized config:")
    print(f"  - GPU compute capability: {hardware_config.gpu_config.gpu_compute_capability}")
    print(f"  - CPU threads: {hardware_config.cpu_config.num_threads}")
    print(f"  - L3 cache size: {hardware_config.cpu_config.l3_cache_size / (1024*1024)}MB")
    print(f"  - SIMD instruction set: {hardware_config.cpu_config.simd_instruction_set}")
    print(f"  - Hardware target: {hardware_config.hardware_target}")


def main():
    """Run all demos."""
    print("Unified Configuration System Demo")
    print("=" * 50)
    
    demo_basic_configuration()
    demo_configuration_serialization()
    demo_different_optimization_levels()
    demo_hardware_optimized_configuration()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()