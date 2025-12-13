"""Demo script for the unified configuration system."""

import torch
import time
from pathlib import Path
import tempfile
import os
import json
from typing import Dict, Any

from src.qwen3_vl.core.config import (
    UnifiedConfig, 
    MemoryConfig, 
    CPUConfig, 
    GPUConfig, 
    PowerManagementConfig,
    OptimizationConfig,
    UnifiedConfigManager,
    get_default_config,
    create_unified_config_manager
)


def demo_basic_configuration():
    """Demo basic configuration usage."""
    print("=== Basic Configuration Demo ===")
    
    # Create default configuration
    config = get_default_config()
    print(f"Default config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Torch dtype: {config.torch_dtype}")
    print(f"  - Optimization level: {config.optimization_level}")
    
    # Create manager and get different optimization levels
    manager = create_unified_config_manager()
    
    minimal_config = manager.get_config("minimal")
    print(f"\nMinimal config: Use MoE={minimal_config.optimization_config.use_moe}, Use sparsity={minimal_config.optimization_config.use_sparsity}")
    
    balanced_config = manager.get_config("balanced")
    print(f"Balanced config: Use MoE={balanced_config.optimization_config.use_moe}, Use sparsity={balanced_config.optimization_config.use_sparsity}")
    
    aggressive_config = manager.get_config("aggressive")
    print(f"Aggressive config: Use MoE={aggressive_config.optimization_config.use_moe}, Use sparsity={aggressive_config.optimization_config.use_sparsity}")


def demo_configuration_validation():
    """Demo configuration validation."""
    print("\n=== Configuration Validation Demo ===")
    
    manager = create_unified_config_manager()
    
    # Valid configuration
    valid_config = UnifiedConfig()
    is_valid = manager.validate_config(valid_config)
    print(f"Valid config validation: {is_valid}")
    
    # Invalid configuration (for demonstration)
    try:
        invalid_config = UnifiedConfig(
            hidden_size=512,
            num_attention_heads=7  # Not a divisor of 512
        )
        is_valid = manager.validate_config(invalid_config)
        print(f"Invalid config validation: {is_valid}")
    except ValueError as e:
        print(f"Caught expected validation error: {e}")


def demo_configuration_serialization():
    """Demo configuration serialization."""
    print("\n=== Configuration Serialization Demo ===")
    
    config = get_default_config()
    
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
        loaded_json_config = UnifiedConfig.from_file(json_file)
        print(f"Loaded from JSON: {loaded_json_config.num_hidden_layers} layers, {loaded_json_config.num_attention_heads} heads")
        
        # Load from YAML
        loaded_yaml_config = UnifiedConfig.from_file(yaml_file)
        print(f"Loaded from YAML: {loaded_yaml_config.num_hidden_layers} layers, {loaded_yaml_config.num_attention_heads} heads")
    finally:
        # Clean up temporary files
        os.unlink(json_file)
        os.unlink(yaml_file)


def demo_hardware_optimized_configuration():
    """Demo hardware-optimized configuration."""
    print("\n=== Hardware-Optimized Configuration Demo ===")
    
    manager = create_unified_config_manager()
    
    # Hardware specs for Intel i5-10210U + NVIDIA SM61
    hardware_specs = {
        "gpu_memory": 6 * 1024 * 1024 * 1024,  # 6GB
        "cpu_cores": 4,
        "memory_gb": 8,
        "storage_type": "nvme"
    }
    
    # Get hardware-optimized config
    hw_config = manager.get_hardware_optimized_config(hardware_specs)
    print(f"Hardware-optimized config:")
    print(f"  - GPU memory: {hw_config.gpu_config.gpu_memory_size / (1024**3):.1f}GB")
    print(f"  - CPU threads: {hw_config.cpu_config.num_threads}")
    print(f"  - Memory pool size: {hw_config.memory_config.memory_pool_size / (1024**3):.1f}GB")
    print(f"  - Use mixed precision: {hw_config.optimization_config.use_mixed_precision}")
    print(f"  - Sparsity ratio: {hw_config.optimization_config.sparsity_ratio}")
    print(f"  - MoE experts: {hw_config.optimization_config.moe_num_experts}")


def demo_runtime_configuration_updates():
    """Demo runtime configuration updates."""
    print("\n=== Runtime Configuration Updates Demo ===")
    
    manager = create_unified_config_manager()
    config = manager.get_config("balanced")
    
    print(f"Original config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
    
    # Update configuration
    updates = {
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "optimization_config": {
            "use_sparsity": True,
            "sparsity_ratio": 0.4,
            "use_moe": True,
            "moe_num_experts": 6,
            "moe_top_k": 3
        }
    }
    
    updated_config = manager.update_config(config, updates)
    print(f"Updated config: {updated_config.num_hidden_layers} layers, {updated_config.num_attention_heads} heads")
    print(f"  - Sparsity ratio: {updated_config.optimization_config.sparsity_ratio}")
    print(f"  - MoE experts: {updated_config.optimization_config.moe_num_experts}")
    print(f"  - MoE top_k: {updated_config.optimization_config.moe_top_k}")


def demo_multiple_sources():
    """Demo configuration from multiple sources."""
    print("\n=== Multiple Sources Configuration Demo ===")
    
    manager = create_unified_config_manager()
    source_manager = manager.source_manager
    
    # Add configuration from different sources
    source_manager.add_source("base_config", {"num_hidden_layers": 32, "num_attention_heads": 32})
    source_manager.add_source("optimization_config", {
        "optimization_config": {
            "use_sparsity": True,
            "sparsity_ratio": 0.3
        }
    })
    
    # Create a config combining sources
    base_dict = source_manager.sources["base_config"]
    opt_dict = source_manager.sources["optimization_config"]
    combined_dict = source_manager.merge_configs(base_dict, opt_dict)
    combined_config = UnifiedConfig.from_dict(combined_dict)
    
    print(f"Combined config: {combined_config.num_hidden_layers} layers, {combined_config.num_attention_heads} heads")
    print(f"  - Sparsity enabled: {combined_config.optimization_config.use_sparsity}")
    print(f"  - Sparsity ratio: {combined_config.optimization_config.sparsity_ratio}")


def demo_backward_compatibility():
    """Demo backward compatibility."""
    print("\n=== Backward Compatibility Demo ===")
    
    # Get legacy-style config
    from src.qwen3_vl.core.config import get_legacy_config, update_legacy_config
    
    legacy_dict = get_legacy_config()
    print(f"Legacy config keys: {len(legacy_dict)}")
    print(f"  - num_hidden_layers: {legacy_dict.get('num_hidden_layers', 'N/A')}")
    print(f"  - num_attention_heads: {legacy_dict.get('num_attention_heads', 'N/A')}")
    print(f"  - hidden_size: {legacy_dict.get('hidden_size', 'N/A')}")
    
    # Update legacy config
    updates = {
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "use_sparsity": True,
        "sparsity_ratio": 0.4
    }
    
    updated_legacy_dict = update_legacy_config(legacy_dict, updates)
    print(f"Updated legacy config:")
    print(f"  - num_hidden_layers: {updated_legacy_dict.get('num_hidden_layers', 'N/A')}")
    print(f"  - num_attention_heads: {updated_legacy_dict.get('num_attention_heads', 'N/A')}")
    print(f"  - use_sparsity: {updated_legacy_dict.get('use_sparsity', 'N/A')}")
    print(f"  - sparsity_ratio: {updated_legacy_dict.get('sparsity_ratio', 'N/A')}")


def demo_performance_with_different_configs():
    """Demo performance comparison with different configurations."""
    print("\n=== Performance Comparison Demo ===")
    
    manager = create_unified_config_manager()
    
    # Create configurations with different optimization levels
    configs = {
        "minimal": manager.get_config("minimal"),
        "balanced": manager.get_config("balanced"),
        "aggressive": manager.get_config("aggressive")
    }
    
    print("Configuration comparison:")
    for name, config in configs.items():
        print(f"  {name.capitalize()} config:")
        print(f"    - Use memory pooling: {config.memory_config.enable_memory_pooling}")
        print(f"    - Use sparsity: {config.optimization_config.use_sparsity}")
        print(f"    - Use MoE: {config.optimization_config.use_moe}")
        print(f"    - Sparsity ratio: {getattr(config.optimization_config, 'sparsity_ratio', 'N/A')}")
        print(f"    - MoE experts: {getattr(config.optimization_config, 'moe_num_experts', 'N/A')}")
        print(f"    - MoE top_k: {getattr(config.optimization_config, 'moe_top_k', 'N/A')}")


def main():
    """Run all demos."""
    print("Unified Configuration System Demo")
    print("=" * 50)
    
    demo_basic_configuration()
    demo_configuration_validation()
    demo_configuration_serialization()
    demo_hardware_optimized_configuration()
    demo_runtime_configuration_updates()
    demo_multiple_sources()
    demo_backward_compatibility()
    demo_performance_with_different_configs()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()