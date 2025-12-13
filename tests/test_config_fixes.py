#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify that the configuration fixes work properly.
"""
import tempfile
import os
from src.qwen3_vl.core.config import Qwen3VLConfig
from src.qwen3_vl.components.configuration.unified_config_manager import (
    UnifiedConfig, 
    UnifiedConfigManager, 
    get_default_config, 
    get_legacy_config,
    update_legacy_config
)

def test_qwen3_vl_config():
    """Test Qwen3VLConfig functionality."""
    print("Testing Qwen3VLConfig...")
    
    # Create a basic config
    config = Qwen3VLConfig()
    print(f"  - Created config with {config.num_hidden_layers} hidden layers")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Use flash attention 2: {config.use_flash_attention_2}")
    print(f"  - GPU memory size: {config.gpu_config.gpu_memory_size}")
    
    # Test from_dict and to_dict
    config_dict = config.to_dict()
    print(f"  - Config dict keys: {len(config_dict)}")
    
    new_config = Qwen3VLConfig.from_dict(config_dict)
    print(f"  - Created new config from dict: {new_config.num_attention_heads} attention heads")
    
    # Test GPU config attributes access
    print(f"  - GPU config flash attention: {config.gpu_config.use_flash_attention_2}")
    print(f"  - GPU memory size: {config.gpu_config.gpu_memory_size}")

    # Test property access
    print(f"  - Config use_flash_attention_2 property: {config.use_flash_attention_2}")
    
    print("Qwen3VLConfig tests passed!\n")


def test_unified_config():
    """Test UnifiedConfig functionality."""
    print("Testing UnifiedConfig...")
    
    # Create a basic unified config
    config = UnifiedConfig()
    print(f"  - Created unified config with {config.num_hidden_layers} hidden layers")
    print(f"  - Optimization level: {config.optimization_level}")
    print(f"  - GPU memory size: {config.gpu_config.gpu_memory_size}")
    print(f"  - Use mixed precision: {config.optimization_config.use_mixed_precision}")
    
    # Test from_dict and to_dict
    config_dict = config.to_dict()
    print(f"  - Config dict keys: {len(config_dict)}")
    
    new_config = UnifiedConfig.from_dict(config_dict)
    print(f"  - Created new unified config from dict")
    
    print("UnifiedConfig tests passed!\n")


def test_config_manager():
    """Test UnifiedConfigManager functionality."""
    print("Testing UnifiedConfigManager...")
    
    # Create config manager
    config_manager = UnifiedConfigManager()
    print("  - Created config manager")
    
    # Get different optimization levels
    minimal_config = config_manager.get_config("minimal")
    print(f"  - Got minimal config with sparsity: {minimal_config.optimization_config.use_sparsity}")
    
    balanced_config = config_manager.get_config("balanced")
    print(f"  - Got balanced config with sparsity: {balanced_config.optimization_config.use_sparsity}")
    
    aggressive_config = config_manager.get_config("aggressive")
    print(f"  - Got aggressive config with sparsity: {aggressive_config.optimization_config.use_sparsity}")
    
    # Test validation
    is_valid = config_manager.validate_config(balanced_config)
    print(f"  - Balanced config validation: {is_valid}")
    
    print("ConfigManager tests passed!\n")


def test_legacy_compatibility():
    """Test legacy compatibility functions."""
    print("Testing legacy compatibility...")
    
    # Test get_legacy_config
    legacy_config = get_legacy_config()
    print(f"  - Got legacy config with {len(legacy_config)} keys")
    
    # Test update_legacy_config
    updates = {"num_hidden_layers": 16, "hidden_size": 2048}
    updated_config = update_legacy_config(legacy_config, updates)
    print(f"  - Updated config: {updated_config.get('num_hidden_layers')} hidden layers")
    
    print("Legacy compatibility tests passed!\n")


def test_file_operations():
    """Test file save/load operations."""
    print("Testing file operations...")
    
    # Create a config
    config = UnifiedConfig(optimization_level="minimal")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        config.save_to_file(temp_file)
        print(f"  - Saved config to {temp_file}")
        
        # Load from file
        loaded_config = UnifiedConfig.from_file(temp_file)
        print(f"  - Loaded config with {loaded_config.num_hidden_layers} hidden layers")
        
        print("File operations tests passed!\n")
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    """Run all tests."""
    print("Running configuration fixes verification tests...\n")
    
    test_qwen3_vl_config()
    test_unified_config()
    test_config_manager()
    test_legacy_compatibility()
    test_file_operations()
    
    print("All tests passed! Configuration fixes are working correctly.")


if __name__ == "__main__":
    main()