#!/usr/bin/env python
"""Test script to verify that the type errors have been corrected."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
from src.qwen3_vl.core.config import Qwen3VLConfig
from src.qwen3_vl.optimization.hardware_specific_optimization import HardwareOptimizedAttention, HardwareOptimizedMLP
from src.qwen3_vl.memory_management.memory_manager import MemoryManager, MemoryConfig

def test_config_creation():
    """Test that the configuration can be created without errors."""
    print("Testing configuration creation...")
    try:
        config = Qwen3VLConfig()
        print(f"[SUCCESS] Configuration created successfully")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        print(f"  - Memory config: {config.memory_config is not None}")
        print(f"  - GPU config: {config.gpu_config is not None}")
        print(f"  - Optimization config: {config.optimization_config is not None}")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration creation failed: {e}")
        return False

def test_hardware_optimized_modules():
    """Test that hardware optimized modules can be created."""
    print("\nTesting hardware optimized modules...")
    try:
        config = Qwen3VLConfig()
        
        # Test attention module
        attention = HardwareOptimizedAttention(config, layer_idx=0)
        print(f"[SUCCESS] HardwareOptimizedAttention created successfully")

        # Test MLP module
        mlp = HardwareOptimizedMLP(config, layer_idx=0)
        print(f"[SUCCESS] HardwareOptimizedMLP created successfully")

        return True
    except Exception as e:
        print(f"[ERROR] Hardware optimized modules creation failed: {e}")
        return False

def test_memory_manager():
    """Test that memory manager can be created."""
    print("\nTesting memory manager...")
    try:
        config = MemoryConfig()
        memory_manager = MemoryManager(config)
        print(f"[SUCCESS] MemoryManager created successfully")

        # Test basic tensor allocation
        tensor = memory_manager.allocate_tensor((100, 200), torch.float32)
        print(f"[SUCCESS] Tensor allocated successfully: {tensor.shape}")

        # Test tensor deallocation
        success = memory_manager.free_tensor(tensor)
        print(f"[SUCCESS] Tensor deallocation successful: {success}")

        return True
    except Exception as e:
        print(f"[ERROR] Memory manager test failed: {e}")
        return False

def test_config_properties():
    """Test that configuration properties work correctly."""
    print("\nTesting configuration properties...")
    try:
        config = Qwen3VLConfig()

        # Test properties that were causing issues
        print(f"[SUCCESS] use_gradient_checkpointing: {config.use_gradient_checkpointing}")
        print(f"[SUCCESS] use_flash_attention_2: {config.use_flash_attention_2}")
        print(f"[SUCCESS] attention_implementation: {config.attention_implementation}")
        print(f"[SUCCESS] use_memory_efficient_attention: {config.use_memory_efficient_attention}")
        print(f"[SUCCESS] kv_cache_strategy: {config.kv_cache_strategy}")
        print(f"[SUCCESS] use_dynamic_sparse_attention: {config.use_dynamic_sparse_attention}")
        print(f"[SUCCESS] sparsity_ratio: {config.sparsity_ratio}")

        return True
    except Exception as e:
        print(f"[ERROR] Configuration properties test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running tests to verify corrections to type errors...\n")

    tests = [
        test_config_creation,
        test_hardware_optimized_modules,
        test_memory_manager,
        test_config_properties,
    ]

    results = []
    for test in tests:
        results.append(test())

    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("[SUCCESS] All tests passed! Type errors appear to be fixed.")
        return 0
    else:
        print("[ERROR] Some tests failed. Type errors may still exist.")
        return 1

if __name__ == "__main__":
    sys.exit(main())