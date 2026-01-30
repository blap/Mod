#!/usr/bin/env python
"""
Simple test for multimodal attention optimization in Qwen3-VL-2B model.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
import torch

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath('.'))

def test_basic_imports():
    """Test basic imports for multimodal attention optimization."""
    print("Testing basic imports...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        print("[OK] Qwen3VL2BConfig imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import Qwen3VL2BConfig: {e}")
        return False
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            Qwen3VL2BMultimodalAttentionOptimizer,
            apply_qwen3_vl_multimodal_attention_optimizations_to_model
        )
        print("[OK] Multimodal attention optimization modules imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import multimodal attention optimization modules: {e}")
        return False
    
    return True

def test_optimizer_creation():
    """Test creating the multimodal attention optimizer."""
    print("\nTesting optimizer creation...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import Qwen3VL2BMultimodalAttentionOptimizer
        
        config = Qwen3VL2BConfig()
        config.hidden_size = 512
        config.num_attention_heads = 8
        config.num_key_value_heads = 2
        
        optimizer = Qwen3VL2BMultimodalAttentionOptimizer(config=config, layer_idx=0)
        print("[OK] Qwen3VL2BMultimodalAttentionOptimizer created successfully")
        print(f"  - Hidden size: {optimizer.hidden_size}")
        print(f"  - Num attention heads: {optimizer.num_attention_heads}")
        print(f"  - Num KV heads: {optimizer.num_key_value_heads}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test the forward pass of the optimizer."""
    print("\nTesting forward pass...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import Qwen3VL2BMultimodalAttentionOptimizer
        
        config = Qwen3VL2BConfig()
        config.hidden_size = 512
        config.num_attention_heads = 8
        config.num_key_value_heads = 2
        
        optimizer = Qwen3VL2BMultimodalAttentionOptimizer(config=config, layer_idx=0)
        
        # Create test inputs
        batch_size = 1
        vision_seq_len = 5
        language_seq_len = 10
        hidden_size = config.hidden_size
        
        vision_input = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_input = torch.randn(batch_size, language_seq_len, hidden_size)
        
        print(f"  - Vision input shape: {vision_input.shape}")
        print(f"  - Language input shape: {language_input.shape}")
        
        # Test forward pass
        output, attn_weights, past_key_value = optimizer(
            vision_hidden_states=vision_input,
            language_hidden_states=language_input
        )
        
        print("[OK] Forward pass completed successfully")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Attention weights: {attn_weights}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Qwen3-VL-2B Multimodal Attention Optimization Implementation")
    print("=" * 65)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Optimizer Creation", test_optimizer_creation),
        ("Forward Pass", test_forward_pass)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
            print(f"  Status: PASSED")
        else:
            print(f"  Status: FAILED")
    
    print("\n" + "=" * 65)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Multimodal attention optimization is working correctly.")
        return True
    else:
        print("FAILURE: Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)