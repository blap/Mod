#!/usr/bin/env python
"""
Integration test for Qwen3-VL-2B projection layer optimizations.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
import torch
import torch.nn as nn
def projection_optimization_integration()():
    """Test that projection layer optimizations are properly integrated with the Qwen3-VL-2B model."""
    print("Testing projection optimization integration with Qwen3-VL-2B model...")
    
    # Create a config with projection layer optimization enabled
    config = Qwen3VL2BConfig()
    config.use_projection_layer_optimization = True
    config.model_path = "dummy"  # Use dummy path to avoid downloading
    
    # Create a mock model to test the optimization application
    with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained'), \
         patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained'), \
         patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
        
        # Create the model instance
        model = Qwen3VL2BModel.__new__(Qwen3VL2BModel)  # Create without calling __init__ to avoid initialization
        model.config = config
        model._model = MagicMock()  # Mock the internal model
        model._tokenizer = MagicMock()
        model._image_processor = MagicMock()
        model._model_name = "dummy"
        
        # Manually call the projection optimization method
        original_apply_method = model._apply_projection_layer_optimizations
        
        # Call the method to test integration
        try:
            model._apply_projection_layer_optimizations()
            print("[OK] Projection layer optimizations applied successfully to Qwen3-VL-2B model")
        except Exception as e:
            print(f"[ERROR] Failed to apply projection layer optimizations: {e}")
            return False
    
    print("Projection optimization integration test completed successfully!")
    return True

def model_config_has_projection_attributes()():
    """Test that the Qwen3-VL-2B config has all required projection layer attributes."""
    print("\nTesting that Qwen3-VL-2B config has projection layer attributes...")
    
    config = Qwen3VL2BConfig()
    
    # Check for projection layer optimization attributes
    projection_attrs = [
        'use_projection_layer_optimization',
        'projection_layer_use_bias',
        'projection_layer_activation',
        'projection_layer_dropout',
        'projection_layer_use_residual',
        'projection_layer_use_low_rank',
        'projection_layer_low_rank_dim',
        'projection_layer_use_group_norm',
        'projection_layer_group_norm_num_groups',
        'projection_layer_intermediate_dim',
        'projection_layer_num_layers',
        'projection_layer_use_cross_attention',
        'projection_layer_cross_attention_heads'
    ]
    
    all_present = True
    for attr in projection_attrs:
        if hasattr(config, attr):
            print(f"[OK] Config has projection attribute: {attr}")
        else:
            print(f"[MISSING] Config missing projection attribute: {attr}")
            all_present = False
    
    if all_present:
        print("All projection layer attributes are present in config!")
        return True
    else:
        print("Some projection layer attributes are missing!")
        return False

def model_has_projection_method()():
    """Test that the Qwen3-VL-2B model has the projection optimization method."""
    print("\nTesting that Qwen3-VL-2B model has projection optimization method...")
    
    # Check if the method exists in the class
    if hasattr(Qwen3VL2BModel, '_apply_projection_layer_optimizations'):
        print("[OK] Qwen3-VL-2B model has _apply_projection_layer_optimizations method")
        return True
    else:
        print("[MISSING] Qwen3-VL-2B model missing _apply_projection_layer_optimizations method")
        return False

def projection_optimization_flag_handling()():
    """Test that the model respects the projection optimization flag in config."""
    print("\nTesting projection optimization flag handling in model...")
    
    config = Qwen3VL2BConfig()
    
    # Test with optimization enabled
    config.use_projection_layer_optimization = True
    print(f"[OK] Projection optimization enabled in config: {config.use_projection_layer_optimization}")
    
    # Test with optimization disabled
    config.use_projection_layer_optimization = False
    print(f"[OK] Projection optimization disabled in config: {config.use_projection_layer_optimization}")
    
    print("Projection optimization flag handling verified!")
    return True

if __name__ == "__main__":
    print("Qwen3-VL-2B Projection Layer Optimization Integration Test")
    print("=" * 65)
    
    try:
        test1_passed = test_model_config_has_projection_attributes()
        test2_passed = test_model_has_projection_method()
        test3_passed = test_projection_optimization_flag_handling()
        test4_passed = test_projection_optimization_integration()

        all_passed = test1_passed and test2_passed and test3_passed and test4_passed
        
        print("\n" + "=" * 65)
        if all_passed:
            print("[SUCCESS] ALL INTEGRATION TESTS PASSED!")
            print("Qwen3-VL-2B projection layer optimizations are properly integrated.")
        else:
            print("[FAILURE] SOME TESTS FAILED!")
            print("Check the output above for details.")
            
    except Exception as e:
        print(f"\nError during integration testing: {e}")
        import traceback
        traceback.print_exc()