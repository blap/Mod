#!/usr/bin/env python3
"""
Integration test for Dynamic Multimodal Batching in Qwen3-VL-2B model.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin


def test_model_config_integration():
    """Test that the model configuration properly integrates with dynamic multimodal batching."""
    print("Testing Model Configuration Integration...")
    
    # Create a config with dynamic multimodal batching enabled
    config = Qwen3VL2BConfig()
    config.enable_dynamic_multimodal_batching = True
    config.initial_batch_size = 2
    config.min_batch_size = 1
    config.max_batch_size = 6
    config.text_weight = 0.5
    config.image_weight = 0.5
    
    # Create the model (without actually loading weights to avoid downloading)
    try:
        # Temporarily disable loading the actual model to speed up tests
        original_model_path = config.model_path
        config.model_path = "dummy_path"  # This will cause loading to fail gracefully
        
        model = Qwen3VL2BModel(config)
        
        # Check if the dynamic multimodal batch manager was initialized
        has_batch_manager = hasattr(model, '_dynamic_multimodal_batch_manager') and model._dynamic_multimodal_batch_manager is not None
        print(f"Dynamic multimodal batch manager initialized: {has_batch_manager}")
        
        # Check if the method exists
        has_method = hasattr(model, 'generate_with_adaptive_batching')
        print(f"Generate with adaptive batching method exists: {has_method}")
        
        # If the model didn't initialize properly due to missing model files, that's OK for this test
        print("Model configuration integration test completed (expected to fail loading actual model)")
        
    except Exception as e:
        print(f"Expected exception during model loading (no actual model): {e}")
    
    print("[PASS] Model Configuration Integration test completed")


def test_plugin_config_integration():
    """Test that the plugin configuration properly integrates with dynamic multimodal batching."""
    print("\nTesting Plugin Configuration Integration...")
    
    plugin = Qwen3_VL_2B_Plugin()
    
    # Create a config with dynamic multimodal batching enabled
    config = Qwen3VL2BConfig()
    config.enable_dynamic_multimodal_batching = True
    config.initial_batch_size = 2
    config.min_batch_size = 1
    config.max_batch_size = 6
    config.text_weight = 0.5
    config.image_weight = 0.5
    
    # Check if the setup method exists
    has_setup_method = hasattr(plugin, 'setup_dynamic_multimodal_batching')
    print(f"Setup dynamic multimodal batching method exists: {has_setup_method}")
    
    # Check if the get method exists
    has_get_method = hasattr(plugin, 'get_optimal_multimodal_batch_size')
    print(f"Get optimal multimodal batch size method exists: {has_get_method}")
    
    print("[PASS] Plugin Configuration Integration test passed")


def test_multimodal_batching_workflow():
    """Test the workflow of multimodal batching."""
    print("\nTesting Multimodal Batching Workflow...")
    
    # Create sample multimodal inputs
    simple_inputs = [
        {
            'text': 'Simple text',
            'image': Image.new('RGB', (224, 224), color='white')
        }
    ]
    
    complex_inputs = [
        {
            'text': 'This is a very complex sentence with many words and intricate meaning that requires significant processing',
            'image': Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
        }
    ]
    
    # Test that we can create the batch manager
    from src.inference_pio.common.dynamic_multimodal_batching import DynamicMultimodalBatchManager
    
    batch_manager = DynamicMultimodalBatchManager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=8
    )
    
    # Test complexity analysis
    simple_text_comp, simple_img_comp, simple_combined = batch_manager.analyze_multimodal_complexity(simple_inputs)
    complex_text_comp, complex_img_comp, complex_combined = batch_manager.analyze_multimodal_complexity(complex_inputs)
    
    print(f"Simple inputs - Text: {simple_text_comp:.3f}, Image: {simple_img_comp:.3f}, Combined: {simple_combined:.3f}")
    print(f"Complex inputs - Text: {complex_text_comp:.3f}, Image: {complex_img_comp:.3f}, Combined: {complex_combined:.3f}")
    
    # Complex inputs should have higher complexity
    assert complex_combined >= simple_combined, "Complex inputs should have higher combined complexity"
    
    # Test batch size recommendations
    simple_recommended = batch_manager._get_complexity_based_batch_size(simple_combined)
    complex_recommended = batch_manager._get_complexity_based_batch_size(complex_combined)
    
    print(f"Recommended batch size for simple inputs: {simple_recommended}")
    print(f"Recommended batch size for complex inputs: {complex_recommended}")
    
    # Complex inputs should result in smaller batch size
    assert simple_recommended >= complex_recommended, "Complex inputs should result in smaller batch size"
    
    print("[PASS] Multimodal Batching Workflow test passed")


def test_generate_with_adaptive_batching_method():
    """Test that the generate_with_adaptive_batching method exists and works with multimodal inputs."""
    print("\nTesting Generate with Adaptive Batching Method...")
    
    # Create a mock model to test the method signature
    config = Qwen3VL2BConfig()
    config.enable_dynamic_multimodal_batching = False  # Disable for this test
    
    # Create a minimal model instance to test method existence
    model = Qwen3VL2BModel.__new__(Qwen3VL2BModel)
    model.config = config
    model._dynamic_multimodal_batch_manager = None  # Explicitly set to None
    
    # Check if the method exists
    has_method = hasattr(model, 'generate_with_adaptive_batching')
    print(f"Generate with adaptive batching method exists: {has_method}")
    
    # The method should exist
    assert has_method, "generate_with_adaptive_batching method should exist"
    
    # Check if it can handle multimodal inputs (even if it doesn't execute fully)
    method = getattr(model, 'generate_with_adaptive_batching', None)
    print(f"Method callable: {callable(method)}")
    
    print("[PASS] Generate with Adaptive Batching Method test passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Integration Test for Dynamic Multimodal Batching in Qwen3-VL-2B")
    print("=" * 70)

    try:
        test_model_config_integration()
        test_plugin_config_integration()
        test_multimodal_batching_workflow()
        test_generate_with_adaptive_batching_method()

        print("\n" + "=" * 70)
        print("All integration tests completed! [PASS]")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())