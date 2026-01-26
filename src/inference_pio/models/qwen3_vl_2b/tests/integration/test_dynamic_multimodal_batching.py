#!/usr/bin/env python3
"""
Test script for Dynamic Multimodal Batching in Qwen3-VL-2B model.

This script tests the dynamic multimodal batching functionality for the Qwen3-VL-2B model.
It verifies that the system correctly adjusts batch sizes based on input complexity.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.common.dynamic_multimodal_batching import (
    DynamicMultimodalBatchManager,
    ImageComplexityAnalyzer,
    get_dynamic_multimodal_batch_manager
)


def test_image_complexity_analyzer():
    """Test the image complexity analyzer."""
    print("Testing Image Complexity Analyzer...")
    
    analyzer = ImageComplexityAnalyzer()
    
    # Create a simple image (low complexity)
    simple_img = Image.new('RGB', (224, 224), color='red')
    simple_complexity = analyzer.analyze_image_complexity(simple_img)
    print(f"Simple image complexity: {simple_complexity:.3f}")
    
    # Create a more complex image with random noise
    complex_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    complex_img = Image.fromarray(complex_array)
    complex_complexity = analyzer.analyze_image_complexity(complex_img)
    print(f"Complex image complexity: {complex_complexity:.3f}")
    
    # Verify that complex image has higher complexity score
    assert complex_complexity > simple_complexity, f"Complex image should have higher complexity ({complex_complexity} > {simple_complexity})"
    assert 0.0 <= simple_complexity <= 1.0, f"Complexity should be between 0 and 1, got {simple_complexity}"
    assert 0.0 <= complex_complexity <= 1.0, f"Complexity should be between 0 and 1, got {complex_complexity}"
    
    print("[PASS] Image Complexity Analyzer test passed")


def test_dynamic_multimodal_batch_manager():
    """Test the dynamic multimodal batch manager."""
    print("\nTesting Dynamic Multimodal Batch Manager...")
    
    # Create batch manager
    batch_manager = DynamicMultimodalBatchManager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=8,
        text_weight=0.4,
        image_weight=0.6,
        complexity_threshold_low=0.3,
        complexity_threshold_high=0.7
    )
    
    # Test with simple inputs (should result in larger batch size)
    simple_inputs = [
        {'text': 'Hello world', 'image': Image.new('RGB', (224, 224), color='white')}
        for _ in range(5)
    ]
    
    simple_batch_size = batch_manager._get_complexity_based_batch_size(0.2)  # Low complexity
    print(f"Simple inputs recommended batch size: {simple_batch_size}")
    assert simple_batch_size == 8, f"Expected max batch size (8) for simple inputs, got {simple_batch_size}"
    
    # Test with complex inputs (should result in smaller batch size)
    complex_inputs = [
        {'text': 'This is a very complex sentence with many words and intricate meaning', 
         'image': Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))}
        for _ in range(5)
    ]
    
    complex_batch_size = batch_manager._get_complexity_based_batch_size(0.8)  # High complexity
    print(f"Complex inputs recommended batch size: {complex_batch_size}")
    assert complex_batch_size == 1, f"Expected min batch size (1) for complex inputs, got {complex_batch_size}"
    
    # Test with moderate inputs
    moderate_batch_size = batch_manager._get_complexity_based_batch_size(0.5)  # Moderate complexity
    print(f"Moderate inputs recommended batch size: {moderate_batch_size}")
    assert 1 <= moderate_batch_size <= 8, f"Moderate batch size should be between 1 and 8, got {moderate_batch_size}"
    
    print("[PASS] Dynamic Multimodal Batch Manager test passed")


def test_get_dynamic_multimodal_batch_manager_singleton():
    """Test that the singleton getter works correctly."""
    print("\nTesting Dynamic Multimodal Batch Manager Singleton...")
    
    manager1 = get_dynamic_multimodal_batch_manager(
        initial_batch_size=2,
        min_batch_size=1,
        max_batch_size=4
    )
    
    manager2 = get_dynamic_multimodal_batch_manager()
    
    # They should be the same instance
    assert manager1 is manager2, "Singleton should return the same instance"
    
    # Check that the configuration was applied
    assert manager1.current_batch_size == 2, f"Initial batch size should be 2, got {manager1.current_batch_size}"
    assert manager1.min_batch_size == 1, f"Min batch size should be 1, got {manager1.min_batch_size}"
    assert manager1.max_batch_size == 4, f"Max batch size should be 4, got {manager1.max_batch_size}"
    
    print("[PASS] Dynamic Multimodal Batch Manager Singleton test passed")


def test_plugin_integration():
    """Test integration with the Qwen3-VL-2B plugin."""
    print("\nTesting Plugin Integration...")
    
    plugin = Qwen3_VL_2B_Plugin()
    
    # Create a config with dynamic multimodal batching enabled
    config = Qwen3VL2BConfig()
    config.enable_dynamic_multimodal_batching = True
    config.initial_batch_size = 2
    config.min_batch_size = 1
    config.max_batch_size = 6
    config.text_weight = 0.5
    config.image_weight = 0.5
    
    # Initialize the plugin with the config
    # Note: This will likely fail to load the actual model, but should set up the batching system
    try:
        success = plugin.initialize(**config.__dict__)
        print(f"Plugin initialization result: {success}")
    except Exception as e:
        print(f"Plugin initialization failed as expected (no model): {e}")
        # This is expected since we don't have the actual model
    
    # Check if the dynamic multimodal batching was set up
    # Even if initialization failed, the setup should have been attempted
    print(f"Dynamic multimodal batching enabled: {hasattr(plugin, '_dynamic_multimodal_batch_manager')}")
    
    # Test the method exists
    assert hasattr(plugin, 'get_optimal_multimodal_batch_size'), "Plugin should have get_optimal_multimodal_batch_size method"
    
    print("[PASS] Plugin Integration test passed")


def test_multimodal_batching_logic():
    """Test the multimodal batching logic with simulated inputs."""
    print("\nTesting Multimodal Batching Logic...")
    
    batch_manager = DynamicMultimodalBatchManager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=8
    )
    
    # Create mixed complexity inputs
    inputs = [
        {
            'text': 'Simple text',
            'image': Image.new('RGB', (224, 224), color='blue')
        },
        {
            'text': 'Another simple text',
            'image': Image.new('RGB', (224, 224), color='green')
        }
    ]
    
    # Analyze complexity
    text_complexity, image_complexity, combined_complexity = batch_manager.analyze_multimodal_complexity(inputs)
    print(f"Text complexity: {text_complexity:.3f}")
    print(f"Image complexity: {image_complexity:.3f}")
    print(f"Combined complexity: {combined_complexity:.3f}")
    
    # The complexity should be reasonable
    assert 0.0 <= text_complexity <= 1.0, "Text complexity should be between 0 and 1"
    assert 0.0 <= image_complexity <= 1.0, "Image complexity should be between 0 and 1"
    assert 0.0 <= combined_complexity <= 1.0, "Combined complexity should be between 0 and 1"
    
    # Test batch type determination
    batch_type = batch_manager.determine_batch_type(inputs)
    print(f"Batch type: {batch_type}")
    
    print("[PASS] Multimodal Batching Logic test passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Dynamic Multimodal Batching Test Suite for Qwen3-VL-2B")
    print("=" * 70)

    try:
        test_image_complexity_analyzer()
        test_dynamic_multimodal_batch_manager()
        test_get_dynamic_multimodal_batch_manager_singleton()
        test_multimodal_batching_logic()
        test_plugin_integration()

        print("\n" + "=" * 70)
        print("All tests passed! [PASS]")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())