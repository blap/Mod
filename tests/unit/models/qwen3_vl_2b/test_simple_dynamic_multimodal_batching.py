#!/usr/bin/env python3
"""
Simple test for Dynamic Multimodal Batching in Qwen3-VL-2B model.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

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
    simple_batch_size = batch_manager._get_complexity_based_batch_size(0.2)  # Low complexity
    print(f"Simple inputs recommended batch size: {simple_batch_size}")
    assert simple_batch_size == 8, f"Expected max batch size (8) for simple inputs, got {simple_batch_size}"
    
    # Test with complex inputs (should result in smaller batch size)
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


def main():
    """Run all tests."""
    print("=" * 70)
    print("Simple Dynamic Multimodal Batching Test for Qwen3-VL-2B")
    print("=" * 70)

    try:
        test_image_complexity_analyzer()
        test_dynamic_multimodal_batch_manager()
        test_get_dynamic_multimodal_batch_manager_singleton()

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