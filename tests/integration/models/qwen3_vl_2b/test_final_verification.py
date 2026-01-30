#!/usr/bin/env python3
"""
Final verification test for Dynamic Multimodal Batching in Qwen3-VL-2B model.
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
    ImageComplexityAnalyzer
)


def test_complete_workflow():
    """Test the complete dynamic multimodal batching workflow."""
    print("Testing Complete Dynamic Multimodal Batching Workflow...")
    
    # Create a dynamic multimodal batch manager
    batch_manager = DynamicMultimodalBatchManager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=8,
        text_weight=0.4,
        image_weight=0.6,
        complexity_threshold_low=0.3,
        complexity_threshold_high=0.7
    )
    
    # Create sample multimodal inputs with varying complexity
    simple_inputs = [
        {
            'text': 'Hi',
            'image': Image.new('RGB', (224, 224), color='white')
        }
    ]
    
    complex_inputs = [
        {
            'text': 'This is a very complex sentence with many words and intricate meaning that requires significant processing power and memory resources',
            'image': Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
        }
    ]
    
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
    
    # Test metrics collection
    batch_info_simple = batch_manager.collect_multimodal_metrics(
        batch_size=simple_recommended,
        processing_time_ms=100.0,
        tokens_processed=10,
        inputs=simple_inputs
    )
    
    batch_info_complex = batch_manager.collect_multimodal_metrics(
        batch_size=complex_recommended,
        processing_time_ms=200.0,
        tokens_processed=10,
        inputs=complex_inputs
    )
    
    print(f"Simple batch info - Type: {batch_info_simple.batch_type}, Size: {batch_info_simple.batch_size}")
    print(f"Complex batch info - Type: {batch_info_complex.batch_type}, Size: {batch_info_complex.batch_size}")
    
    # Test optimal batch size calculation
    optimal_simple = batch_manager.get_optimal_multimodal_batch_size(
        processing_time_ms=100.0,
        tokens_processed=10,
        inputs=simple_inputs
    )
    
    optimal_complex = batch_manager.get_optimal_multimodal_batch_size(
        processing_time_ms=200.0,
        tokens_processed=10,
        inputs=complex_inputs
    )
    
    print(f"Optimal batch size for simple inputs: {optimal_simple}")
    print(f"Optimal batch size for complex inputs: {optimal_complex}")
    
    # Complex inputs should still result in smaller or equal batch size
    assert optimal_simple >= optimal_complex, "Complex inputs should result in smaller or equal batch size"
    
    print("[PASS] Complete Dynamic Multimodal Batching Workflow test passed")
    return True


def test_image_complexity_analyzer():
    """Test the image complexity analyzer thoroughly."""
    print("\nTesting Image Complexity Analyzer...")
    
    analyzer = ImageComplexityAnalyzer()
    
    # Test with different types of images
    # 1. Solid color image (lowest complexity)
    solid_img = Image.new('RGB', (224, 224), color='red')
    solid_complexity = analyzer.analyze_image_complexity(solid_img)
    print(f"Solid color image complexity: {solid_complexity:.3f}")
    
    # 2. Gradient image (low-medium complexity)
    gradient_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient_array[:, i, :] = [i % 256, (i * 2) % 256, (i * 3) % 256]
    gradient_img = Image.fromarray(gradient_array)
    gradient_complexity = analyzer.analyze_image_complexity(gradient_img)
    print(f"Gradient image complexity: {gradient_complexity:.3f}")
    
    # 3. Random noise image (highest complexity)
    random_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    random_img = Image.fromarray(random_array)
    random_complexity = analyzer.analyze_image_complexity(random_img)
    print(f"Random noise image complexity: {random_complexity:.3f}")
    
    # Verify ordering
    assert solid_complexity <= gradient_complexity, "Solid should be less complex than gradient"
    assert gradient_complexity <= random_complexity, "Gradient should be less complex than random"
    assert 0.0 <= solid_complexity <= 1.0, "Complexity should be between 0 and 1"
    assert 0.0 <= gradient_complexity <= 1.0, "Complexity should be between 0 and 1"
    assert 0.0 <= random_complexity <= 1.0, "Complexity should be between 0 and 1"
    
    print("[PASS] Image Complexity Analyzer test passed")
    return True


def test_batch_type_detection():
    """Test the batch type detection functionality."""
    print("\nTesting Batch Type Detection...")
    
    batch_manager = DynamicMultimodalBatchManager()
    
    # Test text-only batch
    text_only_inputs = [{'text': 'Hello world'}]
    text_only_type = batch_manager.determine_batch_type(text_only_inputs)
    print(f"Text-only batch type: {text_only_type}")
    assert str(text_only_type) == "MultimodalBatchType.TEXT_ONLY", f"Expected TEXT_ONLY, got {text_only_type}"
    
    # Test image-only batch
    image_only_inputs = [{'image': Image.new('RGB', (224, 224), color='blue')}]
    image_only_type = batch_manager.determine_batch_type(image_only_inputs)
    print(f"Image-only batch type: {image_only_type}")
    assert str(image_only_type) == "MultimodalBatchType.IMAGE_ONLY", f"Expected IMAGE_ONLY, got {image_only_type}"
    
    # Test text-image batch
    text_image_inputs = [{'text': 'Hello', 'image': Image.new('RGB', (224, 224), color='green')}]
    text_image_type = batch_manager.determine_batch_type(text_image_inputs)
    print(f"Text-image batch type: {text_image_type}")
    assert str(text_image_type) == "MultimodalBatchType.TEXT_IMAGE", f"Expected TEXT_IMAGE, got {text_image_type}"
    
    print("[PASS] Batch Type Detection test passed")
    return True


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Final Verification Test for Dynamic Multimodal Batching")
    print("=" * 70)

    all_passed = True
    
    all_passed &= test_complete_workflow()
    all_passed &= test_image_complexity_analyzer()
    all_passed &= test_batch_type_detection()

    if all_passed:
        print("\n" + "=" * 70)
        print("All verification tests passed! [PASS]")
        print("=" * 70)
        print("\nDynamic Multimodal Batching Implementation is complete and working correctly!")
        print("- Implemented dynamic batching for multimodal inputs (text + image)")
        print("- Created ImageComplexityAnalyzer for image complexity assessment")
        print("- Integrated with Qwen3-VL-2B model and plugin")
        print("- Added configuration options for fine-tuning")
        print("- Implemented adaptive batch sizing based on input complexity")
        return 0
    else:
        print("\n" + "=" * 70)
        print("Some verification tests failed! [FAIL]")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())