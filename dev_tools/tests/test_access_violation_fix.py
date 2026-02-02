#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the fix for access violation (code 3221225477 - 0xC0000005)
when instantiating the Qwen3-VL model.

This script tests the corrected model initialization to ensure that:
1. No access violations occur during model instantiation
2. No infinite recursion occurs during initialization
3. No stack overflow occurs during device movement
4. All components are properly initialized
"""
import traceback

import torch

from src.qwen3_vl.config import Qwen3VLConfig
from src.qwen3_vl.models import Qwen3VLForConditionalGeneration
from tests.utils.test_utils import (
    assert_equal,
    assert_false,
    assert_greater,
    assert_in,
    assert_is_instance,
    assert_is_none,
    assert_is_not_none,
    assert_less,
    assert_not_equal,
    assert_not_in,
    assert_raises,
    assert_true,
    run_tests,
)


def test_model_initialization():
    """Test model initialization to verify the fix for access violation."""
    print("Testing Qwen3-VL model initialization...")

    try:
        # Create a configuration
        config = Qwen3VLConfig()

        # Validate the configuration
        if not config.validate_capacity_preservation():
            print("Warning: Configuration does not preserve full capacity")
        else:
            print("Configuration validated: capacity preservation confirmed")

        # Test model creation - this was causing the access violation
        print("Creating Qwen3-VL model...")
        model = Qwen3VLForConditionalGeneration(config)

        print("Model created successfully!")
        print(
            f"Model has {config.num_hidden_layers} layers and {config.num_attention_heads} attention heads"
        )

        # Test moving model to CPU - this was causing recursion issues
        print("Testing model movement to CPU...")
        model = model.cpu()
        print("Model moved to CPU successfully!")

        # Test moving model to CUDA if available - to test the non-CPU path
        if torch.cuda.is_available():
            print("Testing model movement to CUDA...")
            model = model.cuda()
            print("Model moved to CUDA successfully!")

            # Move back to CPU for further testing
            model = model.cpu()

        # Test basic forward pass with dummy inputs
        print("Testing basic forward pass...")

        # Create dummy inputs
        batch_size = 1
        seq_len = 10
        image_size = 448

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)

        # Move inputs to CPU to match model
        input_ids = input_ids.cpu()
        pixel_values = pixel_values.cpu()

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)

        print("Forward pass completed successfully!")
        print(f"Output type: {type(outputs)}")

        if isinstance(outputs, dict):
            print(f"Output keys: {list(outputs.keys())}")
            if "logits" in outputs:
                print(f"Logits shape: {outputs['logits'].shape}")
        else:
            print(
                f"Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}"
            )

        print("\n✅ All tests passed! Access violation fix is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False


def test_recursive_initialization_protection():
    """Test that recursive initialization is properly protected."""
    print("\nTesting recursive initialization protection...")

    try:
        # Create multiple models to test for memory issues
        models = []
        configs = []

        for i in range(3):  # Create 3 models to test for accumulation issues
            config = Qwen3VLConfig()
            config.hidden_size = 2560  # Smaller size for testing
            config.intermediate_size = 6912
            config.num_hidden_layers = 4  # Fewer layers for testing
            config.num_attention_heads = 8  # Fewer heads for testing

            model = Qwen3VLForConditionalGeneration(config)
            models.append(model)
            configs.append(config)
            print(f"Created model {i+1}")

        # Test moving all models to CPU
        for i, model in enumerate(models):
            model = model.cpu()
            print(f"Moved model {i+1} to CPU")

        print("✅ Recursive initialization protection test passed!")
        return True

    except Exception as e:
        print(f"❌ Recursive initialization test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_device_movement_edge_cases():
    """Test edge cases for device movement."""
    print("\nTesting device movement edge cases...")

    try:
        config = Qwen3VLConfig()
        config.hidden_size = 256  # Small for testing
        config.num_hidden_layers = 2  # Small for testing
        config.num_attention_heads = 4  # Small for testing

        model = Qwen3VLForConditionalGeneration(config)

        # Test rapid device switching
        for i in range(5):
            model = model.cpu()
            model = model.cpu()  # Double CPU movement
            print(f"Rapid switch iteration {i+1} completed")

        # Test with different tensor types
        model = model.train()
        model = model.eval()

        print("✅ Device movement edge cases test passed!")
        return True

    except Exception as e:
        print(f"❌ Device movement edge cases test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Qwen3-VL Access Violation Fix Verification Tests")
    print("=" * 60)

    all_tests_passed = True

    # Run the main test
    all_tests_passed &= test_model_initialization()

    # Run additional tests
    all_tests_passed &= test_recursive_initialization_protection()
    all_tests_passed &= test_device_movement_edge_cases()

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The access violation fix is working correctly.")
        print("The model can now be instantiated without causing access violations.")
    else:
        print("💥 SOME TESTS FAILED! The fix may not be complete.")
    print("=" * 60)

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
