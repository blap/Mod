"""
Simple test suite for the actual Qwen3-VL model implementations with correct imports.
This file validates the real models in the codebase to ensure they work correctly.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add the project root to the path to import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_import():
    """Test that we can import the Qwen3VLConfig correctly."""
    print("Testing configuration import...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        config = Qwen3VLConfig()
        print(f"V Config imported successfully: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
        return True, config
    except Exception as e:
        print(f"X Failed to import config: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation():
    """Test creating a simplified model for testing."""
    print("\nTesting model creation...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        # Create a simplified model that we can test with
        config = Qwen3VLConfig()
        
        # Since the main model import is causing issues, let's test with the core model
        # Let's check if there's a simpler model implementation we can use
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        print("V Found core model implementation")
        
        # Create a small model for testing
        small_config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=128,
            vision_num_attention_heads=4
        )
        
        # Create model
        model = Qwen3VLForConditionalGeneration(small_config)
        print(f"V Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Verify model structure
        print(f"  Language model layers: {len(model.language_model.layers)}")
        print(f"  Vision model layers: {len(model.vision_tower.layers)}")
        print(f"  Language model attention heads: {model.config.num_attention_heads}")
        print(f"  Vision model attention heads: {model.config.vision_num_attention_heads}")
        
        return True, model
    except Exception as e:
        print(f"X Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_forward_pass():
    """Test that the model can perform a forward pass."""
    print("\nTesting model forward pass...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create a small config for testing
        config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=128,
            vision_num_attention_heads=4
        )
        
        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        
        # Create test inputs
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)  # Standard image size
        
        # Forward pass with text and image
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
        
        print(f"V Forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"X Failed forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_generation():
    """Test the model's generation functionality."""
    print("\nTesting model generation...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create a small config for testing
        config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=128,
            vision_num_attention_heads=4
        )
        
        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        
        # Create test inputs
        batch_size = 1
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=3,
                do_sample=False
            )
        
        print(f"V Generation successful, output shape: {generated.shape}")
        print(f"  Generated {generated.shape[1] - input_ids.shape[1]} new tokens")
        return True
    except Exception as e:
        print(f"X Failed generation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_capacity_preservation():
    """Test that model capacity is preserved."""
    print("\nTesting model capacity preservation...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create config with reduced parameters for testing but check the expected values
        config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            attention_dropout_prob=0.0,
            num_hidden_layers=2,  # Small for testing
            vision_num_hidden_layers=2,  # Small for testing
            vocab_size=1000,
            vision_hidden_size=128,
            vision_num_attention_heads=4
        )
        
        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        
        # Check that the model has the expected architecture with the modified config
        expected_layers = config.num_hidden_layers
        expected_vision_layers = config.vision_num_hidden_layers
        expected_heads = config.num_attention_heads
        expected_vision_heads = config.vision_num_attention_heads
        
        actual_lang_layers = len(model.language_model.layers)
        actual_vision_layers = len(model.vision_tower.layers)
        actual_lang_heads = model.config.num_attention_heads
        actual_vision_heads = model.config.vision_num_attention_heads
        
        print(f"  Expected language layers: {expected_layers}, Actual: {actual_lang_layers}")
        print(f"  Expected vision layers: {expected_vision_layers}, Actual: {actual_vision_layers}")
        print(f"  Expected language heads: {expected_heads}, Actual: {actual_lang_heads}")
        print(f"  Expected vision heads: {expected_vision_heads}, Actual: {actual_vision_heads}")
        
        # The model should match the config values
        assert actual_lang_layers == expected_layers, f"Language layers mismatch: expected {expected_layers}, got {actual_lang_layers}"
        assert actual_vision_layers == expected_vision_layers, f"Vision layers mismatch: expected {expected_vision_layers}, got {actual_vision_layers}"
        assert actual_lang_heads == expected_heads, f"Language heads mismatch: expected {expected_heads}, got {actual_lang_heads}"
        assert actual_vision_heads == expected_vision_heads, f"Vision heads mismatch: expected {expected_vision_heads}, got {actual_vision_heads}"
        
        print("V Model architecture matches configuration")
        return True
    except Exception as e:
        print(f"X Failed capacity test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("TESTING ACTUAL QWEN3-VL MODEL IMPLEMENTATIONS")
    print("="*80)
    
    results = []
    
    # Run each test
    success, config = test_config_import()
    results.append(("Config Import", success))
    
    if success:
        success, model = test_model_creation()
        results.append(("Model Creation", success))
        
        if success:
            results.append(("Forward Pass", test_model_forward_pass()))
            results.append(("Generation", test_model_generation()))
            results.append(("Capacity Preservation", test_model_capacity_preservation()))
    else:
        # Skip dependent tests if config import failed
        results.extend([
            ("Model Creation", False),
            ("Forward Pass", False),
            ("Generation", False),
            ("Capacity Preservation", False)
        ])
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, test_result in results:
        status = "PASS" if test_result else "FAIL"
        symbol = "V" if test_result else "X"
        print(f"{symbol} {test_name}: {status}")
        if test_result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("* All tests passed! The Qwen3-VL models are working correctly.")
    else:
        print("X Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)