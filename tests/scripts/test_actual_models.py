"""
Test suite for the actual Qwen3-VL model implementations with proper imports.
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
        return False, None


def test_model_import():
    """Test that we can import the main model."""
    print("\nTesting model import...")
    try:
        # Try importing from the correct location
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        print("✓ Model imported successfully")
        return True, Qwen3VLForConditionalGeneration
    except Exception as e:
        print(f"✗ Failed to import model: {e}")
        return False, None


def test_model_creation():
    """Test creating the full model with proper configuration."""
    print("\nTesting model creation...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create config with full capacity (32 layers, 32 heads)
        config = Qwen3VLConfig()
        print(f"Using config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
        
        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        print(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Verify model structure
        print(f"  Language model layers: {len(model.language_model.layers)}")
        print(f"  Vision model layers: {len(model.vision_tower.layers)}")
        print(f"  Language attention heads: {model.config.num_attention_heads}")
        print(f"  Vision attention heads: {model.config.vision_num_attention_heads}")
        
        return True, model
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_forward_pass():
    """Test that the model can perform a forward pass."""
    print("\nTesting model forward pass...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create a smaller config for faster testing while preserving capacity ratios
        config = Qwen3VLConfig()
        config.hidden_size = 512  # Reduce for testing
        config.intermediate_size = 1024  # Reduce for testing
        config.vision_hidden_size = 384  # Reduce for testing
        config.vision_intermediate_size = 1536  # Reduce for testing
        config.num_attention_heads = 8  # Reduce for testing
        config.vision_num_attention_heads = 6  # Reduce for testing
        config.num_hidden_layers = 2  # Reduce for testing
        config.vision_num_hidden_layers = 2  # Reduce for testing
        
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        
        # Create small test inputs
        batch_size, seq_len = 1, 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_capacity_preservation():
    """Test that model capacity is preserved (when using full config)."""
    print("\nTesting model capacity preservation...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create config with reduced layers for testing but maintain the ratios
        config = Qwen3VLConfig()
        config.hidden_size = 256  # Further reduce for testing
        config.intermediate_size = 512  # Further reduce for testing
        config.vision_hidden_size = 192  # Further reduce for testing
        config.vision_intermediate_size = 384  # Further reduce for testing
        config.num_attention_heads = 4  # Further reduce for testing
        config.vision_num_attention_heads = 4  # Further reduce for testing
        config.num_hidden_layers = 2  # Reduce for testing
        config.vision_num_hidden_layers = 2  # Reduce for testing
        
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
        
        print("✓ Model architecture matches configuration")
        return True
    except Exception as e:
        print(f"✗ Failed capacity test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_functionality():
    """Test the model's generation functionality."""
    print("\nTesting generation functionality...")
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create config with reduced dimensions for testing
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.intermediate_size = 512
        config.vision_hidden_size = 192
        config.vision_intermediate_size = 384
        config.num_attention_heads = 4
        config.vision_num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_num_hidden_layers = 2
        
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
        
        print(f"✓ Generation successful, output shape: {generated.shape}")
        print(f"  Generated {generated.shape[1] - input_ids.shape[1]} new tokens")
        return True
    except Exception as e:
        print(f"✗ Failed generation test: {e}")
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
    results.append(("Config Import", test_config_import()[0]))
    results.append(("Model Import", test_model_import()[0]))
    results.append(("Model Creation", test_model_creation()[0]))
    results.append(("Forward Pass", test_model_forward_pass()))
    results.append(("Capacity Preservation", test_model_capacity_preservation()))
    results.append(("Generation", test_generation_functionality()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, test_result in results:
        status = "PASS" if test_result else "FAIL"
        symbol = "✓" if test_result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if test_result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Qwen3-VL models are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)