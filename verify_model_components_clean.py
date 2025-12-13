"""
Simple test script to verify that the Qwen3-VL model components work correctly
"""
import torch
import sys
import os
from pathlib import Path

# Add the project root to the path to import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports of model components."""
    print("Testing basic imports...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        print("V Qwen3VLConfig imported successfully")
        
        config = Qwen3VLConfig()
        print(f"V Config created with {config.num_hidden_layers} layers and {config.num_attention_heads} attention heads")
        
        # Verify full capacity is preserved
        assert config.num_hidden_layers == 32, f"Expected 32 hidden layers, got {config.num_hidden_layers}"
        assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"
        print("V Full capacity preserved (32 layers, 32 attention heads)")
        
    except Exception as e:
        print(f"X Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_creation():
    """Test model creation with a smaller configuration."""
    print("\\nTesting model creation...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create a smaller config for testing
        config = Qwen3VLConfig()
        config.hidden_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_hidden_size = 128
        config.vision_num_attention_heads = 4
        config.vision_num_hidden_layers = 2
        config.vocab_size = 1000
        
        print("Creating model with smaller config for testing...")
        model = Qwen3VLForConditionalGeneration(config)
        print(f"V Model created successfully")
        
        # Verify model structure
        print(f"  Language model layers: {len(model.language_model.layers)}")
        print(f"  Vision model layers: {len(model.vision_tower.layers)}")
        print(f"  Language attention heads: {model.config.num_attention_heads}")
        
        assert len(model.language_model.layers) == config.num_hidden_layers
        assert len(model.vision_tower.layers) == config.vision_num_hidden_layers
        print("V Model architecture matches configuration")
        
    except Exception as e:
        print(f"X Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_forward_pass():
    """Test model forward pass."""
    print("\\nTesting model forward pass...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create a smaller config for testing
        config = Qwen3VLConfig()
        config.hidden_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_hidden_size = 128
        config.vision_num_attention_heads = 4
        config.vision_num_hidden_layers = 2
        config.vocab_size = 1000
        
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        print("V Model initialized for forward pass")
        
        # Create test inputs
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        print("Created test inputs")
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
        
        print(f"V Forward pass completed successfully, output shape: {output.shape}")
        assert torch.all(torch.isfinite(output)), "Output should contain finite values"
        print("V Output contains finite values")
        
    except Exception as e:
        print(f"X Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model_generation():
    """Test model generation."""
    print("\\nTesting model generation...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.models.modeling_qwen3_vl_nas import Qwen3VLForConditionalGeneration
        
        # Create a smaller config for testing
        config = Qwen3VLConfig()
        config.hidden_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_hidden_size = 128
        config.vision_num_attention_heads = 4
        config.vision_num_hidden_layers = 2
        config.vocab_size = 1000
        
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        print("V Model initialized for generation")
        
        # Create test inputs
        batch_size = 1
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        print("Created test inputs for generation")
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=3,
                do_sample=False
            )
        
        print(f"V Generation completed successfully, output shape: {generated.shape}")
        assert generated.shape[0] == batch_size, "Batch size should be preserved"
        print("V Generation preserves batch size")
        
    except Exception as e:
        print(f"X Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests."""
    print("="*80)
    print("RUNNING QWEN3-VL MODEL COMPONENT VERIFICATION")
    print("="*80)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_model_forward_pass),
        ("Generation", test_model_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    print("\\n" + "="*80)
    print("VERIFICATION RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "V" if success else "X"
        print(f"{symbol} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\\nV ALL TESTS PASSED! Qwen3-VL model components are working correctly.")
        return True
    else:
        print(f"\\nX {len(results) - passed} TEST(S) FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)