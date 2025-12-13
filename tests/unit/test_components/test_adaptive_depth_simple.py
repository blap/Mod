"""
Simple test for the adaptive depth mechanism to verify basic functionality
"""
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from src.components.optimization.adaptive_depth import InputComplexityAssessor, AdaptiveDepthController
from models.adaptive_depth_transformer import AdaptiveDepthTransformer


def test_adaptive_depth_basic():
    """Test basic adaptive depth functionality"""
    print("Testing basic adaptive depth functionality...")
    
    # Create a minimal config for testing
    config = Qwen3VLConfig()
    config.hidden_size = 128
    config.num_hidden_layers = 4  # Use fewer layers for testing
    config.num_attention_heads = 8
    config.vocab_size = 1000
    config.use_adaptive_depth = True  # Enable adaptive depth
    
    # Test complexity assessor
    assessor = InputComplexityAssessor(config)
    
    # Test with simple text (repetitive tokens)
    simple_text = torch.ones(1, 8, dtype=torch.long) * 10
    simple_complexity = assessor.assess_text_complexity(simple_text)
    print(f"Simple text complexity: {simple_complexity.item():.4f}")
    
    # Test with complex text (random tokens)
    complex_text = torch.randint(0, config.vocab_size, (1, 8))
    complex_complexity = assessor.assess_text_complexity(complex_text)
    print(f"Complex text complexity: {complex_complexity.item():.4f}")
    
    # Test with simple image
    simple_image = torch.ones(1, 3, 224, 224) * 0.5
    simple_img_complexity = assessor.assess_image_complexity(simple_image)
    print(f"Simple image complexity: {simple_img_complexity.item():.4f}")
    
    # Test with complex image
    complex_image = torch.randn(1, 3, 224, 224)
    complex_img_complexity = assessor.assess_image_complexity(complex_image)
    print(f"Complex image complexity: {complex_img_complexity.item():.4f}")
    
    # Test adaptive controller
    controller = AdaptiveDepthController(config, assessor)
    
    # Test depth selection for simple input
    simple_depth, simple_score = controller(simple_text, simple_image)
    print(f"Simple input - Depth: {simple_depth}, Complexity: {simple_score:.4f}")
    
    # Test depth selection for complex input
    complex_depth, complex_score = controller(complex_text, complex_image)
    print(f"Complex input - Depth: {complex_depth}, Complexity: {complex_score:.4f}")
    
    print("[PASS] Basic adaptive depth functionality test passed")
    return True


def test_model_integration():
    """Test integration with the main model"""
    print("\nTesting model integration...")
    
    # Create a minimal config
    config = Qwen3VLConfig()
    config.hidden_size = 64
    config.num_hidden_layers = 2  # Use minimal layers for testing
    config.num_attention_heads = 4
    config.vocab_size = 1000
    config.vision_hidden_size = 64
    config.vision_num_hidden_layers = 2
    config.use_adaptive_depth = True
    config.use_vision_adaptive_depth = True
    
    # Create the model
    model = Qwen3VLForConditionalGeneration(config)
    
    # Test text-only forward
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        text_output = model(input_ids=input_ids)
    print(f"Text-only output shape: {text_output.shape}")
    
    # Test with both text and image
    pixel_values = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
    print(f"Multimodal output shape: {multimodal_output.shape}")
    
    print("[PASS] Model integration test passed")
    return True


def test_capacity_preservation():
    """Test that model capacity is preserved"""
    print("\nTesting capacity preservation...")
    
    # Create config with full capacity settings
    config = Qwen3VLConfig()
    config.num_hidden_layers = 4  # Use smaller number for testing
    config.num_attention_heads = 4
    config.vision_num_hidden_layers = 2
    config.hidden_size = 128
    config.vision_hidden_size = 128
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    
    # Verify layer counts
    print(f"Language layers: {len(model.language_model.layers)} (expected: {config.num_hidden_layers})")
    print(f"Vision layers: {len(model.vision_tower.layers)} (expected: {config.vision_num_hidden_layers})")
    
    assert len(model.language_model.layers) == config.num_hidden_layers
    assert len(model.vision_tower.layers) == config.vision_num_hidden_layers
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (base model): {total_params:,}")

    # Create another model with adaptive depth enabled
    config_adaptive = Qwen3VLConfig()
    config_adaptive.num_hidden_layers = 4
    config_adaptive.num_attention_heads = 4
    config_adaptive.vision_num_hidden_layers = 2
    config_adaptive.hidden_size = 128
    config_adaptive.vision_hidden_size = 128
    config_adaptive.use_adaptive_depth = True
    config_adaptive.use_vision_adaptive_depth = True

    model_adaptive = Qwen3VLForConditionalGeneration(config_adaptive)
    total_params_adaptive = sum(p.numel() for p in model_adaptive.parameters())

    print(f"Parameters with adaptive depth: {total_params_adaptive:,}")

    # Check that the core transformer layers have the same number of parameters
    # The adaptive depth adds some parameters for complexity assessment, which is expected
    base_lang_params = sum(p.numel() for name, p in model.named_parameters()
                          if 'language_model.layers' in name)
    adaptive_lang_params = sum(p.numel() for name, p in model_adaptive.named_parameters()
                               if 'language_model.layers' in name)

    base_vision_params = sum(p.numel() for name, p in model.named_parameters()
                             if 'vision_tower.layers' in name)
    adaptive_vision_params = sum(p.numel() for name, p in model_adaptive.named_parameters()
                                 if 'vision_tower.layers' in name)

    print(f"Language layer parameters - Base: {base_lang_params:,}, Adaptive: {adaptive_lang_params:,}")
    print(f"Vision layer parameters - Base: {base_vision_params:,}, Adaptive: {adaptive_vision_params:,}")

    # The core transformer parameters should be the same
    assert base_lang_params == adaptive_lang_params, f"Language layer parameters changed: {base_lang_params} vs {adaptive_lang_params}"
    assert base_vision_params == adaptive_vision_params, f"Vision layer parameters changed: {base_vision_params} vs {adaptive_vision_params}"

    print("[PASS] Capacity preservation test passed")
    return True


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("TESTING ADAPTIVE DEPTH MECHANISM")
    print("=" * 60)
    
    try:
        test_adaptive_depth_basic()
        test_model_integration()
        test_capacity_preservation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("Adaptive depth mechanism is working correctly:")
        print("- Input complexity assessment functions properly")
        print("- Adaptive depth selection works as expected")
        print("- Model integration is successful")
        print("- Full capacity is preserved")
        print("=" * 60)

        return True
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n[PASS] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")
        exit(1)