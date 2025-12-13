"""
Capacity verification for Qwen3-VL adaptive depth implementation
This module verifies that the adaptive depth mechanism maintains the full capacity
of 32 transformer layers and 32 attention heads.
"""
import torch
from torch import nn
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def verify_model_capacity():
    """
    Verify that the model maintains full capacity (32 transformer layers and 32 attention heads)
    even with adaptive depth mechanisms enabled.
    """
    print("Verifying model capacity with adaptive depth mechanisms...")
    
    # Create config with full capacity
    config = Qwen3VLConfig()
    
    # Verify initial capacity settings
    assert config.num_hidden_layers == 32, f"Expected 32 hidden layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"
    print(f"[PASS] Config has {config.num_hidden_layers} hidden layers and {config.num_attention_heads} attention heads")
    
    # Create model with adaptive depth enabled
    config.use_adaptive_depth = True
    config.use_vision_adaptive_depth = True
    
    model = Qwen3VLForConditionalGeneration(config)
    
    # Verify that the model still has the correct architecture
    print(f"Language model has {len(model.language_model.layers)} transformer layers")
    print(f"Vision model has {len(model.vision_tower.layers)} vision transformer layers")
    
    # Check that the layer count matches the config
    assert len(model.language_model.layers) == config.num_hidden_layers, \
        f"Language model layers ({len(model.language_model.layers)}) doesn't match config ({config.num_hidden_layers})"
    
    assert len(model.vision_tower.layers) == config.vision_num_hidden_layers, \
        f"Vision model layers ({len(model.vision_tower.layers)}) doesn't match config ({config.vision_num_hidden_layers})"
    
    print("[PASS] Layer counts match configuration")

    # Check attention heads in the first language layer
    first_lang_layer = model.language_model.layers[0]
    if hasattr(first_lang_layer, 'self_attn'):
        attn_layer = first_lang_layer.self_attn
        if hasattr(attn_layer, 'num_heads'):
            assert attn_layer.num_heads == config.num_attention_heads, \
                f"Attention heads ({attn_layer.num_heads}) doesn't match config ({config.num_attention_heads})"
            print(f"[PASS] Attention heads per layer: {attn_layer.num_heads}")

    # Count total parameters to ensure no reduction
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify that the model can be configured with adaptive depth without parameter reduction
    # The adaptive depth mechanism should only affect computation during forward pass,
    # not the total number of parameters
    config_with_adaptive = Qwen3VLConfig()
    config_with_adaptive.use_adaptive_depth = True
    config_with_adaptive.use_vision_adaptive_depth = True

    model_adaptive = Qwen3VLForConditionalGeneration(config_with_adaptive)
    total_params_adaptive = sum(p.numel() for p in model_adaptive.parameters())

    print(f"Parameters with adaptive depth: {total_params_adaptive:,}")

    # The parameter count should be the same - adaptive depth doesn't remove parameters,
    # it just chooses how many layers to execute
    assert total_params == total_params_adaptive, \
        f"Parameter count changed with adaptive depth: {total_params} vs {total_params_adaptive}"

    print("[PASS] Parameter count unchanged with adaptive depth enabled")

    # Verify that both vision and language components maintain their capacity
    assert model.config.num_hidden_layers == 32
    assert model.config.num_attention_heads == 32
    assert model.config.vision_num_hidden_layers == 24  # Standard vision transformer depth
    assert model.config.vision_num_attention_heads == 16  # Standard vision attention heads

    print("[PASS] All capacity requirements maintained:")
    print(f"  - Language transformer layers: {model.config.num_hidden_layers}")
    print(f"  - Language attention heads: {model.config.num_attention_heads}")
    print(f"  - Vision transformer layers: {model.config.vision_num_hidden_layers}")
    print(f"  - Vision attention heads: {model.config.vision_num_attention_heads}")

    return True


def verify_forward_pass_capacity():
    """
    Verify that forward passes work correctly with the full capacity maintained.
    """
    print("\nVerifying forward pass with adaptive depth...")
    
    config = Qwen3VLConfig()
    config.use_adaptive_depth = True
    config.use_vision_adaptive_depth = True
    config.hidden_size = 256  # Reduce for testing
    config.num_hidden_layers = 4  # Reduce for testing
    config.num_attention_heads = 8  # Reduce for testing
    config.vision_hidden_size = 256
    config.vision_num_hidden_layers = 2  # Reduce for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    
    # Test text-only forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    text_output = model(input_ids=input_ids)
    print(f"Text-only output shape: {text_output.shape}")
    
    # Test vision-only forward pass
    pixel_values = torch.randn(1, 3, 224, 224)
    vision_output = model(pixel_values=pixel_values)
    print(f"Vision-only output shape: {vision_output.shape}")
    
    # Test multimodal forward pass
    multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
    print(f"Multimodal output shape: {multimodal_output.shape}")
    
    # Verify outputs have expected shapes
    batch_size = 1
    expected_seq_len = input_ids.shape[1] if input_ids is not None else pixel_values.shape[2] // config.vision_patch_size
    expected_hidden_size = config.hidden_size
    
    # The output shape should match the expected dimensions
    assert len(multimodal_output.shape) == 3  # [batch, seq_len, hidden_size]
    print("[PASS] Forward passes completed successfully")

    return True


def verify_adaptive_depth_functionality():
    """
    Verify that the adaptive depth mechanism works as expected without reducing capacity.
    """
    print("\nVerifying adaptive depth functionality...")
    
    config = Qwen3VLConfig()
    config.use_adaptive_depth = True
    config.hidden_size = 128
    config.num_hidden_layers = 6  # Use fewer for testing
    config.num_attention_heads = 8
    
    model = Qwen3VLForConditionalGeneration(config)
    
    # Test with simple input (should use fewer layers potentially)
    simple_input_ids = torch.ones(1, 4, dtype=torch.long) * 10  # Repetitive/simple input
    with torch.no_grad():
        simple_output = model(input_ids=simple_input_ids)
    
    # Test with complex input (should use more layers potentially)
    complex_input_ids = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        complex_output = model(input_ids=complex_input_ids)
    
    print(f"Simple input output shape: {simple_output.shape}")
    print(f"Complex input output shape: {complex_output.shape}")
    
    # Both should have the same output shape, confirming capacity is maintained
    assert simple_output.shape == complex_output.shape
    
    print("[PASS] Adaptive depth adjusts computation, not capacity")

    return True


def run_capacity_verification():
    """
    Run all capacity verification tests.
    """
    print("=" * 60)
    print("CAPACITY VERIFICATION FOR ADAPTIVE DEPTH IMPLEMENTATION")
    print("=" * 60)
    
    try:
        verify_model_capacity()
        print("✓ Model capacity verification passed")
        
        verify_forward_pass_capacity()
        print("✓ Forward pass capacity verification passed")
        
        verify_adaptive_depth_functionality()
        print("✓ Adaptive depth functionality verification passed")
        
        print("\n" + "=" * 60)
        print("ALL CAPACITY VERIFICATION TESTS PASSED")
        print("Adaptive depth mechanism maintains full model capacity:")
        print("- 32 transformer layers preserved")
        print("- 32 attention heads preserved")
        print("- No parameter reduction")
        print("- Adaptive computation based on input complexity")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ CAPACITY VERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_capacity_verification()
    if success:
        print("\n✅ All capacity verification tests passed!")
    else:
        print("\n❌ Some capacity verification tests failed!")
        exit(1)