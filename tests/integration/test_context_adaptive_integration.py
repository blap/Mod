"""
Integration test for context-adaptive positional representations in Qwen3-VL architecture.
This test verifies that the learned context-adaptive positional representations are properly
integrated with the existing Qwen3-VL architecture.
"""
import torch
import torch.nn as nn
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def test_context_adaptive_positional_encoding_integration():
    """Test the integration of context-adaptive positional encoding in Qwen3-VL."""
    print("Testing context-adaptive positional encoding integration...")
    
    # Create a simplified config for testing
    config = Qwen3VLConfig()
    config.num_hidden_layers = 4  # Reduced for testing
    config.num_attention_heads = 8  # Reduced for testing
    config.hidden_size = 256
    config.intermediate_size = 512
    config.vocab_size = 1000
    config.vision_hidden_size = 256
    config.vision_num_hidden_layers = 4
    config.vision_num_attention_heads = 8
    config.vision_intermediate_size = 512
    
    # Enable context-adaptive positional encoding
    config.use_context_adaptive_positional_encoding = True
    
    # Create the model
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Test text-only processing
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        text_output = model(input_ids=input_ids)
    
    assert text_output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected text output shape {(batch_size, seq_len, config.hidden_size)}, got {text_output.shape}"
    
    print("PASS: Text-only processing with context-adaptive positional encoding works")
    
    # Test vision-only processing
    image_size = 224
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    with torch.no_grad():
        vision_output = model(pixel_values=pixel_values)
    
    # Vision output shape depends on patch size and image dimensions
    patch_size = config.vision_patch_size
    expected_patches = (image_size // patch_size) ** 2
    assert vision_output.shape == (batch_size, expected_patches, config.hidden_size), \
        f"Expected vision output shape {(batch_size, expected_patches, config.hidden_size)}, got {vision_output.shape}"
    
    print("PASS: Vision-only processing with context-adaptive positional encoding works")
    
    # Test multimodal processing
    with torch.no_grad():
        multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Multimodal output combines vision and text features
    combined_seq_len = expected_patches + seq_len
    assert multimodal_output.shape == (batch_size, combined_seq_len, config.hidden_size), \
        f"Expected multimodal output shape {(batch_size, combined_seq_len, config.hidden_size)}, got {multimodal_output.shape}"
    
    print("PASS: Multimodal processing with context-adaptive positional encoding works")
    
    # Verify that the model has the positional encoder components
    assert hasattr(model.language_model, 'positional_encoder'), \
        "Language model should have positional_encoder when use_context_adaptive_positional_encoding=True"
    
    assert hasattr(model.vision_tower, 'positional_encoder'), \
        "Vision transformer should have positional_encoder when use_context_adaptive_positional_encoding=True"
    
    print("PASS: Model components properly initialized")
    
    print("All integration tests passed!")


def test_legacy_vs_context_adaptive_positional_encoding():
    """Compare behavior between legacy fixed positional encoding and context-adaptive encoding."""
    print("\nTesting legacy vs context-adaptive positional encoding...")
    
    # Create config with legacy positional encoding
    legacy_config = Qwen3VLConfig()
    legacy_config.num_hidden_layers = 2
    legacy_config.num_attention_heads = 4
    legacy_config.hidden_size = 128
    legacy_config.intermediate_size = 256
    legacy_config.vocab_size = 1000
    legacy_config.vision_hidden_size = 128
    legacy_config.vision_num_hidden_layers = 2
    legacy_config.vision_num_attention_heads = 4
    legacy_config.vision_intermediate_size = 256
    # Legacy config does not enable context-adaptive positional encoding
    
    # Create config with context-adaptive positional encoding
    adaptive_config = Qwen3VLConfig()
    adaptive_config.num_hidden_layers = 2
    adaptive_config.num_attention_heads = 4
    adaptive_config.hidden_size = 128
    adaptive_config.intermediate_size = 256
    adaptive_config.vocab_size = 1000
    adaptive_config.vision_hidden_size = 128
    adaptive_config.vision_num_hidden_layers = 2
    adaptive_config.vision_num_attention_heads = 4
    adaptive_config.vision_intermediate_size = 256
    adaptive_config.use_context_adaptive_positional_encoding = True
    
    # Create models
    legacy_model = Qwen3VLForConditionalGeneration(legacy_config)
    adaptive_model = Qwen3VLForConditionalGeneration(adaptive_config)
    
    # Set both models to eval mode and same parameters
    legacy_model.eval()
    adaptive_model.eval()
    
    # Set same random seed for initialization
    torch.manual_seed(42)
    legacy_model.load_state_dict(legacy_model.state_dict())  # Reset to original initialization
    adaptive_model.load_state_dict(adaptive_model.state_dict())  # Reset to original initialization
    
    # Create test inputs
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, legacy_config.vocab_size, (batch_size, seq_len))
    image_size = 224
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    with torch.no_grad():
        legacy_output = legacy_model(input_ids=input_ids)
        adaptive_output = adaptive_model(input_ids=input_ids)
    
    # Outputs should have the same shape but different values due to different positional encoding approaches
    assert legacy_output.shape == adaptive_output.shape, \
        f"Output shapes should match: {legacy_output.shape} vs {adaptive_output.shape}"
    
    # The outputs should be different since they use different positional encoding mechanisms
    are_different = not torch.allclose(legacy_output, adaptive_output, atol=1e-5)
    assert are_different, "Outputs should be different with different positional encoding approaches"
    
    print("PASS: Legacy and context-adaptive positional encoding produce different outputs as expected")
    print("PASS: Both approaches maintain the same output shape")
    
    print("Legacy vs adaptive comparison test passed!")


def test_gradient_flow_through_context_adaptive_encoding():
    """Test that gradients flow properly through the context-adaptive positional encoding."""
    print("\nTesting gradient flow through context-adaptive positional encoding...")
    
    config = Qwen3VLConfig()
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.hidden_size = 128
    config.intermediate_size = 256
    config.vocab_size = 1000
    config.vision_hidden_size = 128
    config.vision_num_hidden_layers = 2
    config.vision_num_attention_heads = 4
    config.vision_intermediate_size = 256
    config.use_context_adaptive_positional_encoding = True
    
    model = Qwen3VLForConditionalGeneration(config)
    model.train()  # Set to train mode for gradient computation
    
    # Create inputs with gradient tracking
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    image_size = 224
    pixel_values = torch.randn(batch_size, 3, image_size, image_size, requires_grad=True)
    
    # Forward pass
    output = model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Create a simple loss
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for key components
    assert pixel_values.grad is not None, "Gradients should exist for pixel_values"
    
    # Check that the context-adaptive positional encoding components have gradients
    assert model.language_model.positional_encoder.position_embeddings.weight.grad is not None, \
        "Gradients should exist for text positional embeddings"
    
    assert model.vision_tower.positional_encoder.row_embeddings.weight.grad is not None, \
        "Gradients should exist for vision row embeddings"
    
    assert model.vision_tower.positional_encoder.col_embeddings.weight.grad is not None, \
        "Gradients should exist for vision column embeddings"
    
    # Check that context adaptation network has gradients
    for param in model.language_model.positional_encoder.context_adaptation.parameters():
        assert param.grad is not None, f"Gradients should exist for context adaptation parameter: {param.shape}"
    
    for param in model.vision_tower.positional_encoder.context_adaptation.parameters():
        assert param.grad is not None, f"Gradients should exist for vision context adaptation parameter: {param.shape}"
    
    print("PASS: Gradients flow properly through context-adaptive positional encoding components")
    print("PASS: All network components have gradients as expected")
    
    print("Gradient flow test passed!")


def test_context_sensitivity():
    """Test that the context-adaptive positional encoding responds to different contexts."""
    print("\nTesting context sensitivity of adaptive positional encoding...")
    
    config = Qwen3VLConfig()
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.hidden_size = 128
    config.intermediate_size = 256
    config.vocab_size = 1000
    config.vision_hidden_size = 128
    config.vision_num_hidden_layers = 2
    config.vision_num_attention_heads = 4
    config.vision_intermediate_size = 256
    config.use_context_adaptive_positional_encoding = True
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Create two different input contexts
    batch_size = 1
    seq_len = 16
    context1 = torch.randint(0, config.vocab_size//2, (batch_size, seq_len))  # Lower vocab range
    context2 = torch.randint(config.vocab_size//2, config.vocab_size, (batch_size, seq_len))  # Higher vocab range
    
    with torch.no_grad():
        output1 = model(input_ids=context1)
        output2 = model(input_ids=context2)
    
    # With context-adaptive positional encoding, different inputs should lead to different
    # positional encodings even at the same positions
    are_different = not torch.allclose(output1, output2, atol=1e-5)
    assert are_different, "Outputs should be different for different input contexts"
    
    print("PASS: Context-adaptive positional encoding responds differently to different contexts")
    print("Context sensitivity test passed!")


def run_all_integration_tests():
    """Run all integration tests for context-adaptive positional representations."""
    print("Running integration tests for Context-Adaptive Positional Representations...")
    print("=" * 70)
    
    test_context_adaptive_positional_encoding_integration()
    test_legacy_vs_context_adaptive_positional_encoding()
    test_gradient_flow_through_context_adaptive_encoding()
    test_context_sensitivity()
    
    print("=" * 70)
    print("All integration tests passed!")


if __name__ == "__main__":
    run_all_integration_tests()