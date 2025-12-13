"""
Backward compatibility test to ensure that the original functionality still works when 
context-adaptive positional encoding is disabled.
"""
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def test_backward_compatibility():
    """Test that original functionality still works when context-adaptive positional encoding is disabled."""
    print("Testing backward compatibility...")
    
    # Create config WITHOUT context-adaptive positional encoding
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
    config.use_context_adaptive_positional_encoding = False  # This should use the original approach
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Verify that the model does NOT have the new positional encoder components
    assert not hasattr(model.language_model, 'positional_encoder'), \
        "Language model should NOT have positional_encoder when use_context_adaptive_positional_encoding=False"
    
    assert not hasattr(model.vision_tower, 'positional_encoder'), \
        "Vision transformer should NOT have positional_encoder when use_context_adaptive_positional_encoding=False"
    
    # But it should still have the original positional embeddings
    assert hasattr(model.vision_tower, 'position_embedding'), \
        "Vision transformer should have original position_embedding when adaptive is disabled"
    
    print("PASS: Model correctly uses original positional encoding approach when adaptive is disabled")
    
    # Test that forward pass still works
    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        text_output = model(input_ids=input_ids)
    
    assert text_output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Text output shape incorrect: {text_output.shape}"
    
    # Test vision
    image_size = 112
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    with torch.no_grad():
        vision_output = model(pixel_values=pixel_values)
    
    patch_size = config.vision_patch_size
    expected_patches = (image_size // patch_size) ** 2
    assert vision_output.shape == (batch_size, expected_patches, config.hidden_size), \
        f"Vision output shape incorrect: {vision_output.shape}"
    
    # Test multimodal
    with torch.no_grad():
        multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
    
    expected_seq_len = seq_len + expected_patches
    assert multimodal_output.shape == (batch_size, expected_seq_len, config.hidden_size), \
        f"Multimodal output shape incorrect: {multimodal_output.shape}"
    
    print("PASS: All processing modes work correctly with original positional encoding")
    print("PASS: Backward compatibility maintained")


def test_config_option():
    """Test that the config option properly controls the behavior."""
    print("Testing config option control...")
    
    # Test with context-adaptive enabled
    adaptive_config = Qwen3VLConfig()
    adaptive_config.num_hidden_layers = 1
    adaptive_config.num_attention_heads = 2
    adaptive_config.hidden_size = 64
    adaptive_config.vocab_size = 500
    adaptive_config.vision_hidden_size = 64
    adaptive_config.vision_num_hidden_layers = 1
    adaptive_config.vision_num_attention_heads = 2
    adaptive_config.use_context_adaptive_positional_encoding = True
    
    adaptive_model = Qwen3VLForConditionalGeneration(adaptive_config)
    
    assert hasattr(adaptive_model.language_model, 'positional_encoder'), \
        "Model with adaptive=True should have positional_encoder"
    assert hasattr(adaptive_model.vision_tower, 'positional_encoder'), \
        "Vision with adaptive=True should have positional_encoder"
    
    print("PASS: Config with adaptive=True enables new positional encoding")
    
    # Test with context-adaptive disabled
    original_config = Qwen3VLConfig()
    original_config.num_hidden_layers = 1
    original_config.num_attention_heads = 2
    original_config.hidden_size = 64
    original_config.vocab_size = 500
    original_config.vision_hidden_size = 64
    original_config.vision_num_hidden_layers = 1
    original_config.vision_num_attention_heads = 2
    original_config.use_context_adaptive_positional_encoding = False
    
    original_model = Qwen3VLForConditionalGeneration(original_config)
    
    assert not hasattr(original_model.language_model, 'positional_encoder'), \
        "Model with adaptive=False should NOT have positional_encoder"
    assert not hasattr(original_model.vision_tower, 'positional_encoder'), \
        "Vision with adaptive=False should NOT have positional_encoder"
    
    print("PASS: Config with adaptive=False uses original positional encoding")
    
    print("PASS: Config option properly controls positional encoding behavior")


def run_compatibility_tests():
    """Run all backward compatibility tests."""
    print("Running Backward Compatibility Tests")
    print("=" * 40)
    
    test_backward_compatibility()
    test_config_option()
    
    print("=" * 40)
    print("All compatibility tests passed!")
    print("The implementation maintains full backward compatibility.")


if __name__ == "__main__":
    run_compatibility_tests()