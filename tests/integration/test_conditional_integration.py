"""
Integration test for conditional feature extraction with the Qwen3-VL model
"""
import sys
import os
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def test_conditional_feature_extraction_integration():
    """Test that conditional feature extraction integrates properly with the main model."""
    print("Testing conditional feature extraction integration...")
    
    # Create configuration with conditional feature extraction enabled
    config = Qwen3VLConfig()
    config.num_hidden_layers = 4  # Reduced for testing
    config.num_attention_heads = 8  # Reduced for testing
    config.hidden_size = 256
    config.vision_hidden_size = 256
    config.vision_num_hidden_layers = 2
    config.vocab_size = 1000
    config.use_conditional_feature_extraction = True  # Enable conditional feature extraction
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Test inputs
    batch_size = 1
    seq_len = 16
    img_size = 224
    
    text_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    image_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print("  - Testing text-only input...")
    with torch.no_grad():
        text_output = model(input_ids=text_input)
    print(f"    Text output shape: {text_output.shape}")
    
    print("  - Testing vision-only input...")
    with torch.no_grad():
        vision_output = model(pixel_values=image_input)
    print(f"    Vision output shape: {vision_output.shape}")
    
    print("  - Testing multimodal input...")
    with torch.no_grad():
        multimodal_output = model(input_ids=text_input, pixel_values=image_input)
    print(f"    Multimodal output shape: {multimodal_output.shape}")
    
    print("Integration test completed successfully!")


def test_backward_compatibility():
    """Test that the model still works when conditional feature extraction is disabled."""
    print("\nTesting backward compatibility...")
    
    # Create configuration with conditional feature extraction disabled
    config = Qwen3VLConfig()
    config.num_hidden_layers = 4  # Reduced for testing
    config.num_attention_heads = 8  # Reduced for testing
    config.hidden_size = 256
    config.vision_hidden_size = 256
    config.vision_num_hidden_layers = 2
    config.vocab_size = 1000
    config.use_conditional_feature_extraction = False  # Disable conditional feature extraction
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Test inputs
    batch_size = 1
    seq_len = 16
    img_size = 224
    
    text_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    image_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print("  - Testing text-only input (original path)...")
    with torch.no_grad():
        text_output = model(input_ids=text_input)
    print(f"    Text output shape: {text_output.shape}")
    
    print("  - Testing vision-only input (original path)...")
    with torch.no_grad():
        vision_output = model(pixel_values=image_input)
    print(f"    Vision output shape: {vision_output.shape}")
    
    print("  - Testing multimodal input (original path)...")
    with torch.no_grad():
        multimodal_output = model(input_ids=text_input, pixel_values=image_input)
    print(f"    Multimodal output shape: {multimodal_output.shape}")
    
    print("Backward compatibility test completed successfully!")


def test_performance_comparison():
    """Compare performance with and without conditional feature extraction."""
    import time
    
    print("\nTesting performance comparison...")
    
    # Create two configurations
    config_conditional = Qwen3VLConfig()
    config_conditional.num_hidden_layers = 2
    config_conditional.num_attention_heads = 4
    config_conditional.hidden_size = 128
    config_conditional.vision_hidden_size = 128
    config_conditional.vision_num_hidden_layers = 2
    config_conditional.vocab_size = 1000
    config_conditional.use_conditional_feature_extraction = True
    
    config_original = Qwen3VLConfig()
    config_original.num_hidden_layers = 2
    config_original.num_attention_heads = 4
    config_original.hidden_size = 128
    config_original.vision_hidden_size = 128
    config_original.vision_num_hidden_layers = 2
    config_original.vocab_size = 1000
    config_original.use_conditional_feature_extraction = False
    
    # Create models
    model_conditional = Qwen3VLForConditionalGeneration(config_conditional)
    model_original = Qwen3VLForConditionalGeneration(config_original)
    
    model_conditional.eval()
    model_original.eval()
    
    # Test inputs
    batch_size = 1
    seq_len = 8
    img_size = 112  # Smaller for faster testing
    
    text_input = torch.randint(0, config_conditional.vocab_size, (batch_size, seq_len))
    image_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Time conditional model
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5):  # Run multiple times for better average
            _ = model_conditional(input_ids=text_input, pixel_values=image_input)
    conditional_time = time.time() - start_time
    
    # Time original model
    start_time = time.time()
    with torch.no_grad():
        for _ in range(5):  # Run multiple times for better average
            _ = model_original(input_ids=text_input, pixel_values=image_input)
    original_time = time.time() - start_time
    
    print(f"  Conditional model time: {conditional_time:.4f}s")
    print(f"  Original model time: {original_time:.4f}s")
    
    print("Performance comparison completed!")


if __name__ == "__main__":
    test_conditional_feature_extraction_integration()
    test_backward_compatibility()
    test_performance_comparison()
    print("\nAll integration tests passed!")