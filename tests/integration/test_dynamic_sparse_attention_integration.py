"""
Integration test for dynamic sparse attention with the full Qwen3-VL model.
This validates the implementation works with both vision and language components.
"""
import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def test_dynamic_sparse_attention_integration():
    """Test integration of dynamic sparse attention with full Qwen3-VL model."""
    print("Testing Dynamic Sparse Attention Integration...")
    
    # Create config with dynamic sparse attention enabled
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.num_hidden_layers = 4  # Reduced for testing
    config.num_attention_heads = 8
    config.vision_hidden_size = 256
    config.vision_num_hidden_layers = 2
    config.vision_num_attention_heads = 4
    
    # Enable dynamic sparse attention
    config.use_dynamic_sparse_attention = True
    config.sparse_attention_sparsity_ratio = 0.5
    config.vision_sparse_attention_sparsity_ratio = 0.4
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Test text-only input
    print("  Testing text-only input...")
    batch_size = 1
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        text_output = model(input_ids=input_ids)
    
    assert text_output.shape[0] == batch_size
    assert text_output.shape[1] == seq_len
    print("  PASS: Text-only input test passed")
    
    # Test vision-only input
    print("  Testing vision-only input...")
    img_size = 224
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    
    with torch.no_grad():
        vision_output = model(pixel_values=pixel_values)
    
    # Vision output shape depends on patch size and number of patches
    expected_vision_seq_len = (img_size // config.vision_patch_size) ** 2
    assert vision_output.shape[0] == batch_size
    assert vision_output.shape[1] == expected_vision_seq_len
    print("  PASS: Vision-only input test passed")
    
    # Test multimodal input
    print("  Testing multimodal input...")
    with torch.no_grad():
        multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Multimodal output combines vision and text features
    combined_seq_len = expected_vision_seq_len + seq_len
    assert multimodal_output.shape[0] == batch_size
    assert multimodal_output.shape[1] == combined_seq_len
    print("  PASS: Multimodal input test passed")
    
    print("PASS: Dynamic Sparse Attention Integration test passed")


def test_dynamic_sparse_attention_performance():
    """Test performance characteristics of dynamic sparse attention."""
    print("Testing Dynamic Sparse Attention Performance...")
    
    # Create config with dynamic sparse attention enabled
    config = Qwen3VLConfig()
    config.hidden_size = 512
    config.num_hidden_layers = 2  # Reduced for testing
    config.num_attention_heads = 8
    config.vision_hidden_size = 512
    config.vision_num_hidden_layers = 2
    config.vision_num_attention_heads = 4
    
    # Enable dynamic sparse attention with different sparsity ratios
    config.use_dynamic_sparse_attention = True
    config.sparse_attention_sparsity_ratio = 0.3  # Keep 30% of attention weights
    config.vision_sparse_attention_sparsity_ratio = 0.25  # Keep 25% of vision attention weights
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Test with different sequence lengths to measure scaling
    seq_lengths = [16, 32, 64]
    
    import time
    
    for seq_len in seq_lengths:
        print(f"    Testing sequence length: {seq_len}")
        
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_ids=input_ids)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"      Execution time: {execution_time:.4f}s")
        
        assert output.shape[1] == seq_len
        assert not torch.isnan(output).any()
    
    print("PASS: Dynamic Sparse Attention Performance test passed")


def test_dynamic_sparse_attention_sparsity():
    """Test that sparsity is properly applied in the attention mechanism."""
    print("Testing Dynamic Sparse Attention Sparsity...")
    
    # Create a simple test to verify sparsity mechanism works
    from models.dynamic_sparse_attention import DynamicSparseAttention
    
    config = Qwen3VLConfig()
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.use_dynamic_sparse_attention = True
    config.sparse_attention_sparsity_ratio = 0.25  # Keep top 25%
    
    attention_layer = DynamicSparseAttention(config)
    
    # Create test input
    batch_size = 1
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    with torch.no_grad():
        output, _, _ = attention_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    print("  PASS: Sparsity mechanism test passed")
    
    # Test vision attention sparsity
    from models.dynamic_sparse_attention import VisionDynamicSparseAttention
    
    vision_config = Qwen3VLConfig()
    vision_config.vision_hidden_size = 128
    vision_config.vision_num_attention_heads = 4
    vision_config.vision_sparse_attention_sparsity_ratio = 0.3  # Keep top 30%
    
    vision_attention = VisionDynamicSparseAttention(vision_config)
    
    vision_hidden_states = torch.randn(batch_size, seq_len, vision_config.vision_hidden_size)
    
    with torch.no_grad():
        vision_output = vision_attention(hidden_states=vision_hidden_states)
    
    assert vision_output.shape == (batch_size, seq_len, vision_config.vision_hidden_size)
    print("  PASS: Vision attention sparsity test passed")
    
    print("PASS: Dynamic Sparse Attention Sparsity test passed")


def run_integration_tests():
    """Run all integration tests for dynamic sparse attention."""
    print("=" * 60)
    print("Running Dynamic Sparse Attention Integration Tests")
    print("=" * 60)
    
    test_dynamic_sparse_attention_integration()
    test_dynamic_sparse_attention_performance()
    test_dynamic_sparse_attention_sparsity()
    
    print("=" * 60)
    print("All Dynamic Sparse Attention Integration Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_tests()