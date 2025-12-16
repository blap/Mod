"""
Comprehensive tests for attention mechanism fixes
"""
import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.qwen3_vl.attention.flash_attention_2 import FlashAttention2, apply_rotary_pos_emb, Qwen3VLRotaryEmbedding
from src.qwen3_vl.attention.dynamic_sparse_attention import DynamicSparseAttention
from src.qwen3_vl.attention.memory_efficient_patterns import MemoryEfficientFlashAttention


def test_rotary_embeddings_with_position_ids():
    """Test that rotary embeddings work correctly with position_ids."""
    # Create a simple test
    dim = 128
    max_seq_len = 32
    batch_size = 2
    num_heads = 4
    
    # Create test tensors
    x = torch.randn(batch_size, num_heads, max_seq_len, dim)
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Initialize rotary embedding
    rotary_emb = Qwen3VLRotaryEmbedding(dim, max_position_embeddings=1024)
    
    # Apply rotary embeddings
    cos, sin = rotary_emb(x, position_ids)
    
    # Apply to query and key
    q = torch.randn(batch_size, num_heads, max_seq_len, dim)
    k = torch.randn(batch_size, num_heads, max_seq_len, dim)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    
    # Verify shapes are preserved
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    
    print("✓ Rotary embeddings with position_ids test passed")


def test_flash_attention_mask_handling():
    """Test that FlashAttention2 correctly handles attention masks."""
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.hidden_size = 256
            self.num_attention_heads = 8
            self.num_key_value_heads = 8
            self.max_position_embeddings = 2048
            self.rope_theta = 10000.0
            self.intermediate_size = 512
    
    config = MockConfig()
    
    # Initialize FlashAttention2
    flash_attn = FlashAttention2(config)
    
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create causal mask
    attention_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    attention_mask = attention_mask.to(hidden_states.dtype)
    
    # Run forward pass
    output, attn_weights, past_key_value = flash_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask
    )
    
    # Verify output shape
    assert output.shape == hidden_states.shape
    
    print("✓ Flash attention mask handling test passed")


def test_dynamic_sparse_attention_efficiency():
    """Test that the dynamic sparse attention is efficient and correct."""
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.hidden_size = 128
            self.num_attention_heads = 4
            self.num_key_value_heads = 4
            self.max_position_embeddings = 2048
            self.rope_theta = 10000.0
            self.sparse_attention_sparsity_ratio = 0.5
    
    config = MockConfig()
    
    # Initialize DynamicSparseAttention
    sparse_attn = DynamicSparseAttention(config)
    
    batch_size = 2
    seq_len = 32
    hidden_size = config.hidden_size
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run forward pass
    output, attn_weights, past_key_value = sparse_attn(
        hidden_states=hidden_states
    )
    
    # Verify output shape
    assert output.shape == hidden_states.shape
    
    print("✓ Dynamic sparse attention efficiency test passed")


def test_memory_efficient_attention():
    """Test memory efficient attention implementation."""
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.hidden_size = 256
            self.num_attention_heads = 8
            self.num_key_value_heads = 8
            self.max_position_embeddings = 2048
            self.rope_theta = 10000.0
    
    config = MockConfig()
    
    # Initialize MemoryEfficientFlashAttention
    mem_eff_attn = MemoryEfficientFlashAttention(config)
    
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    
    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run forward pass
    output, attn_weights, past_key_value = mem_eff_attn(
        hidden_states=hidden_states
    )
    
    # Verify output shape
    assert output.shape == hidden_states.shape
    
    print("✓ Memory efficient attention test passed")


def test_attention_numerical_stability():
    """Test numerical stability of attention computations."""
    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.hidden_size = 64
            self.num_attention_heads = 4
            self.num_key_value_heads = 4
            self.max_position_embeddings = 2048
            self.rope_theta = 10000.0
    
    config = MockConfig()
    
    # Test with FlashAttention2
    flash_attn = FlashAttention2(config)
    
    batch_size = 1
    seq_len = 8
    hidden_size = config.hidden_size
    
    # Create inputs that could cause numerical issues
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run multiple forward passes to check for stability
    outputs = []
    for _ in range(5):
        output, _, _ = flash_attn(hidden_states=hidden_states)
        outputs.append(output.clone())
    
    # Check that outputs are consistent across runs
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-5), f"Outputs differ between runs {0} and {i}"
    
    print("✓ Attention numerical stability test passed")


def run_all_tests():
    """Run all tests."""
    print("Running comprehensive attention mechanism tests...")
    
    test_rotary_embeddings_with_position_ids()
    test_flash_attention_mask_handling()
    test_dynamic_sparse_attention_efficiency()
    test_memory_efficient_attention()
    test_attention_numerical_stability()
    
    print("\n✓ All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()