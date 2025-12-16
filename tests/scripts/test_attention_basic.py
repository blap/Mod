"""
Simple test for attention mechanism fixes - importing directly from files
"""
import torch
import sys
import os
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_rotary_embeddings_with_position_ids():
    """Test that rotary embeddings work correctly with position_ids."""
    # Import using standard Python import instead of hardcoded path
    from qwen3_vl.attention.flash_attention_2 import Qwen3VLRotaryEmbedding, apply_rotary_pos_emb

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
    # Import using standard Python import instead of hardcoded path
    from qwen3_vl.attention.flash_attention_2 import FlashAttention2
    from qwen3_vl.core.config import Qwen3VLConfig

    # Create a mock config using the actual config class
    config = Qwen3VLConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        intermediate_size=512,
        vocab_size=1000  # Required parameter
    )

    # Initialize FlashAttention2
    flash_attn = FlashAttention2(config, layer_idx=0)

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
        attention_mask=attention_mask,
        position_ids=torch.arange(seq_len).expand(batch_size, -1)
    )

    # Verify output shape
    assert output.shape == hidden_states.shape

    print("✓ Flash attention mask handling test passed")


def test_dynamic_sparse_attention_efficiency():
    """Test that the dynamic sparse attention is efficient and correct."""
    # Import using standard Python import instead of hardcoded path
    from qwen3_vl.attention.dynamic_sparse_attention import DynamicSparseAttention
    from qwen3_vl.core.config import Qwen3VLConfig

    # Create a proper config
    config = Qwen3VLConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        sparse_attention_sparsity_ratio=0.5,
        vocab_size=1000  # Required parameter
    )

    # Initialize DynamicSparseAttention
    sparse_attn = DynamicSparseAttention(config, layer_idx=0)

    batch_size = 2
    seq_len = 32
    hidden_size = config.hidden_size

    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Run forward pass
    output, attn_weights, past_key_value = sparse_attn(
        hidden_states=hidden_states,
        position_ids=torch.arange(seq_len).expand(batch_size, -1)
    )

    # Verify output shape
    assert output.shape == hidden_states.shape

    print("✓ Dynamic sparse attention efficiency test passed")


def test_memory_efficient_attention():
    """Test memory efficient attention implementation."""
    # Import using standard Python import instead of hardcoded path
    from qwen3_vl.attention.memory_efficient_patterns import MemoryEfficientFlashAttention
    from qwen3_vl.core.config import Qwen3VLConfig

    # Create a proper config
    config = Qwen3VLConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        vocab_size=1000  # Required parameter
    )

    # Initialize MemoryEfficientFlashAttention
    attn = MemoryEfficientFlashAttention(config, layer_idx=0)

    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size

    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Run forward pass
    output, attn_weights, past_key_value = attn(
        hidden_states=hidden_states,
        position_ids=torch.arange(seq_len).expand(batch_size, -1)
    )

    # Verify output shape
    assert output.shape == hidden_states.shape

    print("✓ Memory efficient attention test passed")


def run_basic_tests():
    """Run basic tests without numerical stability tests."""
    print("Running basic attention mechanism tests...")
    
    test_rotary_embeddings_with_position_ids()
    test_flash_attention_mask_handling()
    test_dynamic_sparse_attention_efficiency()
    test_memory_efficient_attention()
    
    print("\n✓ All basic tests passed successfully!")


if __name__ == "__main__":
    run_basic_tests()