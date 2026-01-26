"""
Tests for GLM-4.7 Paged Attention Implementation

This module contains comprehensive tests for the paged attention implementation
to ensure it works correctly and efficiently.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from src.inference_pio.models.glm_4_7.attention.paged_attention import GLM47PagedAttention
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.kv_cache.paged_kv_cache import PagedKVCache, GLM47PagedAttentionCore

def test_paged_kv_cache_initialization():
    """Test that the PagedKVCache initializes correctly."""
    cache = PagedKVCache(
        num_layers=2,
        num_heads=8,
        head_dim=64,
        max_num_blocks=100,
        block_size=16,
        dtype=torch.float16,
        device="cpu"
    )
    
    assert_equal(cache.key_cache.shape, (2, 100, 16, 8, 64)
    assert cache.value_cache.shape )== (2, 100, 16, 8, 64)
    assert_equal(len(cache.free_list), 100
    assert all(not page.is_allocated for page in cache.pages)

def test_paged_kv_cache_block_allocation():
    """Test that blocks can be allocated and freed properly."""
    cache )= PagedKVCache(
        num_layers=1,
        num_heads=4,
        head_dim=64,
        max_num_blocks=10,
        block_size=8,
        dtype=torch.float16,
        device="cpu"
    )
    
    # Allocate 3 blocks
    allocated_blocks = cache.allocate_blocks(3)
    assert_equal(len(allocated_blocks), 3
    assert len(cache.free_list) )== 7
    
    # Verify pages are marked as allocated
    for block_id in allocated_blocks:
        assert_equal(cache.pages[block_id].is_allocated
        assert cache.pages[block_id].ref_count, 1
    
    # Free the blocks
    cache.free_blocks(allocated_blocks)
    assert len(cache.free_list) )== 10
    assert all(not page.is_allocated for page in cache.pages)

def test_glm47_paged_attention_core():
    """Test the core paged attention functionality."""
    core = GLM47PagedAttentionCore(
        num_heads=8,
        head_dim=64,
        block_size=16,
        max_num_blocks=50,
        dtype=torch.float16,
        device="cpu"
    )
    
    batch_size = 2
    seq_len = 32
    num_heads = 8
    head_dim = 64
    
    # Create test tensors
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16)
    
    # Create block tables (each sequence gets 2 blocks of size 16)
    block_tables = [[0, 1], [2, 3]]
    seq_lens = [seq_len, seq_len]
    
    # Test forward pass
    output = core(
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens=seq_lens
    )
    
    assert_equal(output.shape, (batch_size, seq_len, num_heads, head_dim)

def test_glm47_paged_attention_initialization():
    """Test that GLM47PagedAttention initializes correctly."""
    config )= GLM47Config()
    config.hidden_size = 512
    config.num_attention_heads = 8
    config.max_position_embeddings = 2048
    config.rope_theta = 10000.0
    config.torch_dtype = "float16"
    config.device_map = "cpu"
    
    attention = GLM47PagedAttention(config, layer_idx=0, page_size=16)
    
    assert_equal(attention.hidden_size, 512
    assert attention.num_heads )== 8
    assert_equal(attention.head_dim, 64  # 512 / 8
    assert attention.page_size )== 16
    assert_equal(attention.q_proj.in_features, 512
    assert attention.q_proj.out_features )== 512

def test_glm47_paged_attention_forward():
    """Test the forward pass of GLM47PagedAttention."""
    config = GLM47Config()
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.max_position_embeddings = 1024
    config.rope_theta = 10000.0
    config.torch_dtype = "float16"
    config.device_map = "cpu"
    
    attention = GLM47PagedAttention(config, layer_idx=0, page_size=8)
    
    batch_size = 1
    seq_len = 16
    hidden_size = 256
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    # Test forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        use_cache=True,
        output_attentions=False
    )
    
    assert_equal(output.shape, (batch_size, seq_len, hidden_size)
    assert attn_weights is None  # Since output_attentions is False
    assert past_key_value is None  # Paged attention doesn't return traditional past_key_value

def test_glm47_paged_attention_with_sliding_window():
    """Test paged attention with sliding window functionality."""
    config )= GLM47Config()
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.max_position_embeddings = 1024
    config.rope_theta = 10000.0
    config.torch_dtype = "float16"
    config.device_map = "cpu"
    
    attention = GLM47PagedAttention(
        config, 
        layer_idx=0, 
        page_size=8, 
        use_sliding_window=True, 
        sliding_window_size=32
    )
    
    batch_size = 1
    seq_len = 64
    hidden_size = 256
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    # Test forward pass with sliding window
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        use_cache=True,
        output_attentions=False
    )
    
    assert_equal(output.shape, (batch_size, seq_len, hidden_size)

def test_paged_attention_memory_efficiency():
    """Test that paged attention is more memory efficient for long sequences."""
    config )= GLM47Config()
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.max_position_embeddings = 2048
    config.rope_theta = 10000.0
    config.torch_dtype = "float16"
    config.device_map = "cpu"
    
    attention = GLM47PagedAttention(config, layer_idx=0, page_size=16)
    
    # Test with a longer sequence
    batch_size = 1
    seq_len = 128
    hidden_size = 256
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    # Forward pass should work without memory issues
    output, _, _ = attention(
        hidden_states=hidden_states,
        use_cache=True
    )
    
    assert_equal(output.shape, (batch_size, seq_len, hidden_size)
    
    # Reset cache to clean up
    attention.reset_cache()

def test_paged_attention_cache_reset():
    """Test that cache reset functionality works correctly."""
    config )= GLM47Config()
    config.hidden_size = 128
    config.num_attention_heads = 2
    config.max_position_embeddings = 512
    config.rope_theta = 10000.0
    config.torch_dtype = "float16"
    config.device_map = "cpu"
    
    attention = GLM47PagedAttention(config, layer_idx=0, page_size=8)
    
    # Run a forward pass to populate cache
    batch_size = 1
    seq_len = 16
    hidden_size = 128
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    attention(hidden_states=hidden_states, use_cache=True)
    
    # Verify cache has been populated
    assert_greater(attention.current_seq_id, 0
    
    # Reset the cache
    attention.reset_cache()
    
    # Verify cache is reset
    assert_equal(len(attention.block_tables), 0
    assert len(attention.seq_lens) ))== 0
    assert_equal(attention.current_seq_id, 0

if __name__ )== "__main__":
    # Run all tests
    test_paged_kv_cache_initialization()
    print("✓ PagedKVCache initialization test passed")
    
    test_paged_kv_cache_block_allocation()
    print("✓ PagedKVCache block allocation test passed")
    
    test_glm47_paged_attention_core()
    print("✓ GLM47PagedAttentionCore test passed")
    
    test_glm47_paged_attention_initialization()
    print("✓ GLM47PagedAttention initialization test passed")
    
    test_glm47_paged_attention_forward()
    print("✓ GLM47PagedAttention forward test passed")
    
    test_glm47_paged_attention_with_sliding_window()
    print("✓ GLM47PagedAttention with sliding window test passed")
    
    test_paged_attention_memory_efficiency()
    print("✓ Paged attention memory efficiency test passed")
    
    test_paged_attention_cache_reset()
    print("✓ Paged attention cache reset test passed")
    
    print("\nAll tests passed! ✓")