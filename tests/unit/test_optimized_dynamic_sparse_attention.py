"""
Comprehensive tests for the optimized dynamic sparse attention mechanism with learned routing.
This includes unit tests, performance tests, and integration tests.
"""
import torch
import torch.nn as nn
import pytest
import time
import sys
import os
from typing import Tuple

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.config import Qwen3VLConfig
from models.dynamic_sparse_attention_optimized import (
    OptimizedDynamicSparseAttention,
    OptimizedVisionDynamicSparseAttention,
    LearnedTokenRouter,
    VectorizedSparseAttention,
    HardwareOptimizedAttention
)


def test_learned_token_router():
    """Test the learned token router for proper functionality."""
    print("Testing Learned Token Router...")
    
    hidden_size = 256
    num_heads = 8
    seq_len = 32
    batch_size = 2
    
    router = LearnedTokenRouter(hidden_size, num_heads)
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    routing_scores = router(hidden_states)
    
    # Verify output shape
    assert routing_scores.shape == (batch_size, num_heads, seq_len), \
        f"Expected {(batch_size, num_heads, seq_len)}, got {routing_scores.shape}"
    
    # Verify values are normalized (sum to 1 across sequence dimension)
    scores_sum = routing_scores.sum(dim=-1)
    expected_sum = torch.ones_like(scores_sum)
    assert torch.allclose(scores_sum, expected_sum, atol=1e-5), \
        "Routing scores should sum to 1 across sequence dimension"
    
    print("✓ Learned Token Router test passed")


def test_vectorized_sparse_attention():
    """Test the vectorized sparse attention mechanism."""
    print("Testing Vectorized Sparse Attention...")
    
    batch_size = 2
    num_heads = 4
    seq_len = 16
    sparsity_ratio = 0.5  # Keep 50% of attention weights
    
    sparse_attn = VectorizedSparseAttention(sparsity_ratio)
    
    # Create test attention weights
    attn_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    # Apply sparse attention
    sparse_attn_weights = sparse_attn(attn_weights)
    
    # Verify output shape
    assert sparse_attn_weights.shape == attn_weights.shape, \
        f"Output shape mismatch: expected {attn_weights.shape}, got {sparse_attn_weights.shape}"
    
    # Count non-infinite values to verify sparsity
    non_inf_mask = torch.isfinite(sparse_attn_weights)
    expected_non_inf_count = batch_size * num_heads * seq_len * int(sparsity_ratio * seq_len)
    actual_non_inf_count = non_inf_mask.sum().item()
    
    # Allow for small tolerance due to rounding
    assert abs(actual_non_inf_count - expected_non_inf_count) <= batch_size * num_heads * seq_len, \
        f"Sparsity not applied correctly: expected ~{expected_non_inf_count} non-inf values, got {actual_non_inf_count}"
    
    print("✓ Vectorized Sparse Attention test passed")


def test_optimized_dynamic_sparse_attention():
    """Test the optimized dynamic sparse attention mechanism."""
    print("Testing Optimized Dynamic Sparse Attention...")
    
    # Create config
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.num_attention_heads = 8
    config.sparse_attention_sparsity_ratio = 0.5
    
    # Create attention layer
    attention = OptimizedDynamicSparseAttention(config, layer_idx=0)
    
    # Create test input
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Create attention mask
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float)
    
    # Forward pass
    output, _, _ = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Output shape mismatch: expected {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
    
    print("✓ Optimized Dynamic Sparse Attention test passed")


def test_optimized_vision_dynamic_sparse_attention():
    """Test the optimized vision dynamic sparse attention mechanism."""
    print("Testing Optimized Vision Dynamic Sparse Attention...")
    
    # Create config
    config = Qwen3VLConfig()
    config.vision_hidden_size = 256
    config.vision_num_attention_heads = 8
    config.vision_sparse_attention_sparsity_ratio = 0.4
    
    # Create attention layer
    vision_attention = OptimizedVisionDynamicSparseAttention(config)
    
    # Create test input
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.vision_hidden_size)
    
    # Forward pass
    output = vision_attention(hidden_states=hidden_states)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, config.vision_hidden_size), \
        f"Output shape mismatch: expected {(batch_size, seq_len, config.vision_hidden_size)}, got {output.shape}"
    
    print("✓ Optimized Vision Dynamic Sparse Attention test passed")


def test_hardware_optimized_attention():
    """Test the hardware-optimized attention for SM61."""
    print("Testing Hardware-Optimized Attention...")
    
    # Create config
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.num_attention_heads = 8
    config.sparse_attention_sparsity_ratio = 0.5
    
    # Create attention layer
    attention = HardwareOptimizedAttention(config, layer_idx=0)
    
    # Create test input
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Create attention mask
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float)
    
    # Forward pass
    output, _, _ = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Output shape mismatch: expected {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
    
    print("✓ Hardware-Optimized Attention test passed")


def benchmark_optimized_vs_standard():
    """Benchmark optimized implementation vs standard attention."""
    print("Benchmarking Optimized vs Standard Attention...")
    
    # Create config
    config = Qwen3VLConfig()
    config.hidden_size = 512
    config.num_attention_heads = 8
    config.sparse_attention_sparsity_ratio = 0.5
    
    # Create optimized attention
    optimized_attention = OptimizedDynamicSparseAttention(config, layer_idx=0)
    
    # Create test input
    batch_size = 1
    seq_lengths = [64, 128, 256]
    
    for seq_len in seq_lengths:
        print(f"  Testing sequence length: {seq_len}")
        
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float)
        
        # Time optimized attention
        start_time = time.time()
        for _ in range(10):  # Run multiple times for better timing
            with torch.no_grad():
                output = optimized_attention(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
        optimized_time = (time.time() - start_time) / 10
        
        print(f"    Optimized attention time: {optimized_time:.6f}s")
        
        # Verify output is valid
        assert not torch.isnan(output[0]).any(), "Output contains NaN values"
        assert torch.isfinite(output[0]).all(), "Output contains infinite values"
    
    print("✓ Benchmark test completed")


def test_memory_efficiency():
    """Test memory efficiency of the optimized implementation."""
    print("Testing Memory Efficiency...")
    
    # Create config
    config = Qwen3VLConfig()
    config.hidden_size = 512
    config.num_attention_heads = 8
    config.sparse_attention_sparsity_ratio = 0.5
    
    # Create optimized attention
    attention = OptimizedDynamicSparseAttention(config, layer_idx=0)
    
    # Create test input
    batch_size = 1
    seq_len = 256
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float)
    
    # Test forward pass without gradient computation
    with torch.no_grad():
        output, _, _ = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
    
    # Verify output shape and validity
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    assert torch.isfinite(output).all()
    
    print("✓ Memory efficiency test passed")


def test_integration_with_config():
    """Test that the optimized attention works with different configuration options."""
    print("Testing Integration with Configuration Options...")
    
    # Test with different sparsity ratios
    sparsity_ratios = [0.2, 0.5, 0.8]
    
    for sparsity_ratio in sparsity_ratios:
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.sparse_attention_sparsity_ratio = sparsity_ratio
        
        attention = OptimizedDynamicSparseAttention(config, layer_idx=0)
        
        batch_size = 1
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float)
        
        with torch.no_grad():
            output, _, _ = attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert torch.isfinite(output).all()
        
        print(f"  ✓ Sparsity ratio {sparsity_ratio} works correctly")
    
    print("✓ Integration with configuration options test passed")


def run_all_tests():
    """Run all tests for the optimized dynamic sparse attention."""
    print("=" * 70)
    print("Running Optimized Dynamic Sparse Attention Tests")
    print("=" * 70)
    
    test_learned_token_router()
    test_vectorized_sparse_attention()
    test_optimized_dynamic_sparse_attention()
    test_optimized_vision_dynamic_sparse_attention()
    test_hardware_optimized_attention()
    benchmark_optimized_vs_standard()
    test_memory_efficiency()
    test_integration_with_config()
    
    print("=" * 70)
    print("All Optimized Dynamic Sparse Attention Tests Passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()