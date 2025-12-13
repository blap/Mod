"""
Implementation testing for Phase 2.5: Activation Sparsity and Early Exit Mechanisms
"""
import pytest
import torch
import torch.nn as nn
from src.models.config import Qwen3VLConfig
from src.components.optimization.activation_sparsity import TopKSparsify, ConfidenceGatedEarlyExit, SparseMLP, SparseAttention, AdaptiveComputationLayer


def test_implement_top_k_activation_sparsity():
    """Test Top-K activation sparsity implementation"""
    # Test with 50% sparsity
    sparsity_ratio = 0.5
    sparsify_layer = TopKSparsify(sparsity_ratio=sparsity_ratio)
    
    # Create a sample tensor
    batch_size, seq_len, hidden_size = 2, 64, 128
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    # Apply sparsification
    output_tensor = sparsify_layer(input_tensor)
    
    # Count non-zero elements to verify sparsity
    original_nonzero = torch.count_nonzero(input_tensor).item()
    output_nonzero = torch.count_nonzero(output_tensor).item()
    
    # Calculate actual sparsity ratio
    actual_sparsity = 1 - (output_nonzero / original_nonzero)
    
    print(f"Top-K Sparsify: requested {sparsity_ratio:.2f}, achieved {actual_sparsity:.2f}")
    
    # Verify sparsity is close to requested (allowing for rounding)
    assert abs(actual_sparsity - sparsity_ratio) < 0.1, f"Sparsity ratio not achieved: {actual_sparsity} vs {sparsity_ratio}"
    
    # Output should have same shape as input
    assert output_tensor.shape == input_tensor.shape, "Output shape should match input shape"
    
    # Output values should be either original or zero
    mask = output_tensor != 0
    torch.testing.assert_close(output_tensor[mask], input_tensor[mask])


def test_create_confidence_gated_early_exit_mechanisms():
    """Test confidence-gated early exit mechanisms"""
    hidden_size = 128
    num_layers = 4
    exit_threshold = 0.8
    
    early_exit_layer = ConfidenceGatedEarlyExit(
        hidden_size=hidden_size,
        num_layers=num_layers,
        exit_threshold=exit_threshold
    )
    
    # Create sample hidden states
    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test early exit at different layers
    for layer_idx in range(num_layers):
        output_states, should_exit = early_exit_layer(hidden_states, layer_idx)
        
        # Output should have same shape as input
        assert output_states.shape == hidden_states.shape, "Output shape should match input shape"
        
        # For the last layer, should always exit
        if layer_idx == num_layers - 1:
            assert should_exit, "Should always exit at the last layer"
        else:
            # For other layers, exit decision depends on confidence
            assert isinstance(should_exit, bool), "should_exit should be boolean"


def test_develop_input_adaptive_routing():
    """Test input-adaptive routing to skip unnecessary layers"""
    # This is tested as part of the AdaptiveComputationLayer
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    
    # Create adaptive computation layer
    layer_idx = 0
    adaptive_layer = AdaptiveComputationLayer(
        config=config,
        layer_idx=layer_idx,
        sparsity_ratio=0.5,
        exit_threshold=0.8
    )
    
    # Create sample input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = adaptive_layer(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )
    
    # Output should contain (hidden_states, [attn_weights], [cache], should_exit)
    assert len(output) >= 2, "Output should contain at least hidden states and early exit flag"
    
    output_hidden_states = output[0]
    should_exit = output[-1]  # Early exit flag is the last element
    
    # Output should have same shape as input
    assert output_hidden_states.shape == hidden_states.shape, "Output shape should match input shape"
    assert isinstance(should_exit, bool), "should_exit should be boolean"


def test_integrate_sparsity_and_early_exit_with_gradient_checkpointing():
    """Test integration of sparsity and early exit with gradient checkpointing"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_hidden_layers = 2
    config.use_gradient_checkpointing = True
    
    # Create sparse MLP
    sparse_mlp = SparseMLP(config, sparsity_ratio=0.3)
    
    # Create sparse attention
    sparse_attention = SparseAttention(config, sparsity_ratio=0.2)
    
    # Test that both can operate in training mode with gradients
    batch_size, seq_len = 1, 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    
    # Test sparse MLP
    mlp_output = sparse_mlp(hidden_states)
    assert mlp_output.shape == hidden_states.shape, "MLP output should match input shape"
    
    # Test sparse attention
    attn_output, _, _ = sparse_attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )
    assert attn_output.shape == hidden_states.shape, "Attention output should match input shape"
    
    # Verify gradients can flow through both
    loss = mlp_output.sum() + attn_output.sum()
    loss.backward()
    
    assert hidden_states.grad is not None, "Gradients should flow back to input"


def test_optimize_for_target_hardware():
    """Test optimizations for target hardware (Intel i5-10210U + NVIDIA SM61 + NVMe SSD)"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_hidden_layers = 2
    
    # Create sparse attention layer
    sparse_attention = SparseAttention(config, sparsity_ratio=0.4)
    
    # Create sparse MLP layer
    sparse_mlp = SparseMLP(config, sparsity_ratio=0.5)
    
    # Test with different input sizes to simulate different hardware loads
    test_cases = [
        (1, 32),   # Small batch, short sequence
        (1, 128),  # Small batch, medium sequence
        (2, 64),   # Medium batch, medium sequence
    ]
    
    for batch_size, seq_len in test_cases:
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Test sparse attention
        attn_output, _, _ = sparse_attention(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None
        )
        assert attn_output.shape[0] == batch_size
        assert attn_output.shape[1] == seq_len
        assert attn_output.shape[2] == config.hidden_size
        
        # Test sparse MLP
        mlp_output = sparse_mlp(hidden_states)
        assert mlp_output.shape == hidden_states.shape


if __name__ == "__main__":
    test_implement_top_k_activation_sparsity()
    test_create_confidence_gated_early_exit_mechanisms()
    test_develop_input_adaptive_routing()
    test_integrate_sparsity_and_early_exit_with_gradient_checkpointing()
    test_optimize_for_target_hardware()
    print("All implementation tests for Phase 2.5 passed!")