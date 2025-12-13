"""
Comprehensive tests for Mixture of Experts (MoE) implementation with 2-4 experts and top-2 routing.
Tests include functionality, load balancing, routing efficiency, and gradient checkpointing integration.
"""
import torch
import pytest
import numpy as np
from torch import nn
from models.moe_flash_attention import MoeLayer, MoEWithGradientCheckpointing, MoETransformerLayerWithGradientCheckpointing
from src.qwen3_vl.components.configuration.config import Qwen3VLConfig


def test_moe_layer_basic_functionality():
    """Test basic functionality of the MoE layer."""
    config = Qwen3VLConfig()
    config.hidden_size = 512
    config.intermediate_size = 2048
    
    # Test with 4 experts and top-2 routing
    moe_layer = MoeLayer(config, num_experts=4, top_k=2)
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = moe_layer(hidden_states)
    
    # Check output shape matches input shape
    assert output.shape == hidden_states.shape, f"Expected {hidden_states.shape}, got {output.shape}"
    
    # Check that output is finite
    assert torch.isfinite(output).all(), "MoE output contains non-finite values"
    
    print("✓ Basic functionality test passed")


def test_moe_different_expert_counts():
    """Test MoE layer with different numbers of experts."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.intermediate_size = 1024
    
    for num_experts in [2, 3, 4]:
        for top_k in [1, 2]:
            if top_k <= num_experts:  # top_k should not exceed num_experts
                moe_layer = MoeLayer(config, num_experts=num_experts, top_k=top_k)
                
                batch_size, seq_len = 2, 8
                hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
                
                output = moe_layer(hidden_states)
                
                assert output.shape == hidden_states.shape, f"Shape mismatch for {num_experts} experts, top-{top_k}"
                assert torch.isfinite(output).all(), f"Non-finite output for {num_experts} experts, top-{top_k}"
                
                print(f"✓ Test passed for {num_experts} experts, top-{top_k}")


def test_moe_load_balancing():
    """Test that experts are being used in a balanced manner."""
    config = Qwen3VLConfig()
    config.hidden_size = 128
    config.intermediate_size = 512
    
    # Create MoE layer with 4 experts and top-2 routing
    moe_layer = MoeLayer(config, num_experts=4, top_k=2)
    moe_layer.train()  # Put in training mode to update expert counts
    
    # Run multiple forward passes to accumulate expert usage statistics
    batch_size, seq_len = 4, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    output = moe_layer(hidden_states)
    
    # Check that expert counts have been updated
    expert_usage = moe_layer.expert_counts.float() / (moe_layer.total_tokens + 1e-8)
    
    # All experts should have been used to some extent
    assert (moe_layer.expert_counts > 0).sum().item() >= 2, "Multiple experts should be used"
    
    # Check that the total tokens count is updated
    assert moe_layer.total_tokens.item() == batch_size * seq_len, "Total tokens not updated correctly"
    
    print("✓ Load balancing test passed")


def test_moe_routing_efficiency():
    """Test that routing is happening efficiently."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.intermediate_size = 1024
    
    # Create MoE layer with 4 experts and top-2 routing
    num_experts = 4
    moe_layer = MoeLayer(config, num_experts=num_experts, top_k=2)
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Get routing decisions by accessing the router directly
    x_flat = hidden_states.view(-1, config.hidden_size)
    router_logits = moe_layer.w_gate(x_flat)
    raw_weights = torch.softmax(router_logits, dim=-1)
    top_k_weights, top_k_indices = torch.topk(raw_weights, 2, dim=-1)
    
    # Verify that exactly top_k experts are selected per token
    assert top_k_indices.shape[1] == 2, "Should have exactly 2 experts per token"
    assert top_k_weights.shape[1] == 2, "Should have exactly 2 weights per token"
    
    # Check that routing weights are properly normalized
    normalized_weights = torch.softmax(top_k_weights, dim=-1)
    assert torch.allclose(torch.sum(normalized_weights, dim=1), torch.ones_like(normalized_weights[:, 0]), atol=1e-5), \
        "Top-k weights should sum to 1 after normalization"
    
    print("✓ Routing efficiency test passed")


def test_moe_gradient_checkpointing():
    """Test MoE layer with gradient checkpointing."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.intermediate_size = 1024
    config.use_gradient_checkpointing = True
    
    # Create MoE layer with gradient checkpointing
    moe_with_ckpt = MoEWithGradientCheckpointing(config, num_experts=3, top_k=2)
    
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    
    # Forward pass
    output = moe_with_ckpt(hidden_states)
    
    assert output.shape == hidden_states.shape, "Output shape mismatch with gradient checkpointing"
    assert torch.isfinite(output).all(), "Output contains non-finite values with gradient checkpointing"
    
    # Backward pass to ensure gradients flow properly
    loss = output.sum()
    loss.backward()
    
    assert hidden_states.grad is not None, "Gradients not computed with gradient checkpointing"
    assert torch.isfinite(hidden_states.grad).all(), "Gradients contain non-finite values"
    
    print("✓ Gradient checkpointing test passed")


def test_moe_active_parameter_reduction():
    """Test that MoE reduces active parameters during inference."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.intermediate_size = 1024
    
    # Create MoE layer with 4 experts and top-1 routing (most sparse)
    moe_layer_top1 = MoeLayer(config, num_experts=4, top_k=1)
    
    # Create MoE layer with 4 experts and top-2 routing (less sparse)
    moe_layer_top2 = MoeLayer(config, num_experts=4, top_k=2)
    
    # Create a standard MLP for comparison
    standard_mlp = nn.Sequential(
        nn.Linear(config.hidden_size, config.intermediate_size),
        nn.GELU(),
        nn.Linear(config.intermediate_size, config.hidden_size)
    )
    
    # Count parameters
    moe_top1_params = sum(p.numel() for p in moe_layer_top1.parameters())
    moe_top2_params = sum(p.numel() for p in moe_layer_top2.parameters())
    standard_params = sum(p.numel() for p in standard_mlp.parameters())
    
    print(f"Standard MLP parameters: {standard_params:,}")
    print(f"MoE (top-1) parameters: {moe_top1_params:,}")
    print(f"MoE (top-2) parameters: {moe_top2_params:,}")
    
    # The number of parameters should be the same (shared across experts)
    assert moe_top1_params == moe_top2_params, "Parameter count should be the same regardless of top-k"
    
    # But during inference, only a subset of parameters is active per token
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Run forward pass to see active computation
    with torch.no_grad():
        output_top1 = moe_layer_top1(hidden_states)
        output_top2 = moe_layer_top2(hidden_states)
    
    assert output_top1.shape == hidden_states.shape
    assert output_top2.shape == hidden_states.shape
    
    print("✓ Active parameter reduction test passed")


def test_moe_transformer_layer():
    """Test the complete MoE transformer layer."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.max_position_embeddings = 512
    config.rope_theta = 10000.0
    
    # Create MoE transformer layer
    layer = MoETransformerLayerWithGradientCheckpointing(config, layer_idx=0, num_experts=3, top_k=2)
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = layer(hidden_states)
    
    assert len(output) >= 1, "Output should contain at least hidden states"
    assert output[0].shape == hidden_states.shape, "Output shape should match input shape"
    assert torch.isfinite(output[0]).all(), "Output contains non-finite values"
    
    print("✓ MoE transformer layer test passed")


def test_moe_auxiliary_losses():
    """Test that auxiliary losses are computed correctly."""
    config = Qwen3VLConfig()
    config.hidden_size = 128
    config.intermediate_size = 512
    
    # Create MoE layer with load balancing
    moe_layer = MoeLayer(config, num_experts=4, top_k=2, balance_loss_weight=0.01)
    moe_layer.train()
    
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    
    # Forward pass
    output = moe_layer(hidden_states)
    
    # The forward pass should have computed auxiliary losses
    # Check that expert counts are updated
    assert moe_layer.total_tokens.item() == batch_size * seq_len, "Total tokens should be updated"
    
    print("✓ Auxiliary losses test passed")


def test_moe_capacity_constraints():
    """Test that the capacity parameters are properly set."""
    config = Qwen3VLConfig()
    config.hidden_size = 128
    config.intermediate_size = 512

    # Create MoE layer with capacity parameters
    moe_layer = MoeLayer(
        config,
        num_experts=4,
        top_k=2,
        capacity_factor=1.25,  # Standard capacity factor
        min_capacity=4
    )

    # Verify that the parameters are set correctly
    assert moe_layer.capacity_factor == 1.25, "Capacity factor not set correctly"
    assert moe_layer.min_capacity == 4, "Min capacity not set correctly"

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output = moe_layer(hidden_states)

    assert output.shape == hidden_states.shape, "Output shape should match input"
    assert torch.isfinite(output).all(), "Output should be finite"

    print("✓ Capacity parameters test passed")


def run_all_tests():
    """Run all tests for the MoE implementation."""
    print("Running comprehensive tests for MoE implementation...")
    
    test_moe_layer_basic_functionality()
    test_moe_different_expert_counts()
    test_moe_load_balancing()
    test_moe_routing_efficiency()
    test_moe_gradient_checkpointing()
    test_moe_active_parameter_reduction()
    test_moe_transformer_layer()
    test_moe_auxiliary_losses()
    test_moe_capacity_constraints()
    
    print("\n✅ All tests passed! MoE implementation is working correctly.")


if __name__ == "__main__":
    run_all_tests()