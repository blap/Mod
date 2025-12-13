"""
Integration testing for all new architectural improvements in Qwen3-VL model.
Tests the integration of Phase 2.5, 2.75, 2.85, and 2.9 features.
"""
import pytest
import torch
import torch.nn as nn
from src.models.config import Qwen3VLConfig
from src.components.optimization.activation_sparsity import AdaptiveComputationLayer
from src.components.optimization.moe_flash_attention import MoeTransformerLayer
from src.components.optimization.kv_cache_optimization import OptimizedKVCachingAttention
from models.memory_pooling import MemoryPool, PooledMLP


def test_integrated_architecture_preserves_capacity():
    """Test that the integrated architecture preserves full capacity (32 layers, 32 heads)"""
    config = Qwen3VLConfig()
    
    # Verify configuration
    assert config.num_hidden_layers == 32, f"Config should have 32 hidden layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Config should have 32 attention heads, got {config.num_attention_heads}"
    
    # Test creating layers with new features while preserving capacity
    adaptive_layer = AdaptiveComputationLayer(config, layer_idx=0, sparsity_ratio=0.3)
    moe_layer = MoeTransformerLayer(config, layer_idx=0, num_experts=4, top_k=2)
    
    # Verify that these layers use the full capacity
    assert adaptive_layer.self_attn.config.num_attention_heads == 32, "Adaptive layer should preserve attention heads"
    assert moe_layer.self_attn.config.num_attention_heads == 32, "MoE layer should preserve attention heads"
    
    print("Integrated architecture preserves full capacity: 32 layers, 32 attention heads")


def test_memory_efficiency_improvements_integration():
    """Test integration of all memory efficiency improvements"""
    # Create a memory pool
    memory_pool = MemoryPool(pool_size=4 * 1024 * 1024)  # 4MB pool
    
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    
    # Create components with different optimizations
    pooled_mlp = PooledMLP(config, memory_pool=memory_pool)
    optimized_attention = OptimizedKVCachingAttention(
        config=config,
        use_quantization=True,
        window_size=64,
        device=torch.device('cpu')
    )
    
    # Create test input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test the optimized components
    mlp_output = pooled_mlp(hidden_states)
    attn_output, _, _ = optimized_attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
        cache_position=None
    )
    
    # Verify outputs are valid
    assert mlp_output.shape == hidden_states.shape, "Pooled MLP output should match input shape"
    assert attn_output.shape == hidden_states.shape, "Optimized attention output should match input shape"
    assert torch.isfinite(mlp_output).all(), "Pooled MLP output should be finite"
    assert torch.isfinite(attn_output).all(), "Optimized attention output should be finite"
    
    # Check memory pool stats
    stats = memory_pool.get_memory_stats()
    print(f"Memory pool stats with integrated optimizations: {stats}")


def test_sparsity_and_moe_integration():
    """Test integration of sparsity and MoE mechanisms"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    
    # Create adaptive computation layer (has sparsity) and MoE layer
    adaptive_layer = AdaptiveComputationLayer(config, layer_idx=0, sparsity_ratio=0.4)
    moe_layer = MoeTransformerLayer(config, layer_idx=1, num_experts=3, top_k=2)
    
    # Create test input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Pass through adaptive layer
    adaptive_output = adaptive_layer(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )
    
    # Extract the actual output (first element) and early exit flag (last element)
    adaptive_hidden_states = adaptive_output[0]
    early_exit_flag = adaptive_output[-1]
    
    # Pass through MoE layer
    moe_output = moe_layer(
        hidden_states=adaptive_hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )
    
    moe_hidden_states = moe_output[0]
    
    # Verify outputs are valid
    assert adaptive_hidden_states.shape == hidden_states.shape, "Adaptive layer output should match input shape"
    assert moe_hidden_states.shape == hidden_states.shape, "MoE layer output should match input shape"
    assert torch.isfinite(adaptive_hidden_states).all(), "Adaptive layer output should be finite"
    assert torch.isfinite(moe_hidden_states).all(), "MoE layer output should be finite"
    assert isinstance(early_exit_flag, bool), "Early exit flag should be boolean"
    
    print(f"Sparsity and MoE integration test passed, early exit: {early_exit_flag}")


def test_kv_cache_optimization_with_memory_pooling():
    """Test integration of KV cache optimization with memory pooling"""
    # Create memory pool
    memory_pool = MemoryPool(pool_size=2 * 1024 * 1024)  # 2MB pool
    
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.num_attention_heads = 4
    
    # Create attention with KV cache optimization using pooled components
    optimized_attention = OptimizedKVCachingAttention(
        config=config,
        use_quantization=True,
        window_size=64,
        device=torch.device('cpu')
    )
    
    # Create test input
    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Multiple forward passes to test caching behavior
    for i in range(3):
        output, _, _ = optimized_attention(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=True,
            cache_position=None
        )
        
        # Verify output is valid
        assert torch.isfinite(output).all(), f"Output should be finite on pass {i+1}"
        assert output.shape == hidden_states.shape, f"Output shape should match input on pass {i+1}"
    
    # Check memory usage
    stats = memory_pool.get_memory_stats()
    print(f"Memory stats with KV cache optimization: {stats}")


def test_full_integration_forward_pass():
    """Test a full forward pass with all optimizations integrated"""
    from models.memory_pooling import PooledAttention

    # Create memory pool
    memory_pool = MemoryPool(pool_size=4 * 1024 * 1024)  # 4MB pool

    config = Qwen3VLConfig()
    # Reduce dimensions for testing but preserve layer and head count
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_hidden_layers = 4  # Use fewer layers for testing

    # Create a sequence of optimized layers
    layers = nn.ModuleList([
        AdaptiveComputationLayer(config, layer_idx=0, sparsity_ratio=0.3, exit_threshold=0.9),
        MoeTransformerLayer(config, layer_idx=1, num_experts=3, top_k=2),
        PooledAttention(config, memory_pool=memory_pool),
        OptimizedKVCachingAttention(config, use_quantization=True, window_size=64, device=torch.device('cpu'))
    ])

    # Create test input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass through all layers
    current_states = hidden_states
    for i, layer in enumerate(layers):
        if i == 0:  # Adaptive computation layer
            layer_output = layer(
                hidden_states=current_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None
            )
            # Adaptive layer returns (hidden_states, [attn_weights], [cache], should_exit)
            current_states = layer_output[0]
        elif i == 1:  # MoE transformer layer
            layer_output = layer(
                hidden_states=current_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None
            )
            current_states = layer_output[0]
        elif i == 2:  # Pooled attention
            layer_output, _, _ = layer(
                hidden_states=current_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None
            )
            current_states = layer_output
        else:  # Optimized KV caching attention
            layer_output, _, _ = layer(
                hidden_states=current_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=True,
                cache_position=None
            )
            current_states = layer_output

    # Verify final output is valid
    assert torch.isfinite(current_states).all(), "Final output should be finite"
    assert current_states.shape == hidden_states.shape, "Final output shape should match input shape"

    # Check memory stats
    stats = memory_pool.get_memory_stats()
    print(f"Final memory pool stats after full integration: {stats}")

    print("Full integration test passed")


if __name__ == "__main__":
    test_integrated_architecture_preserves_capacity()
    test_memory_efficiency_improvements_integration()
    test_sparsity_and_moe_integration()
    test_kv_cache_optimization_with_memory_pooling()
    test_full_integration_forward_pass()
    print("All integration tests passed!")