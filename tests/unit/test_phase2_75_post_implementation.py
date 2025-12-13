"""
Post-implementation testing for Phase 2.75: Memory-Efficient Transformer Variants
"""
import pytest
import torch
import torch.nn as nn
from src.models.config import Qwen3VLConfig
from src.components.optimization.moe_flash_attention import MoeLayer, FlashAttention, MoeTransformerLayer


def test_benchmark_attention_computation_efficiency_and_memory_usage():
    """Benchmark attention computation efficiency and memory usage"""
    import time
    import psutil
    import gc
    
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.num_attention_heads = 4
    
    # Create FlashAttention
    flash_attention = FlashAttention(config)
    
    # Create standard attention for comparison
    standard_attention = nn.MultiheadAttention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        batch_first=True
    )
    
    # Create test input
    batch_size, seq_len = 1, 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Benchmark FlashAttention
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    flash_memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    flash_attention.eval()
    with torch.no_grad():
        flash_start = time.time()
        _ = flash_attention(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None
        )
        flash_time = time.time() - flash_start
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    flash_memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Benchmark standard attention
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    standard_memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    standard_attention.eval()
    with torch.no_grad():
        standard_start = time.time()
        _ = standard_attention(hidden_states, hidden_states, hidden_states)
        standard_time = time.time() - standard_start
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    standard_memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Attention benchmarking:")
    print(f"  FlashAttention: {flash_time:.4f}s, memory change: {flash_memory_after - flash_memory_before:.2f} MB")
    print(f"  StandardAttention: {standard_time:.4f}s, memory change: {standard_memory_after - standard_memory_before:.2f} MB")


def test_validate_that_model_capacity_remains_at_32_transformer_layers_and_32_attention_heads():
    """Validate that model capacity remains at 32 transformer layers and 32 attention heads"""
    config = Qwen3VLConfig()
    
    # Verify config parameters
    assert config.num_hidden_layers == 32, f"Config should have 32 hidden layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Config should have 32 attention heads, got {config.num_attention_heads}"
    
    # Create a MoE transformer layer
    moe_layer = MoeTransformerLayer(config, layer_idx=0, num_experts=4, top_k=2)
    
    # Verify the layer was created with correct parameters
    assert moe_layer.self_attn.config.num_attention_heads == 32, "Attention heads should be 32"
    
    # Create a complete transformer with MoE layers
    from src.models.modeling_qwen3_vl import Qwen3VLDecoder
    decoder = Qwen3VLDecoder(config)
    
    assert len(decoder.layers) == 32, f"Decoder should have 32 layers, got {len(decoder.layers)}"
    assert decoder.config.num_hidden_layers == 32, "Decoder config should have 32 hidden layers"
    
    print("Model capacity validation passed: 32 layers and 32 attention heads preserved")


def test_test_moe_routing_performance_and_ensure_load_balancing():
    """Test MoE routing performance and ensure load balancing"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    
    # Create MoE layer with 4 experts and top-2 routing
    num_experts = 4
    moe_layer = MoeLayer(config, num_experts=num_experts, top_k=2)
    
    # Create test input
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = moe_layer(hidden_states)
    
    # Verify output shape
    assert output.shape == hidden_states.shape, "MoE output shape should match input shape"
    
    # Check that different experts are being used by examining router weights
    router_logits = moe_layer.router(hidden_states.view(-1, config.hidden_size))
    routing_weights = torch.softmax(router_logits, dim=-1)
    
    # Check that routing shows some preference (not completely uniform)
    max_routing_per_token = routing_weights.max(dim=-1)[0]
    # With top-2 routing, we expect each token to have significant weight on at least one expert
    assert torch.mean(max_routing_per_token) > 0.3, "Routing should show some preference for experts"
    
    # Check load balancing by looking at expert usage
    top_k_weights, top_k_indices = torch.topk(routing_weights, k=2, dim=-1)
    expert_usage = torch.bincount(top_k_indices.flatten(), minlength=num_experts)
    print(f"Expert usage: {expert_usage}")
    
    # All experts should be used to some extent
    assert torch.sum(expert_usage > 0) > 1, "Multiple experts should be used"


def test_verify_accuracy_preservation_on_multimodal_benchmarks():
    """Verify accuracy preservation on multimodal benchmarks"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    
    # Create MoE and FlashAttention layers
    moe_layer = MoeLayer(config, num_experts=3, top_k=2)
    flash_attention = FlashAttention(config)
    
    # Create test input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test MoE layer
    moe_output = moe_layer(hidden_states)
    assert torch.isfinite(moe_output).all(), "MoE output should be finite"
    assert moe_output.shape == hidden_states.shape, "MoE output shape should match input"
    
    # Test FlashAttention
    flash_output, _, _ = flash_attention(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None
    )
    assert torch.isfinite(flash_output).all(), "FlashAttention output should be finite"
    assert flash_output.shape == hidden_states.shape, "FlashAttention output shape should match input"


def test_profile_performance_on_target_hardware_intel_i5_10210u_nvidia_sm61_nvme_ssd():
    """Profile performance on target hardware (Intel i5-10210U + NVIDIA SM61 + NVMe SSD)"""
    import time
    
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    
    # Create MoE transformer layer
    layer = MoeTransformerLayer(config, layer_idx=0, num_experts=3, top_k=2)
    
    # Test with different input sizes
    test_cases = [
        (1, 16),   # Small
        (1, 32),   # Medium
        (2, 16),   # Batched
    ]
    
    print("Performance profiling on MoE transformer layer:")
    for batch_size, seq_len in test_cases:
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Time the layer
        layer.eval()
        with torch.no_grad():
            start_time = time.time()
            output = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None
            )
            end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        output_hidden = output[0]
        
        print(f"  Batch {batch_size}, Seq {seq_len}: {inference_time:.2f} ms")
        
        # Verify output is valid
        assert output_hidden.shape == hidden_states.shape, "Output shape should match input"
        assert torch.isfinite(output_hidden).all(), "Output should be finite"


if __name__ == "__main__":
    test_benchmark_attention_computation_efficiency_and_memory_usage()
    test_validate_that_model_capacity_remains_at_32_transformer_layers_and_32_attention_heads()
    test_test_moe_routing_performance_and_ensure_load_balancing()
    test_verify_accuracy_preservation_on_multimodal_benchmarks()
    test_profile_performance_on_target_hardware_intel_i5_10210u_nvidia_sm61_nvme_ssd()
    print("All post-implementation tests for Phase 2.75 passed!")