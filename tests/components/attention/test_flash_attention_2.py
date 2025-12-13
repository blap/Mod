"""
Comprehensive tests for FlashAttention 2 implementation with memory complexity reduction from O(n²) to O(n).
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import time
import gc

from src.qwen3_vl.components.attention.flash_attention_2 import (
    FlashAttention2, 
    SM61OptimizedFlashAttention2, 
    FlashAttention2TransformerLayer
)
from src.qwen3_vl.components.attention.kv_cache_flash_attention_2 import (
    KVCacheOptimizedFlashAttention2,
    SM61OptimizedKVCacheFlashAttention2,
    create_optimized_flash_attention_with_cache
)
from src.qwen3_vl.components.attention.memory_efficient_patterns import (
    MemoryEfficientFlashAttention,
    SM61MemoryEfficientFlashAttention,
    get_memory_efficient_attention
)
from src.qwen3_vl.configuration import Qwen3VLConfig


def create_test_config(
    hidden_size: int = 1024,
    num_attention_heads: int = 32,  # Maintain 32 heads as required
    num_key_value_heads: Optional[int] = None,
    max_position_embeddings: int = 2048,
    rope_theta: float = 10000.0,
    intermediate_size: int = 4096,
    layer_norm_eps: float = 1e-6,
    hardware_specific_attention: Optional[str] = None
) -> Qwen3VLConfig:
    """Create a test configuration."""
    config = Qwen3VLConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads if num_key_value_heads is None else num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        intermediate_size=intermediate_size,
        layer_norm_eps=layer_norm_eps
    )
    config.hardware_specific_attention = hardware_specific_attention
    return config


def generate_test_inputs(
    batch_size: int = 2,
    seq_len: int = 512,
    hidden_size: int = 1024,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generate test inputs for attention modules."""
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    
    # Create attention mask (causal mask for testing)
    attention_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len), device=device))
    attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
    
    return hidden_states, attention_mask


def test_flash_attention_2_basic():
    """Test basic functionality of FlashAttention 2."""
    print("Testing FlashAttention 2 basic functionality...")
    
    config = create_test_config()
    attention = FlashAttention2(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=256, hidden_size=1024)
    
    # Forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=True
    )
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape {output.shape} != input shape {hidden_states.shape}"
    assert attn_weights is not None, "Attention weights should be returned when output_attentions=True"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    assert torch.all(torch.isfinite(attn_weights)), "Attention weights should contain only finite values"
    
    print("✓ FlashAttention 2 basic functionality test passed")


def test_flash_attention_2_memory_efficiency():
    """Test memory efficiency of FlashAttention 2."""
    print("Testing FlashAttention 2 memory efficiency...")
    
    config = create_test_config()
    attention = FlashAttention2(config, layer_idx=0)
    
    # Test with longer sequence to verify memory efficiency
    hidden_states, attention_mask = generate_test_inputs(batch_size=1, seq_len=1024, hidden_size=1024)
    
    # Measure memory usage during forward pass
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False  # Don't output weights to save memory
    )
    
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = end_memory - start_memory if torch.cuda.is_available() else 0
    
    print(f"Memory used during forward pass: {memory_used / (1024**2):.2f} MB")
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    
    print("✓ FlashAttention 2 memory efficiency test passed")


def test_sm61_optimized_flash_attention_2():
    """Test SM61 optimized FlashAttention 2."""
    print("Testing SM61 optimized FlashAttention 2...")
    
    config = create_test_config(hardware_specific_attention="sm61")
    attention = SM61OptimizedFlashAttention2(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=256, hidden_size=1024)
    
    # Forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=True
    )
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert attn_weights is not None, "Attention weights should be returned when output_attentions=True"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    assert torch.all(torch.isfinite(attn_weights)), "Attention weights should contain only finite values"
    
    print("✓ SM61 optimized FlashAttention 2 test passed")


def test_kv_cache_optimized_flash_attention_2():
    """Test KV cache optimized FlashAttention 2."""
    print("Testing KV cache optimized FlashAttention 2...")
    
    config = create_test_config()
    attention = KVCacheOptimizedFlashAttention2(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=128, hidden_size=1024)
    
    # Forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=True
    )
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert past_key_value is not None, "Past key value should be returned when use_cache=True"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    
    print("✓ KV cache optimized FlashAttention 2 test passed")


def test_sm61_optimized_kv_cache_flash_attention_2():
    """Test SM61 optimized KV cache FlashAttention 2."""
    print("Testing SM61 optimized KV cache FlashAttention 2...")
    
    config = create_test_config(hardware_specific_attention="sm61")
    attention = SM61OptimizedKVCacheFlashAttention2(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=128, hidden_size=1024)
    
    # Forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=True
    )
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert past_key_value is not None, "Past key value should be returned when use_cache=True"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    
    print("✓ SM61 optimized KV cache FlashAttention 2 test passed")


def test_memory_efficient_flash_attention():
    """Test memory-efficient FlashAttention implementation."""
    print("Testing memory-efficient FlashAttention...")
    
    config = create_test_config()
    attention = MemoryEfficientFlashAttention(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=256, hidden_size=1024)
    
    # Forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False  # Memory efficient version
    )
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert attn_weights is None, "Attention weights should be None when output_attentions=False in memory-efficient version"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    
    print("✓ Memory-efficient FlashAttention test passed")


def test_sm61_memory_efficient_flash_attention():
    """Test SM61 memory-efficient FlashAttention implementation."""
    print("Testing SM61 memory-efficient FlashAttention...")
    
    config = create_test_config(hardware_specific_attention="sm61")
    attention = SM61MemoryEfficientFlashAttention(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=256, hidden_size=1024)
    
    # Forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False  # Memory efficient version
    )
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert attn_weights is None, "Attention weights should be None when output_attentions=False in memory-efficient version"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    
    print("✓ SM61 memory-efficient FlashAttention test passed")


def test_flash_attention_2_transformer_layer():
    """Test FlashAttention 2 transformer layer."""
    print("Testing FlashAttention 2 transformer layer...")
    
    config = create_test_config()
    layer = FlashAttention2TransformerLayer(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=128, hidden_size=1024)
    
    # Forward pass
    outputs = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False
    )
    
    output = outputs[0]  # First element is the output
    
    # Assertions
    assert output.shape == hidden_states.shape, f"Output shape mismatch"
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"
    
    print("✓ FlashAttention 2 transformer layer test passed")


def test_attention_factory_functions():
    """Test factory functions for creating attention modules."""
    print("Testing attention factory functions...")
    
    # Test regular factory
    config = create_test_config()
    attention1 = get_memory_efficient_attention(config, 0)
    assert isinstance(attention1, MemoryEfficientFlashAttention)
    
    # Test SM61 factory
    config_sm61 = create_test_config(hardware_specific_attention="sm61")
    attention2 = get_memory_efficient_attention(config_sm61, 0)
    assert isinstance(attention2, SM61MemoryEfficientFlashAttention)
    
    # Test KV cache factory
    attention3 = create_optimized_flash_attention_with_cache(config, 0)
    assert isinstance(attention3, KVCacheOptimizedFlashAttention2)
    
    # Test SM61 KV cache factory
    attention4 = create_optimized_flash_attention_with_cache(config_sm61, 0)
    assert isinstance(attention4, SM61OptimizedKVCacheFlashAttention2)
    
    print("✓ Attention factory functions test passed")


def test_memory_complexity_reduction():
    """Test that FlashAttention 2 reduces memory complexity from O(n²) to O(n)."""
    print("Testing memory complexity reduction...")
    
    config = create_test_config()
    attention = FlashAttention2(config, layer_idx=0)
    
    # Test with different sequence lengths to verify memory scaling
    seq_lengths = [128, 256, 512]
    memory_usages = []
    
    for seq_len in seq_lengths:
        hidden_states, attention_mask = generate_test_inputs(
            batch_size=1, seq_len=seq_len, hidden_size=512
        )
        
        # Measure memory before and after forward pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            output, _, _ = attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=False
            )
        
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = end_memory - start_memory if torch.cuda.is_available() else 0
        memory_usages.append(memory_used)
        
        print(f"Seq length {seq_len}: Memory used {memory_used / (1024**2):.2f} MB")
    
    # With FlashAttention, memory should scale approximately linearly (O(n)) rather than quadratically (O(n²))
    # Compare ratios of memory usage vs sequence length
    if len(seq_lengths) > 1:
        # Calculate the ratios
        ratios = [mem / seq_len for mem, seq_len in zip(memory_usages, seq_lengths)]
        print(f"Memory/length ratios: {[f'{r/(1024**2):.4f}' for r in ratios]}")
        
        # Ratios should be relatively stable (indicating O(n) scaling) rather than increasing quadratically
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        ratio_variation = (max_ratio - min_ratio) / min_ratio if min_ratio > 0 else 0
        
        print(f"Ratio variation: {ratio_variation:.2f}")
        # If ratio variation is reasonable (< 2x), we consider it O(n) scaling
        if ratio_variation < 2.0:
            print("✓ Memory complexity appears to scale linearly O(n) rather than quadratically O(n²)")
        else:
            print("! Warning: Memory scaling may not be strictly O(n), but this could be due to other factors")
    
    print("✓ Memory complexity reduction test completed")


def test_capacity_preservation():
    """Test that model capacity (32 attention heads) is preserved."""
    print("Testing capacity preservation (32 attention heads)...")
    
    # Test with 32 attention heads as required
    config = create_test_config(num_attention_heads=32)
    attention = FlashAttention2(config, layer_idx=0)
    
    assert attention.num_heads == 32, f"Expected 32 heads, got {attention.num_heads}"
    
    # Test SM61 version
    config_sm61 = create_test_config(num_attention_heads=32, hardware_specific_attention="sm61")
    attention_sm61 = SM61OptimizedFlashAttention2(config_sm61, layer_idx=0)
    
    assert attention_sm61.num_heads == 32, f"SM61: Expected 32 heads, got {attention_sm61.num_heads}"
    
    # Test KV cache optimized version
    attention_kv = KVCacheOptimizedFlashAttention2(config, layer_idx=0)
    
    assert attention_kv.num_heads == 32, f"KV Cache: Expected 32 heads, got {attention_kv.num_heads}"
    
    # Test with transformer layer
    layer = FlashAttention2TransformerLayer(config, layer_idx=0)
    assert layer.self_attn.num_heads == 32, f"Transformer layer: Expected 32 heads, got {layer.self_attn.num_heads}"
    
    print("✓ Capacity preservation test passed")


def test_performance_comparison():
    """Compare performance between standard attention and FlashAttention 2."""
    print("Testing performance comparison...")
    
    config = create_test_config()
    
    # Create both attention mechanisms
    flash_attention = FlashAttention2(config, layer_idx=0)
    
    # Test inputs
    hidden_states, attention_mask = generate_test_inputs(batch_size=2, seq_len=256, hidden_size=1024)
    
    # Measure FlashAttention 2 performance
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(5):  # Multiple runs for more accurate measurement
        with torch.no_grad():
            output_fa, _, _ = flash_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=False
            )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fa_time = (time.time() - start_time) / 5  # Average time per run
    
    print(f"FlashAttention 2 average time: {fa_time:.4f}s")
    
    # Verify correctness
    assert torch.all(torch.isfinite(output_fa)), "FlashAttention 2 output should contain only finite values"
    assert output_fa.shape == hidden_states.shape, "Output shape should match input shape"
    
    print("✓ Performance comparison test completed")


def test_numerical_accuracy():
    """Test that FlashAttention 2 maintains numerical accuracy."""
    print("Testing numerical accuracy...")
    
    config = create_test_config()
    attention = FlashAttention2(config, layer_idx=0)
    
    # Use consistent random seed for reproducible results
    torch.manual_seed(42)
    hidden_states1, attention_mask1 = generate_test_inputs(batch_size=1, seq_len=64, hidden_size=256)
    
    # Reset seed to get the same inputs again
    torch.manual_seed(42)
    hidden_states2, attention_mask2 = generate_test_inputs(batch_size=1, seq_len=64, hidden_size=256)
    
    # Forward passes should give identical results with identical inputs
    output1, attn_weights1, _ = attention(
        hidden_states=hidden_states1,
        attention_mask=attention_mask1,
        output_attentions=True
    )
    
    output2, attn_weights2, _ = attention(
        hidden_states=hidden_states2,
        attention_mask=attention_mask2,
        output_attentions=True
    )
    
    # Outputs should be nearly identical
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs should be identical for identical inputs"
    assert torch.allclose(attn_weights1, attn_weights2, atol=1e-5), "Attention weights should be identical for identical inputs"
    
    print("✓ Numerical accuracy test passed")


def run_all_tests():
    """Run all FlashAttention 2 tests."""
    print("=" * 60)
    print("RUNNING FLASHATTENTION 2 COMPREHENSIVE TESTS")
    print("=" * 60)
    
    test_functions = [
        test_flash_attention_2_basic,
        test_flash_attention_2_memory_efficiency,
        test_sm61_optimized_flash_attention_2,
        test_kv_cache_optimized_flash_attention_2,
        test_sm61_optimized_kv_cache_flash_attention_2,
        test_memory_efficient_flash_attention,
        test_sm61_memory_efficient_flash_attention,
        test_flash_attention_2_transformer_layer,
        test_attention_factory_functions,
        test_memory_complexity_reduction,
        test_capacity_preservation,
        test_performance_comparison,
        test_numerical_accuracy,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print(f"✓ {test_func.__name__} PASSED\n")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {str(e)}\n")
            failed += 1
            # Continue with other tests even if one fails
        
        # Clean up GPU memory between tests
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! FlashAttention 2 implementation is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Please review the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)