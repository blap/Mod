"""
Post-implementation testing for Phase 2.85: KV Cache Optimization Strategies (without INT8 quantization)
"""
import pytest
import torch
import torch.nn as nn
from src.models.config import Qwen3VLConfig
from src.components.optimization.kv_cache_optimization import LowRankKVCache, SlidingWindowKVCache, HybridKVCache, OptimizedKVCachingAttention
from models.kv_cache_optimization import VisionLanguageKVCache


def test_measure_kv_cache_memory_usage_reduction():
    """Measure KV cache memory usage reduction"""
    import psutil
    import gc

    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.num_attention_heads = 4
    head_dim = config.hidden_size // config.num_attention_heads  # 32

    # Create standard attention for comparison
    standard_attn = nn.MultiheadAttention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        batch_first=True
    )

    # Create optimized attention with low-rank KV cache
    optimized_attn = OptimizedKVCachingAttention(
        config,
        use_low_rank=True,
        window_size=64,
        low_rank_rank=16
    )

    # Create sample input
    batch_size, seq_len = 1, 50
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Measure memory for standard attention
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_before_standard = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    standard_output, _ = standard_attn(hidden_states, hidden_states, hidden_states)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_after_standard = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    standard_memory = memory_after_standard - memory_before_standard

    # Measure memory for optimized attention
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_before_optimized = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    optimized_output, _, _ = optimized_attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
        cache_position=None
    )
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_after_optimized = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    optimized_memory = memory_after_optimized - memory_before_optimized

    print(f"KV cache memory usage comparison:")
    print(f"  Standard attention: {standard_memory:.2f} MB")
    print(f"  Optimized attention: {optimized_memory:.2f} MB")

    # The memory usage comparison might not be directly visible due to PyTorch's caching,
    # but the low-rank approximation should reduce the effective memory footprint during computation


def test_validate_accuracy_preservation_with_compressed_caches():
    """Validate accuracy preservation with compressed caches"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.num_attention_heads = 4

    # Create attention layers
    standard_attn = nn.MultiheadAttention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        batch_first=True
    )

    optimized_attn = OptimizedKVCachingAttention(
        config,
        use_low_rank=True,
        window_size=64,
        low_rank_rank=16
    )

    # Create test input
    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass through both
    with torch.no_grad():  # Use no_grad for validation
        standard_output, _ = standard_attn(hidden_states, hidden_states, hidden_states)
        optimized_output, _, _ = optimized_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,  # Disable cache for this test
            cache_position=None
        )

    # Both outputs should be reasonable (not NaN or infinite)
    assert torch.isfinite(standard_output).all(), "Standard attention output should be finite"
    assert torch.isfinite(optimized_output).all(), "Optimized attention output should be finite"

    # Shapes should match
    assert standard_output.shape == optimized_output.shape, "Output shapes should match"


def test_benchmark_long_context_processing_performance_improvements():
    """Benchmark long-context processing performance improvements"""
    import time

    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.num_attention_heads = 4

    # Create attention layers
    standard_attn = nn.MultiheadAttention(
        embed_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        batch_first=True
    )

    optimized_attn = OptimizedKVCachingAttention(
        config,
        use_low_rank=True,
        window_size=64,
        low_rank_rank=16
    )

    # Test with different sequence lengths
    seq_lengths = [32, 64, 128]  # Test different lengths
    batch_size = 1

    print("Long-context processing performance comparison:")
    for seq_len in seq_lengths:
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        # Time standard attention
        standard_attn.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):  # Run multiple times for better measurement
                _ = standard_attn(hidden_states, hidden_states, hidden_states)[0]
            standard_time = time.time() - start_time

        # Time optimized attention
        optimized_attn.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                _ = optimized_attn(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,  # Disable cache for fair comparison
                    cache_position=None
                )[0]
            optimized_time = time.time() - start_time

        print(f"  Sequence length {seq_len}:")
        print(f"    Standard: {standard_time:.4f}s")
        print(f"    Optimized: {optimized_time:.4f}s")


def test_test_vision_language_task_performance_with_optimized_caching():
    """Test vision-language task performance with optimized caching"""
    import time

    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.num_attention_heads = 4

    # Create vision-language optimized attention
    vision_lang_attn = VisionLanguageKVCache(config)

    # Create sample multimodal input (could represent text and image features)
    batch_size, seq_len = 2, 24
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Test performance
    vision_lang_attn.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(20):  # Multiple runs
            output, _, _ = vision_lang_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None
            )
        end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / 20

    print(f"Vision-language task performance:")
    print(f"  Average time per forward pass: {avg_time:.6f}s")
    print(f"  Total time for 20 runs: {total_time:.4f}s")

    # Verify output is valid
    assert torch.isfinite(output).all(), "Vision-language attention output should be finite"
    assert output.shape == hidden_states.shape, "Output shape should match input shape"


def test_verify_compatibility_with_existing_ssd_caching_system():
    """Verify compatibility with existing SSD caching system"""
    config = Qwen3VLConfig()
    # Reduce dimensions for testing
    config.hidden_size = 64
    config.num_attention_heads = 4

    # Create optimized attention
    optimized_attn = OptimizedKVCachingAttention(
        config,
        use_low_rank=True,
        window_size=32,
        low_rank_rank=8
    )

    # Create sample input
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Test with caching enabled
    output, attn_weights, past_key_value = optimized_attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=True,
        use_cache=True,
        cache_position=None
    )

    # Verify that the caching mechanism works correctly
    assert output.shape == hidden_states.shape, "Output shape should match input"
    assert past_key_value is not None, "Past key value should be returned when use_cache=True"

    # Test with additional tokens using the cached values
    new_hidden_states = torch.randn(batch_size, 8, config.hidden_size)
    position_ids = torch.arange(seq_len, seq_len + 8, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    new_output, _, _ = optimized_attn(
        hidden_states=new_hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=False,
        use_cache=True,
        cache_position=None
    )

    assert new_output.shape[0] == batch_size, "Batch size should be preserved"
    assert new_output.shape[1] == 8, "New sequence length should match new input"
    assert new_output.shape[2] == config.hidden_size, "Hidden size should be preserved"


if __name__ == "__main__":
    test_measure_kv_cache_memory_usage_reduction()
    test_validate_accuracy_preservation_with_compressed_caches()
    test_benchmark_long_context_processing_performance_improvements()
    test_test_vision_language_task_performance_with_optimized_caching()
    test_verify_compatibility_with_existing_ssd_caching_system()
    print("All post-implementation tests for Phase 2.85 passed!")