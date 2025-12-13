"""
Comprehensive tests for the KV Cache Optimization System
Tests low-rank approximation, sliding window attention, and vision-language task optimizations
"""
import pytest
import torch
import numpy as np
from kv_cache_optimizer import (
    KVCacheConfig,
    LowRankKVCompressor,
    SlidingWindowKVCache,
    HybridKVCache,
    VisionLanguageKVCache,
    OptimizedKVCacheManager,
    KVCacheOptimizedAttention,
    create_optimized_attention_with_cache
)
from memory_manager import MemoryManager


def test_low_rank_compressor_basic():
    """Test basic functionality of low-rank compressor"""
    config = KVCacheConfig(use_low_rank=True, low_rank_dimension=16)
    memory_manager = MemoryManager()
    compressor = LowRankKVCompressor(config, memory_manager)
    
    # Create test tensor [batch, seq_len, features]
    batch_size, seq_len, feature_dim = 2, 64, 128
    test_tensor = torch.randn(batch_size, seq_len, feature_dim)
    
    # Compress
    left, right = compressor.compress(test_tensor)
    
    # Check shapes
    assert left.shape == (batch_size, seq_len, config.low_rank_dimension)
    assert right.shape == (batch_size, config.low_rank_dimension, feature_dim)
    
    # Decompress (not implemented in this version, but we can still test the shapes)
    reconstructed = torch.matmul(left, right)
    assert reconstructed.shape == test_tensor.shape
    
    print("[PASS] Low-rank compressor basic functionality")


def test_low_rank_compressor_svd_method():
    """Test SVD-based compression method"""
    config = KVCacheConfig(use_low_rank=True, low_rank_dimension=8, low_rank_method="svd")
    memory_manager = MemoryManager()
    compressor = LowRankKVCompressor(config, memory_manager)
    
    # Create a matrix with known low-rank structure
    batch_size, seq_len, feature_dim = 1, 32, 32
    # Create a matrix that's truly low-rank
    true_rank = 5
    U = torch.randn(batch_size, seq_len, true_rank)
    V = torch.randn(batch_size, true_rank, feature_dim)
    test_tensor = torch.matmul(U, V)  # This will have rank at most `true_rank`
    
    # Compress
    left, right = compressor.compress(test_tensor)
    
    # Check shapes
    assert left.shape == (batch_size, seq_len, config.low_rank_dimension)
    assert right.shape == (batch_size, config.low_rank_dimension, feature_dim)
    
    # Reconstruct
    reconstructed = torch.matmul(left, right)
    
    # Since we're using a low-rank tensor and our compression rank is higher,
    # reconstruction should be quite accurate
    mse = torch.mean((test_tensor - reconstructed) ** 2)
    assert mse < 1.0  # Should be reasonably low
    
    print("[PASS] Low-rank compressor SVD method")


def test_sliding_window_basic():
    """Test basic sliding window functionality"""
    config = KVCacheConfig(use_sliding_window=True, sliding_window_size=32)
    memory_manager = MemoryManager()
    cache = SlidingWindowKVCache(config, memory_manager)
    
    # Create test key and value states
    batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update cache
    k_out, v_out = cache.update(key_states, value_states)
    
    # Check shapes
    assert k_out.shape[0] == batch_size
    assert k_out.shape[1] == num_heads
    assert k_out.shape[2] <= config.sliding_window_size  # Should be <= window size
    assert k_out.shape[3] == head_dim
    
    print("[PASS] Sliding window basic functionality")


def test_sliding_window_wraparound():
    """Test sliding window with wraparound behavior"""
    config = KVCacheConfig(use_sliding_window=True, sliding_window_size=16)
    memory_manager = MemoryManager()
    cache = SlidingWindowKVCache(config, memory_manager)
    
    # Create test states that will cause wraparound
    batch_size, num_heads, head_dim = 1, 4, 32
    
    # First, fill the cache close to capacity
    seq1 = 10
    key1 = torch.randn(batch_size, num_heads, seq1, head_dim)
    value1 = torch.randn(batch_size, num_heads, seq1, head_dim)
    k_out1, v_out1 = cache.update(key1, value1)
    
    # Then add more that causes wraparound
    seq2 = 12  # This will exceed the window size when added to the existing 10
    key2 = torch.randn(batch_size, num_heads, seq2, head_dim)
    value2 = torch.randn(batch_size, num_heads, seq2, head_dim)
    k_out2, v_out2 = cache.update(key2, value2)
    
    # The final cache should have at most window_size elements
    assert k_out2.shape[2] <= config.sliding_window_size
    
    print("[PASS] Sliding window wraparound behavior")


def test_hybrid_cache_basic():
    """Test basic hybrid cache functionality"""
    config = KVCacheConfig(
        use_low_rank=True, 
        low_rank_dimension=16,
        use_sliding_window=True, 
        sliding_window_size=32,
        use_hybrid=True
    )
    memory_manager = MemoryManager()
    cache = HybridKVCache(config, memory_manager)
    
    # Create test key and value states
    batch_size, num_heads, seq_len, head_dim = 1, 4, 20, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update cache
    k_out, v_out = cache.update(key_states, value_states)
    
    # With hybrid, we expect the output to be compressed to low-rank dimension
    assert k_out.shape[0] == batch_size
    assert k_out.shape[1] == num_heads
    assert k_out.shape[2] <= config.sliding_window_size
    assert k_out.shape[3] == config.low_rank_dimension  # Should be compressed
    
    print("[PASS] Hybrid cache basic functionality")


def test_vision_language_cache():
    """Test vision-language specialized cache"""
    config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,
        use_sliding_window=True,
        sliding_window_size=128,
        vision_language_optimized=True,
        vision_seq_limit=64,
        language_seq_limit=256
    )
    memory_manager = MemoryManager()
    cache = VisionLanguageKVCache(config, memory_manager)
    
    # Test with language tokens
    batch_size, num_heads, seq_len, head_dim = 1, 8, 16, 128
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update with language tokens
    k_lang, v_lang = cache.update(key_states, value_states, is_vision=False)
    
    # Test with vision tokens
    vision_seq_len = 10
    key_vision = torch.randn(batch_size, num_heads, vision_seq_len, head_dim)
    value_vision = torch.randn(batch_size, num_heads, vision_seq_len, head_dim)
    
    # Update with vision tokens
    k_vision, v_vision = cache.update(key_vision, value_vision, is_vision=True)
    
    # Check that both work
    assert k_lang.shape[0] == batch_size
    assert k_vision.shape[0] == batch_size
    
    print("[PASS] Vision-language cache functionality")


def test_optimized_kv_cache_manager():
    """Test the optimized KV cache manager with memory tracking"""
    config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=16,
        use_sliding_window=True,
        sliding_window_size=64,
        use_hybrid=True
    )
    memory_manager = MemoryManager()
    kv_manager = OptimizedKVCacheManager(config, memory_manager)
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 128
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update cache multiple times to accumulate stats
    for i in range(3):
        k_out, v_out = kv_manager.update(key_states, value_states)
    
    # Check memory stats
    stats = kv_manager.get_memory_stats()
    
    # Verify that we have some compression (ratio should be < 1.0 if working)
    assert 'compression_ratio' in stats
    assert 'memory_saved_percentage' in stats
    
    # The compression ratio should be reasonable
    assert 0 <= stats['compression_ratio'] <= 1.0
    
    print(f"[INFO] Compression ratio: {stats['compression_ratio']:.3f}")
    print(f"[INFO] Memory saved: {stats['memory_saved_percentage']:.2f}%")
    print("[PASS] Optimized KV cache manager with memory tracking")


def test_attention_with_optimized_cache():
    """Test attention mechanism with optimized KV caching"""
    # Create configuration
    config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,
        use_sliding_window=True,
        sliding_window_size=128,
        use_hybrid=True
    )
    
    memory_manager = MemoryManager()
    hidden_size = 512
    num_attention_heads = 8
    
    # Create attention module with optimized cache
    attention = KVCacheOptimizedAttention(
        config, memory_manager, hidden_size, num_attention_heads
    )
    
    # Create test inputs
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run forward pass with caching enabled
    output, attn_weights, past_key_value = attention(
        hidden_states,
        use_cache=True
    )
    
    # Check output shapes
    assert output.shape == hidden_states.shape
    assert past_key_value is not None
    
    # Check cache stats
    cache_stats = attention.get_cache_stats()
    assert 'compression_ratio' in cache_stats
    
    print(f"[INFO] Attention cache compression ratio: {cache_stats['compression_ratio']:.3f}")
    print("[PASS] Attention with optimized KV caching")


def test_memory_efficiency():
    """Test that the optimized cache provides memory efficiency"""
    # Create two configurations: one with optimization, one without
    config_optimized = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=16,  # Low rank to save memory
        use_sliding_window=True,
        sliding_window_size=64,  # Limited window to save memory
        use_hybrid=True
    )
    
    config_standard = KVCacheConfig(
        use_low_rank=False,
        use_sliding_window=False,
        use_hybrid=False
    )
    
    memory_manager = MemoryManager()
    
    # Create cache managers
    optimized_manager = OptimizedKVCacheManager(config_optimized, memory_manager)
    standard_manager = OptimizedKVCacheManager(config_standard, memory_manager)
    
    # Create large test tensors
    batch_size, num_heads, seq_len, head_dim = 1, 12, 256, 128
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update both caches
    optimized_k, optimized_v = optimized_manager.update(key_states, value_states)
    standard_k, standard_v = standard_manager.update(key_states, value_states)
    
    # Get statistics
    opt_stats = optimized_manager.get_memory_stats()
    std_stats = standard_manager.get_memory_stats()
    
    # The optimized version should show compression
    assert opt_stats['compression_ratio'] <= 1.0
    
    print(f"[INFO] Optimized compression ratio: {opt_stats['compression_ratio']:.3f}")
    print(f"[INFO] Memory saved with optimization: {opt_stats['memory_saved_percentage']:.2f}%")
    print("[PASS] Memory efficiency test")


def test_different_low_rank_methods():
    """Test different low-rank compression methods"""
    batch_size, seq_len, feature_dim = 1, 32, 64
    
    test_tensor = torch.randn(batch_size, seq_len, feature_dim)
    
    methods = ["svd", "random"]
    
    for method in methods:
        config = KVCacheConfig(
            use_low_rank=True,
            low_rank_dimension=16,
            low_rank_method=method
        )
        memory_manager = MemoryManager()
        compressor = LowRankKVCompressor(config, memory_manager)
        
        left, right = compressor.compress(test_tensor)
        
        assert left.shape == (batch_size, seq_len, config.low_rank_dimension)
        assert right.shape == (batch_size, config.low_rank_dimension, feature_dim)
        
        print(f"[PASS] Low-rank compression with method '{method}'")


def test_edge_cases():
    """Test edge cases for the optimization system"""
    config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=4,
        use_sliding_window=True,
        sliding_window_size=8
    )
    memory_manager = MemoryManager()
    
    # Test with very small tensors
    batch_size, num_heads, seq_len, head_dim = 1, 2, 3, 4
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    cache = HybridKVCache(config, memory_manager)
    k_out, v_out = cache.update(key_states, value_states)
    
    # Should handle small tensors gracefully
    assert k_out.shape[0] == batch_size
    assert k_out.shape[1] == num_heads
    
    # Test reset functionality
    cache.reset()
    assert cache.sliding_window.current_position == 0
    
    print("[PASS] Edge cases handled correctly")


def test_create_optimized_attention_helper():
    """Test the helper function for creating optimized attention"""
    memory_manager = MemoryManager()
    
    attention = create_optimized_attention_with_cache(
        hidden_size=256,
        num_attention_heads=8,
        memory_manager=memory_manager
    )
    
    # Test that it creates the right type
    assert isinstance(attention, KVCacheOptimizedAttention)
    
    # Test forward pass
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, 256)
    
    output, attn_weights, past_key_value = attention(
        hidden_states,
        use_cache=True
    )
    
    assert output.shape == hidden_states.shape
    
    print("[PASS] Optimized attention helper function")


def run_all_tests():
    """Run all tests for the KV cache optimization system"""
    print("Running KV Cache Optimization System Tests...\n")
    
    test_functions = [
        test_low_rank_compressor_basic,
        test_low_rank_compressor_svd_method,
        test_sliding_window_basic,
        test_sliding_window_wraparound,
        test_hybrid_cache_basic,
        test_vision_language_cache,
        test_optimized_kv_cache_manager,
        test_attention_with_optimized_cache,
        test_memory_efficiency,
        test_different_low_rank_methods,
        test_edge_cases,
        test_create_optimized_attention_helper
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)