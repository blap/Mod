"""
KV Cache Optimization System - Documentation and Usage Examples

This module implements KV cache optimization techniques for the Qwen3-VL model,
achieving 30-60% reduction in KV cache memory usage while maintaining accuracy
for long-context tasks and vision-language processing.

Key Features:
1. Low-rank approximation techniques for KV cache compression
2. Sliding window attention to limit cache size
3. Efficient KV cache allocation for vision-language tasks
4. Integration with existing caching mechanisms
"""

import torch
# Import the main components
from src.qwen3_vl.memory_management.kv_cache_optimizer import (
    KVCacheConfig,
    OptimizedKVCacheManager,
    KVCacheOptimizedAttention,
    LowRankKVCompressor,
    SlidingWindowKVCache,
    HybridKVCache,
    VisionLanguageKVCache,
    create_optimized_attention_with_cache
)


# Define the SimpleMemoryManager at module level
class SimpleMemoryManager:
    def __init__(self):
        pass

    def allocate_tensor(self, shape, dtype=torch.float32, device=None):
        device = device or torch.device("cpu")
        return torch.empty(shape, dtype=dtype, device=device)

    def free_tensor(self, tensor):
        return True


# Example 1: Basic usage with default configuration
def example_basic_usage():
    """Example of using the optimized KV cache with default settings."""

    # Create configuration for KV cache optimization
    config = KVCacheConfig(
        use_low_rank=True,           # Enable low-rank approximation
        low_rank_dimension=64,       # Rank for low-rank approximation
        use_sliding_window=True,     # Enable sliding window
        sliding_window_size=1024,    # Window size for sliding window
        use_hybrid=True,             # Combine low-rank and sliding window
        vision_language_optimized=True  # Optimize for vision-language tasks
    )

    # Create the optimized KV cache manager
    memory_manager = SimpleMemoryManager()
    kv_cache_manager = OptimizedKVCacheManager(config, memory_manager)

    # Use the manager to update KV caches

    # Example key and value states from attention computation
    batch_size, num_heads, seq_len, head_dim = 1, 12, 512, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Update the cache with optimization
    optimized_k, optimized_v = kv_cache_manager.update(key_states, value_states)

    # Get memory usage statistics
    stats = kv_cache_manager.get_memory_stats()
    print(f"Memory saved: {stats['memory_saved_percentage']:.2f}%")
    print(f"Compression ratio: {stats['compression_ratio']:.3f}")

    return kv_cache_manager


# Example 2: Creating optimized attention layer
def example_attention_layer():
    """Example of creating an attention layer with optimized KV caching."""
    
    # Create optimized attention with KV cache optimizations
    attention = create_optimized_attention_with_cache(
        hidden_size=4096,
        num_attention_heads=32,
        # Memory manager is optional, will use simple allocation if not provided
    )
    
    import torch
    
    # Example usage in a forward pass
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 4096)
    
    # Forward pass with KV cache enabled
    output, attn_weights, past_key_value = attention(
        hidden_states,
        use_cache=True
    )
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get cache statistics
    cache_stats = attention.get_cache_stats()
    print(f"Cache compression ratio: {cache_stats['compression_ratio']:.3f}")
    
    return attention


# Example 3: Vision-language specific optimizations
def example_vision_language():
    """Example of using vision-language specific KV cache optimizations."""

    # Create configuration optimized for vision-language tasks
    config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,
        use_sliding_window=True,
        sliding_window_size=256,
        vision_language_optimized=True,
        vision_seq_limit=64,      # Limit for vision tokens
        language_seq_limit=512    # Limit for language tokens
    )

    memory_manager = SimpleMemoryManager()
    vl_cache = VisionLanguageKVCache(config, memory_manager)

    # Example with language tokens
    batch_size, num_heads, lang_seq_len, head_dim = 1, 8, 32, 64
    lang_k = torch.randn(batch_size, num_heads, lang_seq_len, head_dim)
    lang_v = torch.randn(batch_size, num_heads, lang_seq_len, head_dim)

    lang_k_out, lang_v_out = vl_cache.update(lang_k, lang_v, is_vision=False)

    # Example with vision tokens
    vision_seq_len = 16
    vision_k = torch.randn(batch_size, num_heads, vision_seq_len, head_dim)
    vision_v = torch.randn(batch_size, num_heads, vision_seq_len, head_dim)

    vision_k_out, vision_v_out = vl_cache.update(vision_k, vision_v, is_vision=True)

    print(f"Language cache output shape: {lang_k_out.shape}")
    print(f"Vision cache output shape: {vision_k_out.shape}")

    return vl_cache


# Example 4: Custom configuration for specific use cases
def example_custom_config():
    """Example of custom configurations for different use cases."""
    
    import torch
    
    # Case 1: Maximize memory savings (for long contexts)
    long_context_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=16,        # Very low rank for max savings
        use_sliding_window=True,
        sliding_window_size=512,      # Small window for long contexts
        use_hybrid=True,
        memory_efficient_allocation=True
    )
    
    # Case 2: Balance between memory and quality (for general use)
    balanced_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=64,        # Balanced rank
        use_sliding_window=True,
        sliding_window_size=1024,     # Medium window
        use_hybrid=True,
        cache_compression_threshold=0.1
    )
    
    # Case 3: Minimal optimization (for quality-critical tasks)
    quality_config = KVCacheConfig(
        use_low_rank=False,           # No low-rank compression
        use_sliding_window=True,      # But still use sliding window
        sliding_window_size=2048,     # Large window
        use_hybrid=False
    )
    
    print("Created three different configurations:")
    print("1. Long context: Max memory savings")
    print("2. Balanced: Memory vs quality trade-off") 
    print("3. Quality: Minimal optimization, maximum quality")
    
    return long_context_config, balanced_config, quality_config


# Example 5: Integration with existing models
def example_integration():
    """Example of how to integrate with existing transformer models."""
    
    import torch
    import torch.nn as nn
    
    class OptimizedTransformerBlock(nn.Module):
        def __init__(self, hidden_size=4096, num_attention_heads=32):
            super().__init__()
            
            # Create optimized attention
            self.attention = create_optimized_attention_with_cache(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads
            )
            
            # Other components remain the same
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        
        def forward(self, x, use_cache=True):
            # Self-attention with optimized KV caching
            attn_output, _, _ = self.attention(
                x,
                use_cache=use_cache
            )
            
            # Add & Norm
            x = self.norm1(x + attn_output)
            
            # MLP
            mlp_output = self.mlp(x)
            
            # Add & Norm
            x = self.norm2(x + mlp_output)
            
            return x
    
    # Create the optimized transformer block
    model = OptimizedTransformerBlock()
    
    # Example usage
    batch_size, seq_len = 1, 64
    hidden_states = torch.randn(batch_size, seq_len, 4096)
    
    output = model(hidden_states)
    
    print(f"Transformer block input: {hidden_states.shape}")
    print(f"Transformer block output: {output.shape}")
    
    # Get attention cache statistics
    cache_stats = model.attention.get_cache_stats()
    print(f"KV cache compression ratio: {cache_stats['compression_ratio']:.3f}")
    
    return model


if __name__ == "__main__":
    print("KV Cache Optimization System - Usage Examples")
    print("=" * 50)
    
    print("\n1. Basic Usage:")
    kv_manager = example_basic_usage()
    
    print("\n2. Attention Layer:")
    attention = example_attention_layer()
    
    print("\n3. Vision-Language Optimization:")
    vl_cache = example_vision_language()
    
    print("\n4. Custom Configurations:")
    configs = example_custom_config()
    
    print("\n5. Model Integration:")
    model = example_integration()
    
    print("\nAll examples completed successfully!")
    print("The KV Cache Optimization System is ready for use.")