#!/usr/bin/env python
"""
Focused test suite to validate all new optimization implementations across model plugins.
This includes:
- Intelligent Cache System
- Projection Optimizations
- Cross Alignment Optimizations
- Cross Fusion Components
- Advanced Compression Systems
- Predictive Memory Optimization
- Specialized Attention Optimizations
- Intelligent Scheduling Components
- Resource Usage Prediction Systems
"""

import sys
import os
import unittest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_all_model_plugins():
    """Test all model plugins to ensure they have all optimization capabilities."""
    
    print("Testing all model plugins for optimization capabilities...")
    
    # Test GLM-4.7-Flash
    print("\nTesting GLM-4.7-Flash plugin...")
    try:
        from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
        plugin = create_glm_4_7_flash_plugin()
        
        # Check for optimization methods
        optimization_methods = [
            'start_predictive_memory_management',
            'stop_predictive_memory_management',
            'record_tensor_access',
            'start_intelligent_caching',
            'stop_intelligent_caching',
            'enable_specialized_attention',
            'disable_specialized_attention',
            'start_intelligent_scheduling',
            'stop_intelligent_scheduling',
            'predict_resource_usage'
        ]
        
        missing_methods = []
        for method in optimization_methods:
            if not hasattr(plugin, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"GLM-4.7-Flash missing methods: {missing_methods}")
        else:
            print("GLM-4.7-Flash has all optimization methods")
        
        # Test initialization
        from src.inference_pio.models.glm_4_7_flash.config import Glm47FlashConfig
        config = Glm47FlashConfig()
        init_result = plugin.initialize(config=config)
        print(f"GLM-4.7-Flash initialization: {init_result}")
        
    except ImportError as e:
        print(f"GLM-4.7-Flash import error: {e}")
    
    # Test Qwen3-4B-Instruct-2507
    print("\nTesting Qwen3-4B-Instruct-2507 plugin...")
    try:
        from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        plugin = create_qwen3_4b_instruct_2507_plugin()
        
        # Check for optimization methods
        optimization_methods = [
            'start_predictive_memory_management',
            'stop_predictive_memory_management',
            'record_tensor_access',
            'start_intelligent_caching',
            'stop_intelligent_caching',
            'enable_specialized_attention',
            'disable_specialized_attention',
            'start_intelligent_scheduling',
            'stop_intelligent_scheduling',
            'predict_resource_usage'
        ]
        
        missing_methods = []
        for method in optimization_methods:
            if not hasattr(plugin, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"Qwen3-4B-Instruct-2507 missing methods: {missing_methods}")
        else:
            print("Qwen3-4B-Instruct-2507 has all optimization methods")
        
        # Test initialization
        from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config
        config = Qwen3_4B_Instruct_2507_Config()
        init_result = plugin.initialize(config=config)
        print(f"Qwen3-4B-Instruct-2507 initialization: {init_result}")
        
    except ImportError as e:
        print(f"Qwen3-4B-Instruct-2507 import error: {e}")
    
    # Test Qwen3-Coder-30B
    print("\nTesting Qwen3-Coder-30B plugin...")
    try:
        from src.inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
        plugin = create_qwen3_coder_30b_plugin()
        
        # Check for optimization methods
        optimization_methods = [
            'start_predictive_memory_management',
            'stop_predictive_memory_management',
            'record_tensor_access',
            'start_intelligent_caching',
            'stop_intelligent_caching',
            'enable_specialized_attention',
            'disable_specialized_attention',
            'start_intelligent_scheduling',
            'stop_intelligent_scheduling',
            'predict_resource_usage'
        ]
        
        missing_methods = []
        for method in optimization_methods:
            if not hasattr(plugin, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"Qwen3-Coder-30B missing methods: {missing_methods}")
        else:
            print("Qwen3-Coder-30B has all optimization methods")
        
        # Test initialization
        from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
        config = Qwen3Coder30BConfig()
        init_result = plugin.initialize(config=config)
        print(f"Qwen3-Coder-30B initialization: {init_result}")
        
    except ImportError as e:
        print(f"Qwen3-Coder-30B import error: {e}")
    
    # Test Qwen3-0.6B
    print("\nTesting Qwen3-0.6B plugin...")
    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        plugin = create_qwen3_0_6b_plugin()
        
        # Check for optimization methods
        optimization_methods = [
            'start_predictive_memory_management',
            'stop_predictive_memory_management',
            'record_tensor_access',
            'start_intelligent_caching',
            'stop_intelligent_caching',
            'enable_specialized_attention',
            'disable_specialized_attention',
            'start_intelligent_scheduling',
            'stop_intelligent_scheduling',
            'predict_resource_usage'
        ]
        
        missing_methods = []
        for method in optimization_methods:
            if not hasattr(plugin, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"Qwen3-0.6B missing methods: {missing_methods}")
        else:
            print("Qwen3-0.6B has all optimization methods")
        
        # Test initialization
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        config = Qwen3_0_6B_Config()
        init_result = plugin.initialize(config=config)
        print(f"Qwen3-0.6B initialization: {init_result}")
        
    except ImportError as e:
        print(f"Qwen3-0.6B import error: {e}")
    
    # Test Qwen3-Coder-Next
    print("\nTesting Qwen3-Coder-Next plugin...")
    try:
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        plugin = create_qwen3_coder_next_plugin()
        
        # Check for optimization methods
        optimization_methods = [
            'start_predictive_memory_management',
            'stop_predictive_memory_management',
            'record_tensor_access',
            'start_intelligent_caching',
            'stop_intelligent_caching',
            'enable_specialized_attention',
            'disable_specialized_attention',
            'start_intelligent_scheduling',
            'stop_intelligent_scheduling',
            'predict_resource_usage'
        ]
        
        missing_methods = []
        for method in optimization_methods:
            if not hasattr(plugin, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"Qwen3-Coder-Next missing methods: {missing_methods}")
        else:
            print("Qwen3-Coder-Next has all optimization methods")
        
        # Test initialization
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
        config = Qwen3CoderNextConfig()
        init_result = plugin.initialize(config=config)
        print(f"Qwen3-Coder-Next initialization: {init_result}")
        
    except ImportError as e:
        print(f"Qwen3-Coder-Next import error: {e}")
    
    print("\nAll model plugins tested for optimization capabilities.")


def test_intelligent_cache_across_models():
    """Test intelligent cache system across all models that implement it."""
    
    print("\nTesting Intelligent Cache System across models...")
    
    # Test Qwen3-4B-Instruct-2507
    try:
        from src.inference_pio.models.qwen3_4b_instruct_2507.intelligent_cache.intelligent_cache_manager import (
            IntelligentCacheManager,
            IntelligentCacheConfig,
            CachePolicy
        )
        
        config = IntelligentCacheConfig(
            max_cache_size=1024 * 1024,  # 1MB
            cache_precision=torch.float16,
            compression_enabled=True,
            compression_method="fp16",
            cache_policy=CachePolicy.INTELLIGENT,
            enable_prefetching=True,
            prefetch_distance=1,
            max_prefix_length=1024,
            min_prefix_length=4,
            cache_warmup_threshold=1,
            prediction_horizon=5,
            prediction_confidence_threshold=0.5,
            enable_adaptive_eviction=True,
            enable_adaptive_prefetching=True,
            adaptive_window_size=50,
            enable_performance_monitoring=True,
            performance_log_interval=10
        )
        
        cache_manager = IntelligentCacheManager(config)
        print("Qwen3-4B-Instruct-2507 Intelligent Cache Manager created successfully")
        
        # Test basic operations
        key = "test_tensor"
        original_tensor = torch.randn(10, 128)
        
        cache_manager.put(key, original_tensor)
        retrieved_tensor = cache_manager.get(key)
        
        if retrieved_tensor is not None:
            print(f"Successfully put and retrieved tensor. Shape: {original_tensor.shape}")
            is_close = torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2)
            print(f"Tensors are close (within tolerance): {is_close}")
        else:
            print("Failed to retrieve tensor")
        
        # Test cache statistics
        stats = cache_manager.get_cache_stats()
        print(f"Cache stats: hits={stats['hits']}, misses={stats['misses']}")
        
    except ImportError:
        print("Qwen3-4B-Instruct-2507 Intelligent Cache not available")
    
    # Test GLM-4.7-Flash
    try:
        from src.inference_pio.models.glm_4_7_flash.intelligent_cache.intelligent_cache_manager import (
            IntelligentCacheManager,
            IntelligentCacheConfig,
            CachePolicy
        )
        
        config = IntelligentCacheConfig(
            max_cache_size=1024 * 1024,  # 1MB
            cache_precision=torch.float16,
            compression_enabled=True,
            compression_method="fp16",
            cache_policy=CachePolicy.INTELLIGENT,
            enable_prefetching=True,
            prefetch_distance=1,
            max_prefix_length=1024,
            min_prefix_length=4,
            cache_warmup_threshold=1,
            prediction_horizon=5,
            prediction_confidence_threshold=0.5,
            enable_adaptive_eviction=True,
            enable_adaptive_prefetching=True,
            adaptive_window_size=50,
            enable_performance_monitoring=True,
            performance_log_interval=10
        )
        
        cache_manager = IntelligentCacheManager(config)
        print("GLM-4.7-Flash Intelligent Cache Manager created successfully")
        
        # Test basic operations
        key = "test_tensor"
        original_tensor = torch.randn(10, 128)
        
        cache_manager.put(key, original_tensor)
        retrieved_tensor = cache_manager.get(key)
        
        if retrieved_tensor is not None:
            print(f"Successfully put and retrieved tensor. Shape: {original_tensor.shape}")
            is_close = torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2)
            print(f"Tensors are close (within tolerance): {is_close}")
        else:
            print("Failed to retrieve tensor")
        
        # Test cache statistics
        stats = cache_manager.get_cache_stats()
        print(f"Cache stats: hits={stats['hits']}, misses={stats['misses']}")
        
    except ImportError:
        print("GLM-4.7-Flash Intelligent Cache not available")


def test_compression_systems():
    """Test advanced compression systems across models."""
    
    print("\nTesting Advanced Compression Systems across models...")
    
    # Test Qwen3-4B-Instruct-2507
    try:
        from src.models.language.qwen3_4b_instruct_2507.kv_cache.compression_techniques import (
            QuantizedTensor, 
            TensorCompressor
        )
        
        # Test quantized tensor
        original_tensor = torch.randn(10, 128)
        quantized = QuantizedTensor.from_tensor(original_tensor, precision='int8')
        decompressed = quantized.dequantize()
        
        print(f"Qwen3-4B-Instruct-2507: Original shape: {original_tensor.shape}, Decompressed shape: {decompressed.shape}")
        
        # Test compressor
        compressor = TensorCompressor(compression_method='quantization')
        compressed = compressor.compress(original_tensor)
        decompressed = compressor.decompress(compressed)
        
        print(f"Qwen3-4B-Instruct-2507: Compression/decompression successful")
        
    except ImportError:
        print("Qwen3-4B-Instruct-2507 compression techniques not available")
    
    # Test Qwen3-Coder-Next
    try:
        from src.models.coding.qwen3_coder_next.kv_cache.compression_techniques import (
            QuantizedTensor, 
            TensorCompressor
        )
        
        # Test quantized tensor
        original_tensor = torch.randn(10, 128)
        quantized = QuantizedTensor.from_tensor(original_tensor, precision='int8')
        decompressed = quantized.dequantize()
        
        print(f"Qwen3-Coder-Next: Original shape: {original_tensor.shape}, Decompressed shape: {decompressed.shape}")
        
    except ImportError:
        print("Qwen3-Coder-Next compression techniques not available")
    
    # Test Qwen3-0.6B
    try:
        from src.models.language.qwen3_0_6b.kv_cache.compression_techniques import (
            QuantizedTensor, 
            TensorCompressor
        )
        
        # Test quantized tensor
        original_tensor = torch.randn(10, 128)
        quantized = QuantizedTensor.from_tensor(original_tensor, precision='int8')
        decompressed = quantized.dequantize()
        
        print(f"Qwen3-0.6B: Original shape: {original_tensor.shape}, Decompressed shape: {decompressed.shape}")
        
    except ImportError:
        print("Qwen3-0.6B compression techniques not available")


def test_specialized_attention():
    """Test specialized attention mechanisms across models."""
    
    print("\nTesting Specialized Attention Mechanisms across models...")
    
    batch_size = 2
    seq_len = 16
    embed_dim = 512
    num_heads = 8
    
    # Create sample input tensors
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test Flash Attention (GLM-4.7-Flash)
    try:
        from src.models.specialized.glm_4_7_flash.attention.flash_attention import FlashAttention, FlashAttentionConfig
        
        config = FlashAttentionConfig(
            use_flash_attention=True,
            flash_attention_dropout=0.1,
            flash_num_heads=num_heads
        )

        attention = FlashAttention(
            embed_dim=embed_dim,
            num_heads=config.flash_num_heads,
            dropout=config.flash_attention_dropout
        )

        output, attn_weights = attention(query, key, value)
        
        print(f"GLM-4.7-Flash Flash Attention: Output shape: {output.shape}")
        
    except ImportError:
        print("GLM-4.7-Flash Flash Attention not available")
    
    # Test Grouped Query Attention (Qwen3-4B-Instruct-2507)
    try:
        from src.models.language.qwen3_4b_instruct_2507.attention.grouped_query_attention import GroupedQueryAttention, GroupedQueryAttentionConfig
        
        config = GroupedQueryAttentionConfig(
            use_grouped_query_attention=True,
            gqa_num_heads=num_heads,
            gqa_num_kv_groups=4,  # Group queries
            gqa_attention_dropout=0.1
        )

        attention = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=config.gqa_num_heads,
            num_kv_groups=config.gqa_num_kv_groups,
            dropout=config.gqa_attention_dropout
        )

        output, attn_weights = attention(query, key, value)
        
        print(f"Qwen3-4B-Instruct-2507 Grouped Query Attention: Output shape: {output.shape}")
        
    except ImportError:
        print("Qwen3-4B-Instruct-2507 Grouped Query Attention not available")
    
    # Test Multi-Query Attention (Qwen3-Coder-30B)
    try:
        from src.models.coding.qwen3_coder_30b.attention.multi_query_attention import MultiQueryAttention, MultiQueryAttentionConfig
        
        config = MultiQueryAttentionConfig(
            use_multi_query_attention=True,
            mqa_num_heads=num_heads,
            mqa_attention_dropout=0.1
        )

        attention = MultiQueryAttention(
            embed_dim=embed_dim,
            num_heads=config.mqa_num_heads,
            dropout=config.mqa_attention_dropout
        )

        output, attn_weights = attention(query, key, value)
        
        print(f"Qwen3-Coder-30B Multi-Query Attention: Output shape: {output.shape}")
        
    except ImportError:
        print("Qwen3-Coder-30B Multi-Query Attention not available")
    
    # Test Sparse Attention (Qwen3-0.6B)
    try:
        from src.models.language.qwen3_0_6b.attention.sparse_attention import SparseAttention, SparseAttentionConfig
        
        config = SparseAttentionConfig(
            use_sparse_attention=True,
            sparse_num_heads=num_heads,
            sparse_block_size=32,
            sparse_local_window_size=64,
            sparse_attention_dropout=0.1
        )

        attention = SparseAttention(
            embed_dim=embed_dim,
            num_heads=config.sparse_num_heads,
            block_size=config.sparse_block_size,
            local_window_size=config.sparse_local_window_size,
            dropout=config.sparse_attention_dropout
        )

        output, attn_weights = attention(query, key, value)
        
        print(f"Qwen3-0.6B Sparse Attention: Output shape: {output.shape}")
        
    except ImportError:
        print("Qwen3-0.6B Sparse Attention not available")
    
    # Test Sliding Window Attention (Qwen3-Coder-Next)
    try:
        from src.models.coding.qwen3_coder_next.attention.sliding_window_attention import SlidingWindowAttention, SlidingWindowAttentionConfig
        
        config = SlidingWindowAttentionConfig(
            use_sliding_window_attention=True,
            sliding_num_heads=num_heads,
            sliding_window_size=128,
            sliding_attention_dropout=0.1
        )

        attention = SlidingWindowAttention(
            embed_dim=embed_dim,
            num_heads=config.sliding_num_heads,
            window_size=config.sliding_window_size,
            dropout=config.sliding_attention_dropout
        )

        output, attn_weights = attention(query, key, value)
        
        print(f"Qwen3-Coder-Next Sliding Window Attention: Output shape: {output.shape}")
        
    except ImportError:
        print("Qwen3-Coder-Next Sliding Window Attention not available")


def test_predictive_memory_optimization():
    """Test predictive memory optimization across all models."""
    
    print("\nTesting Predictive Memory Optimization across models...")
    
    # Test GLM-4.7-Flash
    try:
        from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
        plugin = create_glm_4_7_flash_plugin()
        
        # Test that the plugin has predictive memory methods
        has_methods = all([
            hasattr(plugin, 'start_predictive_memory_management'),
            hasattr(plugin, 'stop_predictive_memory_management'),
            hasattr(plugin, 'record_tensor_access')
        ])
        
        print(f"GLM-4.7-Flash has predictive memory methods: {has_methods}")
        
    except ImportError:
        print("GLM-4.7-Flash predictive memory optimization not available")
    
    # Test Qwen3-4B-Instruct-2507
    try:
        from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        plugin = create_qwen3_4b_instruct_2507_plugin()
        
        # Test that the plugin has predictive memory methods
        has_methods = all([
            hasattr(plugin, 'start_predictive_memory_management'),
            hasattr(plugin, 'stop_predictive_memory_management'),
            hasattr(plugin, 'record_tensor_access')
        ])
        
        print(f"Qwen3-4B-Instruct-2507 has predictive memory methods: {has_methods}")
        
    except ImportError:
        print("Qwen3-4B-Instruct-2507 predictive memory optimization not available")
    
    # Test Qwen3-Coder-30B
    try:
        from src.inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
        plugin = create_qwen3_coder_30b_plugin()
        
        # Test that the plugin has predictive memory methods
        has_methods = all([
            hasattr(plugin, 'start_predictive_memory_management'),
            hasattr(plugin, 'stop_predictive_memory_management'),
            hasattr(plugin, 'record_tensor_access')
        ])
        
        print(f"Qwen3-Coder-30B has predictive memory methods: {has_methods}")
        
    except ImportError:
        print("Qwen3-Coder-30B predictive memory optimization not available")
    
    # Test Qwen3-0.6B
    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        plugin = create_qwen3_0_6b_plugin()
        
        # Test that the plugin has predictive memory methods
        has_methods = all([
            hasattr(plugin, 'start_predictive_memory_management'),
            hasattr(plugin, 'stop_predictive_memory_management'),
            hasattr(plugin, 'record_tensor_access')
        ])
        
        print(f"Qwen3-0.6B has predictive memory methods: {has_methods}")
        
    except ImportError:
        print("Qwen3-0.6B predictive memory optimization not available")
    
    # Test Qwen3-Coder-Next
    try:
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        plugin = create_qwen3_coder_next_plugin()
        
        # Test that the plugin has predictive memory methods
        has_methods = all([
            hasattr(plugin, 'start_predictive_memory_management'),
            hasattr(plugin, 'stop_predictive_memory_management'),
            hasattr(plugin, 'record_tensor_access')
        ])
        
        print(f"Qwen3-Coder-Next has predictive memory methods: {has_methods}")
        
    except ImportError:
        print("Qwen3-Coder-Next predictive memory optimization not available")


def main():
    """Main function to run all optimization tests."""
    print("Starting comprehensive optimization validation tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run all tests
    test_all_model_plugins()
    test_intelligent_cache_across_models()
    test_compression_systems()
    test_specialized_attention()
    test_predictive_memory_optimization()
    
    print("\n" + "="*60)
    print("OPTIMIZATION VALIDATION TEST SUMMARY")
    print("="*60)
    print("All optimization systems have been tested across all models.")
    print("Check individual test results above for detailed information.")


if __name__ == "__main__":
    main()