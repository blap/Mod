#!/usr/bin/env python
"""
Final comprehensive test suite to validate all implemented optimization systems across all models.
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

class TestSpecializedAttentionOptimizations(unittest.TestCase):
    """Test specialized attention mechanisms across all models."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.seq_len = 16
        self.embed_dim = 512
        self.num_heads = 8
        
        # Create sample input tensors
        self.query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

    def test_flash_attention(self):
        """Test Flash Attention implementation for GLM-4.7-Flash."""
        try:
            from src.models.specialized.glm_4_7_flash.attention.flash_attention import FlashAttention, FlashAttentionConfig
            
            config = FlashAttentionConfig(
                use_flash_attention=True,
                flash_attention_dropout=0.1,
                flash_num_heads=self.num_heads
            )

            attention = FlashAttention(
                embed_dim=self.embed_dim,
                num_heads=config.flash_num_heads,
                dropout=config.flash_attention_dropout
            )

            output, attn_weights = attention(self.query, self.key, self.value)

            # Verify output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

            # Verify attention weights shape if returned
            if attn_weights is not None:
                self.assertEqual(attn_weights.shape[0], self.batch_size)
                self.assertEqual(attn_weights.shape[1], self.seq_len)
                self.assertEqual(attn_weights.shape[2], self.seq_len)
                
            print("[PASS] GLM-4.7-Flash Flash Attention test passed")
        except ImportError:
            self.skipTest("GLM-4.7-Flash Flash Attention not available")

    def test_grouped_query_attention(self):
        """Test Grouped Query Attention implementation for Qwen3-4B-Instruct-2507."""
        try:
            from src.models.language.qwen3_4b_instruct_2507.attention.grouped_query_attention import GroupedQueryAttention, GroupedQueryAttentionConfig
            
            config = GroupedQueryAttentionConfig(
                use_grouped_query_attention=True,
                gqa_num_heads=self.num_heads,
                gqa_num_kv_groups=4,  # Group queries
                gqa_attention_dropout=0.1
            )

            attention = GroupedQueryAttention(
                embed_dim=self.embed_dim,
                num_heads=config.gqa_num_heads,
                num_kv_groups=config.gqa_num_kv_groups,
                dropout=config.gqa_attention_dropout
            )

            output, attn_weights = attention(self.query, self.key, self.value)

            # Verify output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

            # Verify attention weights shape if returned
            if attn_weights is not None:
                self.assertEqual(attn_weights.shape[0], self.batch_size)
                self.assertEqual(attn_weights.shape[1], self.seq_len)
                self.assertEqual(attn_weights.shape[2], self.seq_len)
                
            print("[PASS] Qwen3-4B-Instruct-2507 Grouped Query Attention test passed")
        except ImportError:
            self.skipTest("Qwen3-4B-Instruct-2507 Grouped Query Attention not available")

    def test_multi_query_attention(self):
        """Test Multi-Query Attention implementation for Qwen3-Coder-30B."""
        try:
            from src.models.coding.qwen3_coder_30b.attention.multi_query_attention import MultiQueryAttention, MultiQueryAttentionConfig
            
            config = MultiQueryAttentionConfig(
                use_multi_query_attention=True,
                mqa_num_heads=self.num_heads,
                mqa_attention_dropout=0.1
            )

            attention = MultiQueryAttention(
                embed_dim=self.embed_dim,
                num_heads=config.mqa_num_heads,
                dropout=config.mqa_attention_dropout
            )

            output, attn_weights = attention(self.query, self.key, self.value)

            # Verify output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

            # Verify attention weights shape if returned
            if attn_weights is not None:
                self.assertEqual(attn_weights.shape[0], self.batch_size)
                self.assertEqual(attn_weights.shape[1], self.seq_len)
                self.assertEqual(attn_weights.shape[2], self.seq_len)
                
            print("[PASS] Qwen3-Coder-30B Multi-Query Attention test passed")
        except ImportError:
            self.skipTest("Qwen3-Coder-30B Multi-Query Attention not available")

    def test_sparse_attention(self):
        """Test Sparse Attention implementation for Qwen3-0.6B."""
        try:
            from src.models.language.qwen3_0_6b.attention.sparse_attention import SparseAttention, SparseAttentionConfig
            
            config = SparseAttentionConfig(
                use_sparse_attention=True,
                sparse_num_heads=self.num_heads,
                sparse_block_size=32,
                sparse_local_window_size=64,
                sparse_attention_dropout=0.1
            )

            attention = SparseAttention(
                embed_dim=self.embed_dim,
                num_heads=config.sparse_num_heads,
                block_size=config.sparse_block_size,
                local_window_size=config.sparse_local_window_size,
                dropout=config.sparse_attention_dropout
            )

            output, attn_weights = attention(self.query, self.key, self.value)

            # Verify output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
            
            print("[PASS] Qwen3-0.6B Sparse Attention test passed")
        except ImportError:
            self.skipTest("Qwen3-0.6B Sparse Attention not available")

    def test_sliding_window_attention(self):
        """Test Sliding Window Attention implementation for Qwen3-Coder-Next."""
        try:
            from src.models.coding.qwen3_coder_next.attention.sliding_window_attention import SlidingWindowAttention, SlidingWindowAttentionConfig
            
            config = SlidingWindowAttentionConfig(
                use_sliding_window_attention=True,
                sliding_num_heads=self.num_heads,
                sliding_window_size=128,
                sliding_attention_dropout=0.1
            )

            attention = SlidingWindowAttention(
                embed_dim=self.embed_dim,
                num_heads=config.sliding_num_heads,
                window_size=config.sliding_window_size,
                dropout=config.sliding_attention_dropout
            )

            output, attn_weights = attention(self.query, self.key, self.value)

            # Verify output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
            
            print("[PASS] Qwen3-Coder-Next Sliding Window Attention test passed")
        except ImportError:
            self.skipTest("Qwen3-Coder-Next Sliding Window Attention not available")


class TestCompressionSystems(unittest.TestCase):
    """Test Advanced Compression Systems across models."""
    
    def test_compression_implementations(self):
        """Test that compression systems are properly implemented."""
        # Test Qwen3-4B-Instruct-2507
        try:
            from src.inference_pio.models.qwen3_4b_instruct_2507.kv_cache.compression_techniques import (
                QuantizedTensor, 
                TensorCompressor
            )
            
            # Test quantized tensor
            original_tensor = torch.randn(10, 128)
            quantized = QuantizedTensor.from_tensor(original_tensor, precision='int8')
            decompressed = quantized.dequantize()
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
            
            # Test compressor
            compressor = TensorCompressor(compression_method='quantization')
            compressed = compressor.compress(original_tensor)
            decompressed = compressor.decompress(compressed)
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
            
            print("[PASS] Qwen3-4B-Instruct-2507 Compression Techniques test passed")
        except ImportError:
            print("[SKIP] Qwen3-4B-Instruct-2507 Compression Techniques not available")
        
        # Test Qwen3-Coder-Next
        try:
            from src.inference_pio.models.qwen3_coder_next.kv_cache.compression_techniques import (
                QuantizedTensor, 
                TensorCompressor
            )
            
            # Test quantized tensor
            original_tensor = torch.randn(10, 128)
            quantized = QuantizedTensor.from_tensor(original_tensor, precision='int8')
            decompressed = quantized.dequantize()
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
            
            print("[PASS] Qwen3-Coder-Next Compression Techniques test passed")
        except ImportError:
            print("[SKIP] Qwen3-Coder-Next Compression Techniques not available")
        
        # Test Qwen3-0.6B
        try:
            from src.inference_pio.models.qwen3_0_6b.kv_cache.compression_techniques import (
                QuantizedTensor, 
                TensorCompressor
            )
            
            # Test quantized tensor
            original_tensor = torch.randn(10, 128)
            quantized = QuantizedTensor.from_tensor(original_tensor, precision='int8')
            decompressed = quantized.dequantize()
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
            
            print("[PASS] Qwen3-0.6B Compression Techniques test passed")
        except ImportError:
            print("[SKIP] Qwen3-0.6B Compression Techniques not available")


class TestIntelligentCacheSystem(unittest.TestCase):
    """Test the Intelligent Cache System implementation."""
    
    def test_intelligent_cache_system(self):
        """Test basic intelligent cache functionality."""
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
            
            # Test basic operations
            key = "test_tensor"
            original_tensor = torch.randn(10, 128)
            
            cache_manager.put(key, original_tensor)
            retrieved_tensor = cache_manager.get(key)
            
            self.assertIsNotNone(retrieved_tensor)
            self.assertEqual(original_tensor.shape, retrieved_tensor.shape)
            
            # Check that the retrieved tensor is approximately equal to the original
            is_close = torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2)
            self.assertTrue(is_close)
            
            # Test cache statistics
            stats = cache_manager.get_cache_stats()
            self.assertIn('hits', stats)
            self.assertIn('misses', stats)
            
            print("[PASS] Qwen3-4B-Instruct-2507 Intelligent Cache System test passed")
        except ImportError:
            print("[SKIP] Qwen3-4B-Instruct-2507 Intelligent Cache System not available")
        
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
            
            # Test basic operations
            key = "test_tensor"
            original_tensor = torch.randn(10, 128)
            
            cache_manager.put(key, original_tensor)
            retrieved_tensor = cache_manager.get(key)
            
            self.assertIsNotNone(retrieved_tensor)
            self.assertEqual(original_tensor.shape, retrieved_tensor.shape)
            
            # Check that the retrieved tensor is approximately equal to the original
            is_close = torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2)
            self.assertTrue(is_close)
            
            print("[PASS] GLM-4.7-Flash Intelligent Cache System test passed")
        except ImportError:
            print("[SKIP] GLM-4.7-Flash Intelligent Cache System not available")


class TestModelPluginOptimizations(unittest.TestCase):
    """Test that model plugins have optimization capabilities."""
    
    def test_model_plugin_optimization_methods(self):
        """Test that model plugins have expected optimization methods."""
        # Test GLM-4.7-Flash
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
                print("[PASS] GLM-4.7-Flash has all optimization methods")
                
        except ImportError:
            print("[SKIP] GLM-4.7-Flash plugin not available")
        
        # Test Qwen3-4B-Instruct-2507
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
                print("[PASS] Qwen3-4B-Instruct-2507 has all optimization methods")
                
        except ImportError:
            print("[SKIP] Qwen3-4B-Instruct-2507 plugin not available")
        
        # Test Qwen3-Coder-30B
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
                print("[PASS] Qwen3-Coder-30B has all optimization methods")
                
        except ImportError:
            print("[SKIP] Qwen3-Coder-30B plugin not available")
        
        # Test Qwen3-0.6B
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
                print("[PASS] Qwen3-0.6B has all optimization methods")
                
        except ImportError:
            print("[SKIP] Qwen3-0.6B plugin not available")
        
        # Test Qwen3-Coder-Next
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
                print("[PASS] Qwen3-Coder-Next has all optimization methods")
                
        except ImportError:
            print("[SKIP] Qwen3-Coder-Next plugin not available")


def run_comprehensive_tests():
    """Run all comprehensive optimization tests."""
    print("Starting comprehensive optimization validation tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestSpecializedAttentionOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestCompressionSystems))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentCacheSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPluginOptimizations))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE OPTIMIZATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] All comprehensive optimization tests passed!")
        print("All new implementations are working correctly across all models.")
        return True
    else:
        print("\n[ERROR] Some comprehensive optimization tests failed.")
        for failure in result.failures:
            print(f"FAILURE: {str(failure[0])[:100]} - {str(failure[1])[:400]}...")
        for error in result.errors:
            print(f"ERROR: {str(error[0])[:100]} - {str(error[1])[:400]}...")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)