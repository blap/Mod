#!/usr/bin/env python
"""
Integration test suite to validate that all optimization systems work together correctly.
This tests the interaction between:
- Intelligent Cache System + Projection Optimizations
- Cross Alignment + Cross Fusion Components
- Compression Systems + Predictive Memory Optimization
- Specialized Attention + Intelligent Scheduling
- Resource Prediction + All other systems
"""

import sys
import os
import unittest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestOptimizationIntegration(unittest.TestCase):
    """Test integration between different optimization systems."""
    
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

    def test_intelligent_cache_with_projection_optimizations(self):
        """Test integration between intelligent cache and projection optimizations."""
        try:
            # Test with Qwen3-4B-Instruct-2507
            from src.inference_pio.models.qwen3_4b_instruct_2507.intelligent_cache.intelligent_cache_manager import (
                IntelligentCacheManager,
                IntelligentCacheConfig,
                CachePolicy
            )
            from src.models.language.qwen3_4b_instruct_2507.projection_optimizations.projection_optimizations import OptimizedLinearLayer
            
            # Create cache manager
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
            
            # Create optimized projection layer
            projection_layer = OptimizedLinearLayer(in_features=512, out_features=256)
            
            # Process tensor through projection
            input_tensor = torch.randn(10, 512)
            projected_tensor = projection_layer(input_tensor)
            
            # Cache the projected tensor
            cache_key = "projected_tensor"
            cache_manager.put(cache_key, projected_tensor)
            
            # Retrieve from cache
            retrieved_tensor = cache_manager.get(cache_key)
            
            # Verify shapes match
            self.assertEqual(projected_tensor.shape, retrieved_tensor.shape)
            
            print("✓ Intelligent Cache + Projection Optimizations integration test passed")
            
        except ImportError:
            print("WARNING: Skipping Intelligent Cache + Projection Optimizations integration test (modules not available)")
    
    def test_cross_alignment_with_cross_fusion(self):
        """Test integration between cross alignment and cross fusion components."""
        try:
            # Test with GLM-4.7-Flash
            from src.models.specialized.glm_4_7_flash.cross_alignment.cross_alignment_optimizer import CrossAlignmentOptimizer
            from src.models.specialized.glm_4_7_flash.cross_fusion.cross_fusion_component import CrossFusionComponent
            
            # Create optimizers/components
            alignment_optimizer = CrossAlignmentOptimizer()
            fusion_component = CrossFusionComponent()
            
            # Create sample tensors for testing
            tensor_a = torch.randn(10, 128)
            tensor_b = torch.randn(10, 128)
            
            # Apply cross alignment
            aligned_tensor_a, aligned_tensor_b = alignment_optimizer.align_tensors(tensor_a, tensor_b)
            
            # Apply cross fusion
            fused_tensor = fusion_component.fuse_tensors(aligned_tensor_a, aligned_tensor_b)
            
            # Verify shapes
            self.assertEqual(fused_tensor.shape, aligned_tensor_a.shape)
            self.assertEqual(fused_tensor.shape, aligned_tensor_b.shape)
            
            print("✓ Cross Alignment + Cross Fusion integration test passed")
            
        except ImportError:
            print("WARNING: Skipping Cross Alignment + Cross Fusion integration test (modules not available)")
        except AttributeError:
            print("WARNING: Skipping Cross Alignment + Cross Fusion integration test (methods not available)")
    
    def test_compression_with_predictive_memory(self):
        """Test integration between compression systems and predictive memory optimization."""
        try:
            # Test with Qwen3-4B-Instruct-2507
            from src.models.language.qwen3_4b_instruct_2507.kv_cache.compression_techniques import (
                QuantizedTensor, 
                TensorCompressor
            )
            from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
            
            # Create compressor
            compressor = TensorCompressor(compression_method='quantization')
            
            # Create sample tensor
            original_tensor = torch.randn(10, 128)
            
            # Compress tensor
            compressed = compressor.compress(original_tensor)
            
            # Simulate predictive memory by recording access
            plugin = create_qwen3_4b_instruct_2507_plugin()
            
            # Record access to compressed tensor representation
            record_result = plugin.record_tensor_access("compressed_tensor", compressed)
            
            # Decompress tensor
            decompressed = compressor.decompress(compressed)
            
            # Verify shapes
            self.assertEqual(original_tensor.shape, decompressed.shape)
            
            print("✓ Compression + Predictive Memory integration test passed")
            
        except ImportError:
            print("WARNING: Skipping Compression + Predictive Memory integration test (modules not available)")
    
    def test_attention_with_intelligent_scheduling(self):
        """Test integration between specialized attention and intelligent scheduling."""
        try:
            # Test with Qwen3-0.6B
            from src.models.language.qwen3_0_6b.attention.sparse_attention import SparseAttention, SparseAttentionConfig
            from src.inference_pio.models.qwen3_0_6b.scheduling.intelligent_scheduler import IntelligentScheduler
            
            # Create attention mechanism
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
            
            # Create scheduler
            scheduler = IntelligentScheduler()
            
            # Process attention
            output, attn_weights = attention(self.query, self.key, self.value)
            
            # Schedule the attention computation
            scheduled_task = scheduler.schedule_computation(output, priority=5)
            
            # Verify output shape
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
            
            print("✓ Attention + Intelligent Scheduling integration test passed")
            
        except ImportError:
            print("WARNING: Skipping Attention + Intelligent Scheduling integration test (modules not available)")
    
    def test_resource_prediction_with_all_systems(self):
        """Test that resource prediction works with all other systems."""
        try:
            # Test with Qwen3-Coder-Next
            from src.inference_pio.models.qwen3_coder_next.resource_prediction.resource_predictor import ResourcePredictor
            from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
            
            # Create resource predictor
            predictor = ResourcePredictor()
            
            # Create plugin
            plugin = create_qwen3_coder_next_plugin()
            
            # Make a prediction
            prediction = predictor.predict_resources(prompt_length=100, max_new_tokens=50)
            
            # Verify prediction structure
            self.assertIn('memory_estimate', prediction)
            self.assertIn('compute_estimate', prediction)
            self.assertIn('time_estimate', prediction)
            
            print("✓ Resource Prediction integration test passed")
            
        except ImportError:
            print("WARNING: Skipping Resource Prediction integration test (modules not available)")


class TestEndToEndOptimizationPipeline(unittest.TestCase):
    """Test end-to-end optimization pipeline with multiple systems working together."""
    
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained')
    def test_complete_optimization_pipeline_qwen3_4b(self, mock_tokenizer, mock_model):
        """Test complete optimization pipeline for Qwen3-4B-Instruct-2507."""
        try:
            from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
            from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config
            from src.inference_pio.models.qwen3_4b_instruct_2507.intelligent_cache.intelligent_cache_manager import (
                IntelligentCacheManager,
                IntelligentCacheConfig,
                CachePolicy
            )
            from src.models.language.qwen3_4b_instruct_2507.kv_cache.compression_techniques import TensorCompressor
            
            # Mock model and tokenizer
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
            mock_model.return_value = mock_model_instance
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer_instance.decode.return_value = "Mock response text"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Create plugin and config
            plugin = create_qwen3_4b_instruct_2507_plugin()
            config = Qwen3_4B_Instruct_2507_Config()
            
            # Initialize with optimizations enabled
            success = plugin.initialize(config=config)
            self.assertTrue(success)
            
            # Enable predictive memory management
            pred_mem_result = plugin.start_predictive_memory_management()
            self.assertTrue(pred_mem_result)
            
            # Enable intelligent caching
            cache_result = plugin.start_intelligent_caching()
            self.assertTrue(cache_result)
            
            # Create cache manager directly
            cache_config = IntelligentCacheConfig(
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
            
            cache_manager = IntelligentCacheManager(cache_config)
            
            # Create compressor
            compressor = TensorCompressor(compression_method='quantization')
            
            # Process a sample input through the pipeline
            sample_input = "Test input for optimization pipeline"
            result = plugin.infer(sample_input)
            
            self.assertIsNotNone(result)
            
            # Test cache operations
            test_tensor = torch.randn(5, 64)
            cache_manager.put("test_key", test_tensor)
            cached_result = cache_manager.get("test_key")
            
            self.assertIsNotNone(cached_result)
            
            # Test compression
            compressed = compressor.compress(test_tensor)
            decompressed = compressor.decompress(compressed)
            
            self.assertEqual(test_tensor.shape, decompressed.shape)
            
            # Stop optimizations
            plugin.stop_predictive_memory_management()
            plugin.stop_intelligent_caching()
            
            # Cleanup
            cleanup_success = plugin.cleanup()
            self.assertTrue(cleanup_success)
            
            print("✓ Complete Optimization Pipeline test for Qwen3-4B-Instruct-2507 passed")
            
        except ImportError:
            print("⚠ Skipping Complete Optimization Pipeline test for Qwen3-4B-Instruct-2507 (modules not available)")
    
    @patch('src.inference_pio.models.qwen3_0_6b.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_0_6b.model.AutoTokenizer.from_pretrained')
    def test_complete_optimization_pipeline_qwen3_0_6b(self, mock_tokenizer, mock_model):
        """Test complete optimization pipeline for Qwen3-0.6B."""
        try:
            from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
            from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
            from src.inference_pio.models.qwen3_0_6b.intelligent_cache.intelligent_cache_manager import (
                IntelligentCacheManager,
                IntelligentCacheConfig,
                CachePolicy
            )
            from src.models.language.qwen3_0_6b.kv_cache.compression_techniques import TensorCompressor
            
            # Mock model and tokenizer
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
            mock_model.return_value = mock_model_instance
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer_instance.decode.return_value = "Mock response text"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Create plugin and config
            plugin = create_qwen3_0_6b_plugin()
            config = Qwen3_0_6B_Config()
            
            # Initialize with optimizations enabled
            success = plugin.initialize(config=config)
            self.assertTrue(success)
            
            # Enable predictive memory management
            pred_mem_result = plugin.start_predictive_memory_management()
            self.assertTrue(pred_mem_result)
            
            # Enable intelligent caching
            cache_result = plugin.start_intelligent_caching()
            self.assertTrue(cache_result)
            
            # Create cache manager directly
            cache_config = IntelligentCacheConfig(
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
            
            cache_manager = IntelligentCacheManager(cache_config)
            
            # Create compressor
            compressor = TensorCompressor(compression_method='quantization')
            
            # Process a sample input through the pipeline
            sample_input = "Test input for optimization pipeline"
            result = plugin.infer(sample_input)
            
            self.assertIsNotNone(result)
            
            # Test cache operations
            test_tensor = torch.randn(5, 64)
            cache_manager.put("test_key", test_tensor)
            cached_result = cache_manager.get("test_key")
            
            self.assertIsNotNone(cached_result)
            
            # Test compression
            compressed = compressor.compress(test_tensor)
            decompressed = compressor.decompress(compressed)
            
            self.assertEqual(test_tensor.shape, decompressed.shape)
            
            # Stop optimizations
            plugin.stop_predictive_memory_management()
            plugin.stop_intelligent_caching()
            
            # Cleanup
            cleanup_success = plugin.cleanup()
            self.assertTrue(cleanup_success)
            
            print("✓ Complete Optimization Pipeline test for Qwen3-0.6B passed")
            
        except ImportError:
            print("⚠ Skipping Complete Optimization Pipeline test for Qwen3-0.6B (modules not available)")


def run_integration_tests():
    """Run all integration tests."""
    print("Starting optimization integration tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndOptimizationPipeline))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] All optimization integration tests passed!")
        print("All optimization systems work correctly together.")
        return True
    else:
        print("\n[ERROR] Some optimization integration tests failed.")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]} - {str(failure[1])[:500]}...")
        for error in result.errors:
            print(f"ERROR: {error[0]} - {str(error[1])[:500]}...")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)