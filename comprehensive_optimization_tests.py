#!/usr/bin/env python
"""
Comprehensive test suite to validate all new optimization implementations across all models.
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

class TestIntelligentCacheSystem(unittest.TestCase):
    """Test the Intelligent Cache System implementation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        from src.inference_pio.models.qwen3_4b_instruct_2507.intelligent_cache.intelligent_cache_manager import (
            IntelligentCacheManager,
            IntelligentCacheConfig,
            CachePolicy
        )
        
        self.config = IntelligentCacheConfig(
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
        
        self.cache_manager = IntelligentCacheManager(self.config)
    
    def test_cache_put_get_operations(self):
        """Test basic put and get operations."""
        key = "test_tensor"
        original_tensor = torch.randn(10, 128)
        
        self.cache_manager.put(key, original_tensor)
        retrieved_tensor = self.cache_manager.get(key)
        
        self.assertIsNotNone(retrieved_tensor)
        self.assertEqual(original_tensor.shape, retrieved_tensor.shape)
        
        # Check that the retrieved tensor is approximately equal to the original
        is_close = torch.allclose(original_tensor.half(), retrieved_tensor, atol=1e-2)
        self.assertTrue(is_close)
    
    def test_cache_statistics(self):
        """Test cache statistics functionality."""
        stats = self.cache_manager.get_cache_stats()
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('evictions', stats)
        self.assertIn('size', stats)


class TestProjectionOptimizations(unittest.TestCase):
    """Test Projection Optimizations across all models."""
    
    def test_glm_4_7_flash_projection(self):
        """Test projection optimizations for GLM-4.7-Flash."""
        from src.models.specialized.glm_4_7_flash.projection_optimizations.projection_optimizations import (
            OptimizedLinearLayer,
            create_optimized_projection_layer
        )
        
        layer = OptimizedLinearLayer(in_features=512, out_features=256)
        x = torch.randn(10, 512)
        output = layer(x)
        
        self.assertEqual(output.shape, (10, 256))
        
        # Test factory function
        config = MagicMock()
        config.use_optimized_projections = True
        layer_from_factory = create_optimized_projection_layer(config, in_features=512, out_features=256)
        self.assertIsInstance(layer_from_factory, OptimizedLinearLayer)
    
    def test_qwen3_4b_projection(self):
        """Test projection optimizations for Qwen3-4B-Instruct-2507."""
        from src.models.language.qwen3_4b_instruct_2507.projection_optimizations.projection_optimizations import (
            OptimizedLinearLayer,
            create_optimized_projection_layer
        )
        
        layer = OptimizedLinearLayer(in_features=512, out_features=256)
        x = torch.randn(10, 512)
        output = layer(x)
        
        self.assertEqual(output.shape, (10, 256))
        
        # Test factory function
        config = MagicMock()
        config.use_optimized_projections = True
        layer_from_factory = create_optimized_projection_layer(config, in_features=512, out_features=256)
        self.assertIsInstance(layer_from_factory, OptimizedLinearLayer)
    
    def test_qwen3_coder_30b_projection(self):
        """Test projection optimizations for Qwen3-Coder-30B."""
        from src.models.coding.qwen3_coder_30b.projection_optimizations.projection_optimizations import (
            OptimizedLinearLayer,
            create_optimized_projection_layer
        )
        
        layer = OptimizedLinearLayer(in_features=512, out_features=256)
        x = torch.randn(10, 512)
        output = layer(x)
        
        self.assertEqual(output.shape, (10, 256))
        
        # Test factory function
        config = MagicMock()
        config.use_optimized_projections = True
        layer_from_factory = create_optimized_projection_layer(config, in_features=512, out_features=256)
        self.assertIsInstance(layer_from_factory, OptimizedLinearLayer)
    
    def test_qwen3_0_6b_projection(self):
        """Test projection optimizations for Qwen3-0.6B."""
        from src.models.language.qwen3_0_6b.projection_optimizations.projection_optimizations import (
            OptimizedLinearLayer,
            create_optimized_projection_layer
        )
        
        layer = OptimizedLinearLayer(in_features=512, out_features=256)
        x = torch.randn(10, 512)
        output = layer(x)
        
        self.assertEqual(output.shape, (10, 256))
        
        # Test factory function
        config = MagicMock()
        config.use_optimized_projections = True
        layer_from_factory = create_optimized_projection_layer(config, in_features=512, out_features=256)
        self.assertIsInstance(layer_from_factory, OptimizedLinearLayer)
    
    def test_qwen3_coder_next_projection(self):
        """Test projection optimizations for Qwen3-Coder-Next."""
        from src.models.coding.qwen3_coder_next.projection_optimizations.projection_optimizations import (
            OptimizedLinearLayer,
            create_optimized_projection_layer
        )
        
        layer = OptimizedLinearLayer(in_features=512, out_features=256)
        x = torch.randn(10, 512)
        output = layer(x)
        
        self.assertEqual(output.shape, (10, 256))
        
        # Test factory function
        config = MagicMock()
        config.use_optimized_projections = True
        layer_from_factory = create_optimized_projection_layer(config, in_features=512, out_features=256)
        self.assertIsInstance(layer_from_factory, OptimizedLinearLayer)


class TestCrossAlignmentOptimizations(unittest.TestCase):
    """Test Cross Alignment Optimizations across all models."""
    
    def test_cross_alignment_implementations(self):
        """Test that cross alignment optimizations are properly implemented."""
        # Test GLM-4.7-Flash
        try:
            from src.models.specialized.glm_4_7_flash.cross_alignment.cross_alignment_optimizer import CrossAlignmentOptimizer
            optimizer = CrossAlignmentOptimizer()
            self.assertIsNotNone(optimizer)
        except ImportError:
            self.fail("Cross alignment optimizer not implemented for GLM-4.7-Flash")
        
        # Test Qwen3-4B-Instruct-2507
        try:
            from src.models.language.qwen3_4b_instruct_2507.cross_alignment.cross_alignment_optimizer import CrossAlignmentOptimizer
            optimizer = CrossAlignmentOptimizer()
            self.assertIsNotNone(optimizer)
        except ImportError:
            self.fail("Cross alignment optimizer not implemented for Qwen3-4B-Instruct-2507")
        
        # Test Qwen3-Coder-30B
        try:
            from src.models.coding.qwen3_coder_30b.cross_alignment.cross_alignment_optimizer import CrossAlignmentOptimizer
            optimizer = CrossAlignmentOptimizer()
            self.assertIsNotNone(optimizer)
        except ImportError:
            self.fail("Cross alignment optimizer not implemented for Qwen3-Coder-30B")
        
        # Test Qwen3-0.6B
        try:
            from src.models.language.qwen3_0_6b.cross_alignment.cross_alignment_optimizer import CrossAlignmentOptimizer
            optimizer = CrossAlignmentOptimizer()
            self.assertIsNotNone(optimizer)
        except ImportError:
            self.fail("Cross alignment optimizer not implemented for Qwen3-0.6B")
        
        # Test Qwen3-Coder-Next
        try:
            from src.models.coding.qwen3_coder_next.cross_alignment.cross_alignment_optimizer import CrossAlignmentOptimizer
            optimizer = CrossAlignmentOptimizer()
            self.assertIsNotNone(optimizer)
        except ImportError:
            self.fail("Cross alignment optimizer not implemented for Qwen3-Coder-Next")


class TestCrossFusionComponents(unittest.TestCase):
    """Test Cross Fusion Components across all models."""
    
    def test_cross_fusion_implementations(self):
        """Test that cross fusion components are properly implemented."""
        # Test GLM-4.7-Flash
        try:
            from src.models.specialized.glm_4_7_flash.cross_fusion.cross_fusion_component import CrossFusionComponent
            component = CrossFusionComponent()
            self.assertIsNotNone(component)
        except ImportError:
            self.fail("Cross fusion component not implemented for GLM-4.7-Flash")
        
        # Test Qwen3-4B-Instruct-2507
        try:
            from src.models.language.qwen3_4b_instruct_2507.cross_fusion.cross_fusion_component import CrossFusionComponent
            component = CrossFusionComponent()
            self.assertIsNotNone(component)
        except ImportError:
            self.fail("Cross fusion component not implemented for Qwen3-4B-Instruct-2507")
        
        # Test Qwen3-Coder-30B
        try:
            from src.models.coding.qwen3_coder_30b.cross_fusion.cross_fusion_component import CrossFusionComponent
            component = CrossFusionComponent()
            self.assertIsNotNone(component)
        except ImportError:
            self.fail("Cross fusion component not implemented for Qwen3-Coder-30B")
        
        # Test Qwen3-0.6B
        try:
            from src.models.language.qwen3_0_6b.cross_fusion.cross_fusion_component import CrossFusionComponent
            component = CrossFusionComponent()
            self.assertIsNotNone(component)
        except ImportError:
            self.fail("Cross fusion component not implemented for Qwen3-0.6B")
        
        # Test Qwen3-Coder-Next
        try:
            from src.models.coding.qwen3_coder_next.cross_fusion.cross_fusion_component import CrossFusionComponent
            component = CrossFusionComponent()
            self.assertIsNotNone(component)
        except ImportError:
            self.fail("Cross fusion component not implemented for Qwen3-Coder-Next")


class TestAdvancedCompressionSystems(unittest.TestCase):
    """Test Advanced Compression Systems across all models."""
    
    def test_compression_implementations(self):
        """Test that compression systems are properly implemented."""
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
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
            
            # Test compressor
            compressor = TensorCompressor(compression_method='quantization')
            compressed = compressor.compress(original_tensor)
            decompressed = compressor.decompress(compressed)
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
        except ImportError:
            self.fail("Compression techniques not implemented for Qwen3-4B-Instruct-2507")
        
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
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
        except ImportError:
            self.fail("Compression techniques not implemented for Qwen3-Coder-Next")
        
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
            
            self.assertIsNotNone(decompressed)
            self.assertEqual(original_tensor.shape, decompressed.shape)
        except ImportError:
            self.fail("Compression techniques not implemented for Qwen3-0.6B")


class TestPredictiveMemoryOptimization(unittest.TestCase):
    """Test Predictive Memory Optimization across all models."""
    
    def test_predictive_memory_implementations(self):
        """Test that predictive memory optimization is properly implemented."""
        # Test GLM-4.7-Flash
        try:
            from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
            plugin = create_glm_4_7_flash_plugin()
            
            # Test that the plugin has predictive memory methods
            self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'stop_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'record_tensor_access'))
        except ImportError:
            self.fail("Predictive memory optimization not implemented for GLM-4.7-Flash")
        
        # Test Qwen3-4B-Instruct-2507
        try:
            from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
            plugin = create_qwen3_4b_instruct_2507_plugin()
            
            # Test that the plugin has predictive memory methods
            self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'stop_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'record_tensor_access'))
        except ImportError:
            self.fail("Predictive memory optimization not implemented for Qwen3-4B-Instruct-2507")
        
        # Test Qwen3-Coder-30B
        try:
            from src.inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
            plugin = create_qwen3_coder_30b_plugin()
            
            # Test that the plugin has predictive memory methods
            self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'stop_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'record_tensor_access'))
        except ImportError:
            self.fail("Predictive memory optimization not implemented for Qwen3-Coder-30B")
        
        # Test Qwen3-0.6B
        try:
            from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
            plugin = create_qwen3_0_6b_plugin()
            
            # Test that the plugin has predictive memory methods
            self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'stop_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'record_tensor_access'))
        except ImportError:
            self.fail("Predictive memory optimization not implemented for Qwen3-0.6B")
        
        # Test Qwen3-Coder-Next
        try:
            from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
            plugin = create_qwen3_coder_next_plugin()
            
            # Test that the plugin has predictive memory methods
            self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'stop_predictive_memory_management'))
            self.assertTrue(hasattr(plugin, 'record_tensor_access'))
        except ImportError:
            self.fail("Predictive memory optimization not implemented for Qwen3-Coder-Next")


class TestSpecializedAttentionOptimizations(unittest.TestCase):
    """Test Specialized Attention Optimizations across all models."""
    
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

    def test_grouped_query_attention(self):
        """Test Grouped Query Attention implementation for Qwen3-4B-Instruct-2507."""
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

    def test_multi_query_attention(self):
        """Test Multi-Query Attention implementation for Qwen3-Coder-30B."""
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

    def test_sparse_attention(self):
        """Test Sparse Attention implementation for Qwen3-0.6B."""
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

    def test_sliding_window_attention(self):
        """Test Sliding Window Attention implementation for Qwen3-Coder-Next."""
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


class TestIntelligentSchedulingComponents(unittest.TestCase):
    """Test Intelligent Scheduling Components across all models."""
    
    def test_scheduling_implementations(self):
        """Test that intelligent scheduling components are properly implemented."""
        # Test GLM-4.7-Flash
        try:
            from src.inference_pio.models.glm_4_7_flash.scheduling.intelligent_scheduler import IntelligentScheduler
            scheduler = IntelligentScheduler()
            self.assertIsNotNone(scheduler)
        except ImportError:
            self.fail("Intelligent scheduler not implemented for GLM-4.7-Flash")
        
        # Test Qwen3-4B-Instruct-2507
        try:
            from src.inference_pio.models.qwen3_4b_instruct_2507.scheduling.intelligent_scheduler import IntelligentScheduler
            scheduler = IntelligentScheduler()
            self.assertIsNotNone(scheduler)
        except ImportError:
            self.fail("Intelligent scheduler not implemented for Qwen3-4B-Instruct-2507")
        
        # Test Qwen3-Coder-30B
        try:
            from src.inference_pio.models.qwen3_coder_30b.scheduling.intelligent_scheduler import IntelligentScheduler
            scheduler = IntelligentScheduler()
            self.assertIsNotNone(scheduler)
        except ImportError:
            self.fail("Intelligent scheduler not implemented for Qwen3-Coder-30B")
        
        # Test Qwen3-0.6B
        try:
            from src.inference_pio.models.qwen3_0_6b.scheduling.intelligent_scheduler import IntelligentScheduler
            scheduler = IntelligentScheduler()
            self.assertIsNotNone(scheduler)
        except ImportError:
            self.fail("Intelligent scheduler not implemented for Qwen3-0.6B")
        
        # Test Qwen3-Coder-Next
        try:
            from src.inference_pio.models.qwen3_coder_next.scheduling.intelligent_scheduler import IntelligentScheduler
            scheduler = IntelligentScheduler()
            self.assertIsNotNone(scheduler)
        except ImportError:
            self.fail("Intelligent scheduler not implemented for Qwen3-Coder-Next")


class TestResourceUsagePredictionSystems(unittest.TestCase):
    """Test Resource Usage Prediction Systems across all models."""
    
    def test_resource_prediction_implementations(self):
        """Test that resource usage prediction systems are properly implemented."""
        # Test GLM-4.7-Flash
        try:
            from src.inference_pio.models.glm_4_7_flash.resource_prediction.resource_predictor import ResourcePredictor
            predictor = ResourcePredictor()
            self.assertIsNotNone(predictor)
        except ImportError:
            self.fail("Resource predictor not implemented for GLM-4.7-Flash")
        
        # Test Qwen3-4B-Instruct-2507
        try:
            from src.inference_pio.models.qwen3_4b_instruct_2507.resource_prediction.resource_predictor import ResourcePredictor
            predictor = ResourcePredictor()
            self.assertIsNotNone(predictor)
        except ImportError:
            self.fail("Resource predictor not implemented for Qwen3-4B-Instruct-2507")
        
        # Test Qwen3-Coder-30B
        try:
            from src.inference_pio.models.qwen3_coder_30b.resource_prediction.resource_predictor import ResourcePredictor
            predictor = ResourcePredictor()
            self.assertIsNotNone(predictor)
        except ImportError:
            self.fail("Resource predictor not implemented for Qwen3-Coder-30B")
        
        # Test Qwen3-0.6B
        try:
            from src.inference_pio.models.qwen3_0_6b.resource_prediction.resource_predictor import ResourcePredictor
            predictor = ResourcePredictor()
            self.assertIsNotNone(predictor)
        except ImportError:
            self.fail("Resource predictor not implemented for Qwen3-0.6B")
        
        # Test Qwen3-Coder-Next
        try:
            from src.inference_pio.models.qwen3_coder_next.resource_prediction.resource_predictor import ResourcePredictor
            predictor = ResourcePredictor()
            self.assertIsNotNone(predictor)
        except ImportError:
            self.fail("Resource predictor not implemented for Qwen3-Coder-Next")


class TestModelIntegrationWithOptimizations(unittest.TestCase):
    """Test that all models properly integrate with all optimizations."""
    
    @patch('src.inference_pio.models.qwen3_0_6b.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_0_6b.model.AutoTokenizer.from_pretrained')
    def test_qwen3_0_6b_full_integration(self, mock_tokenizer, mock_model):
        """Test full integration of Qwen3-0.6B with all optimizations."""
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        
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
        
        # Test initialization with various optimization flags
        success = plugin.initialize(config=config)
        self.assertTrue(success)
        
        # Test that the plugin has all expected optimization methods
        self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
        self.assertTrue(hasattr(plugin, 'start_intelligent_caching'))
        self.assertTrue(hasattr(plugin, 'enable_specialized_attention'))
        
        # Test basic inference
        result = plugin.infer("Hello, test!")
        self.assertIsNotNone(result)
        
        # Test cleanup
        cleanup_success = plugin.cleanup()
        self.assertTrue(cleanup_success)
    
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained')
    def test_qwen3_4b_instruct_2507_full_integration(self, mock_tokenizer, mock_model):
        """Test full integration of Qwen3-4B-Instruct-2507 with all optimizations."""
        from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config
        
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
        
        # Test initialization with various optimization flags
        success = plugin.initialize(config=config)
        self.assertTrue(success)
        
        # Test that the plugin has all expected optimization methods
        self.assertTrue(hasattr(plugin, 'start_predictive_memory_management'))
        self.assertTrue(hasattr(plugin, 'start_intelligent_caching'))
        self.assertTrue(hasattr(plugin, 'enable_specialized_attention'))
        
        # Test basic inference
        result = plugin.infer("Hello, test!")
        self.assertIsNotNone(result)
        
        # Test cleanup
        cleanup_success = plugin.cleanup()
        self.assertTrue(cleanup_success)


def run_comprehensive_tests():
    """Run all comprehensive optimization tests."""
    print("Starting comprehensive optimization tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentCacheSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestProjectionOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossAlignmentOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossFusionComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedCompressionSystems))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveMemoryOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecializedAttentionOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentSchedulingComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceUsagePredictionSystems))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegrationWithOptimizations))
    
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
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] All comprehensive optimization tests passed!")
        print("All new implementations are working correctly across all models.")
        return True
    else:
        print("\n[ERROR] Some comprehensive optimization tests failed.")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]} - {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]} - {error[1]}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)