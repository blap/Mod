"""
Comprehensive Test Suite for Qwen3-VL-2B Model with Intelligent Multimodal Caching

This module provides a comprehensive test suite for the Qwen3-VL-2B model implementation,
including all optimizations and the new intelligent multimodal caching system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)
from unittest.mock import patch, MagicMock


import torch
import tempfile
import os

class TestQwen3VL2BCompleteImplementation:
    """Comprehensive test cases for Qwen3-VL-2B model implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Import here to avoid circular dependencies
        from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        config = Qwen3VL2BConfig()
        config.enable_intelligent_multimodal_caching = True
        config.intelligent_multimodal_cache_size_gb = 0.1  # Small cache for testing
        return config

    def config_attributes(self):
        """Test that all required configuration attributes are present."""
        from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

        config = Qwen3VL2BConfig()

        # Check that caching-related attributes exist
        assert_true(hasattr(config, 'enable_intelligent_multimodal_caching'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_size_gb'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_eviction_policy'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_enable_similarity'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_similarity_threshold'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_default_ttl'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_enable_compression'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_compression_ratio'))
        assert_true(hasattr(config, 'intelligent_multimodal_cache_predictive_eviction'))

        # Check default values
        assert_true(config.enable_intelligent_multimodal_caching)
        assert_equal(config.intelligent_multimodal_cache_size_gb, 2.0)
        assert_equal(config.intelligent_multimodal_cache_eviction_policy, "predictive")
        assert_true(config.intelligent_multimodal_cache_enable_similarity)
        assert_equal(config.intelligent_multimodal_cache_similarity_threshold, 0.85)

    def visual_compression_config_attributes(self):
        """Test that visual compression configuration attributes are present."""
        from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

        config = Qwen3VL2BConfig()

        # Check that visual compression-related attributes exist
        assert_true(hasattr(config, 'enable_visual_resource_compression'))
        assert_true(hasattr(config, 'visual_compression_method'))
        assert_true(hasattr(config, 'visual_compression_ratio'))
        assert_true(hasattr(config, 'visual_quantization_bits'))
        assert_true(hasattr(config, 'visual_quantization_method'))
        assert_true(hasattr(config, 'visual_enable_compression_cache'))
        assert_true(hasattr(config, 'visual_compression_cache_size'))
        assert_true(hasattr(config, 'visual_enable_adaptive_compression'))
        assert_true(hasattr(config, 'visual_compression_target_size'))
        assert_true(hasattr(config, 'visual_compression_quality_factor'))
        assert_true(hasattr(config, 'visual_compression_encoding_format'))
        assert_true(hasattr(config, 'visual_compression_cache_ttl'))
        assert_true(hasattr(config, 'visual_enable_progressive_compression'))

        # Check default values
        assert_true(config.enable_visual_resource_compression)
        assert_equal(config.visual_compression_method, 'quantization')
        assert_equal(config.visual_compression_ratio, 0.5)
        assert_equal(config.visual_quantization_bits, 8)
        assert_equal(config.visual_quantization_method, 'linear')
        assert_true(config.visual_enable_compression_cache)
        assert_equal(config.visual_compression_cache_size, 1.0)
        assert_true(config.visual_enable_adaptive_compression)

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def model_creation_with_intelligent_caching(self, mock_processor, mock_tokenizer, mock_model):
        """Test Qwen3-VL-2B model creation with intelligent multimodal caching enabled."""
        from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
        from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

        # Mock the model loading
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        # Create config with caching enabled
        config = Qwen3VL2BConfig()
        config.enable_intelligent_multimodal_caching = True
        config.intelligent_multimodal_cache_size_gb = 0.1  # Small cache for testing

        # Create model with caching enabled
        model = Qwen3VL2BModel(config)

        # Verify that caching manager is initialized
        assert_is_not_none(model._caching_manager)
        assert_true(hasattr(model, '_caching_manager'))
        assert_true(hasattr(model, 'cache_text_input'))
        assert_true(hasattr(model, 'cache_image_input'))
        assert_true(hasattr(model, 'get_cached_text_input'))
        assert_true(hasattr(model, 'get_cached_image_input'))
        assert_true(hasattr(model, 'find_similar_text'))

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def plugin_creation_with_intelligent_caching(self, mock_processor, mock_tokenizer, mock_model):
        """Test Qwen3-VL-2B plugin creation with intelligent multimodal caching enabled."""
        from inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin

        # Mock the model loading
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        # Create plugin
        plugin = Qwen3_VL_2B_Instruct_Plugin()

        # Initialize with caching enabled
        success = plugin.initialize(
            enable_intelligent_multimodal_caching=True,
            intelligent_multimodal_cache_size_gb=0.1,
            device="cpu"
        )

        assert_true(success)
        assert_true(hasattr(plugin, '_caching_manager'))

        # Verify that the model has caching capabilities
        if hasattr(plugin, '_model') and plugin._model:
            assert_true(hasattr(plugin._model, 'cache_text_input'))

    def intelligent_caching_manager_creation(self):
        """Test creation of intelligent caching manager."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager

        # Create caching manager
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.85
        )

        assert_is_not_none(caching_manager)
        assert_is_not_none(caching_manager.cache)
        assert_true(hasattr(caching_manager, 'cache_text_input'))
        assert_true(hasattr(caching_manager, 'cache_image_input'))
        assert_true(hasattr(caching_manager, 'get_cached_text_input'))
        assert_true(hasattr(caching_manager, 'get_cached_image_input'))
        assert_true(hasattr(caching_manager, 'find_similar_text'))
        assert_true(hasattr(caching_manager, 'find_similar_image'))

    def caching_functionality(self):
        """Test basic caching functionality."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager
        from PIL import Image
        import torch

        # Create caching manager
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.85
        )

        # Test text caching
        text = "This is a test text for caching."
        text_tensor = torch.randn(1, 10, 768)  # Using a standard hidden size

        caching_manager.cache_text_input(text, text_tensor)
        cached_result = caching_manager.get_cached_text_input(text)

        assert_is_not_none(cached_result)
        assert_true(torch.equal(text_tensor, cached_result))

        # Test image caching
        image = Image.new('RGB', (224, 224), color='red')
        image_tensor = torch.randn(1, 197, 768)  # Using a standard hidden size

        caching_manager.cache_image_input(image, image_tensor)
        cached_image_result = caching_manager.get_cached_image_input(image)

        assert_is_not_none(cached_image_result)
        assert_true(torch.equal(image_tensor, cached_image_result))

    def caching_with_different_eviction_policies(self):
        """Test caching with different eviction policies."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager, CacheEvictionPolicy

        # Test LRU policy
        lru_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            eviction_policy=CacheEvictionPolicy.LRU,
            enable_similarity_caching=False
        )

        assert_equal(lru_manager.cache.eviction_policy, CacheEvictionPolicy.LRU)

        # Test predictive policy
        pred_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            eviction_policy=CacheEvictionPolicy.PREDICTIVE,
            enable_similarity_caching=False
        )

        assert_equal(pred_manager.cache.eviction_policy, CacheEvictionPolicy.PREDICTIVE)

    def caching_statistics(self):
        """Test caching statistics reporting."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager
        import torch

        # Create caching manager
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.85
        )

        # Get initial stats
        initial_stats = caching_manager.get_cache_stats()

        # Add some data to cache
        text = "Test text for statistics."
        text_tensor = torch.randn(1, 5, 768)  # Using a standard hidden size
        caching_manager.cache_text_input(text, text_tensor)

        # Get updated stats
        updated_stats = caching_manager.get_cache_stats()

        # Verify stats changed appropriately
        assert_greater(updated_stats['active_entries'], initial_stats['active_entries'])
        assert_greater(updated_stats['current_size_bytes'], initial_stats['current_size_bytes'])
        assert_equal(updated_stats['eviction_policy'], 'predictive')
        assert_true(updated_stats['similarity_caching_enabled'])

    def caching_integration_with_model(self):
        """Test that caching integrates properly with the model."""
        from inference_pio.common.intelligent_multimodal_caching import apply_intelligent_multimodal_caching_to_model, create_qwen3_vl_intelligent_caching_manager
        import torch.nn as nn

        # Create a mock model
        mock_model = nn.Module()

        # Create caching manager
        caching_manager = create_qwen3_vl_intelligent_caching_manager(cache_size_gb=0.1)

        # Apply caching to model
        enhanced_model = apply_intelligent_multimodal_caching_to_model(mock_model, caching_manager)

        # Verify that the enhanced model has caching methods
        assert_true(hasattr(enhanced_model, 'cache_text_input'))
        assert_true(hasattr(enhanced_model, 'cache_image_input'))
        assert_true(hasattr(enhanced_model, 'get_cached_text_input'))

    def caching_clear_functionality(self):
        """Test clearing the cache functionality."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager
        import torch

        # Create caching manager
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.85
        )

        # Add some data to cache
        text = "Test text for clearing."
        text_tensor = torch.randn(1, 5, 768)  # Using a standard hidden size
        caching_manager.cache_text_input(text, text_tensor)

        # Verify cache has entries
        stats_before = caching_manager.get_cache_stats()
        assert_greater(stats_before['active_entries'], 0)

        # Clear the cache
        caching_manager.clear_cache()

        # Verify cache is empty
        stats_after = caching_manager.get_cache_stats()
        assert_equal(stats_after['active_entries'], 0)
        assert_equal(stats_after['current_size_bytes'], 0)

    def caching_compression_functionality(self):
        """Test caching with compression enabled."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager
        import torch

        # Create caching manager with compression
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.85,
            enable_compression=True,
            compression_ratio=0.5
        )

        # Add some data to cache
        text = "Test text for compression."
        text_tensor = torch.randn(1, 10, 768)  # Using a standard hidden size
        caching_manager.cache_text_input(text, text_tensor)

        # Verify that compression was applied by checking stats
        stats = caching_manager.get_cache_stats()
        assert_true(stats['compression_enabled'])

    def caching_ttl_functionality(self):
        """Test caching with TTL (Time-To-Live) enabled."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager
        import torch

        # Create caching manager with TTL
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.85,
            enable_ttl=True,
            default_ttl=1.0  # 1 second TTL for testing
        )

        # Add some data to cache
        text = "Test text for TTL."
        text_tensor = torch.randn(1, 5, 768)  # Using a standard hidden size
        caching_manager.cache_text_input(text, text_tensor)

        # Verify that TTL was applied by checking stats
        stats = caching_manager.get_cache_stats()
        assert_true(stats['enable_ttl'])

        # Wait for TTL to expire and verify entry is removed
        import time
        time.sleep(1.1)  # Wait slightly more than TTL

        # Clear expired entries
        caching_manager.cache.clear_expired()

        # Check that cache is now empty
        stats_after_expiry = caching_manager.get_cache_stats()
        assert_equal(stats_after_expiry['active_entries'], 0)

    def caching_similarity_detection(self):
        """Test similarity detection in caching."""
        from inference_pio.common.intelligent_multimodal_caching import Qwen3VL2BIntelligentCachingManager
        import torch

        # Create caching manager with similarity detection
        caching_manager = Qwen3VL2BIntelligentCachingManager(
            cache_size_gb=0.1,
            enable_similarity_caching=True,
            similarity_threshold=0.8,
            enable_ttl=False  # Disable TTL for this test
        )

        # Add a text to cache
        original_text = "This is the original text for similarity testing."
        original_tensor = torch.randn(1, 5, 768)  # Using a standard hidden size
        caching_manager.cache_text_input(original_text, original_tensor)

        # Try to find a similar text (exact match)
        similar_text = "This is the original text for similarity testing."
        result = caching_manager.find_similar_text(similar_text)

        # Verify that similar text was found
        assert_is_not_none(result)
        assert_equal(len(result), 2)  # Should return (key, data) tuple
        if result:
            _, cached_tensor = result
            assert_true(torch.equal(original_tensor, cached_tensor))

    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def model_creation_with_visual_compression(self, mock_processor, mock_tokenizer, mock_model):
        """Test Qwen3-VL-2B model creation with visual resource compression enabled."""
        from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
        from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

        # Mock the model loading
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        # Create config with visual compression enabled
        config = Qwen3VL2BConfig()
        config.enable_visual_resource_compression = True
        config.visual_compression_method = 'quantization'
        config.visual_quantization_bits = 8

        # Create model with visual compression enabled
        model = Qwen3VL2BModel(config)

        # Verify that visual compressor is initialized
        assert_is_not_none(model._visual_compressor)
        assert_true(hasattr(model, '_visual_compressor'))

    def visual_compression_functionality(self):
        """Test basic visual compression functionality."""
        from inference_pio.models.qwen3_vl_2b.visual_resource_compression import (
            VisualCompressionConfig,
            VisualResourceCompressor,
            CompressionMethod
        )
        import torch

        # Create visual compression config
        compression_config = VisualCompressionConfig(
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_bits=8,
            enable_compression_cache=True
        )

        # Create compressor
        compressor = VisualResourceCompressor(compression_config)

        # Create a sample image tensor
        image_tensor = torch.randn(1, 3, 224, 224)

        # Compress the image
        compressed, metadata = compressor.compress(image_tensor, key="test_image")

        # Verify compression worked
        assert_is_not_none(compressed)
        assert_is_not_none(metadata)
        # Note: We can't check dtype directly as it depends on implementation

        # Decompress the image
        decompressed = compressor.decompress(compressed, metadata)

        # Verify decompression worked
        assert_is_not_none(decompressed)
        assert_equal(decompressed.shape, image_tensor.shape)

        # Check that values are approximately the same (allowing for quantization error)
        mse = torch.mean((image_tensor - decompressed) ** 2)
        assert_less(mse.item(), 0.1)  # Reasonable threshold for quantization error

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any cached data
        import gc
        gc.collect()

def run_complete_tests():
    """Run all Qwen3-VL-2B tests including intelligent caching tests."""
    print("Running Qwen3-VL-2B Complete Implementation Tests with Intelligent Multimodal Caching...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3VL2BCompleteImplementation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Results: {result.testsRun} tests run")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_complete_tests()
    if not success:
        exit(1)