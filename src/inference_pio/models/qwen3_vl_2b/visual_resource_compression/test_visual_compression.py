"""
Tests for Visual Resource Compression System for Qwen3-VL-2B Model

This module contains comprehensive tests for the visual resource compression system
implemented for the Qwen3-VL-2B model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import numpy as np
from src.inference_pio.models.qwen3_vl_2b.visual_resource_compression import (
    CompressionMethod,
    VisualCompressionConfig,
    VisualResourceCompressor,
    VisualFeatureCompressor,
    create_visual_compressor,
    apply_visual_compression_to_model
)
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

# TestVisualCompressionConfig

    """Test cases for VisualCompressionConfig."""

    def default_config_values(self)():
        """Test that default configuration values are set correctly."""
        config = VisualCompressionConfig()

        assert_equal(config.compression_method, CompressionMethod.QUANTIZATION)
        assert_equal(config.compression_ratio, 0.5)
        assert_equal(config.quantization_bits, 8)
        assert_equal(config.quantization_method, "linear")
        assert_true(config.enable_compression_cache)
        assert_equal(config.compression_cache_size)
        assert_true(config.enable_adaptive_compression)

    def custom_config_values(self)():
        """Test that custom configuration values are set correctly."""
        config = VisualCompressionConfig(
            compression_method=CompressionMethod.PCA,
            compression_ratio=0.7,
            pca_components_ratio=0.8,
            quantization_bits=4,
            enable_compression_cache=False,
            compression_cache_size=500
        )

        assert_equal(config.compression_method, CompressionMethod.PCA)
        assert_equal(config.compression_ratio, 0.7)
        assert_equal(config.pca_components_ratio, 0.8)
        assert_equal(config.quantization_bits, 4)
        assert_false(config.enable_compression_cache)
        assert_equal(config.compression_cache_size)

# TestVisualResourceCompressor

    """Test cases for VisualResourceCompressor."""

    def setup_helper():
        """Set up test fixtures."""
        config = VisualCompressionConfig(
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_bits=8,
            quantization_method="linear",
            enable_compression_cache=True,
            compression_cache_size=10
        )
        compressor = VisualResourceCompressor(config)

    def compress_and_decompress_quantization(self)():
        """Test compression and decompression with quantization."""
        # Create a sample tensor
        x = torch.randn(2, 3, 224, 224)  # Sample image tensor
        
        # Compress
        compressed, metadata = compressor.compress(x, key="test_key")
        
        # Check that compressed tensor is smaller in terms of conceptual size
        assert_equal(type(compressed), torch.Tensor)
        assert_equal(compressed.dtype, torch.uint8)  # Quantized to 8-bit
        
        # Decompress
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check that decompressed tensor has same shape as original
        assert_equal(decompressed.shape, x.shape)
        
        # Check that values are approximately the same (allowing for quantization error)
        mse = torch.mean((x - decompressed) ** 2)
        # The MSE should be reasonably small for 8-bit quantization
        assert_less(mse.item(), 0.1)

    def compress_and_decompress_with_cache(self)():
        """Test compression and decompression with caching."""
        # Create a sample tensor
        x = torch.randn(2, 4, 16, 16)
        
        # Compress with key (should cache)
        compressed1, metadata1 = compressor.compress(x, key="cached_key")
        
        # Compress again with same key (should retrieve from cache)
        compressed2, metadata2 = compressor.compress(x, key="cached_key")
        
        # The results should be identical
        assert_true(torch.equal(compressed1))
        assert_equal(metadata1['original_shape'], metadata2['original_shape'])

    def compress_without_cache(self)():
        """Test compression without caching."""
        # Create a sample tensor
        x = torch.randn(1, 512)
        
        # Compress without key (should not cache)
        compressed, metadata = compressor.compress(x)
        
        # Check that compression worked
        assert_equal(type(compressed), torch.Tensor)
        
        # Decompress
        decompressed = compressor.decompress(compressed, metadata)
        
        # Check that decompressed tensor has same shape as original
        assert_equal(decompressed.shape, x.shape)

    def forward_pass(self)():
        """Test forward pass of the compressor."""
        # Create a sample tensor
        x = torch.randn(1, 3, 32, 32)
        
        # Forward pass (compress and decompress)
        output = compressor(x, key="forward_test")
        
        # Check that output has same shape as input
        assert_equal(output.shape, x.shape)
        
        # Check that values are approximately the same
        mse = torch.mean((x - output) ** 2)
        assert_less(mse.item(), 0.1)

# TestVisualFeatureCompressor

    """Test cases for VisualFeatureCompressor."""

    def setup_helper():
        """Set up test fixtures."""
        compression_config = VisualCompressionConfig(
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_bits=8,
            enable_compression_cache=True
        )
        
        model_config = Qwen3VL2BConfig()
        model_config.hidden_size = 512
        model_config.vision_hidden_size = 1024
        
        feature_compressor = VisualFeatureCompressor(
            compression_config, 
            model_config
        )

    def compress_features(self)():
        """Test compressing visual features."""
        # Create sample features
        features = torch.randn(2, 197, 1024)  # Batch of patch embeddings
        
        # Compress features
        compressed, metadata = feature_compressor.compress_features(
            features, 
            layer_name="vision_encoder", 
            feature_type="vision"
        )
        
        # Check that compression worked
        assert_equal(type(compressed), torch.Tensor)
        assert_in('layer_name', metadata)
        assert_in('feature_type', metadata)
        assert_in('compression_ratio', metadata)
        
        # Check that metadata contains expected values
        assert_equal(metadata['layer_name'], 'vision_encoder')
        assert_equal(metadata['feature_type'], 'vision')

    def decompress_features(self)():
        """Test decompressing visual features."""
        # Create sample features
        features = torch.randn(1, 50, 512)
        
        # Compress features
        compressed, metadata = feature_compressor.compress_features(
            features,
            layer_name="mlp_layer",
            feature_type="mlp"
        )
        
        # Decompress features
        decompressed = feature_compressor.decompress_features(compressed, metadata)
        
        # Check that decompressed features have same shape as original
        assert_equal(decompressed.shape, features.shape)

    def get_compression_stats(self)():
        """Test getting compression statistics."""
        # Create sample features and compress them
        features1 = torch.randn(1, 25, 256)
        features2 = torch.randn(1, 50, 512)
        
        feature_compressor.compress_features(features1, "layer1", "vision")
        feature_compressor.compress_features(features2, "layer2", "mlp")
        
        # Get statistics
        stats = feature_compressor.get_compression_stats()
        
        # Check that statistics are reasonable
        assertGreaterEqual(stats['compression_calls'], 2)
        assertGreaterEqual(stats['avg_compression_ratio'], 0.0)
        assert_equal(len(stats['compression_ratios']), 2)

# TestFactoryFunctions

    """Test cases for factory functions."""

    def setup_helper():
        """Set up test fixtures."""
        compression_config = VisualCompressionConfig(
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_bits=8
        )
        
        model_config = Qwen3VL2BConfig()
        model_config.hidden_size = 256

    def create_visual_compressor(self)():
        """Test creating visual compressor."""
        compressor = create_visual_compressor(compression_config, model_config)
        
        assert_is_instance(compressor, VisualFeatureCompressor)
        assert_equal(compressor.config, compression_config)
        assert_equal(compressor.model_config, model_config)

    def apply_visual_compression_to_model(self)():
        """Test applying visual compression to a model."""
        # Create a mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                dummy_param = nn.Parameter(torch.randn(10))

        model = MockModel()
        
        # Apply visual compression
        compressed_model = apply_visual_compression_to_model(
            model, 
            model_config, 
            compression_config
        )
        
        # Check that the model now has a visual compressor
        assert_true(hasattr(compressed_model))
        assert_is_instance(compressed_model.visual_compressor, VisualFeatureCompressor)

# TestIntegrationWithQwen3VL2B

    """Integration tests for visual compression with Qwen3-VL-2B model."""

    def setup_helper():
        """Set up test fixtures."""
        model_config = Qwen3VL2BConfig()
        model_config.hidden_size = 256
        model_config.vision_hidden_size = 512
        
        compression_config = VisualCompressionConfig(
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_bits=8,
            enable_compression_cache=True
        )

    @unittest.skip("Skipping integration test that requires actual model loading")
    def integration_with_real_model(self)():
        """Test integration with a real Qwen3-VL-2B model structure."""
        # This test would require the actual model implementation
        # For now, we'll just verify the configuration works
        assert_true(hasattr(model_config))
        assert_true(hasattr(compression_config))

if __name__ == '__main__':
    print("Running Visual Resource Compression tests...")
    run_tests(test_functions)