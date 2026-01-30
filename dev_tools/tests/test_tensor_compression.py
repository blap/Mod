"""
Test suite for Tensor Compression functionality in model plugins.

This test verifies that the tensor compression system works correctly across all model plugins.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.tensor_compression import (
    TensorCompressor,
    AdaptiveTensorCompressor,
    get_tensor_compressor,
    compress_model_weights,
    decompress_model_weights
)
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

# TestTensorCompression

    """Test cases for tensor compression functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugins = [
            GLM_4_7_Plugin(),
            Qwen3_4B_Instruct_2507_Plugin(),
            Qwen3_Coder_30B_Plugin(),
            Qwen3_VL_2B_Plugin()
        ]
        
        # Create a simple test model for compression tests
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear1 = nn.Linear(100, 50)
                relu = nn.ReLU()
                linear2 = nn.Linear(50, 10)
                
            def forward(self, x):
                x = linear1(x)
                x = relu(x)
                x = linear2(x)
                return x
        
        simple_model = SimpleModel()

    def tensor_compressor_creation(self)():
        """Test that the tensor compressor can be created."""
        compressor = TensorCompressor(
            compression_method="incremental_pca",
            compression_ratio=0.5,
            max_components=25
        )
        assert_is_instance(compressor, TensorCompressor)

    def adaptive_tensor_compressor_creation(self)():
        """Test that the adaptive tensor compressor can be created."""
        compressor = AdaptiveTensorCompressor(
            compression_method="incremental_pca",
            base_compression_ratio=0.5,
            max_components=25
        )
        assert_is_instance(compressor, AdaptiveTensorCompressor)

    def global_tensor_compressor(self)():
        """Test getting global tensor compressor."""
        compressor1 = get_tensor_compressor()
        compressor2 = get_tensor_compressor()
        
        # Both should be the same instance
        assertIs(compressor1, compressor2)

    def compress_decompress_tensor(self)():
        """Test basic tensor compression and decompression."""
        original_tensor = torch.randn(100, 50)
        
        compressor = TensorCompressor(
            compression_method="incremental_pca",
            compression_ratio=0.5,
            max_components=25
        )
        
        # Compress the tensor
        compressed_data, metadata = compressor.compress_tensor(original_tensor, "test_tensor")
        
        # Decompress the tensor
        decompressed_tensor = compressor.decompress_tensor(compressed_data, metadata)
        
        # Check shapes match
        assert_equal(original_tensor.shape, decompressed_tensor.shape)

    def compress_model_weights(self)():
        """Test compressing model weights."""
        # Test compressing model weights
        compressed_model, compression_metadata = compress_model_weights(
            simple_model,
            compression_ratio=0.5,
            device="cpu"
        )
        
        # Should return a model and metadata
        assert_is_instance(compressed_model, nn.Module)
        assert_is_instance(compression_metadata, dict)
        
        # Should have compressed some parameters
        assert_greater(len(compression_metadata), 0)

    def decompress_model_weights(self)():
        """Test decompressing model weights."""
        # First compress the model
        compressed_model, compression_metadata = compress_model_weights(
            simple_model,
            compression_ratio=0.5,
            device="cpu"
        )
        
        # Then decompress
        decompressed_model = decompress_model_weights(compressed_model, compression_metadata)
        
        # Should return a model
        assert_is_instance(decompressed_model, nn.Module)
        
        # Should have the same number of parameters
        orig_params = sum(p.numel() for p in simple_model.parameters())
        decomp_params = sum(p.numel() for p in decompressed_model.parameters())
        assert_equal(orig_params, decomp_params)

    def different_compression_methods(self)():
        """Test different compression methods."""
        original_tensor = torch.randn(100, 50)
        
        methods = ["incremental_pca", "svd"]
        
        for method in methods:
            with subTest(method=method):
                compressor = TensorCompressor(
                    compression_method=method,
                    compression_ratio=0.5,
                    max_components=25
                )
                
                compressed_data, metadata = compressor.compress_tensor(original_tensor, f"test_tensor_{method}")
                decompressed_tensor = compressor.decompress_tensor(compressed_data, metadata)
                
                # Check shapes match
                assert_equal(original_tensor.shape, decompressed_tensor.shape)

    def plugin_tensor_compression_setup(self)():
        """Test that all plugins can set up tensor compression."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize(enable_tensor_compression=True)
            assert_true(success)
            
            # Check that tensor compression methods are available
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def plugin_compress_model_weights(self)():
        """Test that plugins can compress model weights."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with tensor compression enabled
            success = plugin.initialize(enable_tensor_compression=True)
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Compress model weights
            compression_success = plugin.compress_model_weights(compression_ratio=0.5)
            
            # Should return True on success
            assert_true(compression_success)

    def plugin_decompress_model_weights(self)():
        """Test that plugins can decompress model weights."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with tensor compression enabled
            success = plugin.initialize(enable_tensor_compression=True)
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # First compress
            compression_success = plugin.compress_model_weights(compression_ratio=0.5)
            assert_true(compression_success)
            
            # Then decompress
            decompression_success = plugin.decompress_model_weights()
            
            # Should return True on success
            assert_true(decompression_success)

    def plugin_compress_activations(self)():
        """Test that plugins can compress activations."""
        for plugin in plugins:
            # Initialize the plugin with tensor compression enabled
            success = plugin.initialize(enable_tensor_compression=True, enable_activation_compression=True)
            assert_true(success)
            
            # Compress activations
            activation_compression_success = plugin.compress_activations()
            
            # Should return True on success
            assert_true(activation_compression_success)

    def plugin_get_compression_stats(self)():
        """Test that plugins can report compression statistics."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)
            
            # Get compression stats (should work even without compression performed)
            stats = plugin.get_compression_stats()
            
            # Should return a dictionary with stats
            assert_is_instance(stats, dict)
            assert_in('compression_enabled', stats)
            assert_in('compressed_parameters_count', stats)

    def plugin_enable_adaptive_compression(self)():
        """Test that plugins can enable adaptive compression."""
        for plugin in plugins:
            # Initialize the plugin
            success = plugin.initialize()
            assert_true(success)
            
            # Enable adaptive compression
            adaptive_success = plugin.enable_adaptive_compression()
            
            # Should return True on success
            assert_true(adaptive_success)

    def adaptive_compression_with_memory_management(self)():
        """Test adaptive compression working with memory management."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with both optimizations
            success = plugin.initialize(
                enable_tensor_compression=True,
                enable_memory_management=True,
                enable_tensor_paging=True
            )
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Compress model weights
            compression_success = plugin.compress_model_weights(compression_ratio=0.5)
            assert_true(compression_success)
            
            # Get compression stats
            stats = plugin.get_compression_stats()
            assert_in('compression_enabled', stats)
            
            # Get memory stats to verify memory management is also working
            memory_stats = plugin.get_memory_stats()
            assert_in('system_memory_percent', memory_stats)

    def tensor_compression_with_other_optimizations(self)():
        """Test tensor compression working with other optimizations."""
        for plugin in plugins[:1]:  # Test with first plugin to avoid long execution
            # Initialize the plugin with tensor compression and other optimizations
            success = plugin.initialize(
                enable_tensor_compression=True,
                enable_kernel_fusion=True,
                enable_adaptive_batching=True,
                enable_disk_offloading=True,
                enable_model_surgery=True,
                enable_activation_offloading=True
            )
            assert_true(success)
            
            # Load the model if not already loaded
            if plugin._model is None:
                plugin.load_model()
            
            # Compress model weights
            compression_success = plugin.compress_model_weights(compression_ratio=0.5)
            assert_true(compression_success)
            
            # Get compression stats
            stats = plugin.get_compression_stats()
            assert_in('compression_enabled', stats)

    def adaptive_compression_under_memory_pressure(self)():
        """Test adaptive compression responding to memory pressure."""
        compressor = AdaptiveTensorCompressor(
            compression_method="incremental_pca",
            base_compression_ratio=0.5,
            max_components=25
        )
        
        # Simulate high memory pressure
        original_ratio = compressor.compression_ratio
        compressor.update_compression_ratio(0.8)  # Higher compression under pressure
        
        assert_not_equal(original_ratio, compressor.compression_ratio)

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up any resources used by the plugins
        for plugin in plugins:
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)