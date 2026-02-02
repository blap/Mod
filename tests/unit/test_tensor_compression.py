"""
Unit tests for tensor compression functionality in plugins.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin


def test_tensor_compression_setup():
    """Test tensor compression setup functionality."""
    plugin = Qwen3_0_6B_Plugin()

    # Test that tensor compression methods exist
    assert hasattr(plugin, "setup_tensor_compression")
    assert hasattr(plugin, "compress_model_weights")
    assert hasattr(plugin, "decompress_model_weights")
    assert hasattr(plugin, "compress_activations")
    assert hasattr(plugin, "get_compression_stats")
    assert hasattr(plugin, "enable_adaptive_compression")

    # Test default implementations return expected values
    assert plugin.setup_tensor_compression() is True
    assert plugin.compress_model_weights() is True
    assert plugin.decompress_model_weights() is True
    assert plugin.compress_activations() is True
    stats = plugin.get_compression_stats()
    assert isinstance(stats, dict)
    assert plugin.enable_adaptive_compression() is True


def test_tensor_compression_with_real_initialization():
    """Test tensor compression methods after plugin initialization."""
    plugin = Qwen3_0_6B_Plugin()

    # Initialize the plugin
    success = plugin.initialize()
    assert success is True

    # Test tensor compression methods still work after initialization
    assert plugin.setup_tensor_compression() is True
    assert plugin.compress_model_weights(compression_ratio=0.5) is True
    assert plugin.decompress_model_weights() is True
    assert plugin.compress_activations() is True

    stats = plugin.get_compression_stats()
    assert isinstance(stats, dict)
    assert plugin.enable_adaptive_compression() is True


def test_compression_stats_format():
    """Test that compression stats returns expected format."""
    plugin = Qwen3_0_6B_Plugin()

    stats = plugin.get_compression_stats()

    # Should be a dictionary (default implementation returns empty dict)
    assert isinstance(stats, dict)


def test_tensor_compression_lifecycle():
    """Test the complete tensor compression lifecycle."""
    plugin = Qwen3_0_6B_Plugin()

    # Setup tensor compression
    setup_result = plugin.setup_tensor_compression()
    assert setup_result is True

    # Compress model weights
    compress_result = plugin.compress_model_weights(compression_ratio=0.7)
    assert compress_result is True

    # Compress activations
    activation_compress_result = plugin.compress_activations()
    assert activation_compress_result is True

    # Enable adaptive compression
    adaptive_result = plugin.enable_adaptive_compression()
    assert adaptive_result is True

    # Get stats
    stats = plugin.get_compression_stats()
    assert isinstance(stats, dict)

    # Decompress model weights
    decompress_result = plugin.decompress_model_weights()
    assert decompress_result is True


def test_tensor_compression_methods_exist_on_interface():
    """Test that all tensor compression-related methods exist on the plugin."""
    plugin = Qwen3_0_6B_Plugin()

    compression_methods = [
        "setup_tensor_compression",
        "compress_model_weights",
        "decompress_model_weights",
        "compress_activations",
        "get_compression_stats",
        "enable_adaptive_compression",
    ]

    for method_name in compression_methods:
        assert hasattr(plugin, method_name)
        method = getattr(plugin, method_name)
        assert callable(method)


def test_weight_compression_with_different_ratios():
    """Test weight compression with different compression ratios."""
    plugin = Qwen3_0_6B_Plugin()

    # Test with different compression ratios
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    for ratio in ratios:
        result = plugin.compress_model_weights(compression_ratio=ratio)
        assert result is True


def test_compression_with_config():
    """Test compression with configuration."""
    plugin = Qwen3_0_6B_Plugin()

    # Test setup with configuration
    config = {"algorithm": "quantization", "precision": 8, "method": "symmetric"}

    result = plugin.setup_tensor_compression(**config)
    assert result is True

    compress_result = plugin.compress_model_weights(compression_ratio=0.5, **config)
    assert compress_result is True


def test_adaptive_compression_with_params():
    """Test adaptive compression with different parameters."""
    plugin = Qwen3_0_6B_Plugin()

    # Test adaptive compression with different parameters
    test_params = [
        {"target_memory_usage": 0.8},
        {"compression_threshold": 0.7},
        {"algorithm": "quantization"},
    ]

    for params in test_params:
        result = plugin.enable_adaptive_compression(**params)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
