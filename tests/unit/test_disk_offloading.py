"""
Unit tests for disk offloading functionality in plugins.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin
from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin


def test_disk_offloading_setup():
    """Test disk offloading setup functionality."""
    plugin = Qwen3_0_6B_Plugin()

    # Test that disk offloading methods exist
    assert hasattr(plugin, "setup_disk_offloading")
    assert hasattr(plugin, "enable_disk_offloading")
    assert hasattr(plugin, "offload_model_parts")
    assert hasattr(plugin, "predict_model_part_access")
    assert hasattr(plugin, "get_offloading_stats")

    # Test default implementations return expected values
    assert plugin.setup_disk_offloading() is True
    assert plugin.enable_disk_offloading() is True
    assert plugin.offload_model_parts() is True
    predicted_access = plugin.predict_model_part_access()
    assert isinstance(predicted_access, dict)
    stats = plugin.get_offloading_stats()
    assert isinstance(stats, dict)


def test_disk_offloading_with_real_initialization():
    """Test disk offloading methods after plugin initialization."""
    # Use the centralized utility to create and initialize the plugin
    plugin = create_and_initialize_plugin(Qwen3_0_6B_Plugin)

    # Test disk offloading methods still work after initialization
    assert plugin.setup_disk_offloading() is True
    assert plugin.enable_disk_offloading() is True
    assert plugin.offload_model_parts() is True

    predicted = plugin.predict_model_part_access()
    assert isinstance(predicted, dict)

    stats = plugin.get_offloading_stats()
    assert isinstance(stats, dict)


def test_model_part_prediction_format():
    """Test that model part prediction returns expected format."""
    plugin = Qwen3_0_6B_Plugin()

    prediction = plugin.predict_model_part_access()

    # Should be a dictionary mapping model part names to access probabilities
    assert isinstance(prediction, dict)


def test_disk_offloading_stats_format():
    """Test that disk offloading stats returns expected format."""
    plugin = Qwen3_0_6B_Plugin()

    stats = plugin.get_offloading_stats()

    # Should be a dictionary (default implementation returns empty dict)
    assert isinstance(stats, dict)


def test_disk_offloading_lifecycle():
    """Test the complete disk offloading lifecycle."""
    plugin = Qwen3_0_6B_Plugin()

    # Setup disk offloading
    setup_result = plugin.setup_disk_offloading()
    assert setup_result is True

    # Enable disk offloading
    enable_result = plugin.enable_disk_offloading()
    assert enable_result is True

    # Offload model parts
    offload_result = plugin.offload_model_parts()
    assert offload_result is True

    # Predict model part access
    predictions = plugin.predict_model_part_access()
    assert isinstance(predictions, dict)

    # Get stats
    stats = plugin.get_offloading_stats()
    assert isinstance(stats, dict)


def test_disk_offloading_methods_exist_on_interface():
    """Test that all disk offloading-related methods exist on the plugin."""
    plugin = Qwen3_0_6B_Plugin()

    disk_methods = [
        "setup_disk_offloading",
        "enable_disk_offloading",
        "offload_model_parts",
        "predict_model_part_access",
        "get_offloading_stats",
    ]

    for method_name in disk_methods:
        assert hasattr(plugin, method_name)
        method = getattr(plugin, method_name)
        assert callable(method)


def test_model_part_predictions_with_params():
    """Test model part predictions with different parameters."""
    plugin = Qwen3_0_6B_Plugin()

    # Test prediction with different parameters
    test_params = [
        {"layer": 1, "module": "attention"},
        {"layer": 5, "module": "mlp"},
        {"batch_size": 2, "seq_len": 128},
    ]

    for params in test_params:
        prediction = plugin.predict_model_part_access(**params)
        assert isinstance(prediction, dict)


def test_disk_offloading_with_config():
    """Test disk offloading with configuration."""
    plugin = Qwen3_0_6B_Plugin()

    # Test setup with configuration
    config = {"strategy": "lru", "threshold": 0.8, "device": "cpu"}

    result = plugin.setup_disk_offloading(**config)
    assert result is True

    enable_result = plugin.enable_disk_offloading(**config)
    assert enable_result is True


if __name__ == "__main__":
    pytest.main([__file__])
