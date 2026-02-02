"""
Unit tests for pipeline functionality in plugins.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin


def test_pipeline_setup():
    """Test pipeline setup functionality."""
    plugin = Qwen3_0_6B_Plugin()

    # Test that pipeline methods exist
    assert hasattr(plugin, "setup_pipeline")
    assert hasattr(plugin, "execute_pipeline")
    assert hasattr(plugin, "create_pipeline_stages")
    assert hasattr(plugin, "get_pipeline_manager")
    assert hasattr(plugin, "get_pipeline_stats")

    # Test default implementations return expected values
    assert plugin.setup_pipeline() is True
    assert plugin.execute_pipeline("test data") == plugin.infer("test data")
    assert plugin.create_pipeline_stages() == []
    assert plugin.get_pipeline_manager() is None
    pipeline_stats = plugin.get_pipeline_stats()
    assert isinstance(pipeline_stats, dict)
    assert "pipeline_enabled" in pipeline_stats


def test_pipeline_with_real_initialization():
    """Test pipeline methods after plugin initialization."""
    plugin = Qwen3_0_6B_Plugin()

    # Initialize the plugin
    success = plugin.initialize()
    assert success is True

    # Test pipeline methods still work after initialization
    assert plugin.setup_pipeline() is True
    result = plugin.execute_pipeline("test prompt")
    assert result is not None  # Should return some result
    assert plugin.create_pipeline_stages() == []
    assert plugin.get_pipeline_manager() is None
    stats = plugin.get_pipeline_stats()
    assert isinstance(stats, dict)


def test_pipeline_execution():
    """Test pipeline execution with different inputs."""
    plugin = Qwen3_0_6B_Plugin()

    # Initialize the plugin
    success = plugin.initialize()
    assert success is True

    # Test pipeline execution with string input
    result1 = plugin.execute_pipeline("Hello, world!")
    assert result1 is not None

    # Test pipeline execution with config
    result2 = plugin.execute_pipeline("Test input", {"batch_size": 1})
    assert result2 is not None


def test_pipeline_stats_format():
    """Test that pipeline stats returns expected format."""
    plugin = Qwen3_0_6B_Plugin()

    pipeline_stats = plugin.get_pipeline_stats()

    # Should be a dictionary with expected keys
    assert isinstance(pipeline_stats, dict)
    assert "pipeline_enabled" in pipeline_stats
    assert "num_stages" in pipeline_stats
    assert "checkpoint_directory" in pipeline_stats
    assert "pipeline_performance" in pipeline_stats


def test_pipeline_lifecycle():
    """Test the complete pipeline lifecycle."""
    plugin = Qwen3_0_6B_Plugin()

    # Setup pipeline
    setup_result = plugin.setup_pipeline()
    assert setup_result is True

    # Create stages (should return empty list by default)
    stages = plugin.create_pipeline_stages()
    assert isinstance(stages, list)

    # Execute pipeline
    result = plugin.execute_pipeline("test data")
    assert result is not None

    # Get pipeline stats
    stats = plugin.get_pipeline_stats()
    assert isinstance(stats, dict)


def test_pipeline_methods_exist_on_interface():
    """Test that all pipeline-related methods exist on the plugin."""
    plugin = Qwen3_0_6B_Plugin()

    pipeline_methods = [
        "setup_pipeline",
        "execute_pipeline",
        "create_pipeline_stages",
        "get_pipeline_manager",
        "get_pipeline_stats",
    ]

    for method_name in pipeline_methods:
        assert hasattr(plugin, method_name)
        method = getattr(plugin, method_name)
        assert callable(method)


def test_pipeline_with_config():
    """Test pipeline execution with configuration."""
    plugin = Qwen3_0_6B_Plugin()

    # Initialize the plugin
    success = plugin.initialize()
    assert success is True

    # Test pipeline execution with various configurations
    configs = [
        {"batch_size": 1},
        {"max_length": 100},
        {"temperature": 0.7},
        {"do_sample": True},
    ]

    for config in configs:
        result = plugin.execute_pipeline("Test with config", config)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
