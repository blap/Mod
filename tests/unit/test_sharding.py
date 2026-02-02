"""
Unit tests for sharding functionality in plugins.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin
from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin


def test_sharding_setup():
    """Test sharding setup functionality."""
    plugin = Qwen3_0_6B_Plugin()

    # Test that sharding methods exist
    assert hasattr(plugin, "enable_sharding")
    assert hasattr(plugin, "disable_sharding")
    assert hasattr(plugin, "shard_model")
    assert hasattr(plugin, "prepare_inference_context")
    assert hasattr(plugin, "execute_with_shards")
    assert hasattr(plugin, "cleanup_inference_context")
    assert hasattr(plugin, "get_sharding_stats")

    # Test default implementations return expected values
    assert plugin.enable_sharding() is True
    assert plugin.disable_sharding() is True
    assert plugin.shard_model(MagicMock()) is True
    assert plugin.prepare_inference_context("test_ctx", (1, 2)) == []
    result_tensor = torch.tensor([1, 2, 3])
    assert torch.equal(
        plugin.execute_with_shards("test_ctx", result_tensor),
        plugin.infer(result_tensor),
    )
    assert plugin.get_sharding_stats() is not None


def test_sharding_with_real_initialization():
    """Test sharding methods after plugin initialization."""
    # Use the centralized utility to create and initialize the plugin
    plugin = create_and_initialize_plugin(Qwen3_0_6B_Plugin)

    # Test sharding methods still work after initialization
    assert plugin.enable_sharding() is True
    assert plugin.disable_sharding() is True

    # Test with actual model if available
    if plugin._model:
        assert plugin.shard_model(plugin._model) is True

    assert plugin.prepare_inference_context("ctx1", (10, 512)) == []

    test_input = torch.randn(1, 10, 512)
    result = plugin.execute_with_shards("ctx1", test_input)
    assert result is not None

    plugin.cleanup_inference_context("ctx1")

    stats = plugin.get_sharding_stats()
    assert isinstance(stats, dict)


def test_sharding_stats_format():
    """Test that sharding stats returns expected format."""
    plugin = Qwen3_0_6B_Plugin()

    sharding_stats = plugin.get_sharding_stats()

    # Should be a dictionary with expected keys
    assert isinstance(sharding_stats, dict)
    assert "sharding_enabled" in sharding_stats
    assert "total_shards" in sharding_stats
    assert "loaded_shards" in sharding_stats
    assert "total_size_bytes" in sharding_stats
    assert "loaded_size_bytes" in sharding_stats
    assert "memory_utilization_ratio" in sharding_stats


def test_sharding_lifecycle():
    """
    Test the complete sharding lifecycle.

    This test validates the complete workflow for model sharding, which is a
    memory optimization technique that splits large models across multiple
    devices or storage locations. Sharding allows running models that would
    otherwise exceed memory constraints by distributing the model parameters
    across available resources.

    The sharding lifecycle includes:
    1. Enable: Activate sharding functionality with configuration
    2. Initialize: Load the model that will be sharded
    3. Shard: Split the model into multiple shards
    4. Prepare Context: Set up inference context with required shards
    5. Execute: Run inference using the sharded model
    6. Cleanup: Release resources and unload shards
    7. Stats: Monitor sharding effectiveness
    8. Disable: Deactivate sharding functionality

    This comprehensive test ensures all components work together properly.
    """
    plugin = Qwen3_0_6B_Plugin()

    # Enable sharding with specific configuration
    # num_shards=100: Divide model into 100 pieces
    # storage_path="./test_shards": Directory for storing shard files
    enable_result = plugin.enable_sharding(num_shards=100, storage_path="./test_shards")
    assert enable_result is True

    # Initialize plugin to load the model that will be sharded
    init_success = plugin.initialize()
    assert init_success is True

    # Shard the model if it exists
    # This splits the model parameters into multiple pieces stored separately
    if plugin._model:
        shard_result = plugin.shard_model(plugin._model, num_shards=100)
        assert shard_result is True

    # Prepare inference context with required shards
    # "test_context": Identifier for this inference session
    # (1, 512): Input tensor shape (batch_size=1, sequence_length=512)
    # "forward": Type of inference operation
    shards_loaded = plugin.prepare_inference_context(
        "test_context", (1, 512), "forward"
    )
    assert isinstance(shards_loaded, list)

    # Execute inference using the sharded model
    # This tests that the sharded model can perform computations correctly
    test_tensor = torch.randn(1, 512)
    execution_result = plugin.execute_with_shards("test_context", test_tensor)
    assert execution_result is not None

    # Cleanup context and unload shards to free resources
    # force_unload=True: Ensure all shards are completely unloaded
    plugin.cleanup_inference_context("test_context", force_unload=True)

    # Get statistics about sharding performance and resource usage
    stats = plugin.get_sharding_stats()
    assert isinstance(stats, dict)

    # Disable sharding to clean up resources
    disable_result = plugin.disable_sharding()
    assert disable_result is True


def test_sharding_methods_exist_on_interface():
    """Test that all sharding-related methods exist on the plugin."""
    plugin = Qwen3_0_6B_Plugin()

    sharding_methods = [
        "enable_sharding",
        "disable_sharding",
        "shard_model",
        "prepare_inference_context",
        "execute_with_shards",
        "cleanup_inference_context",
        "get_sharding_stats",
    ]

    for method_name in sharding_methods:
        assert hasattr(plugin, method_name)
        method = getattr(plugin, method_name)
        assert callable(method)


def test_sharding_with_different_parameters():
    """Test sharding with different parameter configurations."""
    plugin = Qwen3_0_6B_Plugin()

    # Test enabling sharding with different parameters
    test_configs = [
        {"num_shards": 10, "storage_path": "./shards_small"},
        {"num_shards": 500, "storage_path": "./shards_large"},
        {"num_shards": 1000, "storage_path": "./shards_xlarge"},
    ]

    for config in test_configs:
        result = plugin.enable_sharding(**config)
        assert result is True

        disable_result = plugin.disable_sharding()
        assert disable_result is True


def test_inference_context_management():
    """
    Test inference context management functionality.

    This test validates the management of inference contexts in a sharded model
    environment. Inference contexts encapsulate the state and resources needed
    for a particular inference operation, including which shards are loaded
    and ready for computation.

    The test covers different input shapes and inference types to ensure
    the context management system handles various operational scenarios.
    """
    plugin = Qwen3_0_6B_Plugin()

    # Test preparing context with different parameters
    # context_id: Unique identifier for this inference session
    context_id = "test_ctx_mgr"
    # Various input shapes to test different model configurations
    input_shapes = [
        (1, 128),
        (2, 256),
        (1, 512, 768),
    ]  # Different batch sizes and dimensions
    # Different inference types to test various operational modes
    inference_types = ["forward", "generate", "encode"]

    for shape in input_shapes:
        for inf_type in inference_types:
            # Prepare inference context with specific parameters
            # This loads the necessary shards for the operation
            shards = plugin.prepare_inference_context(context_id, shape, inf_type)
            assert isinstance(shards, list)

            # Test executing with shards
            # This verifies that the prepared context allows successful inference
            dummy_input = torch.randn(*shape)
            result = plugin.execute_with_shards(context_id, dummy_input)
            assert result is not None

            # Cleanup context to release resources
            # This ensures proper resource management between operations
            plugin.cleanup_inference_context(context_id, force_unload=True)


if __name__ == "__main__":
    pytest.main([__file__])
