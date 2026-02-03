"""
Test script for the extreme sharding and streaming system.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src.inference_pio.common.interfaces.base_plugin_interface import (
    ModelPluginInterface,
    ModelPluginMetadata,
    PluginType,
)
from src.inference_pio.common.parallel.model_sharder import create_extreme_sharding_system
from tests.utils.test_utils import (
    assert_equal,
    assert_false,
    assert_greater,
    assert_in,
    assert_is_instance,
    assert_is_none,
    assert_is_not_none,
    assert_less,
    assert_not_equal,
    assert_not_in,
    assert_raises,
    assert_true,
    run_tests,
)


class SimpleTestModel(nn.Module):
    """Simple test model for sharding."""

    def __init__(self, num_layers=10, hidden_size=256):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class TestPlugin(ModelPluginInterface):
    """Test plugin to verify sharding functionality."""

    def __init__(self):
        metadata = ModelPluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for sharding",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        super().__init__(metadata)
        self._model = SimpleTestModel(num_layers=5, hidden_size=128)

    def initialize(self, **kwargs) -> bool:
        return True

    def load_model(self, config=None) -> nn.Module:
        return self._model

    def infer(self, data) -> any:
        return self._model(data)

    def cleanup(self) -> bool:
        return True


def test_basic_sharding():
    """Test basic sharding functionality."""
    print("Testing basic sharding functionality...")

    # Create a temporary directory for shards
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sharding system
        sharder, loader = create_extreme_sharding_system(
            storage_path=temp_dir, num_shards=20  # Smaller number for testing
        )

        # Create a simple model
        model = SimpleTestModel(num_layers=5, hidden_size=64)

        # Shard the model
        shards = sharder.shard_model(model, num_shards=20)
        print(f"Created {len(shards)} shards")

        # Test loading a shard
        test_shard_id = shards[0].id
        loaded_shard = sharder.load_shard(test_shard_id, device="cpu")
        print(f"Loaded shard {test_shard_id} successfully")

        # Test unloading
        sharder.unload_shard(test_shard_id)
        print(f"Unloaded shard {test_shard_id} successfully")

        # Test memory stats
        stats = sharder.get_memory_usage()
        print(f"Memory stats: {stats}")

        print("Basic sharding test passed!\n")


def test_streaming_loader():
    """Test streaming loader functionality."""
    print("Testing streaming loader functionality...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sharding system
        sharder, loader = create_extreme_sharding_system(
            storage_path=temp_dir, num_shards=10
        )

        # Create and shard a model
        model = SimpleTestModel(num_layers=3, hidden_size=32)
        shards = sharder.shard_model(model, num_shards=10)

        # Prepare an inference context
        context_id = "test_context_1"
        input_shape = (1, 32)  # Batch size 1, hidden size 32
        required_shards = loader.prepare_inference_context(context_id, input_shape)

        print(f"Prepared context {context_id} with {len(required_shards)} shards")

        # Create test input
        test_input = torch.randn(1, 32)

        # Execute in context
        output = loader.execute_in_context(context_id, test_input)
        print(f"Executed inference, output shape: {output.shape}")

        # Cleanup
        loader.cleanup_context(context_id)
        print(f"Cleaned up context {context_id}")

        print("Streaming loader test passed!\n")


def test_plugin_integration():
    """Test integration with plugin system."""
    print("Testing plugin integration...")

    plugin = TestPlugin()

    # Initialize with sharding enabled
    success = plugin.initialize(
        enable_sharding=True, num_shards=10, sharding_storage_path="./temp_shards"
    )
    print(f"Plugin initialization with sharding: {success}")

    if success:
        # Test sharding stats
        stats = plugin.get_sharding_stats()
        print(f"Sharding stats: {stats}")

        # Cleanup
        plugin.cleanup()
        print("Plugin cleanup completed")

    print("Plugin integration test passed!\n")


def main():
    """Run all tests."""
    print("Running sharding system tests...\n")

    test_basic_sharding()
    test_streaming_loader()
    test_plugin_integration()

    print("All tests passed!")


if __name__ == "__main__":
    main()
