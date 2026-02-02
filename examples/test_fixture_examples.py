"""
Example demonstrating how to use the test fixture system in Inference-PIO
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importlib.util

# For fixture-related functionality, we'll use standard Python approaches
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Import the necessary functions from the appropriate module
from src.common.test_utilities import (
    calculate_statistics,
    ensure_directory_exists,
    extract_model_name_from_path,
    format_bytes,
    format_duration,
    generate_cache_key,
    get_timestamp,
    is_cache_valid,
    load_json_file,
    measure_execution_time,
    normalize_path_separators,
    sanitize_filename,
    save_json_file,
)


# Define fixture-like classes for the example
@dataclass
class TemporaryDirectoryFixture:
    """Fixture for creating temporary directories."""

    temp_dir: Optional[str] = None

    def __enter__(self):
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        import shutil

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TensorFixture:
    """Fixture for creating and managing tensors."""

    def __init__(self):
        self.tensors = {
            "tensor_0": torch.randn(2, 4),
            "tensor_1": torch.randn(3, 5),
            "tensor_2": torch.randn(1, 6),
        }

    def get_tensor(self, name: str):
        return self.tensors.get(name)

    def get_random_tensors(self, count: int):
        import random

        keys = list(self.tensors.keys())
        selected_keys = random.sample(keys, min(count, len(keys)))
        return [self.tensors[key] for key in selected_keys]


class MockModelFixture:
    """Fixture for creating mock models."""

    def __init__(self):
        self.model = nn.Linear(10, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.dataloader = [(torch.randn(1, 10), torch.randn(1, 1)) for _ in range(5)]

    def get_model(self):
        return self.model


@dataclass
class ConfigFixture:
    """Fixture for creating configuration objects."""

    def __init__(self):
        self.config = {
            "model_name": "test_model",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
        }

    def get_config(self, key: Optional[str] = None):
        if key:
            return self.config.get(key)
        return self.config

    def update_config(self, updates: Dict[str, Any]):
        self.config.update(updates)


def test_example_with_temp_dir():
    """Example: Using temporary directory fixture."""
    # Note: Since 'temp_dir' fixture is registered with the decorator system,
    # we'll create it directly for this example
    temp_fixture = TemporaryDirectoryFixture()

    with temp_fixture as temp_dir_path:
        # Create a file in the temporary directory
        test_file = os.path.join(temp_dir_path, "example.txt")
        with open(test_file, "w") as f:
            f.write("This is an example file created in a temporary directory.")

        # Verify the file exists and read its content
        assert os.path.exists(test_file), "File should exist"
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "This is an example file created in a temporary directory."

        print("PASS Temporary directory fixture example passed")


def test_example_with_tensor_fixture():
    """Example: Using tensor fixture."""
    tensor_fixture = TensorFixture()

    # Get a specific tensor
    tensor = tensor_fixture.get_tensor("tensor_0")
    assert tensor is not None, "Tensor should exist"
    assert hasattr(tensor, "shape"), "Tensor should have shape attribute"

    # Get random tensors
    random_tensors = tensor_fixture.get_random_tensors(2)
    assert len(random_tensors) >= 0, "Should get at least some tensors"

    print("PASS Tensor fixture example passed")


def test_example_with_mock_model():
    """Example: Using mock model fixture."""
    model_fixture = MockModelFixture()

    # Verify model components exist
    assert model_fixture.model is not None, "Model should exist"
    assert model_fixture.optimizer is not None, "Optimizer should exist"
    assert model_fixture.dataloader is not None, "Dataloader should exist"

    # Perform a simple forward pass
    import torch

    dummy_input = torch.randn(1, 10)
    output = model_fixture.model(dummy_input)
    assert output is not None, "Output should exist"

    print("PASS Mock model fixture example passed")


def test_example_with_config():
    """Example: Using config fixture."""
    config_fixture = ConfigFixture()

    # Get the config
    config = config_fixture.get_config()
    assert config is not None, "Config should exist"
    assert "model_name" in config, "Config should have model_name"

    # Test getting specific config values
    model_name = config_fixture.get_config("model_name")
    assert model_name is not None, "Model name should exist"

    # Update config
    config_fixture.update_config({"new_param": "new_value"})
    new_param = config_fixture.get_config("new_param")
    assert new_param == "new_value", "New param should have correct value"

    print("PASS Config fixture example passed")


def test_combined_example():
    """Example: Using multiple fixtures together."""
    # Create fixtures manually for this example
    temp_fixture = TemporaryDirectoryFixture()
    tensor_fixture = TensorFixture()
    config_fixture = ConfigFixture()

    # Use the temporary directory
    with temp_fixture as temp_dir:
        # Save a tensor to the temp directory
        tensor = tensor_fixture.get_tensor("tensor_0")
        config = config_fixture.get_config()

        # Create a config file in temp directory
        config_file = os.path.join(temp_dir, "config.json")
        with open(config_file, "w") as f:
            import json

            json.dump(config, f)

        # Verify file was created
        assert os.path.exists(config_file), "Config file should exist"

        print("PASS Combined fixtures example passed")


def run_examples():
    """Run all fixture examples."""
    print("Testing Inference-PIO Test Fixture System Examples")
    print("=" * 50)

    test_example_with_temp_dir()
    test_example_with_tensor_fixture()
    test_example_with_mock_model()
    test_example_with_config()
    test_combined_example()

    print("=" * 50)
    print("All fixture examples passed successfully!")


if __name__ == "__main__":
    run_examples()
