"""
Example test file demonstrating the use of shared test utilities and fixtures.

This file shows how to use the unified test architecture with shared fixtures
and utilities to reduce code duplication and improve maintainability.
"""

import tempfile
from pathlib import Path

import pytest
import torch

# Import shared fixtures
from tests.shared.fixtures.plugin_fixtures import (
    mock_torch_model,
    realistic_test_plugin,
    sample_config,
    sample_metadata,
    sample_plugin_manifest,
    sample_tensor_data,
    sample_text_data,
    temp_dir,
)

# Import shared assertions
from tests.shared.utils.assertions import (
    assert_dict_contains_keys,
    assert_plugin_interface_implemented,
    assert_response_format,
    assert_tensor_properties,
)

# Import shared utilities
from tests.shared.utils.test_utils import (
    assert_tensor_shape,
    assert_tensor_values_close,
    compare_dicts,
    create_sample_tensor_data,
    create_sample_text_data,
)


def test_shared_temp_dir_fixture(temp_dir):
    """Test the shared temporary directory fixture."""
    # Verify that temp_dir is a Path object
    assert isinstance(temp_dir, Path)

    # Create a test file in the temp directory
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")

    # Verify the file exists
    assert test_file.exists()

    # The fixture should automatically clean up after the test
    print(f"Temporary directory created at: {temp_dir}")


def test_shared_text_data_fixture(sample_text_data):
    """Test the shared sample text data fixture."""
    # Verify that sample_text_data is a list
    assert isinstance(sample_text_data, list)

    # Verify that it has the expected number of items
    assert len(sample_text_data) == 5

    # Verify that all items are strings
    for text in sample_text_data:
        assert isinstance(text, str)

    print(f"Sample text data: {sample_text_data[:2]}...")


def test_shared_tensor_data_fixture(sample_tensor_data):
    """Test the shared sample tensor data fixture."""
    # Verify that sample_tensor_data is a tensor
    assert isinstance(sample_tensor_data, torch.Tensor)

    # Verify the expected shape (should be [4, 10, 128])
    assert sample_tensor_data.shape[0] == 4  # batch size
    assert sample_tensor_data.shape[1] == 10  # sequence length
    assert sample_tensor_data.shape[2] == 128  # hidden size

    print(f"Sample tensor shape: {sample_tensor_data.shape}")


def test_shared_mock_torch_model(mock_torch_model):
    """Test the shared mock torch model fixture."""
    # Verify that mock_torch_model is a torch module
    assert isinstance(mock_torch_model, torch.nn.Module)

    # Test that it can process input
    input_tensor = torch.randn(1, 10)  # batch_size=1, input_dim=10
    output = mock_torch_model(input_tensor)

    # Verify output shape
    assert output.shape == (1, 1)  # batch_size=1, output_dim=1

    print(f"Mock model input shape: {input_tensor.shape}, output shape: {output.shape}")


def test_shared_sample_metadata(sample_metadata):
    """Test the shared sample metadata fixture."""
    # Import the expected class
    from src.common.improved_base_plugin_interface import PluginMetadata

    # Verify that sample_metadata is a PluginMetadata instance
    assert isinstance(sample_metadata, PluginMetadata)

    # Verify some expected attributes
    assert sample_metadata.name == "SamplePlugin"
    assert sample_metadata.version == "1.0.0"
    assert sample_metadata.author == "Sample Author"

    print(f"Sample metadata: {sample_metadata.name} v{sample_metadata.version}")


def test_shared_realistic_test_plugin(realistic_test_plugin):
    """Test the shared realistic test plugin fixture."""
    # Use the shared assertion utility
    assert_plugin_interface_implemented(realistic_test_plugin)

    # Test plugin functionality
    assert realistic_test_plugin.initialize() == True
    result = realistic_test_plugin.infer("test input")
    assert "Processed: test input" == result
    assert realistic_test_plugin.cleanup() == True

    print(f"Realistic test plugin name: {realistic_test_plugin.metadata.name}")


def test_shared_sample_config(sample_config):
    """Test the shared sample config fixture."""
    # Verify that sample_config is a dictionary
    assert isinstance(sample_config, dict)

    # Verify expected keys exist
    expected_keys = [
        "model_path",
        "batch_size",
        "max_seq_len",
        "device",
        "precision",
        "use_flash_attention",
        "use_quantization",
        "num_workers",
    ]

    for key in expected_keys:
        assert key in sample_config

    print(f"Sample config keys: {list(sample_config.keys())}")


def test_shared_plugin_manifest(sample_plugin_manifest):
    """Test the shared plugin manifest fixture."""
    # Verify that sample_plugin_manifest is a dictionary
    assert isinstance(sample_plugin_manifest, dict)

    # Verify expected keys exist
    expected_keys = [
        "name",
        "version",
        "author",
        "description",
        "plugin_type",
        "dependencies",
        "compatibility",
        "created_at",
        "updated_at",
    ]

    for key in expected_keys:
        assert key in sample_plugin_manifest

    print(f"Plugin manifest name: {sample_plugin_manifest['name']}")


def test_shared_utility_functions():
    """Test various shared utility functions."""
    # Test creating sample text data
    text_data = create_sample_text_data(num_samples=3, max_length=10)
    assert isinstance(text_data, list)
    assert len(text_data) == 3
    for text in text_data:
        assert isinstance(text, str)

    # Test creating sample tensor data
    tensor_data = create_sample_tensor_data(batch_size=2, seq_len=5, hidden_size=64)
    assert isinstance(tensor_data, torch.Tensor)
    assert tensor_data.shape == (2, 5, 64)

    # Test tensor shape assertion
    test_tensor = torch.randn(3, 4, 5)
    assert_tensor_shape(test_tensor, (3, 4, 5))

    # Test tensor values closeness assertion
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0001, 2.0001, 3.0001])
    assert_tensor_values_close(tensor1, tensor2, tolerance=1e-3)

    # Test dictionary comparison
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"a": 1, "b": 2, "c": 3, "d": 4}  # Extra key
    assert not compare_dicts(dict1, dict2)  # Should be False due to extra key
    assert compare_dicts(dict1, dict2, ignore_keys=["d"])  # Should be True ignoring 'd'

    print("All shared utility functions tested successfully")


def test_shared_assertion_functions(realistic_test_plugin):
    """Test various shared assertion functions."""
    # Test plugin interface assertion
    assert_plugin_interface_implemented(realistic_test_plugin)

    # Test tensor properties assertion
    test_tensor = torch.randn(2, 3, 4)
    assert_tensor_properties(test_tensor, expected_dtype=torch.float32, expected_dims=3)

    # Test dictionary contains keys assertion
    test_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}
    assert_dict_contains_keys(test_dict, ["key1", "key2"])

    # Test response format assertion
    def sample_func():
        return "response string"

    result = sample_func()
    assert_response_format(result, str)

    print("All shared assertion functions tested successfully")


# Example of parametrized test using shared fixtures
@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("hello", "Processed: hello"),
        ("world", "Processed: world"),
        ("test", "Processed: test"),
    ],
)
def test_plugin_with_parametrized_inputs(
    realistic_test_plugin, input_text, expected_output
):
    """Test plugin with parametrized inputs."""
    realistic_test_plugin.initialize()
    result = realistic_test_plugin.infer(input_text)
    assert result == expected_output
    realistic_test_plugin.cleanup()


if __name__ == "__main__":
    # Run tests manually if executed as script
    import os
    import sys

    sys.path.insert(0, os.path.abspath("."))

    # Create fixtures manually for demonstration
    from tests.shared.fixtures.plugin_fixtures import (
        mock_torch_model,
        realistic_test_plugin,
        sample_config,
        sample_metadata,
        sample_plugin_manifest,
        sample_tensor_data,
        sample_text_data,
        temp_dir,
    )

    print("Running example tests...")

    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        test_shared_temp_dir_fixture(temp_path)

    # Create sample data
    text_data = create_sample_text_data(5, 20)
    test_shared_text_data_fixture(text_data)

    tensor_data = create_sample_tensor_data(4, 10, 128)
    test_shared_tensor_data_fixture(tensor_data)

    model = create_sample_tensor_data(
        1, 10
    )  # This won't work as intended, just for demo
    # Need to create actual model
    from tests.shared.utils.test_utils import create_mock_model

    mock_model = create_mock_model(10, 1)
    test_shared_mock_torch_model(mock_model)

    from datetime import datetime

    from src.common.improved_base_plugin_interface import PluginMetadata, PluginType

    metadata = PluginMetadata(
        name="SamplePlugin",
        version="1.0.0",
        author="Sample Author",
        description="Sample Description",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    test_shared_sample_metadata(metadata)

    # Create realistic test plugin
    plugin = realistic_test_plugin
    test_shared_realistic_test_plugin(plugin)

    config = {
        "model_path": "/tmp/test_model",
        "batch_size": 4,
        "max_seq_len": 512,
        "device": "cpu",
        "precision": "fp32",
        "use_flash_attention": False,
        "use_quantization": False,
        "num_workers": 2,
    }
    test_shared_sample_config(config)

    manifest = {
        "name": "TestPlugin",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "A test model plugin",
        "plugin_type": "MODEL_COMPONENT",
        "dependencies": ["torch"],
        "compatibility": {"torch_version": ">=2.0.0"},
        "created_at": "2026-01-31T00:00:00",
        "updated_at": "2026-01-31T00:00:00",
        "model_architecture": "TestArch",
        "model_size": "1.0B",
        "required_memory_gb": 1.0,
        "supported_modalities": ["text"],
        "license": "MIT",
        "tags": ["test", "model"],
        "model_family": "TestFamily",
        "num_parameters": 1000000,
        "test_coverage": 1.0,
        "validation_passed": True,
        "main_class_path": "src.models.test_plugin.plugin.TestPlugin",
        "entry_point": "create_test_plugin",
        "input_types": ["text"],
        "output_types": ["text"],
    }
    test_shared_plugin_manifest(manifest)

    test_shared_utility_functions()
    test_shared_assertion_functions(plugin)

    print("All example tests completed successfully!")
