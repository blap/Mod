"""
Shared Test Fixtures Module

This module contains common pytest fixtures used across the test suite to eliminate
code duplication and ensure consistent setup/teardown behavior.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock

import pytest
import torch

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    PluginMetadata,
    PluginType,
    TextModelPluginInterface,
)
from tests.shared.utils.test_utils import (
    cleanup_temp_directory,
    create_mock_model,
    create_sample_tensor_data,
    create_sample_text_data,
    create_temp_directory,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing and clean it up afterwards.

    Yields:
        Path object to the temporary directory
    """
    temp_path = create_temp_directory()
    yield temp_path
    cleanup_temp_directory(temp_path)


@pytest.fixture
def sample_text_data() -> list:
    """
    Provide sample text data for testing.

    Returns:
        List of sample text strings
    """
    return create_sample_text_data(num_samples=5, max_length=20)


@pytest.fixture
def sample_tensor_data() -> torch.Tensor:
    """
    Provide sample tensor data for testing.

    Returns:
        Sample tensor data
    """
    return create_sample_tensor_data(batch_size=4, seq_len=10, hidden_size=128)


@pytest.fixture
def mock_torch_model() -> torch.nn.Module:
    """
    Provide a mock PyTorch model for testing.

    Returns:
        A simple PyTorch model instance
    """
    return create_mock_model(input_dim=10, output_dim=1)


@pytest.fixture
def sample_metadata() -> PluginMetadata:
    """
    Provide sample metadata for testing plugins.

    Returns:
        PluginMetadata instance with sample data
    """
    return PluginMetadata(
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


@pytest.fixture
def realistic_test_plugin(sample_metadata) -> TextModelPluginInterface:
    """
    Provide a realistic test plugin instance.

    Args:
        sample_metadata: Metadata to use for the plugin

    Returns:
        Instance of a realistic test plugin
    """

    class RealisticTestPlugin(TextModelPluginInterface):
        """A realistic test plugin implementation that can be reused across tests."""

        def __init__(self, name="TestPlugin", metadata=None):
            if metadata is None:
                metadata = sample_metadata
                metadata.name = name
            super().__init__(metadata)
            self._initialized = False
            self._model = None

        def initialize(self, **kwargs) -> bool:
            self._initialized = True
            return True

        def load_model(self, config=None):
            # Create a simple torch model
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)

                def forward(self, x):
                    return self.linear(x)

            self._model = SimpleModel()
            return self._model

        def infer(self, data):
            if not self._initialized:
                self.initialize()

            if isinstance(data, str):
                return f"Processed: {data}"
            elif isinstance(data, (list, tuple)):
                return [f"Processed: {item}" for item in data]
            else:
                return f"Processed: {str(data)}"

        def cleanup(self) -> bool:
            self._model = None
            self._initialized = False
            return True

        def supports_config(self, config) -> bool:
            return config is None or isinstance(config, dict)

        def tokenize(self, text: str, **kwargs):
            if not isinstance(text, str):
                raise TypeError("Text must be a string")
            tokens = text.split()
            token_map = {word: idx + 1 for idx, word in enumerate(set(tokens))}
            return [token_map[word] for word in tokens]

        def detokenize(self, token_ids, **kwargs) -> str:
            if isinstance(token_ids, (list, tuple)):
                return " ".join([f"token_{tid}" for tid in token_ids])
            else:
                return f"token_{token_ids}"

        def generate_text(
            self, prompt: str, max_new_tokens: int = 512, **kwargs
        ) -> str:
            if not self._initialized:
                self.initialize()
            return f"{prompt} [GENERATED TEXT]"

    return RealisticTestPlugin()


@pytest.fixture
def mock_plugin_dependencies() -> Dict[str, Any]:
    """
    Provide mock dependencies for plugin testing.

    Returns:
        Dictionary with mock dependencies
    """
    return {
        "torch": Mock(),
        "transformers": Mock(),
        "numpy": Mock(),
        "json": Mock(),
        "datetime": Mock(),
    }


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Provide a sample configuration for testing.

    Returns:
        Sample configuration dictionary
    """
    return {
        "model_path": "/tmp/test_model",
        "batch_size": 4,
        "max_seq_len": 512,
        "device": "cpu",
        "precision": "fp32",
        "use_flash_attention": False,
        "use_quantization": False,
        "num_workers": 2,
    }


@pytest.fixture
def mock_plugin_interface_methods() -> list:
    """
    Provide a list of required plugin interface methods for testing.

    Returns:
        List of method names that should be implemented by plugins
    """
    return [
        "initialize",
        "load_model",
        "infer",
        "cleanup",
        "supports_config",
        "tokenize",
        "detokenize",
        "generate_text",
    ]


@pytest.fixture
def sample_plugin_manifest() -> Dict[str, Any]:
    """
    Provide a sample plugin manifest for testing.

    Returns:
        Sample plugin manifest dictionary
    """
    return {
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
        "main_class_path": "src.inference_pio.models.test_plugin.plugin.TestPlugin",
        "entry_point": "create_test_plugin",
        "input_types": ["text"],
        "output_types": ["text"],
    }
