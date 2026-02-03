"""
Unit tests for the model factory functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.inference_pio.common.interfaces.improved_base_plugin_interface import TextModelPluginInterface
from src.inference_pio.core.model_factory import ModelFactory, create_model


def test_create_model_qwen3_0_6b():
    """Test creating Qwen3-0.6B model."""
    try:
        plugin = create_model("qwen3-0.6b")

        # Verify the plugin is created correctly
        assert plugin is not None
        assert isinstance(plugin, TextModelPluginInterface)
        assert plugin.metadata.name == "Qwen3-0.6B"

        # Verify required methods exist
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "load_model")
        assert hasattr(plugin, "infer")
        assert hasattr(plugin, "cleanup")
        assert hasattr(plugin, "supports_config")
        assert hasattr(plugin, "tokenize")
        assert hasattr(plugin, "detokenize")
        assert hasattr(plugin, "generate_text")
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for Qwen3-0.6B not available")


def test_create_model_qwen3_vl_2b():
    """Test creating Qwen3-VL-2B model."""
    try:
        plugin = create_model("qwen3-vl-2b")

        # Verify the plugin is created correctly
        assert plugin is not None
        assert isinstance(plugin, TextModelPluginInterface)

        # Verify required methods exist
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "load_model")
        assert hasattr(plugin, "infer")
        assert hasattr(plugin, "cleanup")
        assert hasattr(plugin, "supports_config")
        assert hasattr(plugin, "tokenize")
        assert hasattr(plugin, "detokenize")
        assert hasattr(plugin, "generate_text")
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for Qwen3-VL-2B not available")


def test_create_model_glm_4_7_flash():
    """Test creating GLM-4.7-Flash model."""
    try:
        plugin = create_model("glm-4-7-flash")

        # Verify the plugin is created correctly
        assert plugin is not None
        assert isinstance(plugin, TextModelPluginInterface)

        # Verify required methods exist
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "load_model")
        assert hasattr(plugin, "infer")
        assert hasattr(plugin, "cleanup")
        assert hasattr(plugin, "supports_config")
        assert hasattr(plugin, "tokenize")
        assert hasattr(plugin, "detokenize")
        assert hasattr(plugin, "generate_text")
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for GLM-4.7-Flash not available")


def test_create_model_qwen3_4b():
    """Test creating Qwen3-4B model."""
    try:
        plugin = create_model("qwen3-4b")

        # Verify the plugin is created correctly
        assert plugin is not None
        assert isinstance(plugin, TextModelPluginInterface)

        # Verify required methods exist
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "load_model")
        assert hasattr(plugin, "infer")
        assert hasattr(plugin, "cleanup")
        assert hasattr(plugin, "supports_config")
        assert hasattr(plugin, "tokenize")
        assert hasattr(plugin, "detokenize")
        assert hasattr(plugin, "generate_text")
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for Qwen3-4B not available")


def test_create_model_qwen3_coder_30b():
    """Test creating Qwen3-Coder-30B model."""
    try:
        plugin = create_model("qwen3-coder-30b")

        # Verify the plugin is created correctly
        assert plugin is not None
        assert isinstance(plugin, TextModelPluginInterface)

        # Verify required methods exist
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "load_model")
        assert hasattr(plugin, "infer")
        assert hasattr(plugin, "cleanup")
        assert hasattr(plugin, "supports_config")
        assert hasattr(plugin, "tokenize")
        assert hasattr(plugin, "detokenize")
        assert hasattr(plugin, "generate_text")
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for Qwen3-Coder-30B not available")


def test_create_model_with_config():
    """Test creating a model with configuration."""
    try:
        config = {"device": "cpu", "dtype": torch.float32}
        plugin = create_model("qwen3-0.6b", config=config)

        # Verify the plugin is created correctly
        assert plugin is not None
        assert isinstance(plugin, TextModelPluginInterface)
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for Qwen3-0.6B not available")


def test_create_model_invalid_name():
    """Test creating a model with an invalid name."""
    with pytest.raises(ValueError):
        create_model("invalid-model-name")


def test_list_supported_models():
    """Test listing supported models."""
    supported_models = ModelFactory.list_supported_models()

    # Verify the list contains expected models
    expected_models = [
        "qwen3-0.6b",
        "qwen3-vl-2b",
        "glm-4-7-flash",
        "qwen3-4b",
        "qwen3-coder-30b",
    ]

    for model in expected_models:
        assert model in supported_models


def test_case_insensitive_model_names():
    """Test that model names are handled case-insensitively."""
    try:
        # Test different case variations
        plugin1 = create_model("QWEN3-0.6B")
        plugin2 = create_model("qwen3_0_6b")
        plugin3 = create_model("Qwen3 0 6b")

        # All should create the same type of plugin
        assert isinstance(plugin1, TextModelPluginInterface)
        assert isinstance(plugin2, TextModelPluginInterface)
        assert isinstance(plugin3, TextModelPluginInterface)
    except ImportError:
        # Skip if dependencies are not available
        pytest.skip("Dependencies for Qwen3-0.6B not available")


if __name__ == "__main__":
    pytest.main([__file__])
