"""
Tests to verify that Qwen3-4B-Instruct-2507 model implements the common interface correctly.
This version uses the standardized test class hierarchies.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

import pytest
import torch

from src.inference_pio.common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface,
    PluginMetadata,
    PluginType,
    StandardPluginInterface,
    TextModelPluginInterface,
)
from tests.base.unit_test_base import ModelUnitTest


class TestQwen3_4B_Instruct_2507_InterfaceCompliance(ModelUnitTest):
    """Test that Qwen3-4B-Instruct-2507 plugin implements the interface correctly."""

    def get_model_plugin_class(self):
        from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
            Qwen3_4B_Instruct_2507_Plugin,
        )

        return Qwen3_4B_Instruct_2507_Plugin

    def test_required_functionality(self):
        """Test that the model plugin implements required functionality."""
        plugin = self.create_model_instance()

        # Verify it inherits from the correct classes
        assert isinstance(plugin, TextModelPluginInterface)
        assert isinstance(plugin, ModelPluginInterface)
        assert isinstance(plugin, StandardPluginInterface)

        # Verify required methods exist
        required_methods = [
            "initialize",
            "load_model",
            "infer",
            "cleanup",
            "supports_config",
            "tokenize",
            "detokenize",
            "generate_text",
        ]

        for method_name in required_methods:
            assert hasattr(plugin, method_name)
            assert callable(getattr(plugin, method_name))


def test_model_factory_creates_qwen3_4b_instruct_2507_plugin_correctly():
    """Test that the model factory creates Qwen3-4B-Instruct-2507 plugin that implements the interface."""

    from src.inference_pio.core.model_factory import create_model
    from src.inference_pio.utils.test_utils import verify_plugin_interface

    # Test that factory creates plugins that implement the interface
    # Using a minimal config to avoid heavy dependencies during testing
    plugin = create_model("qwen3-4b-instruct-2507")

    # Initialize with real model
    init_result = plugin.initialize(device="cpu", use_mock_model=False)

    # Verify it implements the required interface using the centralized utility
    interface_ok = verify_plugin_interface(plugin)
    assert interface_ok, "Plugin should implement the required interface"

    # Additional verification of interface compliance
    assert isinstance(plugin, TextModelPluginInterface)
    assert isinstance(plugin, ModelPluginInterface)
    assert isinstance(plugin, StandardPluginInterface)


if __name__ == "__main__":
    pytest.main([__file__])