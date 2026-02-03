"""
Tests to verify that Qwen3-0.6B model implements the common interface correctly.
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


class TestQwen3_0_6B_InterfaceCompliance(ModelUnitTest):
    """Test that Qwen3-0.6B plugin implements the interface correctly."""

    def get_model_plugin_class(self):
        from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin

        return Qwen3_0_6B_Plugin

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

    def test_interface_method_signatures(self):
        """Test that the plugin has the correct method signatures."""

        plugin = self.create_model_instance()

        # Test initialize method signature
        import inspect

        init_sig = inspect.signature(plugin.initialize)
        assert list(init_sig.parameters.keys()) == ["kwargs"]
        assert init_sig.return_annotation == bool

        # Test load_model method signature
        load_sig = inspect.signature(plugin.load_model)
        params = list(load_sig.parameters.keys())
        assert "config" in params
        assert load_sig.return_annotation == torch.nn.Module

        # Test infer method signature
        infer_sig = inspect.signature(plugin.infer)
        params = list(infer_sig.parameters.keys())
        assert "data" in params
        assert infer_sig.return_annotation == Any

        # Test cleanup method signature
        cleanup_sig = inspect.signature(plugin.cleanup)
        assert len(cleanup_sig.parameters) == 0
        assert cleanup_sig.return_annotation == bool

        # Test supports_config method signature
        supports_sig = inspect.signature(plugin.supports_config)
        params = list(supports_sig.parameters.keys())
        assert "config" in params
        assert supports_sig.return_annotation == bool

    def test_text_model_interface_methods(self):
        """Test that text model interface methods have correct signatures."""

        plugin = self.create_model_instance()

        import inspect

        # Test tokenize method signature
        tokenize_sig = inspect.signature(plugin.tokenize)
        params = list(tokenize_sig.parameters.keys())
        assert "text" in params
        assert tokenize_sig.return_annotation == Any

        # Test detokenize method signature
        detokenize_sig = inspect.signature(plugin.detokenize)
        params = list(detokenize_sig.parameters.keys())
        assert "token_ids" in params
        assert detokenize_sig.return_annotation == str

        # Test generate_text method signature
        gen_sig = inspect.signature(plugin.generate_text)
        params = list(gen_sig.parameters.keys())
        assert "prompt" in params
        assert "max_new_tokens" in params
        assert gen_sig.return_annotation == str


def test_model_factory_creates_qwen3_0_6b_plugin_correctly():
    """Test that the model factory creates Qwen3-0.6B plugin that implements the interface."""

    from src.inference_pio.core.model_factory import create_model
    from src.inference_pio.utils.testing_utils import verify_plugin_interface

    # Test that factory creates plugins that implement the interface
    # Using a minimal config to avoid heavy dependencies during testing
    plugin = create_model("qwen3-0.6b")

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