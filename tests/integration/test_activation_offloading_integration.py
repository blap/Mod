"""
Integration tests for activation offloading functionality in plugins.
"""

import pytest
import torch

from tests.base.integration_test_base import ModelIntegrationTest
from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin


class TestActivationOffloadingIntegration(ModelIntegrationTest):
    """Integration test for activation offloading functionality in plugins."""

    def get_model_plugin_class(self):
        from src.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin

        return Qwen3_0_6B_Plugin

    def create_model_instance(self, **kwargs):
        """Create an instance of the model plugin with test configuration."""
        plugin_class = self.get_model_plugin_class()

        # Use the centralized utility to create and initialize the plugin
        return create_and_initialize_plugin(plugin_class, **kwargs)

    def get_related_components(self):
        """Return related components that integrate with the model."""
        # For activation offloading, related components might include:
        # - Memory management systems
        # - Device management systems
        # - Cache systems
        return []

    def test_integration_scenario(self):
        """Test the core integration scenario for activation offloading."""
        # Create integrated environment
        env = self.create_integrated_environment()
        model_plugin = env["model_plugin"]

        # Initialize the model
        init_result = model_plugin.initialize()
        self.assertTrue(
            init_result, "Model initialization should succeed in integrated environment"
        )

        # Test activation offloading setup in integrated environment
        setup_result = model_plugin.setup_activation_offloading()
        self.assertTrue(
            setup_result,
            "Activation offloading setup should succeed in integrated environment",
        )

        # Test activation offloading functionality with other components
        enable_result = model_plugin.enable_activation_offloading()
        self.assertTrue(
            enable_result,
            "Activation offloading enable should succeed in integrated environment",
        )

        # Perform inference to test integration
        mock_input = torch.randn(1, 10)
        result = model_plugin.infer(mock_input)
        self.assertIsNotNone(
            result,
            "Inference should work with activation offloading in integrated environment",
        )

    def test_model_plugin_integration_with_activation_offloading(self):
        """Test that the model plugin with activation offloading integrates correctly with other components."""
        env = self.create_integrated_environment()
        model_plugin = env["model_plugin"]
        components = env["components"]

        # Verify that the model plugin and related components exist
        self.assertIsNotNone(
            model_plugin, "Model plugin should exist in integrated environment"
        )
        self.assertIsNotNone(
            components, "Related components should exist in integrated environment"
        )

        # Test activation offloading functionality
        setup_result = model_plugin.setup_activation_offloading()
        self.assertTrue(
            setup_result, "Activation offloading setup should work in integration test"
        )

        enable_result = model_plugin.enable_activation_offloading()
        self.assertTrue(
            enable_result,
            "Activation offloading enable should work in integration test",
        )

        # Test that activation offloading works with mock data
        mock_input = torch.randn(2, 10)  # Using batch size 2 as per integration config
        result = model_plugin.infer(mock_input)
        self.assertIsNotNone(
            result, "Inference should produce result in integration test"
        )


if __name__ == "__main__":
    pytest.main([__file__])
