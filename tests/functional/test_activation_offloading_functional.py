"""
Functional tests for activation offloading functionality in plugins.
"""

import pytest
import torch

from tests.base.functional_test_base import ModelFunctionalTest
from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin


class TestActivationOffloadingFunctional(ModelFunctionalTest):
    """Functional test for activation offloading functionality in plugins."""

    def get_model_plugin_class(self):
        from src.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin

        return Qwen3_0_6B_Plugin

    def create_model_instance(self, **kwargs):
        """Create an instance of the model plugin with test configuration."""
        plugin_class = self.get_model_plugin_class()

        # Use the centralized utility to create and initialize the plugin
        return create_and_initialize_plugin(plugin_class, **kwargs)

    def test_functional_requirement(self):
        """Test the core functional requirement for activation offloading."""
        # Test the complete workflow of activation offloading
        plugin = self.create_model_instance()

        # Initialize the model
        init_result = plugin.initialize()
        self.assertTrue(init_result, "Model initialization should succeed")

        # Setup activation offloading
        setup_result = plugin.setup_activation_offloading()
        self.assertTrue(setup_result, "Activation offloading setup should succeed")

        # Enable activation offloading
        enable_result = plugin.enable_activation_offloading()
        self.assertTrue(enable_result, "Activation offloading enable should succeed")

        # Perform inference with activation offloading enabled
        # Use text input instead of random tensor for the model
        text_input = "This is a test prompt for the model."
        try:
            # Try using generate_text method if available
            if hasattr(plugin, "generate_text"):
                result = plugin.generate_text(text_input, max_new_tokens=10)
            else:
                # Try tokenizing and using infer method
                tokenized_input = plugin.tokenize(text_input)
                result = plugin.infer(tokenized_input)
        except Exception:
            # If all else fails, try with a simple tensor
            mock_input = torch.randint(0, 1000, (1, 10))
            result = plugin.infer(mock_input)

        self.assertIsNotNone(
            result,
            "Inference should produce a result with activation offloading enabled",
        )

        # Get activation offloading statistics
        stats = plugin.get_activation_offloading_stats()
        self.assertIsInstance(stats, dict, "Stats should be returned as a dictionary")

        # Test activation prediction
        prediction = plugin.predict_activation_access()
        self.assertIsInstance(
            prediction, dict, "Prediction should be returned as a dictionary"
        )

    def test_complete_model_workflow_with_activation_offloading(self):
        """Test the complete workflow of a model with activation offloading from initialization to inference."""
        plugin = self.create_model_instance()

        # Initialize the model
        init_result = plugin.initialize()
        self.assertTrue(init_result, "Model initialization should succeed")

        # Setup and enable activation offloading
        setup_result = plugin.setup_activation_offloading()
        self.assertTrue(setup_result, "Activation offloading setup should succeed")

        enable_result = plugin.enable_activation_offloading()
        self.assertTrue(enable_result, "Activation offloading enable should succeed")

        # Perform multiple inferences to test the complete workflow
        for i in range(3):
            text_input = f"This is test prompt {i+1} for the model."
            try:
                # Try using generate_text method if available
                if hasattr(plugin, "generate_text"):
                    result = plugin.generate_text(text_input, max_new_tokens=10)
                else:
                    # Try tokenizing and using infer method
                    tokenized_input = plugin.tokenize(text_input)
                    result = plugin.infer(tokenized_input)
            except Exception:
                # If all else fails, try with a simple tensor
                mock_input = torch.randint(0, 1000, (1, 10))
                result = plugin.infer(mock_input)

            self.assertIsNotNone(
                result,
                f"Inference {i+1} should produce a result with activation offloading enabled",
            )

        # Check that activation offloading statistics are being collected
        stats = plugin.get_activation_offloading_stats()
        self.assertIsNotNone(stats, "Stats should be available after inferences")


if __name__ == "__main__":
    pytest.main([__file__])
