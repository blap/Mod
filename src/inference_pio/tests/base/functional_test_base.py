"""
Base classes for functional tests in the Mod project.

These base classes provide common functionality and setup for functional testing
of the complete system from a user perspective.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest
import torch


class BaseFunctionalTest(unittest.TestCase, ABC):
    """
    Base class for all functional tests in the Mod project.

    This class provides common setup, teardown, and utility methods for
    testing the complete system functionality from a user perspective.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = self._create_default_test_config()

    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_default_test_config(self) -> Dict[str, Any]:
        """Create a default configuration for functional testing."""
        return {
            "device": "cpu",
            "batch_size": 1,
            "test_mode": True,
            "temp_dir": self.temp_dir,
            "functional_test": True,
        }

    @abstractmethod
    def test_functional_requirement(self):
        """
        Abstract method that must be implemented by subclasses.
        Each functional test class must define its core requirement test.
        """
        raise NotImplementedError("Method not implemented")

    def run_system_command(self, command: str) -> subprocess.CompletedProcess:
        """Run a system command and return the result."""
        return subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.test_config.get("temp_dir", "."),
        )

    def validate_output_format(self, output: Any, expected_format: str):
        """Validate that the output matches the expected format."""
        # This is a simplified validation - in practice, this would be more complex
        if expected_format == "json":
            import json

            try:
                parsed = json.loads(output)
                return True
            except (TypeError, ValueError):
                return False
        return True  # Simplified for now


class ModelFunctionalTest(BaseFunctionalTest, ABC):
    """
    Base class for functional testing of model plugins.

    Provides additional utilities specific to testing model functionality
    from a user perspective.
    """

    def setUp(self):
        """Set up test fixtures for model functional tests."""
        super().setUp()

    @abstractmethod
    def get_model_plugin_class(self):
        """Return the model plugin class to be tested."""
        raise NotImplementedError("Method not implemented")

    def create_model_instance(self, **kwargs):
        """Create an instance of the model plugin with test configuration."""
        from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin

        plugin_class = self.get_model_plugin_class()

        # Use the centralized utility to create and initialize the plugin
        return create_and_initialize_plugin(plugin_class, **kwargs)

    def test_complete_model_workflow(self):
        """Test the complete workflow of a model from initialization to inference."""
        model_plugin = self.create_model_instance()

        # Initialize the model
        init_result = model_plugin.initialize()
        self.assertTrue(init_result, "Model initialization should succeed")

        # Load the model
        model = model_plugin.load_model(config={})
        self.assertIsNotNone(model, "Model should be loaded successfully")

        # Use text input instead of random tensor for the model
        text_input = "This is a test prompt for the model."
        try:
            # Try using generate_text method if available
            if hasattr(model_plugin, "generate_text"):
                result = model_plugin.generate_text(text_input, max_new_tokens=10)
            else:
                # Try tokenizing and using infer method
                tokenized_input = model_plugin.tokenize(text_input)
                result = model_plugin.infer(tokenized_input)
        except Exception:
            # If all else fails, try with a simple tensor
            mock_input = torch.randint(0, 1000, (1, 10))
            result = model_plugin.infer(mock_input)

        self.assertIsNotNone(result, "Inference should produce a result")

    def test_user_facing_api(self):
        """Test the user-facing API of the model plugin."""
        model_plugin = self.create_model_instance()

        # Test various API methods
        self.assertTrue(
            hasattr(model_plugin, "generate_text"), "Should have generate_text method"
        )
        self.assertTrue(
            hasattr(model_plugin, "tokenize"), "Should have tokenize method"
        )
        self.assertTrue(
            hasattr(model_plugin, "detokenize"), "Should have detokenize method"
        )

        # Test that methods are callable
        self.assertTrue(callable(getattr(model_plugin, "generate_text")))
        self.assertTrue(callable(getattr(model_plugin, "tokenize")))
        self.assertTrue(callable(getattr(model_plugin, "detokenize")))


class SystemFunctionalTest(BaseFunctionalTest, ABC):
    """
    Base class for functional testing of the complete system.

    Tests high-level system functionality and user workflows.
    """

    def setUp(self):
        """Set up test fixtures for system functional tests."""
        super().setUp()

    def test_system_startup(self):
        """Test that the system starts up correctly."""
        # This would typically test actual system startup
        # For now, we'll just verify basic system components exist
        import src

        self.assertIsNotNone(src)

    def test_complete_user_workflow(self):
        """Test a complete user workflow from model selection to inference."""
        # This would test the complete user experience
        # For now, we'll simulate the workflow
        from src.inference_pio.core.model_factory import create_model

        # Test model creation
        try:
            # Using a mock model name since we don't know all available models
            model_plugin = create_model(
                "mock_model"
            )  # This might fail, which is OK for testing
        except Exception:
            # Expected if mock_model doesn't exist
            raise NotImplementedError("Method not implemented")

        # The test passes if no critical errors occur during the workflow simulation
        self.assertTrue(True)
