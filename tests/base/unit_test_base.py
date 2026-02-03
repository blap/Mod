"""
Base classes for unit tests in the Mod project.

These base classes provide common functionality and setup for unit testing
individual components of the system.
"""

import os
import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest
import torch

from src.inference_pio.utils.testing_utils import (
    create_temp_test_config,
    cleanup_temp_config,
    create_mock_model,
    assert_tensor_shape,
    assert_tensor_values_close,
    create_test_model_instance,
    verify_plugin_interface,
    run_basic_functionality_tests
)


class BaseUnitTest(unittest.TestCase, ABC):
    """
    Base class for all unit tests in the Mod project.

    This class provides common setup, teardown, and utility methods for
    testing individual units of code in isolation.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = create_temp_test_config()

    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        # Clean up temporary directory
        cleanup_temp_config({"temp_dir": self.temp_dir})

    @abstractmethod
    def test_required_functionality(self):
        """
        Abstract method that must be implemented by subclasses.
        Each unit test class must define its core functionality test.
        """
        pass

    def assert_tensor_shape(self, tensor: torch.Tensor, expected_shape: tuple):
        """Assert that a tensor has the expected shape."""
        self.assertTrue(
            assert_tensor_shape(tensor, expected_shape),
            f"Expected shape {expected_shape}, got {tensor.shape if hasattr(tensor, 'shape') else 'unknown'}",
        )

    def assert_tensor_values_close(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-6
    ):
        """Assert that two tensors have close values within a tolerance."""
        self.assertTrue(
            assert_tensor_values_close(tensor1, tensor2, tolerance),
            f"Tensors are not close within tolerance {tolerance}",
        )


class ModelUnitTest(BaseUnitTest, ABC):
    """
    Base class for unit testing model plugins.

    Provides additional utilities specific to testing model implementations.
    """

    def setUp(self):
        """Set up test fixtures for model unit tests."""
        super().setUp()
        # Don't create mock model, use real models for testing
        pass

    @abstractmethod
    def get_model_plugin_class(self):
        """Return the model plugin class to be tested."""
        pass

    def create_model_instance(self, **kwargs):
        """Create an instance of the model plugin with test configuration."""
        plugin_class = self.get_model_plugin_class()

        # Use the shared utility function
        return create_test_model_instance(plugin_class, **kwargs)

    def test_model_initialization(self):
        """Test that the model plugin initializes correctly."""
        plugin = self.create_model_instance()
        self.assertIsNotNone(plugin)

    def test_model_interface_compliance(self):
        """Test that the model plugin implements required interfaces."""
        plugin = self.create_model_instance()
        # Use the shared utility function
        compliant = verify_plugin_interface(plugin)
        self.assertTrue(compliant, "Plugin should implement required interfaces")


class PluginUnitTest(BaseUnitTest, ABC):
    """
    Base class for unit testing plugin components.

    Provides additional utilities specific to testing plugin implementations.
    """

    def setUp(self):
        """Set up test fixtures for plugin unit tests."""
        super().setUp()

    @abstractmethod
    def get_plugin_class(self):
        """Return the plugin class to be tested."""
        pass

    def create_plugin_instance(self, **kwargs):
        """Create an instance of the plugin with test configuration."""
        from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin

        plugin_class = self.get_plugin_class()
        config = self.test_config.copy()
        config.update(kwargs)
        return plugin_class(**config)

    def test_plugin_initialization(self):
        """Test that the plugin initializes correctly."""
        plugin = self.create_plugin_instance()
        self.assertIsNotNone(plugin)

    def test_plugin_interface_compliance(self):
        """Test that the plugin implements required interfaces."""
        plugin = self.create_plugin_instance()
        # Use the shared utility function
        compliant = verify_plugin_interface(plugin)
        self.assertTrue(compliant, "Plugin should implement required interfaces")
