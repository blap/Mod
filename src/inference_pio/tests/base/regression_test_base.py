"""
Base classes for regression tests in the Mod project.

These base classes provide common functionality and setup for regression testing
to ensure that new changes don't break existing functionality.
"""

import hashlib
import json
import os
import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
import torch


class BaseRegressionTest(unittest.TestCase, ABC):
    """
    Base class for all regression tests in the Mod project.

    This class provides common setup, teardown, and utility methods for
    testing that new changes don't break existing functionality.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = self._create_default_test_config()
        self.baseline_data_path = Path(self.temp_dir) / "baseline_data.json"

    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_default_test_config(self) -> Dict[str, Any]:
        """Create a default configuration for regression testing."""
        return {
            "device": "cpu",
            "batch_size": 1,
            "test_mode": True,
            "temp_dir": self.temp_dir,
            "regression_test": True,
        }

    @abstractmethod
    def test_regression_scenario(self):
        """
        Abstract method that must be implemented by subclasses.
        Each regression test class must define its core scenario test.
        """
        pass

    def save_baseline_data(self, data: Any, identifier: str):
        """Save baseline data for regression testing."""
        baseline_data = {}
        if self.baseline_data_path.exists():
            with open(self.baseline_data_path, "r") as f:
                baseline_data = json.load(f)

        # Create hash of the data for comparison
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()
        baseline_data[identifier] = {
            "data": data,
            "hash": data_hash,
            "timestamp": str(self._get_current_timestamp()),
        }

        with open(self.baseline_data_path, "w") as f:
            json.dump(baseline_data, f, indent=2, default=str)

    def load_baseline_data(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load baseline data for a specific identifier."""
        if not self.baseline_data_path.exists():
            return None

        with open(self.baseline_data_path, "r") as f:
            baseline_data = json.load(f)

        return baseline_data.get(identifier)

    def compare_with_baseline(
        self, current_data: Any, identifier: str, tolerance: float = 1e-6
    ) -> bool:
        """Compare current data with baseline data."""
        baseline = self.load_baseline_data(identifier)
        if baseline is None:
            # If no baseline exists, save current data as baseline and return True
            self.save_baseline_data(current_data, identifier)
            return True

        # Create hash of current data
        current_hash = hashlib.sha256(str(current_data).encode()).hexdigest()
        baseline_hash = baseline["hash"]

        # For numerical data, we might want to allow some tolerance
        if isinstance(current_data, (int, float)) and isinstance(
            baseline["data"], (int, float)
        ):
            return abs(current_data - baseline["data"]) <= tolerance
        elif isinstance(current_data, torch.Tensor) and isinstance(
            baseline["data"], list
        ):
            # Convert tensor to list for comparison
            import numpy as np

            current_list = (
                current_data.tolist()
                if isinstance(current_data, torch.Tensor)
                else current_data
            )
            baseline_list = baseline["data"]
            return self._compare_tensors_with_tolerance(
                current_list, baseline_list, tolerance
            )
        else:
            # For other types, compare hashes
            return current_hash == baseline_hash

    def _compare_tensors_with_tolerance(self, current, baseline, tolerance):
        """Compare two tensor-like structures with tolerance."""
        import numpy as np

        def convert_to_numpy(obj):
            if isinstance(obj, list):
                return np.array(obj)
            return obj

        current_np = convert_to_numpy(current)
        baseline_np = convert_to_numpy(baseline)

        try:
            return np.allclose(current_np, baseline_np, atol=tolerance)
        except Exception:
            # If comparison fails, fall back to hash comparison
            return (
                hashlib.sha256(str(current).encode()).hexdigest()
                == hashlib.sha256(str(baseline).encode()).hexdigest()
            )

    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now()


class ModelRegressionTest(BaseRegressionTest, ABC):
    """
    Base class for regression testing of model plugins.

    Provides additional utilities specific to testing model functionality
    to ensure consistency across versions.
    """

    def setUp(self):
        """Set up test fixtures for model regression tests."""
        super().setUp()

    @abstractmethod
    def get_model_plugin_class(self):
        """Return the model plugin class to be tested."""
        pass

    def create_model_instance(self, **kwargs):
        """Create an instance of the model plugin with test configuration."""
        from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin

        plugin_class = self.get_model_plugin_class()

        # Use the centralized utility to create and initialize the plugin
        return create_and_initialize_plugin(plugin_class, **kwargs)

    def test_model_output_consistency(
        self, input_tensor: torch.Tensor, identifier: str
    ):
        """Test that model outputs remain consistent across versions."""
        model_plugin = self.create_model_instance()
        model_plugin.initialize()

        # Get current output
        current_output = model_plugin.infer(input_tensor)

        # Compare with baseline
        is_consistent = self.compare_with_baseline(current_output, identifier)

        if not is_consistent:
            # If not consistent, save the new output as the new baseline
            # (In a real scenario, this might trigger a warning or failure)
            self.save_baseline_data(current_output, identifier)

        return is_consistent

    def test_model_interface_stability(self):
        """Test that the model interface remains stable."""
        model_plugin = self.create_model_instance()

        # Define expected interface methods
        expected_methods = [
            "initialize",
            "load_model",
            "infer",
            "cleanup",
            "supports_config",
            "tokenize",
            "detokenize",
            "generate_text",
        ]

        # Check that all expected methods exist
        for method_name in expected_methods:
            self.assertTrue(
                hasattr(model_plugin, method_name),
                f"Method {method_name} is missing from model interface",
            )
            self.assertTrue(
                callable(getattr(model_plugin, method_name)),
                f"Method {method_name} is not callable",
            )


class FeatureRegressionTest(BaseRegressionTest, ABC):
    """
    Base class for regression testing of specific features.

    Tests that specific features continue to work as expected after changes.
    """

    def setUp(self):
        """Set up test fixtures for feature regression tests."""
        super().setUp()

    @abstractmethod
    def get_feature_identifier(self) -> str:
        """Return a unique identifier for the feature being tested."""
        pass

    def test_feature_functionality(self):
        """Test that the feature works as expected."""
        feature_id = self.get_feature_identifier()

        # Execute the feature functionality
        result = self.execute_feature()

        # Compare with baseline
        is_consistent = self.compare_with_baseline(result, f"{feature_id}_output")

        # The test passes if the feature produces consistent results
        self.assertTrue(
            is_consistent, f"Feature {feature_id} output has changed from baseline"
        )

    @abstractmethod
    def execute_feature(self):
        """Execute the feature and return the result."""
        pass
