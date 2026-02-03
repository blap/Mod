"""
Regression Testing Module for Mod Project

This module provides independent functionality for regression testing
of different aspects of the Mod project. Each model/plugin is independent 
with its own configuration, tests and benchmarks.
"""

import unittest
import sys
import os
import hashlib
import json
from typing import Type, Any, Dict, List, Callable
import pickle
import logging
from pathlib import Path

# Add the src directory to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)


class RegressionTestBase(unittest.TestCase):
    """
    Base class for regression tests.
    Provides common functionality for comparing current behavior with baseline.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        logger.info(f"Setting up regression test: {self._testMethodName}")
        self.baseline_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'baseline_data'))
        self.baseline_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        logger.info(f"Tearing down regression test: {self._testMethodName}")
    
    def save_baseline_result(self, test_name: str, result: Any):
        """
        Save a baseline result for comparison in future tests.
        
        Args:
            test_name: Name of the test
            result: Result to save as baseline
        """
        baseline_file = self.baseline_dir / f"{test_name}_baseline.pkl"
        with open(baseline_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved baseline result for {test_name}")
    
    def load_baseline_result(self, test_name: str) -> Any:
        """
        Load a baseline result for comparison.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Baseline result, or None if not found
        """
        baseline_file = self.baseline_dir / f"{test_name}_baseline.pkl"
        if baseline_file.exists():
            with open(baseline_file, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"Loaded baseline result for {test_name}")
            return result
        else:
            logger.warning(f"No baseline found for {test_name}")
            return None
    
    def compare_results(self, current_result: Any, baseline_result: Any, tolerance: float = 1e-6) -> bool:
        """
        Compare current result with baseline result.
        
        Args:
            current_result: Current test result
            baseline_result: Baseline result to compare against
            tolerance: Tolerance for floating point comparisons
            
        Returns:
            True if results match within tolerance, False otherwise
        """
        if type(current_result) != type(baseline_result):
            return False
        
        if isinstance(current_result, (int, float)):
            return abs(current_result - baseline_result) <= tolerance
        elif isinstance(current_result, (list, tuple)):
            if len(current_result) != len(baseline_result):
                return False
            return all(self.compare_results(c, b, tolerance) 
                      for c, b in zip(current_result, baseline_result))
        elif isinstance(current_result, dict):
            if set(current_result.keys()) != set(baseline_result.keys()):
                return False
            return all(self.compare_results(current_result[k], baseline_result[k], tolerance) 
                      for k in current_result.keys())
        else:
            return current_result == baseline_result
    
    def assert_regression_match(self, current_result: Any, test_name: str, tolerance: float = 1e-6):
        """
        Assert that current result matches the baseline result.
        
        Args:
            current_result: Current test result
            test_name: Name of the test
            tolerance: Tolerance for floating point comparisons
        """
        baseline_result = self.load_baseline_result(test_name)
        if baseline_result is None:
            # If no baseline exists, save current result as baseline
            self.save_baseline_result(test_name, current_result)
            self.skipTest(f"Baseline created for {test_name}, skipping comparison")
        
        match = self.compare_results(current_result, baseline_result, tolerance)
        self.assertTrue(match, f"Current result does not match baseline for {test_name}")


class ModelRegressionTest(RegressionTestBase):
    """
    Regression test class for model plugins.
    Tests that model outputs remain consistent across versions.
    """
    
    def get_model_plugin_class(self):
        """Override this method to return the model plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_model_plugin_class")
    
    def setUp(self):
        """Set up the model plugin for regression testing."""
        super().setUp()
        self.model_plugin_class = self.get_model_plugin_class()
        self.model_instance = None
        
        # Initialize the model plugin
        try:
            self.model_instance = self.model_plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize model plugin: {str(e)}")
    
    def test_model_output_consistency(self):
        """Test that model outputs remain consistent with baseline."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        test_inputs = [
            "Simple test input",
            "Another test case",
            "Input with numbers 12345",
            "Special characters !@#$%"
        ]
        
        for i, test_input in enumerate(test_inputs):
            with self.subTest(input_index=i):
                try:
                    current_output = self.model_instance.process(test_input)
                    test_name = f"model_output_consistency_{i}"
                    self.assert_regression_match(current_output, test_name)
                except Exception as e:
                    self.fail(f"Model output consistency test failed: {str(e)}")
    
    def test_model_configuration_stability(self):
        """Test that model configuration remains stable."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        try:
            current_config = self.model_instance.get_config()
            test_name = "model_configuration_stability"
            self.assert_regression_match(current_config, test_name)
        except Exception as e:
            self.fail(f"Model configuration stability test failed: {str(e)}")
    
    def test_model_interface_compatibility(self):
        """Test that model interface remains compatible."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        # Test that required methods still exist
        required_methods = ['process', 'get_config', 'get_name']
        for method_name in required_methods:
            self.assertTrue(
                hasattr(self.model_instance, method_name),
                f"Model should have method '{method_name}'"
            )
        
        # Test that method signatures haven't changed
        try:
            # Test process method with standard input
            result = self.model_instance.process("test compatibility")
            self.assertIsNotNone(result, "Process method should return a result")
        except Exception as e:
            self.fail(f"Interface compatibility test failed: {str(e)}")


class FeatureRegressionTest(RegressionTestBase):
    """
    Regression test class for specific features.
    Tests that specific features behave consistently across versions.
    """
    
    def get_feature_component(self):
        """Override this method to return the component with the feature to test."""
        raise NotImplementedError("Subclasses must implement get_feature_component")
    
    def get_feature_name(self):
        """Override this method to return the name of the feature being tested."""
        raise NotImplementedError("Subclasses must implement get_feature_name")
    
    def setUp(self):
        """Set up the feature component for regression testing."""
        super().setUp()
        self.feature_component = self.get_feature_component()
        self.feature_name = self.get_feature_name()
        
        if self.feature_component is None:
            self.skipTest(f"Feature component for {self.feature_name} not available")
    
    def test_feature_behavior_consistency(self):
        """Test that feature behavior remains consistent with baseline."""
        if self.feature_component is None:
            self.skipTest("Feature component not available")
        
        try:
            # Execute the feature with standard inputs
            current_result = self.execute_feature_test()
            test_name = f"feature_{self.feature_name}_behavior"
            self.assert_regression_match(current_result, test_name)
        except Exception as e:
            self.fail(f"Feature behavior consistency test failed: {str(e)}")
    
    def execute_feature_test(self):
        """
        Execute the specific feature test.
        Override this method to implement the actual feature test logic.
        """
        raise NotImplementedError("Subclasses must implement execute_feature_test")


class SystemRegressionTest(RegressionTestBase):
    """
    Regression test class for system-level functionality.
    Tests that system-level behaviors remain consistent.
    """
    
    def get_system_component(self):
        """Override this method to return the system component to test."""
        raise NotImplementedError("Subclasses must implement get_system_component")
    
    def setUp(self):
        """Set up the system component for regression testing."""
        super().setUp()
        self.system_component = self.get_system_component()
        
        if self.system_component is None:
            self.skipTest("System component not available")
    
    def test_system_state_consistency(self):
        """Test that system state remains consistent."""
        if self.system_component is None:
            self.skipTest("System component not available")
        
        try:
            current_state = self.system_component.get_state()
            test_name = "system_state_consistency"
            self.assert_regression_match(current_state, test_name)
        except Exception as e:
            self.fail(f"System state consistency test failed: {str(e)}")
    
    def test_system_workflow_consistency(self):
        """Test that system workflows remain consistent."""
        if self.system_component is None:
            self.skipTest("System component not available")
        
        try:
            # Execute a standard workflow
            workflow_result = self.execute_standard_workflow()
            test_name = "system_workflow_consistency"
            self.assert_regression_match(workflow_result, test_name)
        except Exception as e:
            self.fail(f"System workflow consistency test failed: {str(e)}")
    
    def execute_standard_workflow(self):
        """
        Execute a standard system workflow.
        Override this method to implement the actual workflow test logic.
        """
        raise NotImplementedError("Subclasses must implement execute_standard_workflow")


def run_regression_tests(test_classes: List[Type[unittest.TestCase]], verbosity: int = 2):
    """
    Run regression tests with specified test classes.
    
    Args:
        test_classes: List of test classes to run
        verbosity: Verbosity level for test output
    
    Returns:
        TestResult object with results
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def regression_test_suite():
    """
    Create a test suite for regression tests.
    
    Returns:
        TestSuite object containing all regression tests
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), '..', 'tests', 'regression'),
        pattern='test_*.py',
        top_level_dir=os.path.join(os.path.dirname(__file__), '..')
    )
    return suite


# Example usage and test runner
if __name__ == "__main__":
    # This would typically be called from the main test runner
    # For demonstration purposes, we'll show the structure
    print("Regression Testing Module loaded successfully")
    print("Available test classes:")
    print("- RegressionTestBase")
    print("- ModelRegressionTest") 
    print("- FeatureRegressionTest")
    print("- SystemRegressionTest")