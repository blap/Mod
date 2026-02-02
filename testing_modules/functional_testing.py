"""
Functional Testing Module for Mod Project

This module provides independent functionality for functional testing
of different aspects of the Mod project. Each model/plugin is independent 
with its own configuration, tests and benchmarks.
"""

import unittest
import sys
import os
from typing import Type, Any, Dict, List
import time
import logging

# Add the src directory to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

class FunctionalTestBase(unittest.TestCase):
    """
    Base class for functional tests.
    Provides common functionality for all functional tests.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        logger.info(f"Setting up functional test: {self._testMethodName}")
        
    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        logger.info(f"Tearing down functional test: {self._testMethodName}")
    
    def assert_valid_output(self, output: Any, expected_type: Type = None):
        """Assert that the output is valid."""
        self.assertIsNotNone(output, "Output should not be None")
        if expected_type:
            self.assertIsInstance(output, expected_type, f"Output should be of type {expected_type.__name__}")


class ModelFunctionalTest(FunctionalTestBase):
    """
    Functional test class for model plugins.
    Each model should inherit from this class and implement required methods.
    """
    
    def get_model_plugin_class(self):
        """Override this method to return the model plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_model_plugin_class")
    
    def setUp(self):
        """Set up the model plugin for testing."""
        super().setUp()
        self.model_plugin_class = self.get_model_plugin_class()
        self.model_instance = None
        
        # Initialize the model plugin
        try:
            self.model_instance = self.model_plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize model plugin: {str(e)}")
    
    def test_required_functionality(self):
        """Test the required functionality of the model."""
        self.assertIsNotNone(self.model_instance, "Model instance should be initialized")
        
        # Test basic functionality
        if hasattr(self.model_instance, 'process'):
            # Example test - adjust based on actual model interface
            try:
                result = self.model_instance.process("test input")
                self.assert_valid_output(result)
            except Exception as e:
                self.fail(f"Model process failed: {str(e)}")
    
    def test_model_interface_compliance(self):
        """Test that the model complies with the expected interface."""
        self.assertIsNotNone(self.model_instance)
        
        # Check required methods exist
        required_methods = ['process', 'get_config', 'get_name']
        for method_name in required_methods:
            self.assertTrue(
                hasattr(self.model_instance, method_name),
                f"Model should have method '{method_name}'"
            )


class PluginFunctionalTest(FunctionalTestBase):
    """
    Functional test class for plugins.
    Each plugin should inherit from this class and implement required methods.
    """
    
    def get_plugin_class(self):
        """Override this method to return the plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_plugin_class")
    
    def setUp(self):
        """Set up the plugin for testing."""
        super().setUp()
        self.plugin_class = self.get_plugin_class()
        self.plugin_instance = None
        
        # Initialize the plugin
        try:
            self.plugin_instance = self.plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize plugin: {str(e)}")
    
    def test_plugin_functionality(self):
        """Test the core functionality of the plugin."""
        self.assertIsNotNone(self.plugin_instance, "Plugin instance should be initialized")
        
        # Test basic functionality
        if hasattr(self.plugin_instance, 'execute'):
            try:
                result = self.plugin_instance.execute("test input")
                self.assert_valid_output(result)
            except Exception as e:
                self.fail(f"Plugin execution failed: {str(e)}")


def run_functional_tests(test_classes: List[Type[unittest.TestCase]], verbosity: int = 2):
    """
    Run functional tests with specified test classes.
    
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


def functional_test_suite():
    """
    Create a test suite for functional tests.
    
    Returns:
        TestSuite object containing all functional tests
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), '..', 'tests', 'functional'),
        pattern='test_*.py',
        top_level_dir=os.path.join(os.path.dirname(__file__), '..')
    )
    return suite


# Example usage and test runner
if __name__ == "__main__":
    # This would typically be called from the main test runner
    # For demonstration purposes, we'll show the structure
    print("Functional Testing Module loaded successfully")
    print("Available test classes:")
    print("- FunctionalTestBase")
    print("- ModelFunctionalTest") 
    print("- PluginFunctionalTest")