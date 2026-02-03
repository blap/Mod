"""
Unit Testing Module for Mod Project

This module provides independent functionality for unit testing
of different aspects of the Mod project. Each model/plugin is independent 
with its own configuration, tests and benchmarks.
"""

import unittest
import sys
import os
from typing import Type, Any, Dict, List
import logging
from unittest.mock import Mock, patch, MagicMock
import inspect

# Add the src directory to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)


class UnitTestBase(unittest.TestCase):
    """
    Base class for unit tests.
    Provides common functionality for testing individual units of code in isolation.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        logger.info(f"Setting up unit test: {self._testMethodName}")
        
    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        logger.info(f"Tearing down unit test: {self._testMethodName}")
    
    def assert_method_signature(self, obj: Any, method_name: str, expected_params: List[str]):
        """
        Assert that a method has the expected signature.
        
        Args:
            obj: Object containing the method
            method_name: Name of the method to check
            expected_params: List of expected parameter names (excluding 'self')
        """
        method = getattr(obj, method_name)
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        # Remove 'self' if present
        if params and params[0] == 'self':
            params = params[1:]
        
        self.assertEqual(params, expected_params, 
                        f"Method {method_name} should have parameters {expected_params}, got {params}")
    
    def create_mock_dependencies(self, dependencies: List[str]) -> Dict[str, Mock]:
        """
        Create mock objects for dependencies.
        
        Args:
            dependencies: List of dependency names to mock
            
        Returns:
            Dictionary mapping dependency names to mock objects
        """
        mocks = {}
        for dep in dependencies:
            mocks[dep] = Mock(name=dep)
        return mocks


class ModelUnitTest(UnitTestBase):
    """
    Unit test class for model plugins.
    Tests individual units of model functionality in isolation.
    """
    
    def get_model_plugin_class(self):
        """Override this method to return the model plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_model_plugin_class")
    
    def setUp(self):
        """Set up the model plugin for unit testing."""
        super().setUp()
        self.model_plugin_class = self.get_model_plugin_class()
        self.model_instance = None
        
        # Initialize the model plugin
        try:
            self.model_instance = self.model_plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize model plugin: {str(e)}")
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.model_instance, "Model instance should be initialized")
        
        # Check that required attributes exist
        required_attributes = ['name', 'version', 'config']
        for attr in required_attributes:
            self.assertTrue(
                hasattr(self.model_instance, attr),
                f"Model should have attribute '{attr}'"
            )
    
    def test_model_required_methods_exist(self):
        """Test that required methods exist on the model."""
        self.assertIsNotNone(self.model_instance, "Model instance should be initialized")
        
        # Check required methods exist
        required_methods = ['process', 'get_config', 'get_name']
        for method_name in required_methods:
            self.assertTrue(
                hasattr(self.model_instance, method_name),
                f"Model should have method '{method_name}'"
            )
    
    def test_model_method_signatures(self):
        """Test that model methods have correct signatures."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        # Test process method signature
        self.assert_method_signature(self.model_instance, 'process', ['input_data'])
        
        # Test get_config method signature
        self.assert_method_signature(self.model_instance, 'get_config', [])
        
        # Test get_name method signature
        self.assert_method_signature(self.model_instance, 'get_name', [])
    
    def test_model_process_method(self):
        """Test the process method in isolation."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        # Test with various inputs
        test_cases = [
            "simple input",
            "",
            "input with spaces",
            "input\nwith\nnewlines",
            "input\twith\ttabs"
        ]
        
        for test_input in test_cases:
            with self.subTest(input=test_input):
                try:
                    result = self.model_instance.process(test_input)
                    # Basic validation - result should not be None
                    self.assertIsNotNone(result, f"Process should return a result for input: {test_input}")
                except Exception as e:
                    self.fail(f"Process method failed for input '{test_input}': {str(e)}")
    
    def test_model_config_access(self):
        """Test that config can be accessed properly."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        try:
            config = self.model_instance.get_config()
            self.assertIsNotNone(config, "Config should not be None")
            self.assertIsInstance(config, (dict, object), "Config should be a dictionary or object")
        except Exception as e:
            self.fail(f"Config access failed: {str(e)}")
    
    def test_model_name_access(self):
        """Test that name can be accessed properly."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        try:
            name = self.model_instance.get_name()
            self.assertIsNotNone(name, "Name should not be None")
            self.assertIsInstance(name, str, "Name should be a string")
            self.assertNotEqual(name.strip(), "", "Name should not be empty")
        except Exception as e:
            self.fail(f"Name access failed: {str(e)}")


class PluginUnitTest(UnitTestBase):
    """
    Unit test class for plugin components.
    Tests individual units of plugin functionality in isolation.
    """
    
    def get_plugin_class(self):
        """Override this method to return the plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_plugin_class")
    
    def setUp(self):
        """Set up the plugin for unit testing."""
        super().setUp()
        self.plugin_class = self.get_plugin_class()
        self.plugin_instance = None
        
        # Initialize the plugin
        try:
            self.plugin_instance = self.plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize plugin: {str(e)}")
    
    def test_plugin_initialization(self):
        """Test that the plugin initializes correctly."""
        self.assertIsNotNone(self.plugin_instance, "Plugin instance should be initialized")
        
        # Check that required attributes exist
        required_attributes = ['name', 'version', 'enabled']
        for attr in required_attributes:
            self.assertTrue(
                hasattr(self.plugin_instance, attr),
                f"Plugin should have attribute '{attr}'"
            )
    
    def test_plugin_required_methods_exist(self):
        """Test that required methods exist on the plugin."""
        self.assertIsNotNone(self.plugin_instance, "Plugin instance should be initialized")
        
        # Check required methods exist
        required_methods = ['execute', 'get_config', 'is_enabled']
        for method_name in required_methods:
            self.assertTrue(
                hasattr(self.plugin_instance, method_name),
                f"Plugin should have method '{method_name}'"
            )
    
    def test_plugin_method_signatures(self):
        """Test that plugin methods have correct signatures."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
        
        # Test execute method signature
        self.assert_method_signature(self.plugin_instance, 'execute', ['input_data'])
        
        # Test get_config method signature
        self.assert_method_signature(self.plugin_instance, 'get_config', [])
        
        # Test is_enabled method signature
        self.assert_method_signature(self.plugin_instance, 'is_enabled', [])
    
    def test_plugin_execute_method(self):
        """Test the execute method in isolation."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
        
        # Test with various inputs
        test_cases = [
            "simple input",
            "",
            "input with spaces",
            "input\nwith\nnewlines",
            "input\twith\ttabs"
        ]
        
        for test_input in test_cases:
            with self.subTest(input=test_input):
                try:
                    result = self.plugin_instance.execute(test_input)
                    # Basic validation - result should not be None
                    self.assertIsNotNone(result, f"Execute should return a result for input: {test_input}")
                except Exception as e:
                    self.fail(f"Execute method failed for input '{test_input}': {str(e)}")
    
    def test_plugin_config_access(self):
        """Test that plugin config can be accessed properly."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
        
        try:
            config = self.plugin_instance.get_config()
            self.assertIsNotNone(config, "Config should not be None")
            self.assertIsInstance(config, (dict, object), "Config should be a dictionary or object")
        except Exception as e:
            self.fail(f"Plugin config access failed: {str(e)}")
    
    def test_plugin_enabled_status(self):
        """Test that plugin enabled status can be checked."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
        
        try:
            enabled = self.plugin_instance.is_enabled()
            self.assertIsInstance(enabled, bool, "Enabled status should be a boolean")
        except Exception as e:
            self.fail(f"Plugin enabled status check failed: {str(e)}")


class ComponentUnitTest(UnitTestBase):
    """
    Unit test class for generic components.
    Tests individual units of component functionality in isolation.
    """
    
    def get_component_class(self):
        """Override this method to return the component class to test."""
        raise NotImplementedError("Subclasses must implement get_component_class")
    
    def setUp(self):
        """Set up the component for unit testing."""
        super().setUp()
        self.component_class = self.get_component_class()
        self.component_instance = None
        
        # Initialize the component
        try:
            self.component_instance = self.component_class()
        except Exception as e:
            self.fail(f"Failed to initialize component: {str(e)}")
    
    def test_component_initialization(self):
        """Test that the component initializes correctly."""
        self.assertIsNotNone(self.component_instance, "Component instance should be initialized")
    
    def test_component_required_methods_exist(self):
        """Test that required methods exist on the component."""
        if not self.component_instance:
            self.skipTest("Component instance not available")
        
        # This is a generic test - subclasses should override with specific methods
        pass
    
    def test_component_attributes(self):
        """Test that component has expected attributes."""
        if not self.component_instance:
            self.skipTest("Component instance not available")
        
        # This is a generic test - subclasses should override with specific attributes
        pass


def run_unit_tests(test_classes: List[Type[unittest.TestCase]], verbosity: int = 2):
    """
    Run unit tests with specified test classes.
    
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


def unit_test_suite():
    """
    Create a test suite for unit tests.
    
    Returns:
        TestSuite object containing all unit tests
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), '..', 'tests', 'unit'),
        pattern='test_*.py',
        top_level_dir=os.path.join(os.path.dirname(__file__), '..')
    )
    return suite


# Example usage and test runner
if __name__ == "__main__":
    # This would typically be called from the main test runner
    # For demonstration purposes, we'll show the structure
    print("Unit Testing Module loaded successfully")
    print("Available test classes:")
    print("- UnitTestBase")
    print("- ModelUnitTest") 
    print("- PluginUnitTest")
    print("- ComponentUnitTest")