"""
Integration Testing Module for Mod Project

This module provides independent functionality for integration testing
of different aspects of the Mod project. Each model/plugin is independent 
with its own configuration, tests and benchmarks.
"""

import unittest
import sys
import os
from typing import Type, Any, Dict, List
import logging
from contextlib import contextmanager

# Add the src directory to the path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)


class IntegrationTestBase(unittest.TestCase):
    """
    Base class for integration tests.
    Provides common functionality for testing interactions between components.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        super().setUp()
        logger.info(f"Setting up integration test: {self._testMethodName}")
        
    def tearDown(self):
        """Clean up after each test method."""
        super().tearDown()
        logger.info(f"Tearing down integration test: {self._testMethodName}")
    
    def assert_component_interaction(self, component_a, component_b, interaction_method: str):
        """Assert that two components can interact properly."""
        self.assertTrue(
            hasattr(component_a, interaction_method),
            f"Component A should have method '{interaction_method}'"
        )
        self.assertTrue(
            hasattr(component_b, interaction_method),
            f"Component B should have method '{interaction_method}'"
        )
    
    @contextmanager
    def temporary_config(self, config_updates: Dict[str, Any]):
        """Context manager to temporarily update configuration."""
        original_values = {}
        # This would typically interact with a config system
        # For now, we'll just log the intended changes
        logger.info(f"Applying temporary config updates: {config_updates}")
        try:
            yield
        finally:
            logger.info("Restoring original configuration")


class ModelIntegrationTest(IntegrationTestBase):
    """
    Integration test class for model plugins interacting with other components.
    Each model should inherit from this class and implement required methods.
    """
    
    def get_model_plugin_class(self):
        """Override this method to return the model plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_model_plugin_class")
    
    def get_related_components(self):
        """Override this method to return related components to test integration with."""
        return []
    
    def setUp(self):
        """Set up the model plugin and related components for integration testing."""
        super().setUp()
        self.model_plugin_class = self.get_model_plugin_class()
        self.model_instance = None
        self.related_components = self.get_related_components()
        
        # Initialize the model plugin
        try:
            self.model_instance = self.model_plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize model plugin: {str(e)}")
    
    def test_model_with_pipeline_integration(self):
        """Test integration between model and pipeline components."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        # Look for pipeline components in related components
        pipeline_components = [comp for comp in self.related_components 
                              if 'pipeline' in comp.__class__.__name__.lower()]
        
        for pipeline_comp in pipeline_components:
            with self.subTest(pipeline_component=pipeline_comp.__class__.__name__):
                # Test that model can work with pipeline
                try:
                    # Example integration test - adjust based on actual interfaces
                    if hasattr(pipeline_comp, 'add_model') and self.model_instance:
                        pipeline_comp.add_model(self.model_instance)
                        self.assertTrue(True, "Model successfully added to pipeline")
                except Exception as e:
                    self.fail(f"Integration with pipeline component failed: {str(e)}")
    
    def test_model_with_plugins_integration(self):
        """Test integration between model and plugin components."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        # Look for plugin components in related components
        plugin_components = [comp for comp in self.related_components 
                            if 'plugin' in comp.__class__.__name__.lower()]
        
        for plugin_comp in plugin_components:
            with self.subTest(plugin_component=plugin_comp.__class__.__name__):
                # Test that model can work with plugins
                try:
                    # Example integration test - adjust based on actual interfaces
                    if hasattr(plugin_comp, 'set_model') and self.model_instance:
                        plugin_comp.set_model(self.model_instance)
                        self.assertTrue(True, "Model successfully integrated with plugin")
                except Exception as e:
                    self.fail(f"Integration with plugin component failed: {str(e)}")
    
    def test_end_to_end_workflow(self):
        """Test a complete workflow involving the model and other components."""
        if not self.model_instance:
            self.skipTest("Model instance not available")
        
        # Example end-to-end workflow test
        try:
            # Initialize components
            # Configure workflow
            # Execute workflow
            # Validate results
            result = self.model_instance.process("test input for end-to-end workflow")
            self.assertIsNotNone(result, "End-to-end workflow should produce a result")
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {str(e)}")


class PipelineIntegrationTest(IntegrationTestBase):
    """
    Integration test class for pipeline components.
    Tests integration between various pipeline elements.
    """
    
    def get_pipeline_class(self):
        """Override this method to return the pipeline class to test."""
        raise NotImplementedError("Subclasses must implement get_pipeline_class")
    
    def get_pipeline_components(self):
        """Override this method to return components to test in the pipeline."""
        return []
    
    def setUp(self):
        """Set up the pipeline and components for integration testing."""
        super().setUp()
        self.pipeline_class = self.get_pipeline_class()
        self.pipeline_instance = None
        self.components = self.get_pipeline_components()
        
        # Initialize the pipeline
        try:
            self.pipeline_instance = self.pipeline_class()
        except Exception as e:
            self.fail(f"Failed to initialize pipeline: {str(e)}")
    
    def test_pipeline_component_flow(self):
        """Test that components flow correctly through the pipeline."""
        if not self.pipeline_instance:
            self.skipTest("Pipeline instance not available")
        
        for component in self.components:
            try:
                # Add component to pipeline
                self.pipeline_instance.add_component(component)
                
                # Verify component was added
                self.assertIn(
                    component, 
                    self.pipeline_instance.get_components(),
                    f"Component {component.__class__.__name__} should be in pipeline"
                )
            except Exception as e:
                self.fail(f"Adding component to pipeline failed: {str(e)}")
    
    def test_pipeline_execution(self):
        """Test executing the entire pipeline."""
        if not self.pipeline_instance:
            self.skipTest("Pipeline instance not available")
        
        try:
            # Execute the pipeline with test data
            result = self.pipeline_instance.execute("test input for pipeline")
            self.assertIsNotNone(result, "Pipeline execution should produce a result")
        except Exception as e:
            self.fail(f"Pipeline execution failed: {str(e)}")


class PluginIntegrationTest(IntegrationTestBase):
    """
    Integration test class for plugin components.
    Tests integration between plugins and other system components.
    """
    
    def get_plugin_class(self):
        """Override this method to return the plugin class to test."""
        raise NotImplementedError("Subclasses must implement get_plugin_class")
    
    def get_host_system_class(self):
        """Override this method to return the host system class to test integration with."""
        return None
    
    def setUp(self):
        """Set up the plugin and host system for integration testing."""
        super().setUp()
        self.plugin_class = self.get_plugin_class()
        self.host_system_class = self.get_host_system_class()
        self.plugin_instance = None
        self.host_system_instance = None
        
        # Initialize the plugin
        try:
            self.plugin_instance = self.plugin_class()
        except Exception as e:
            self.fail(f"Failed to initialize plugin: {str(e)}")
        
        # Initialize the host system if provided
        if self.host_system_class:
            try:
                self.host_system_instance = self.host_system_class()
            except Exception as e:
                self.fail(f"Failed to initialize host system: {str(e)}")
    
    def test_plugin_registration(self):
        """Test that the plugin can register with the host system."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
        
        if not self.host_system_instance:
            self.skipTest("Host system instance not available")
        
        try:
            # Register plugin with host system
            self.host_system_instance.register_plugin(self.plugin_instance)
            
            # Verify registration
            registered_plugins = self.host_system_instance.get_registered_plugins()
            self.assertIn(
                self.plugin_instance,
                registered_plugins,
                "Plugin should be registered with host system"
            )
        except Exception as e:
            self.fail(f"Plugin registration failed: {str(e)}")
    
    def test_plugin_execution_in_context(self):
        """Test that the plugin executes properly in the host system context."""
        if not self.plugin_instance:
            self.skipTest("Plugin instance not available")
        
        if not self.host_system_instance:
            self.skipTest("Host system instance not available")
        
        try:
            # Register and execute plugin
            self.host_system_instance.register_plugin(self.plugin_instance)
            result = self.host_system_instance.execute_plugin(
                self.plugin_instance.__class__.__name__, 
                "test input"
            )
            self.assertIsNotNone(result, "Plugin execution in context should produce a result")
        except Exception as e:
            self.fail(f"Plugin execution in context failed: {str(e)}")


def run_integration_tests(test_classes: List[Type[unittest.TestCase]], verbosity: int = 2):
    """
    Run integration tests with specified test classes.
    
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


def integration_test_suite():
    """
    Create a test suite for integration tests.
    
    Returns:
        TestSuite object containing all integration tests
    """
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), '..', 'tests', 'integration'),
        pattern='test_*.py',
        top_level_dir=os.path.join(os.path.dirname(__file__), '..')
    )
    return suite


# Example usage and test runner
if __name__ == "__main__":
    # This would typically be called from the main test runner
    # For demonstration purposes, we'll show the structure
    print("Integration Testing Module loaded successfully")
    print("Available test classes:")
    print("- IntegrationTestBase")
    print("- ModelIntegrationTest") 
    print("- PipelineIntegrationTest")
    print("- PluginIntegrationTest")