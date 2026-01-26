"""
Tests for security resource management and isolation.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import shutil

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, 
    assert_is_none, assert_is_not_none, assert_in, assert_not_in, 
    assert_greater, assert_less, assert_is_instance, assert_raises, 
    run_tests
)


def test_resource_manager_creation():
    """Test creation of resource manager."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    assert_is_not_none(resource_manager, "Resource manager should be created successfully")
    assert_is_instance(resource_manager, ResourceManager, "Should be instance of ResourceManager")


def test_plugin_isolation_initialization():
    """Test initialization of plugin isolation."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for a plugin
    init_result = resource_manager.initialize_plugin_isolation(plugin_id="isolation_test")
    assert_true(init_result, "Plugin isolation initialization should succeed")
    
    # Verify that the plugin is registered
    registered_plugins = resource_manager.list_registered_plugins()
    assert_in("isolation_test", registered_plugins, "Initialized plugin should be registered")


def test_resource_limit_enforcement():
    """Test enforcement of resource limits."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for a plugin
    resource_manager.initialize_plugin_isolation(plugin_id="limits_test")
    
    # Enforce resource limits
    limits_result = resource_manager.enforce_resource_limits("limits_test")
    assert_true(limits_result, "Resource limit enforcement should succeed")


def test_path_access_validation():
    """Test validation of path access."""
    from src.inference_pio.security.resource_manager import ResourceManager
    import tempfile
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for a plugin
    resource_manager.initialize_plugin_isolation(plugin_id="path_test")
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
    
    try:
        # Validate access to the temporary file
        access_result = resource_manager.validate_path_access("path_test", tmp_file_path)
        # The result depends on the implementation, but it should return some kind of boolean or result object
        assert_is_instance(access_result, (bool, type(None)), "Path access validation should return a boolean or None")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)


def test_operation_tracking():
    """Test tracking of operations."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for a plugin
    resource_manager.initialize_plugin_isolation(plugin_id="tracking_test")
    
    # Begin an operation
    token = resource_manager.begin_operation("tracking_test")
    assert_is_not_none(token, "Operation token should be returned")
    
    # End the operation
    end_result = resource_manager.end_operation("tracking_test")
    assert_true(end_result, "Ending operation should succeed")


def test_resource_usage_monitoring():
    """Test monitoring of plugin resource usage."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for a plugin
    resource_manager.initialize_plugin_isolation(plugin_id="monitoring_test")
    
    # Get resource usage for the plugin
    usage = resource_manager.get_plugin_resource_usage("monitoring_test")
    assert_is_not_none(usage, "Resource usage should be returned")
    assert_in("plugin_id", usage, "Usage info should contain plugin ID")
    assert_equal(usage["plugin_id"], "monitoring_test", "Usage info should have correct plugin ID")


def test_resource_manager_cleanup():
    """Test cleanup of resource manager."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for a plugin
    resource_manager.initialize_plugin_isolation(plugin_id="cleanup_test")
    
    # Verify the plugin is registered
    registered_before = resource_manager.list_registered_plugins()
    assert_in("cleanup_test", registered_before, "Plugin should be registered before cleanup")
    
    # Perform cleanup
    cleanup_result = resource_manager.cleanup_plugin_resources("cleanup_test")
    assert_true(cleanup_result, "Plugin resource cleanup should succeed")
    
    # Verify the plugin is no longer registered
    registered_after = resource_manager.list_registered_plugins()
    assert_not_in("cleanup_test", registered_after, "Plugin should not be registered after cleanup")


def test_multiple_plugin_isolation():
    """Test isolation between multiple plugins."""
    from src.inference_pio.security.resource_manager import ResourceManager
    
    resource_manager = ResourceManager()
    
    # Initialize isolation for multiple plugins
    resource_manager.initialize_plugin_isolation(plugin_id="multi_test_1")
    resource_manager.initialize_plugin_isolation(plugin_id="multi_test_2")
    
    # Both plugins should be registered
    registered_plugins = resource_manager.list_registered_plugins()
    assert_in("multi_test_1", registered_plugins, "First plugin should be registered")
    assert_in("multi_test_2", registered_plugins, "Second plugin should be registered")
    
    # Resources for each plugin should be isolated
    usage1 = resource_manager.get_plugin_resource_usage("multi_test_1")
    usage2 = resource_manager.get_plugin_resource_usage("multi_test_2")
    
    assert_is_not_none(usage1, "First plugin should have resource usage info")
    assert_is_not_none(usage2, "Second plugin should have resource usage info")
    assert_not_equal(id(usage1), id(usage2), "Resource usage objects should be different")


def run_tests():
    """Run all resource management tests."""
    print("Running resource management tests...")
    
    test_functions = [
        test_resource_manager_creation,
        test_plugin_isolation_initialization,
        test_resource_limit_enforcement,
        test_path_access_validation,
        test_operation_tracking,
        test_resource_usage_monitoring,
        test_resource_manager_cleanup,
        test_multiple_plugin_isolation
    ]
    
    all_passed = True
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All resource management tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)