"""
Tests for security contexts and isolation mechanisms.
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


def test_security_context_creation():
    """Test creation of security contexts."""
    from src.inference_pio.security.context import SecurityContext
    
    context = SecurityContext(
        plugin_id="test_plugin",
        allowed_paths=["/tmp"],
        allowed_resources={"cpu": 1.0, "memory": "1GB"},
        permissions=["read", "write"]
    )
    
    assert_equal(context.plugin_id, "test_plugin", "Security context should have correct plugin ID")
    assert_is_not_none(context.allowed_paths, "Security context should have allowed paths")
    assert_is_not_none(context.allowed_resources, "Security context should have allowed resources")
    assert_is_not_none(context.permissions, "Security context should have permissions")


def test_security_context_isolation():
    """Test security context isolation mechanisms."""
    from src.inference_pio.security.context import SecurityContext
    
    # Create two different security contexts
    context1 = SecurityContext(
        plugin_id="plugin_1",
        allowed_paths=["/tmp/plugin1"],
        allowed_resources={"cpu": 0.5, "memory": "512MB"},
        permissions=["read"]
    )
    
    context2 = SecurityContext(
        plugin_id="plugin_2",
        allowed_paths=["/tmp/plugin2"],
        allowed_resources={"cpu": 0.5, "memory": "512MB"},
        permissions=["write"]
    )
    
    # Contexts should be isolated from each other
    assert_not_equal(context1.plugin_id, context2.plugin_id, "Different plugins should have different IDs")
    assert_not_equal(context1.allowed_paths, context2.allowed_paths, "Different plugins should have different allowed paths")
    assert_not_equal(context1.permissions, context2.permissions, "Different plugins should have different permissions")


def test_security_manager_creation():
    """Test creation of security manager."""
    from src.inference_pio.security.manager import SecurityManager
    
    security_manager = SecurityManager()
    assert_is_not_none(security_manager, "Security manager should be created successfully")
    assert_is_instance(security_manager, SecurityManager, "Should be instance of SecurityManager")


def test_security_context_lifecycle():
    """Test the lifecycle of security contexts."""
    from src.inference_pio.security.manager import SecurityManager
    
    security_manager = SecurityManager()
    
    # Create a security context
    context = security_manager.create_security_context(
        plugin_id="lifecycle_test",
        allowed_paths=["/tmp/lifecycle"],
        allowed_resources={"cpu": 1.0, "memory": "1GB"}
    )
    
    assert_is_not_none(context, "Security context should be created")
    assert_equal(context.plugin_id, "lifecycle_test", "Context should have correct plugin ID")
    
    # Get the security context
    retrieved_context = security_manager.get_security_context("lifecycle_test")
    assert_is_not_none(retrieved_context, "Retrieved context should not be None")
    assert_equal(retrieved_context.plugin_id, "lifecycle_test", "Retrieved context should have correct plugin ID")
    
    # Cleanup the security context
    cleanup_result = security_manager.cleanup_security_context("lifecycle_test")
    assert_true(cleanup_result, "Cleanup should succeed")
    
    # After cleanup, the context should no longer be retrievable
    cleaned_context = security_manager.get_security_context("lifecycle_test")
    assert_is_none(cleaned_context, "Context should not be retrievable after cleanup")


def test_security_context_registry():
    """Test the security context registry functionality."""
    from src.inference_pio.security.manager import SecurityManager
    
    security_manager = SecurityManager()
    
    # Register multiple contexts
    context1 = security_manager.create_security_context(plugin_id="registry_test_1")
    context2 = security_manager.create_security_context(plugin_id="registry_test_2")
    
    assert_is_not_none(context1, "First context should be created")
    assert_is_not_none(context2, "Second context should be created")
    
    # List registered contexts
    registered_ids = security_manager.list_registered_contexts()
    assert_greater(len(registered_ids), 1, "Should have at least 2 registered contexts")
    assert_in("registry_test_1", registered_ids, "Should contain first context")
    assert_in("registry_test_2", registered_ids, "Should contain second context")


def test_security_context_update():
    """Test updating security contexts."""
    from src.inference_pio.security.manager import SecurityManager
    
    security_manager = SecurityManager()
    
    # Create initial context
    initial_context = security_manager.create_security_context(
        plugin_id="update_test",
        allowed_paths=["/tmp/initial"],
        permissions=["read"]
    )
    
    assert_equal(initial_context.allowed_paths, ["/tmp/initial"], "Initial context should have correct paths")
    assert_equal(initial_context.permissions, ["read"], "Initial context should have correct permissions")
    
    # Update the context
    update_result = security_manager.update_security_context(
        plugin_id="update_test",
        allowed_paths=["/tmp/updated"],
        permissions=["read", "write"]
    )
    
    assert_true(update_result, "Context update should succeed")
    
    # Verify the update
    updated_context = security_manager.get_security_context("update_test")
    assert_equal(updated_context.allowed_paths, ["/tmp/updated"], "Updated context should have new paths")
    assert_equal(updated_context.permissions, ["read", "write"], "Updated context should have new permissions")


def run_tests():
    """Run all security context tests."""
    print("Running security context tests...")
    
    test_functions = [
        test_security_context_creation,
        test_security_context_isolation,
        test_security_manager_creation,
        test_security_context_lifecycle,
        test_security_context_registry,
        test_security_context_update
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
        print("\n✓ All security context tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)