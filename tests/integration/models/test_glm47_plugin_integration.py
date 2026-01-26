"""
Standardized Test for Plugin Integration - GLM-4.7

This module tests the plugin system integration for the GLM-4.7 model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from inference_pio.plugin_system.plugin_manager import PluginManager, get_plugin_manager


def test_plugin_registration():
    """Test that the plugin can be registered with the plugin manager."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register the plugin
    success = manager.register_plugin(plugin)
    assert_true(success)

    # Verify it's in the registry
    registered_plugins = manager.list_plugins()
    assert_in(plugin.metadata.name, registered_plugins)

    # Get the plugin back
    retrieved_plugin = manager.get_plugin(plugin.metadata.name)
    assert_is_not_none(retrieved_plugin)
    assert_equal(retrieved_plugin.metadata.name, plugin.metadata.name)


def test_plugin_activation():
    """Test that the plugin can be activated."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register the plugin first
    manager.register_plugin(plugin)

    # Activate the plugin
    success = manager.activate_plugin(plugin.metadata.name, device="cpu")
    assert_true(success)

    # Verify it's active
    active_plugins = manager.list_active_plugins()
    assert_in(plugin.metadata.name, active_plugins)

    # Verify the plugin is actually initialized
    assert_true(hasattr(plugin, '_initialized') and plugin._initialized)


def test_plugin_execution():
    """Test executing the plugin through the plugin manager."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register and activate the plugin
    manager.register_plugin(plugin)
    success = manager.activate_plugin(plugin.metadata.name)
    assert_true(success)

    # Execute the plugin
    result = manager.execute_plugin(plugin.metadata.name)
    assert_is_not_none(result)


def test_plugin_deactivation():
    """Test that the plugin can be deactivated."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register and activate the plugin
    manager.register_plugin(plugin)
    success = manager.activate_plugin(plugin.metadata.name)
    assert_true(success)

    # Verify it's active
    active_plugins = manager.list_active_plugins()
    assert_in(plugin.metadata.name, active_plugins)

    # Deactivate the plugin
    success = manager.deactivate_plugin(plugin.metadata.name)
    assert_true(success)

    # Verify it's no longer active
    active_plugins_after = manager.list_active_plugins()
    assert_not_in(plugin.metadata.name, active_plugins)


def test_multiple_plugins_management():
    """Test managing multiple plugins."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register multiple instances or different plugins
    manager.register_plugin(plugin)

    # Register another plugin (same type for testing purposes)
    plugin2 = create_glm_4_7_flash_plugin()
    # Change name slightly to avoid conflicts
    plugin2.metadata.name = "GLM-4.7-Flash-Test2"
    manager.register_plugin(plugin2)

    # Verify both are registered
    registered_plugins = manager.list_plugins()
    assert_in(plugin.metadata.name, registered_plugins)
    assert_in(plugin2.metadata.name, registered_plugins)

    # Activate both
    success1 = manager.activate_plugin(plugin.metadata.name, device="cpu")
    success2 = manager.activate_plugin(plugin2.metadata.name, device="cpu")
    assert_true(success1)
    assert_true(success2)

    # Verify both are active
    active_plugins = manager.list_active_plugins()
    assert_in(plugin.metadata.name, active_plugins)
    assert_in(plugin2.metadata.name, active_plugins)


def test_plugin_metadata_consistency():
    """Test that plugin metadata is consistent."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register the plugin
    manager.register_plugin(plugin)

    # Retrieve and check metadata
    retrieved_plugin = manager.get_plugin(plugin.metadata.name)
    assert_equal(retrieved_plugin.metadata.name, plugin.metadata.name)
    assert_equal(retrieved_plugin.metadata.version, plugin.metadata.version)
    assert_equal(retrieved_plugin.metadata.description, plugin.metadata.description)


def test_plugin_cleanup_on_deactivation():
    """Test that plugin cleanup occurs during deactivation."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Register and activate the plugin
    manager.register_plugin(plugin)
    success = manager.activate_plugin(plugin.metadata.name, device="cpu")
    assert_true(success)

    # Verify plugin is loaded
    assert_true(hasattr(plugin, 'is_loaded') and plugin.is_loaded)

    # Deactivate the plugin
    success = manager.deactivate_plugin(plugin.metadata.name)
    assert_true(success)


def test_plugin_error_handling():
    """Test error handling in plugin operations."""
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    # Try to get a non-existent plugin
    non_existent = manager.get_plugin("NonExistentPlugin")
    assert_is_none(non_existent)

    # Try to activate a non-existent plugin
    success = manager.activate_plugin("NonExistentPlugin")
    assert_false(success)

    # Try to execute a non-existent plugin
    result = manager.execute_plugin("NonExistentPlugin")
    assert_is_none(result)


def test_plugin_lifecycle():
    """Test complete plugin lifecycle: register -> activate -> execute -> deactivate -> cleanup."""
    plugin = create_glm_4_7_flash_plugin()
    manager = get_plugin_manager()
    # Clear any existing plugins for clean test
    manager.plugins.clear()
    manager.active_plugins.clear()

    plugin_name = plugin.metadata.name

    # Register
    registered = manager.register_plugin(plugin)
    assert_true(registered)

    # Activate
    activated = manager.activate_plugin(plugin_name)
    assert_true(activated)

    # Execute
    result = manager.execute_plugin(plugin_name)
    assert_is_not_none(result)

    # Deactivate
    deactivated = manager.deactivate_plugin(plugin_name)
    assert_true(deactivated)

    # Verify deactivation
    active_list = manager.list_active_plugins()
    assert_not_in(plugin_name, active_list)


def test_global_plugin_manager_access():
    """Test accessing plugin manager through global interface."""
    plugin = create_glm_4_7_flash_plugin()
    # Get global plugin manager
    global_manager = get_plugin_manager()

    # Should be the same instance
    assert_is_instance(global_manager, PluginManager)

    # Register through global interface
    from inference_pio.plugin_system import register_plugin
    register_success = register_plugin(plugin)
    assert_true(register_success)

    # Verify registration
    registered_plugins = global_manager.list_plugins()
    assert_in(plugin.metadata.name, registered_plugins)


if __name__ == '__main__':
    run_tests([
        test_plugin_registration,
        test_plugin_activation,
        test_plugin_execution,
        test_plugin_deactivation,
        test_multiple_plugins_management,
        test_plugin_metadata_consistency,
        test_plugin_cleanup_on_deactivation,
        test_plugin_error_handling,
        test_plugin_lifecycle,
        test_global_plugin_manager_access
    ])