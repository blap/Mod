"""
Standardized Test for Plugin Integration - Qwen3-4B-Instruct-2507

This module tests the plugin system integration for the Qwen3-4B-Instruct-2507 model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
from inference_pio.plugin_system.plugin_manager import PluginManager, get_plugin_manager

# TestQwen34BInstruct2507PluginIntegration

    """Test cases for Qwen3-4B-Instruct-2507 plugin system integration."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugin = create_qwen3_4b_instruct_2507_plugin()
        manager = get_plugin_manager()
        # Clear any existing plugins for clean test
        manager.plugins.clear()
        manager.active_plugins.clear()

    def plugin_registration(self)():
        """Test that the plugin can be registered with the plugin manager."""
        # Register the plugin
        success = manager.register_plugin(plugin)
        assert_true(success)
        
        # Verify it's in the registry
        registered_plugins = manager.list_plugins()
        assert_in(plugin.metadata.name)
        
        # Get the plugin back
        retrieved_plugin = manager.get_plugin(plugin.metadata.name)
        assert_is_not_none(retrieved_plugin)
        assert_equal(retrieved_plugin.metadata.name)

    def plugin_activation(self)():
        """Test that the plugin can be activated."""
        # Register the plugin first
        manager.register_plugin(plugin)
        
        # Activate the plugin
        success = manager.activate_plugin(plugin.metadata.name, device="cpu")
        assert_true(success)
        
        # Verify it's active
        active_plugins = manager.list_active_plugins()
        assert_in(plugin.metadata.name)
        
        # Verify the plugin is actually initialized
        assert_true(plugin._initialized)

    def plugin_execution(self)():
        """Test executing the plugin through the plugin manager."""
        # Register and activate the plugin
        manager.register_plugin(plugin)
        success = manager.activate_plugin(plugin.metadata.name)
        assert_true(success)
        
        # Execute the plugin
        result = manager.execute_plugin(plugin.metadata.name)))
        assert_is_not_none(result)

    def plugin_deactivation(self)():
        """Test that the plugin can be deactivated."""
        # Register and activate the plugin
        manager.register_plugin(plugin)
        success = manager.activate_plugin(plugin.metadata.name)
        assert_true(success)
        
        # Verify it's active
        active_plugins = manager.list_active_plugins()
        assertIn(plugin.metadata.name)
        
        # Deactivate the plugin
        success = manager.deactivate_plugin(plugin.metadata.name)
        assert_true(success)
        
        # Verify it's no longer active
        active_plugins_after = manager.list_active_plugins()
        assert_not_in(plugin.metadata.name)

    def multiple_plugins_management(self)():
        """Test managing multiple plugins."""
        # Register multiple instances or different plugins
        manager.register_plugin(plugin)
        
        # Register another plugin (same type for testing purposes)
        plugin2 = create_qwen3_4b_instruct_2507_plugin()
        # Change name slightly to avoid conflicts
        plugin2.metadata.name = "Qwen3-4B-Instruct-2507-Test2"
        manager.register_plugin(plugin2)
        
        # Verify both are registered
        registered_plugins = manager.list_plugins()
        assertIn(plugin.metadata.name, registered_plugins)
        assert_in(plugin2.metadata.name, registered_plugins)
        
        # Activate both
        success1 = manager.activate_plugin(plugin.metadata.name, device="cpu")
        success2 = manager.activate_plugin(plugin2.metadata.name, device="cpu")
        assert_true(success1)
        assertTrue(success2)
        
        # Verify both are active
        active_plugins = manager.list_active_plugins()
        assert_in(plugin.metadata.name)
        assertIn(plugin2.metadata.name, active_plugins)

    def plugin_metadata_consistency(self)():
        """Test that plugin metadata is consistent."""
        # Register the plugin
        manager.register_plugin(plugin)
        
        # Retrieve and check metadata
        retrieved_plugin = manager.get_plugin(plugin.metadata.name)
        assert_equal(retrieved_plugin.metadata.name, plugin.metadata.name)
        assert_equal(retrieved_plugin.metadata.version, plugin.metadata.version)
        assert_equal(retrieved_plugin.metadata.description, plugin.metadata.description)

    def plugin_cleanup_on_deactivation(self)():
        """Test that plugin cleanup occurs during deactivation."""
        # Register and activate the plugin
        manager.register_plugin(plugin)
        success = manager.activate_plugin(plugin.metadata.name, device="cpu")
        assert_true(success)
        
        # Verify plugin is loaded
        assertTrue(plugin.is_loaded)
        
        # Deactivate the plugin
        success = manager.deactivate_plugin(plugin.metadata.name)
        assertTrue(success)

    def plugin_error_handling(self)():
        """Test error handling in plugin operations."""
        # Try to get a non-existent plugin
        non_existent = manager.get_plugin("NonExistentPlugin")
        assert_is_none(non_existent)
        
        # Try to activate a non-existent plugin
        success = manager.activate_plugin("NonExistentPlugin")
        assert_false(success)
        
        # Try to execute a non-existent plugin
        result = manager.execute_plugin("NonExistentPlugin")
        assertIsNone(result)

    def plugin_lifecycle(self)():
        """Test complete plugin lifecycle: register -> activate -> execute -> deactivate -> cleanup."""
        plugin_name = plugin.metadata.name
        
        # Register
        registered = manager.register_plugin(plugin)
        assertTrue(registered)
        
        # Activate
        activated = manager.activate_plugin(plugin_name)
        assert_true(activated)
        
        # Execute
        result = manager.execute_plugin(plugin_name)))
        assert_is_not_none(result)
        
        # Deactivate
        deactivated = manager.deactivate_plugin(plugin_name)
        assert_true(deactivated)
        
        # Verify deactivation
        active_list = manager.list_active_plugins()
        assertNotIn(plugin_name)

    def global_plugin_manager_access(self)():
        """Test accessing plugin manager through global interface."""
        # Get global plugin manager
        global_manager = get_plugin_manager()
        
        # Should be the same instance
        assertIs(global_manager)
        
        # Register through global interface
        from inference_pio.plugin_system import register_plugin
        register_success = register_plugin(plugin)
        assert_true(register_success)
        
        # Verify registration
        registered_plugins = global_manager.list_plugins()
        assertIn(plugin.metadata.name)

    def cleanup_helper():
        """Clean up after each test method."""
        # Clean up all plugins
        manager.plugins.clear()
        manager.active_plugins.clear()
        
        if hasattr(plugin) and plugin.is_loaded:
            plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)