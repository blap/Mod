#!/usr/bin/env python
"""
Test script to verify that the plugin system integration works correctly.
"""

import sys
import os
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_plugin_system():
    """Test basic plugin system functionality."""
    print("Testing plugin system integration...")
    
    # Test importing core components
    from qwen3_vl.plugin_system.core import (
        BasePlugin, PluginMetadata, PluginType, PluginState, PluginConfig
    )
    print("+ Core components imported successfully")

    # Test importing registry
    from qwen3_vl.plugin_system.registry import PluginRegistry, get_plugin_registry
    print("+ Registry components imported successfully")

    # Test importing lifecycle
    from qwen3_vl.plugin_system.lifecycle import PluginLifecycleManager, get_lifecycle_manager
    print("+ Lifecycle components imported successfully")

    # Test importing other modules
    from qwen3_vl.plugin_system.discovery import PluginDiscovery, PluginLoader
    from qwen3_vl.plugin_system.config import PluginConfigurationManager
    from qwen3_vl.plugin_system.validation import PluginValidator
    from qwen3_vl.plugin_system.compatibility import LegacyPluginWrapper, CompatibilityLayer
    print("+ All plugin system modules imported successfully")

    # Test creating a simple plugin
    class TestPlugin(BasePlugin):
        def __init__(self):
            metadata = PluginMetadata(
                name="test_plugin",
                version="1.0.0",
                description="Test plugin for integration verification",
                author="Test Author",
                plugin_type=PluginType.CUSTOM,
                dependencies=[],
                compatibility=["1.0.0"]
            )
            super().__init__(metadata)

        def activate(self) -> bool:
            return super().activate()

        def deactivate(self) -> bool:
            return super().deactivate()

    # Test plugin creation and basic functionality
    test_plugin = TestPlugin()
    print(f"+ Plugin created: {test_plugin.get_metadata().name}")
    print(f"+ Initial state: {test_plugin.state}")

    # Test plugin initialization
    config = PluginConfig("test_plugin", {"test_param": "test_value"})
    init_result = test_plugin.initialize(config)
    print(f"+ Plugin initialized: {init_result}")
    print(f"+ State after initialization: {test_plugin.state}")

    # Test plugin activation
    activate_result = test_plugin.activate()
    print(f"+ Plugin activated: {activate_result}")
    print(f"+ State after activation: {test_plugin.state}")

    # Test registry functionality
    registry = get_plugin_registry()
    registry.register_plugin(test_plugin)
    print("+ Plugin registered in registry")

    retrieved_plugin = registry.get_plugin("test_plugin")
    print(f"+ Plugin retrieved from registry: {retrieved_plugin is not None}")

    # Test lifecycle manager
    lifecycle_manager = get_lifecycle_manager()
    deactivate_result = lifecycle_manager.deactivate_plugin("test_plugin")
    print(f"+ Plugin deactivated via lifecycle manager: {deactivate_result}")

    # Clean up
    lifecycle_manager.unload_plugin("test_plugin")
    print("+ Plugin unloaded from registry")

    print("\n+ All plugin system integration tests passed!")


if __name__ == "__main__":
    test_plugin_system()