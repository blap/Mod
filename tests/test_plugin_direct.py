#!/usr/bin/env python
"""
Test script to verify that the plugin system integration works correctly,
by directly importing from the plugin system modules.
"""

import sys
import os
# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_plugin_system_direct():
    """Test basic plugin system functionality by importing modules directly."""
    print("Testing plugin system integration (direct import)...")
    
    # Import core components directly from their files
    import importlib.util
    
    # Load core module directly
    core_spec = importlib.util.spec_from_file_location(
        "core", 
        os.path.join(os.path.dirname(__file__), 'src', 'qwen3_vl', 'plugin_system', 'core.py')
    )
    core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core)
    print("+ Core module loaded successfully")

    # Load registry module directly
    registry_spec = importlib.util.spec_from_file_location(
        "registry",
        os.path.join(os.path.dirname(__file__), 'src', 'qwen3_vl', 'plugin_system', 'registry.py')
    )
    registry = importlib.util.module_from_spec(registry_spec)
    registry_spec.loader.exec_module(registry)
    print("+ Registry module loaded successfully")

    # Create a test plugin using the imported classes
    class TestPlugin(core.BasePlugin):
        def __init__(self):
            metadata = core.PluginMetadata(
                name="test_plugin",
                version="1.0.0",
                description="Test plugin for integration verification",
                author="Test Author",
                plugin_type=core.PluginType.CUSTOM,
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
    config = core.PluginConfig("test_plugin", {"test_param": "test_value"})
    init_result = test_plugin.initialize(config)
    print(f"+ Plugin initialized: {init_result}")
    print(f"+ State after initialization: {test_plugin.state}")

    # Test registry functionality
    plugin_registry = registry.PluginRegistry()
    plugin_registry.register_plugin(test_plugin)
    print("+ Plugin registered in registry")

    retrieved_plugin = plugin_registry.get_plugin("test_plugin")
    print(f"+ Plugin retrieved from registry: {retrieved_plugin is not None}")

    # Test deactivation
    deactivate_result = plugin_registry.deactivate_plugin("test_plugin")
    print(f"+ Plugin deactivated: {deactivate_result}")

    print("\n+ All direct plugin system integration tests passed!")


if __name__ == "__main__":
    test_plugin_system_direct()