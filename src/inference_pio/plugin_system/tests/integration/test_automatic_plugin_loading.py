"""
Test script for automatic plugin loading functionality.
"""
import sys
from pathlib import Path

# Add the src directory to the Python path so we can import inference_pio
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_is_instance, assert_raises,
    run_tests
)

from inference_pio import (
    get_plugin_manager,
    discover_and_load_plugins,
    activate_plugin,
    execute_plugin
)


def test_automatic_plugin_discovery():
    """Test the automatic plugin discovery and loading functionality."""
    
    print("Testing automatic plugin discovery and loading...")
    
    # Get the plugin manager
    pm = get_plugin_manager()
    
    # Print initial state
    print(f"Initial plugins count: {len(pm.list_plugins())}")
    print(f"Initial plugins: {pm.list_plugins()}")
    
    # Discover and load all plugins
    loaded_count = discover_and_load_plugins()
    print(f"Plugins loaded via auto-discovery: {loaded_count}")
    
    # List all discovered plugins
    all_plugins = pm.list_plugins()
    print(f"All discovered plugins ({len(all_plugins)}): {all_plugins}")
    
    # Check if expected plugins are present
    expected_plugins = [
        "GLM-4.7",
        "Qwen3-4B-Instruct-2507", 
        "Qwen3-Coder-30B",
        "Qwen3-VL-2B"
    ]
    
    found_expected = []
    for expected in expected_plugins:
        for plugin in all_plugins:
            if expected.lower() in plugin.lower():
                found_expected.append(plugin)
                break
    
    print(f"Expected plugins found: {found_expected}")
    
    # Test activating and running a plugin if available
    if all_plugins:
        sample_plugin = all_plugins[0]
        print(f"\nTesting plugin: {sample_plugin}")
        
        # Try to activate the plugin
        try:
            activation_result = activate_plugin(sample_plugin, device="cpu")
            print(f"Activation result for {sample_plugin}: {activation_result}")
            
            # If activation was successful, try a simple inference
            if activation_result:
                try:
                    # Use a simple test input
                    result = execute_plugin(sample_plugin, "Hello, world!")
                    print(f"Inference result: {result}")
                except Exception as e:
                    print(f"Inference failed for {sample_plugin}: {e}")
        except Exception as e:
            print(f"Activation failed for {sample_plugin}: {e}")
    
    print("\nAutomatic plugin discovery test completed.")


if __name__ == "__main__":
    test_automatic_plugin_discovery()