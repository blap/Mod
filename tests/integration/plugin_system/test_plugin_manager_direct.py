"""
Test for plugin discovery functionality without importing all models.
"""
import sys
from pathlib import Path
import importlib.util

# Add the src directory to the Python path so we can import inference_pio
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tests.utils.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_is_instance, assert_raises,
    run_tests
)


def test_plugin_manager_directly():
    """Test the plugin manager directly without importing all models."""
    
    print("Testing plugin manager directly...")
    
    # Import the plugin manager directly
    plugin_manager_path = Path(__file__).parent / "src/inference_pio/plugin_system/plugin_manager.py"
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("plugin_manager", plugin_manager_path)
    plugin_manager_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_manager_module)
    
    # Get the plugin manager
    pm = plugin_manager_module.get_plugin_manager()
    
    # Print initial state
    print(f"Initial plugins count: {len(pm.list_plugins())}")
    print(f"Initial plugins: {pm.list_plugins()}")
    
    # Discover and load all plugins from the models directory
    models_dir = Path(__file__).parent / "src" / "inference_pio" / "models"
    loaded_count = pm.discover_and_load_plugins(models_dir)
    print(f"Plugins loaded via auto-discovery: {loaded_count}")
    
    # List all discovered plugins
    all_plugins = pm.list_plugins()
    print(f"All discovered plugins ({len(all_plugins)}): {all_plugins}")
    
    print("Direct plugin manager test completed.")


if __name__ == "__main__":
    test_plugin_manager_directly()