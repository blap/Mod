"""
Test for plugin discovery functionality by patching the system.
"""
import sys
from pathlib import Path
import os

# Add the src directory to the Python path so we can import inference_pio
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Add the src directory to Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Change the working directory to make relative imports work
original_cwd = os.getcwd()
os.chdir(str(Path(__file__).parent / "src"))

from tests.utils.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_is_instance, assert_raises,
    run_tests
)

try:
    from inference_pio.plugin_system.plugin_manager import get_plugin_manager, discover_and_load_plugins
finally:
    os.chdir(original_cwd)

def test_plugin_discovery():
    """Test the plugin discovery functionality."""
    
    print("Testing plugin discovery...")
    
    # Get the plugin manager
    pm = get_plugin_manager()
    
    # Print initial state
    print(f"Initial plugins count: {len(pm.list_plugins())}")
    print(f"Initial plugins: {pm.list_plugins()}")
    
    # Discover and load all plugins from the models directory
    models_dir = Path(__file__).parent / "src" / "inference_pio" / "models"
    loaded_count = discover_and_load_plugins(models_dir)
    print(f"Plugins loaded via auto-discovery: {loaded_count}")
    
    # List all discovered plugins
    all_plugins = pm.list_plugins()
    print(f"All discovered plugins ({len(all_plugins)}): {all_plugins}")
    
    print("Plugin discovery test completed.")


if __name__ == "__main__":
    test_plugin_discovery()