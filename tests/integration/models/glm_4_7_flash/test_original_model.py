import sys
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)

sys.path.insert(0, '.')

from pathlib import Path
import importlib.util

# Test loading the current plugin directly (no backup directory needed)
current_path = Path(__file__).parent.parent / 'src' / 'inference_pio' / 'models' / 'glm_4_7_flash'
plugin_file = current_path / 'plugin.py'

print(f"Looking for plugin file: {plugin_file}")
print(f"Plugin file exists: {plugin_file.exists()}")

if plugin_file.exists():
    # Import using the full module path
    spec = importlib.util.spec_from_file_location("current_glm_4_7_flash_plugin", plugin_file)
    current_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current_module)

    # Get the factory function
    create_plugin_func = getattr(current_module, 'create_glm_4_7_flash_plugin', None)
    print(f"Factory function exists: {create_plugin_func is not None}")

    if create_plugin_func:
        print("Successfully found the create_glm_4_7_flash_plugin function")
    else:
        print("Could not find create_glm_4_7_flash_plugin function")
        print(f"Available functions: {[attr for attr in dir(current_module) if not attr.startswith('_')]}")
else:
    print("Plugin file does not exist!")