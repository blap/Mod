"""
Unit test for the enhanced plugin manager functionality.
"""
import tempfile
import os
from pathlib import Path

# Add the src directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from tests.utils.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_is_instance, assert_raises,
    run_tests
)
from inference_pio.plugin_system.plugin_manager import PluginManager


def test_discover_and_load_plugins_method_exists():
    """Test that the discover_and_load_plugins method exists."""
    pm = PluginManager()
    assert_true(hasattr(pm, 'discover_and_load_plugins'))
    assert_true(callable(getattr(pm, 'discover_and_load_plugins')))


def test_discover_and_load_plugins_with_invalid_directory():
    """Test discover_and_load_plugins with invalid directory."""
    pm = PluginManager()
    result = pm.discover_and_load_plugins("/nonexistent/directory")
    assert_equal(result, 0)


def test_discover_and_load_plugins_empty_directory():
    """Test discover_and_load_plugins with empty directory."""
    pm = PluginManager()
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        result = pm.discover_and_load_plugins(temp_dir)
        assert_equal(result, 0)
        assert_equal(len(pm.list_plugins()), 0)
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_manual_plugin_registration():
    """Manual test to verify the plugin manager works."""
    print("Testing manual plugin registration...")

    # Create a mock plugin for testing
    from inference_pio.common.base_plugin_interface import ModelPluginInterface
    from inference_pio.common.standard_plugin_interface import PluginMetadata, PluginType
    from datetime import datetime

    class MockPlugin(ModelPluginInterface):
        def __init__(self):
            metadata = PluginMetadata(
                name="MockPlugin",
                version="1.0.0",
                author="Test",
                description="A mock plugin for testing",
                plugin_type=PluginType.MODEL_COMPONENT,
                dependencies=[],
                compatibility={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            super().__init__(metadata)

        def initialize(self, **kwargs):
            return True

        def load_model(self, config=None):
            return None

        def infer(self, data):
            return "mock result"

        def cleanup(self):
            return True

        def supports_config(self, config):
            return True

        def tokenize(self, text, **kwargs):
            return [1, 2, 3]

        def detokenize(self, token_ids, **kwargs):
            return "mock text"

        def generate_text(self, prompt, max_new_tokens=512, **kwargs):
            return "generated text"

    # Create plugin manager
    pm = PluginManager()

    # Register a mock plugin
    mock_plugin = MockPlugin()
    success = pm.register_plugin(mock_plugin)
    print(f"Plugin registration success: {success}")

    # Check that plugin is registered
    plugins = pm.list_plugins()
    print(f"Registered plugins: {plugins}")

    # Test activation
    activation_success = pm.activate_plugin("MockPlugin")
    print(f"Plugin activation success: {activation_success}")

    # Test execution
    result = pm.execute_plugin("MockPlugin", "test input")
    print(f"Plugin execution result: {result}")

    print("Manual plugin test completed.")


if __name__ == "__main__":
    print("Running unit tests...")
    run_tests([
        test_discover_and_load_plugins_method_exists,
        test_discover_and_load_plugins_with_invalid_directory,
        test_discover_and_load_plugins_empty_directory
    ])

    print("\nRunning manual plugin test...")
    test_manual_plugin_registration()