"""
Sample test for plugin system to verify test discovery system.
"""

from src.inference_pio.test_utils import assert_equal, assert_true


def test_sample_plugin_functionality():
    """Test basic plugin functionality."""
    # Simple test to verify the test discovery system works
    result = 3 * 3
    assert_equal(result, 9, "Basic multiplication should work")
    

def test_plugin_configuration():
    """Test plugin configuration."""
    config = {"enabled": True}
    assert_true(config["enabled"], "Plugin should be enabled by default")