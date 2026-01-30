"""
Sample test for Qwen3 VL 2B model to verify test discovery system.
"""

from tests.utils.test_utils import assert_equal, assert_true


def test_sample_model_functionality():
    """Test basic model functionality."""
    # Simple test to verify the test discovery system works
    result = 2 + 2
    assert_equal(result, 4, "Basic arithmetic should work")
    

def test_another_model_feature():
    """Test another model feature."""
    text = "hello world"
    assert_true("world" in text, "Text should contain 'world'")


def helper_function():
    """This is not a test function since it doesn't start with 'test_'."""
    return "helper"


class TestModelFeatures:
    """Test class with test methods."""
    
    def test_model_class_method(self):
        """Test a method in a test class."""
        assert_true(True, "Class method test should pass")