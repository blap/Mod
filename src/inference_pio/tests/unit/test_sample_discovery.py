"""
Sample test for main tests to verify test discovery system.
"""

from src.inference_pio.test_utils import assert_equal, assert_true


def test_main_functionality():
    """Test basic main functionality."""
    # Simple test to verify the test discovery system works
    result = 5 - 3
    assert_equal(result, 2, "Basic subtraction should work")
    

def test_main_feature():
    """Test main feature."""
    items = [1, 2, 3]
    assert_true(len(items) == 3, "List should have 3 items")