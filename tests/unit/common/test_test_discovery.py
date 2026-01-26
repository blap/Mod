"""
Test for the Test Discovery System

This module contains tests for the test discovery system itself.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference_pio.test_discovery import (
    discover_test_functions_from_file,
    discover_test_functions_from_directory,
    discover_tests_for_model,
    discover_tests_for_plugin_system,
    discover_all_tests,
    run_discovered_tests
)


def test_discover_test_functions_from_file():
    """Test discovering test functions from a single file."""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
def test_example_function():
    assert 1 + 1 == 2

def regular_function():
    return "not a test"

class TestClass:
    def test_method(self):
        assert True
""")
        temp_file = f.name

    try:
        # Discover test functions from the file
        test_functions = discover_test_functions_from_file(temp_file)

        # Check that we found the right number of test functions
        assert len(test_functions) == 2, f"Expected 2 test functions, got {len(test_functions)}"

        # Check that the function names are correct
        function_names = [f.__name__ for f in test_functions]
        assert 'test_example_function' in function_names
        # The class method will have a different name format
        class_method_found = any('test_method' in name for name in function_names)
        assert class_method_found, f"Expected to find test_method in function names: {function_names}"

        print("test_discover_test_functions_from_file: PASSED")
    finally:
        # Clean up
        os.unlink(temp_file)


def test_discover_test_functions_from_directory():
    """Test discovering test functions from a directory."""
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = os.path.join(temp_dir, 'test_sample.py')
        with open(test_file, 'w') as f:
            f.write("""
def test_sample_function():
    assert True

def test_another_function():  # This should start with 'test_' to be discovered
    assert 2 * 2 == 4
""")

        # Discover test functions from the directory
        test_functions = discover_test_functions_from_directory(temp_dir)

        # Check that we found the right number of test functions
        assert len(test_functions) == 2, f"Expected 2 test functions, got {len(test_functions)}"

        print("test_discover_test_functions_from_directory: PASSED")


def test_run_discovered_tests():
    """Test running discovered tests."""
    # Create some test functions
    def test_passing_function():
        assert 1 == 1
    
    def test_failing_function():
        assert 1 == 2  # This should fail
    
    def test_another_passing_function():
        assert "hello".upper() == "HELLO"
    
    # Run the tests
    test_functions = [test_passing_function, test_failing_function, test_another_passing_function]
    
    # Note: This will show failure output, but we expect it
    success = run_discovered_tests(test_functions, verbose=False)
    
    # Since one test fails, the overall result should be False
    assert success == False, "Expected overall test run to fail due to one failing test"
    
    print("test_run_discovered_tests: PASSED")


def test_empty_test_discovery():
    """Test discovering tests from an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_functions = discover_test_functions_from_directory(temp_dir)
        assert len(test_functions) == 0, f"Expected 0 test functions, got {len(test_functions)}"
        
        print("test_empty_test_discovery: PASSED")


def run_all_tests():
    """Run all tests for the test discovery system."""
    test_functions = [
        test_discover_test_functions_from_file,
        test_discover_test_functions_from_directory,
        test_run_discovered_tests,
        test_empty_test_discovery
    ]
    
    print("Running tests for the test discovery system...")
    success = run_discovered_tests(test_functions)
    
    if success:
        print("\nAll tests for the test discovery system PASSED!")
    else:
        print("\nSome tests for the test discovery system FAILED!")
    
    return success


if __name__ == '__main__':
    run_all_tests()