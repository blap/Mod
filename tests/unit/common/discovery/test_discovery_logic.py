"""
Tests for the core discovery logic of the unified test discovery system.
"""

import unittest
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from inference_pio.unified_test_discovery import UnifiedTestDiscovery, TestType


class TestDiscoveryLogic(unittest.TestCase):
    """Test cases for the core discovery logic."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.temp_dir: str = tempfile.mkdtemp()
        self.discovery: UnifiedTestDiscovery = UnifiedTestDiscovery(search_paths=[self.temp_dir])

    def tearDown(self) -> None:
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_file(self, filename: str, content: str) -> str:
        """Helper method to create a test file with given content."""
        filepath: str = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    def test_is_test_function_recognizes_standard_patterns(self) -> None:
        """Test that _is_test_function recognizes standard test naming patterns."""
        # Test standard test_ prefix
        self.assertTrue(self.discovery._is_test_function('test_something'))
        self.assertTrue(self.discovery._is_test_function('testSomething'))
        self.assertTrue(self.discovery._is_test_function('test'))

        # Test other patterns
        self.assertTrue(self.discovery._is_test_function('should_work'))
        self.assertTrue(self.discovery._is_test_function('when_condition_met'))
        self.assertTrue(self.discovery._is_test_function('verify_result'))
        self.assertTrue(self.discovery._is_test_function('validate_input'))
        self.assertTrue(self.discovery._is_test_function('check_output'))

        # Test case insensitivity
        self.assertTrue(self.discovery._is_test_function('TEST_something'))
        self.assertTrue(self.discovery._is_test_function('Should_work'))

        # Test non-matching patterns
        self.assertFalse(self.discovery._is_test_function('not_a_test'))
        self.assertFalse(self.discovery._is_test_function('helper_function'))
        self.assertFalse(self.discovery._is_test_function('calculate_value'))
        self.assertFalse(self.discovery._is_test_function(''))

    def test_is_benchmark_function_recognizes_standard_patterns(self) -> None:
        """Test that _is_benchmark_function recognizes standard benchmark naming patterns."""
        # Test standard patterns
        self.assertTrue(self.discovery._is_benchmark_function('run_performance'))
        self.assertTrue(self.discovery._is_benchmark_function('benchmark_speed'))
        self.assertTrue(self.discovery._is_benchmark_function('perf_test'))
        self.assertTrue(self.discovery._is_benchmark_function('measure_latency'))
        self.assertTrue(self.discovery._is_benchmark_function('profile_memory'))
        self.assertTrue(self.discovery._is_benchmark_function('time_execution'))
        self.assertTrue(self.discovery._is_benchmark_function('speed_test'))
        self.assertTrue(self.discovery._is_benchmark_function('stress_test'))
        self.assertTrue(self.discovery._is_benchmark_function('load_test'))

        # Test case insensitivity
        self.assertTrue(self.discovery._is_benchmark_function('RUN_performance'))
        self.assertTrue(self.discovery._is_benchmark_function('Benchmark_speed'))

        # Test non-matching patterns
        self.assertFalse(self.discovery._is_benchmark_function('not_a_benchmark'))
        self.assertFalse(self.discovery._is_benchmark_function('helper_function'))
        self.assertFalse(self.discovery._is_benchmark_function('calculate_value'))
        self.assertFalse(self.discovery._is_benchmark_function(''))

    def test_determine_test_type(self) -> None:
        """Test that _determine_test_type returns correct TestType."""
        self.assertEqual(self.discovery._determine_test_type('unit'), TestType.UNIT_TEST)
        self.assertEqual(self.discovery._determine_test_type('integration'), TestType.INTEGRATION_TEST)
        self.assertEqual(self.discovery._determine_test_type('performance'), TestType.PERFORMANCE_TEST)
        self.assertEqual(self.discovery._determine_test_type('other'), TestType.UNKNOWN)

    def test_extract_model_name_from_path(self) -> None:
        """Test that _extract_model_name correctly extracts model names from paths."""
        # Test path with models directory
        path_with_models: str = "/some/path/models/qwen3_vl_2b/tests/unit/test_something.py"
        self.assertEqual(self.discovery._extract_model_name(path_with_models), "qwen3_vl_2b")

        # Test path with plugin_system
        path_with_plugin: str = "/some/path/src/inference_pio/plugin_system/tests/unit/test_something.py"
        self.assertEqual(self.discovery._extract_model_name(path_with_plugin), "plugin_system")

        # Test path without models or plugin_system
        path_general: str = "/some/path/tests/unit/test_something.py"
        self.assertEqual(self.discovery._extract_model_name(path_general), "general")

    def test_find_test_functions_in_module(self) -> None:
        """Test that _find_test_functions correctly finds test functions in a module."""
        # Create a temporary test file
        test_content: str = '''
def test_basic_function():
    """A basic test function."""
    assert True

def should_work_correctly():
    """A function with 'should' prefix."""
    assert True

def helper_function():
    """A helper function that should not be detected as a test."""
    pass

class TestClass:
    def test_method(self):
        """A test method in a class."""
        assert True

    def should_pass_validation(self):
        """A method with 'should' prefix."""
        assert True

    def helper_method(self):
        """A helper method that should not be detected."""
        pass
'''
        test_file: str = self.create_test_file('unit/test_sample.py', test_content)

        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_sample", test_file)
        module: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find test functions
        test_functions: List[Dict[str, Any]] = self.discovery._find_test_functions(module, 'test_sample', test_file, 'unit')

        # Should find 4 test functions: 2 standalone + 2 class methods
        self.assertEqual(len(test_functions), 4)

        # Check that test functions are found
        function_names: List[str] = [f['name'] for f in test_functions]
        self.assertIn('test_basic_function', function_names)
        self.assertIn('should_work_correctly', function_names)
        self.assertIn('TestClass.test_method', function_names)
        self.assertIn('TestClass.should_pass_validation', function_names)

        # Check that non-test functions are not found
        self.assertNotIn('helper_function', function_names)
        for func_info in test_functions:
            self.assertNotEqual(func_info['name'], 'TestClass.helper_method')

    def test_find_benchmark_functions_in_module(self) -> None:
        """Test that _find_benchmark_functions correctly finds benchmark functions in a module."""
        # Create a temporary benchmark file
        benchmark_content: str = '''
def run_performance_test():
    """A basic benchmark function."""
    return {"result": "ok"}

def benchmark_speed():
    """A benchmark function."""
    return {"time": 0.1}

def perf_analysis():
    """A performance analysis function."""
    return {"memory": "low"}

def helper_function():
    """A helper function that should not be detected as a benchmark."""
    pass
'''
        benchmark_file: str = self.create_test_file('performance/benchmark_sample.py', benchmark_content)

        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("benchmark_sample", benchmark_file)
        module: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find benchmark functions
        benchmark_functions: List[Dict[str, Any]] = self.discovery._find_benchmark_functions(module, 'benchmark_sample', benchmark_file, 'performance')

        # Should find 3 benchmark functions
        self.assertEqual(len(benchmark_functions), 3)

        # Check that benchmark functions are found
        function_names: List[str] = [f['name'] for f in benchmark_functions]
        self.assertIn('run_performance_test', function_names)
        self.assertIn('benchmark_speed', function_names)
        self.assertIn('perf_analysis', function_names)

        # Check that non-benchmark functions are not found
        self.assertNotIn('helper_function', function_names)


if __name__ == '__main__':
    print("Running tests for unified test discovery logic...")
    unittest.main(verbosity=2)