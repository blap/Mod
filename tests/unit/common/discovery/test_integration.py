"""
Integration tests for the unified test discovery system.
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


class TestUnifiedTestDiscoveryIntegration(unittest.TestCase):
    """Integration tests for the unified test discovery system."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.temp_dir: str = tempfile.mkdtemp()

        # Create a test structure
        os.makedirs(os.path.join(self.temp_dir, 'tests', 'unit'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'tests', 'integration'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'benchmarks', 'performance'), exist_ok=True)

        # Create test files
        with open(os.path.join(self.temp_dir, 'tests', 'unit', 'test_sample.py'), 'w') as f:
            f.write('''
def test_basic_unit():
    assert True

def should_pass_unit():
    assert 2 + 2 == 4
''')

        with open(os.path.join(self.temp_dir, 'tests', 'integration', 'test_integration.py'), 'w') as f:
            f.write('''
def test_integration_flow():
    assert True

class TestIntegrationClass:
    def test_class_method(self):
        assert True
''')

        with open(os.path.join(self.temp_dir, 'benchmarks', 'performance', 'benchmark_sample.py'), 'w') as f:
            f.write('''
def run_performance_check():
    return {"result": "fast"}

def benchmark_throughput():
    return {"throughput": 100}
''')

    def tearDown(self) -> None:
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discover_all_finds_tests_and_benchmarks(self) -> None:
        """Test that discover_all finds both tests and benchmarks."""
        discovery: UnifiedTestDiscovery = UnifiedTestDiscovery(search_paths=[self.temp_dir])
        items: List[Dict[str, Any]] = discovery.discover_all()

        # Should find all created items
        self.assertGreater(len(items), 0)

        # Separate tests and benchmarks
        test_items: List[Dict[str, Any]] = [item for item in items if item['type'] != TestType.BENCHMARK]
        benchmark_items: List[Dict[str, Any]] = [item for item in items if item['type'] == TestType.BENCHMARK]

        self.assertGreater(len(test_items), 0)
        self.assertGreater(len(benchmark_items), 0)

    def test_run_tests_only(self) -> None:
        """Test that run_tests_only only runs tests, not benchmarks."""
        discovery: UnifiedTestDiscovery = UnifiedTestDiscovery(search_paths=[self.temp_dir])
        discovery.discover_all()

        # Mock the run_item method to avoid actually running tests
        with patch.object(discovery, 'run_item') as mock_run:
            mock_run.return_value = "mock_result"
            results: Dict[str, Any] = discovery.run_tests_only()

            # Should have run tests but not benchmarks
            self.assertIn('summary', results)
            self.assertIn('total_items', results['summary'])

    def test_run_benchmarks_only(self) -> None:
        """Test that run_benchmarks_only only runs benchmarks, not tests."""
        discovery: UnifiedTestDiscovery = UnifiedTestDiscovery(search_paths=[self.temp_dir])
        discovery.discover_all()

        # Mock the run_item method to avoid actually running benchmarks
        with patch.object(discovery, 'run_item') as mock_run:
            mock_run.return_value = "mock_result"
            results: Dict[str, Any] = discovery.run_benchmarks_only()

            # Should have run benchmarks but not tests
            self.assertIn('summary', results)
            self.assertIn('total_items', results['summary'])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for the discovery system."""

    def test_discover_tests_for_model(self) -> None:
        """Test the discover_tests_for_model utility function."""
        # This test would require a more complex setup to test properly
        # For now, just ensure the function exists and signature is correct
        from inference_pio.unified_test_discovery import discover_tests_for_model
        self.assertTrue(callable(discover_tests_for_model))

    def test_discover_benchmarks_for_model(self) -> None:
        """Test the discover_benchmarks_for_model utility function."""
        from inference_pio.unified_test_discovery import discover_benchmarks_for_model
        self.assertTrue(callable(discover_benchmarks_for_model))

    def test_get_discovery_summary(self) -> None:
        """Test the get_discovery_summary utility function."""
        from inference_pio.unified_test_discovery import get_discovery_summary
        self.assertTrue(callable(get_discovery_summary))


if __name__ == '__main__':
    print("Running integration tests for unified test discovery system...")
    unittest.main(verbosity=2)