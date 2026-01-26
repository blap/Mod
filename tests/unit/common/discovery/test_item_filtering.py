"""
Tests for the item filtering and retrieval functionality of the unified test discovery system.
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


class TestItemFiltering(unittest.TestCase):
    """Test cases for item filtering and retrieval functionality."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.temp_dir: str = tempfile.mkdtemp()
        self.discovery: UnifiedTestDiscovery = UnifiedTestDiscovery(search_paths=[self.temp_dir])

    def tearDown(self) -> None:
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_items_by_type(self) -> None:
        """Test that get_items_by_type filters items correctly by type."""
        # Create mock items
        mock_items: List[Dict[str, Any]] = [
            {'name': 'test1', 'type': TestType.UNIT_TEST},
            {'name': 'test2', 'type': TestType.INTEGRATION_TEST},
            {'name': 'benchmark1', 'type': TestType.BENCHMARK}
        ]
        self.discovery.discovered_items = mock_items

        # Test filtering by unit test type
        unit_tests: List[Dict[str, Any]] = self.discovery.get_items_by_type(TestType.UNIT_TEST)
        self.assertEqual(len(unit_tests), 1)
        self.assertEqual(unit_tests[0]['name'], 'test1')

        # Test filtering by benchmark type
        benchmarks: List[Dict[str, Any]] = self.discovery.get_items_by_type(TestType.BENCHMARK)
        self.assertEqual(len(benchmarks), 1)
        self.assertEqual(benchmarks[0]['name'], 'benchmark1')

    def test_get_items_by_category(self) -> None:
        """Test that get_items_by_category filters items correctly by category."""
        # Create mock items
        mock_items: List[Dict[str, Any]] = [
            {'name': 'test1', 'category': 'unit'},
            {'name': 'test2', 'category': 'integration'},
            {'name': 'test3', 'category': 'unit'}
        ]
        self.discovery.discovered_items = mock_items

        # Test filtering by unit category
        unit_items: List[Dict[str, Any]] = self.discovery.get_items_by_category('unit')
        self.assertEqual(len(unit_items), 2)
        names: List[str] = [item['name'] for item in unit_items]
        self.assertIn('test1', names)
        self.assertIn('test3', names)

    def test_get_items_by_model(self) -> None:
        """Test that get_items_by_model filters items correctly by model name."""
        # Create mock items
        mock_items: List[Dict[str, Any]] = [
            {'name': 'test1', 'model_name': 'qwen3_vl_2b'},
            {'name': 'test2', 'model_name': 'glm_4_7_flash'},
            {'name': 'test3', 'model_name': 'qwen3_vl_2b'}
        ]
        self.discovery.discovered_items = mock_items

        # Test filtering by model name
        qwen_items: List[Dict[str, Any]] = self.discovery.get_items_by_model('qwen3_vl_2b')
        self.assertEqual(len(qwen_items), 2)
        names: List[str] = [item['name'] for item in qwen_items]
        self.assertIn('test1', names)
        self.assertIn('test3', names)

    def test_run_item_success(self) -> None:
        """Test that run_item executes functions successfully."""
        def sample_test() -> str:
            return "success"

        result: str = self.discovery.run_item(sample_test)
        self.assertEqual(result, "success")

    def test_run_item_error_handling(self) -> None:
        """Test that run_item handles errors gracefully."""
        def failing_test() -> None:
            raise ValueError("Test error")

        result: Dict[str, Any] = self.discovery.run_item(failing_test)
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('Test error', result['error'])


if __name__ == '__main__':
    print("Running tests for unified test discovery item filtering...")
    unittest.main(verbosity=2)