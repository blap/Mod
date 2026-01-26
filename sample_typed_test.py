"""
Sample test file demonstrating proper type hints for test functions.
This serves as a reference for adding type hints to existing test files.
"""

import unittest
from typing import Any, Dict, List, Optional, Union
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class SampleTypedTest(unittest.TestCase):
    """Sample test class with proper type hints."""
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.temp_dir: str = tempfile.mkdtemp()
        self.test_data: Dict[str, Any] = {"key": "value", "number": 42}
        
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
    
    def test_example_with_parameters(self) -> None:
        """Example test with typed parameters and return value."""
        test_value: int = 42
        expected: int = 42
        
        self.assertEqual(test_value, expected)
    
    def test_with_optional_parameter(self, param: Optional[str] = None) -> None:
        """Example test with optional parameter."""
        if param is None:
            param = "default"
            
        self.assertIsInstance(param, str)
    
    def test_returns_boolean_result(self) -> bool:
        """Example test that returns a boolean result."""
        result: bool = True
        self.assertTrue(result)
        return result
    
    def test_complex_types(self) -> Dict[str, Union[str, int]]:
        """Example test with complex type annotations."""
        result: Dict[str, Union[str, int]] = {
            "status": "success",
            "count": 5
        }
        self.assertIn("status", result)
        return result


def test_standalone_function() -> None:
    """Example of a standalone test function with type hints."""
    value: int = 10
    doubled: int = value * 2
    assert doubled == 20


def test_with_parameters_and_return(x: int, y: int) -> bool:
    """Example of a standalone test function with parameters."""
    result: bool = x + y == 30
    if not result:
        print(f"Test failed: {x} + {y} != 30")
    return result


def test_with_complex_input(data: List[Dict[str, Any]]) -> int:
    """Example test with complex input type."""
    count: int = len(data)
    for item in data:
        assert isinstance(item, dict)
    return count


if __name__ == "__main__":
    unittest.main(verbosity=2)