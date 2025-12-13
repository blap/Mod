"""
Tests for the utils module of the Flexible Model System.
"""

import pytest
import time
import logging
from unittest.mock import Mock
from src.utils.helpers import (
    setup_logger, 
    time_it, 
    flatten_list, 
    safe_get, 
    validate_type, 
    batch_iterate
)


def test_setup_logger():
    """Test that setup_logger creates a logger correctly."""
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


def test_time_it_decorator(capfd):
    """Test that time_it decorator measures execution time."""
    @time_it
    def sample_function():
        time.sleep(0.01)  # Sleep briefly to measure time
        return "result"
    
    result = sample_function()
    assert result == "result"
    
    # Capture printed output
    out, err = capfd.readouterr()
    assert "sample_function took" in out


def test_flatten_list():
    """Test flattening nested lists."""
    nested = [[1, 2], [3, [4, 5]], 6]
    flattened = flatten_list(nested)
    assert flattened == [1, 2, 3, 4, 5, 6]
    
    # Test with already flat list
    flat = [1, 2, 3]
    assert flatten_list(flat) == [1, 2, 3]
    
    # Test with empty list
    assert flatten_list([]) == []


def test_safe_get():
    """Test safely getting values from dictionaries."""
    d = {"a": 1, "b": 2}
    
    # Test getting existing key
    assert safe_get(d, "a") == 1
    
    # Test getting non-existing key with default
    assert safe_get(d, "c", "default") == "default"
    
    # Test getting non-existing key without default
    assert safe_get(d, "c") is None


def test_validate_type():
    """Test type validation."""
    # Valid type checks
    assert validate_type(42, int) is True
    assert validate_type("hello", str) is True
    assert validate_type([1, 2, 3], list) is True
    
    # Invalid type checks
    assert validate_type(42, str) is False
    assert validate_type("hello", int) is False


def test_batch_iterate():
    """Test batching iteration."""
    data = [1, 2, 3, 4, 5, 6, 7]
    
    batches = list(batch_iterate(data, 3))
    expected = [[1, 2, 3], [4, 5, 6], [7]]
    assert batches == expected
    
    # Test with exact division
    batches = list(batch_iterate([1, 2, 3, 4], 2))
    expected = [[1, 2], [3, 4]]
    assert batches == expected
    
    # Test with batch size larger than data
    batches = list(batch_iterate([1, 2], 5))
    expected = [[1, 2]]
    assert batches == expected