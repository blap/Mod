"""
Test suite for debug logging cleanup functionality.
"""
import os
import logging
import pytest
from unittest.mock import patch
from src.qwen3_vl.utils.general_utils import is_debug_mode, get_debug_logger


def test_debug_mode_detection():
    """Test that debug mode detection works correctly."""
    # Test when debug mode is not set
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']
    assert is_debug_mode() is False
    
    # Test when debug mode is set to '1'
    os.environ['QWEN3_VL_DEBUG'] = '1'
    assert is_debug_mode() is True
    
    # Test when debug mode is set to 'true'
    os.environ['QWEN3_VL_DEBUG'] = 'true'
    assert is_debug_mode() is True
    
    # Test when debug mode is set to 'yes'
    os.environ['QWEN3_VL_DEBUG'] = 'yes'
    assert is_debug_mode() is True
    
    # Test when debug mode is set to 'on'
    os.environ['QWEN3_VL_DEBUG'] = 'on'
    assert is_debug_mode() is True
    
    # Test when debug mode is set to 'false' (should be False)
    os.environ['QWEN3_VL_DEBUG'] = 'false'
    assert is_debug_mode() is False
    
    # Clean up
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']


def test_get_debug_logger():
    """Test that debug logger is only returned when debug mode is enabled."""
    # When debug mode is disabled, should return None
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']
    assert get_debug_logger() is None
    
    # When debug mode is enabled, should return a logger
    os.environ['QWEN3_VL_DEBUG'] = '1'
    logger = get_debug_logger()
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    
    # Clean up
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']


if __name__ == "__main__":
    test_debug_mode_detection()
    test_get_debug_logger()
    print("All tests passed!")