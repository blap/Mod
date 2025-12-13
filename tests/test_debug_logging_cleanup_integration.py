"""
Test suite to verify debug logging cleanup functionality.
"""
import os
import tempfile
from unittest.mock import patch

from src.qwen3_vl.utils.debug_utils import conditional_debug, get_conditional_logger, debug_if_enabled
from src.qwen3_vl.utils.general_utils import is_debug_mode


def test_conditional_debug():
    """Test the conditional debug utility function."""
    # Test when debug mode is disabled
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']
    
    # Mock a logger to capture calls
    import logging
    logger = logging.getLogger('test_logger')
    
    # Patch the logger's debug method to capture calls
    with patch.object(logger, 'debug') as mock_debug:
        conditional_debug(logger, "Test message")
        # The debug method should not be called when debug mode is off
        mock_debug.assert_not_called()
    
    # Test when debug mode is enabled
    os.environ['QWEN3_VL_DEBUG'] = '1'
    
    with patch.object(logger, 'debug') as mock_debug:
        conditional_debug(logger, "Test message")
        # The debug method should be called when debug mode is on
        mock_debug.assert_called_once_with("Test message")
    
    # Clean up
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']


def test_get_conditional_logger():
    """Test the get_conditional_logger utility function."""
    # Test when debug mode is disabled
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']
    
    logger, is_enabled = get_conditional_logger('test_module')
    assert logger is None
    assert is_enabled is False
    
    # Test when debug mode is enabled
    os.environ['QWEN3_VL_DEBUG'] = '1'
    
    logger, is_enabled = get_conditional_logger('test_module')
    assert logger is not None
    assert is_enabled is True
    
    # Clean up
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']


def test_debug_if_enabled():
    """Test the debug_if_enabled utility function."""
    # Test when debug mode is disabled
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']
    
    import logging
    with patch.object(logging, 'getLogger') as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        debug_if_enabled("Test message", logger_name='test')
        # The debug method should not be called when debug mode is off
        mock_logger.debug.assert_not_called()
    
    # Test when debug mode is enabled
    os.environ['QWEN3_VL_DEBUG'] = '1'
    
    with patch.object(logging, 'getLogger') as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        debug_if_enabled("Test message", logger_name='test')
        # The debug method should be called when debug mode is on
        mock_logger.debug.assert_called_once_with("Test message")
    
    # Clean up
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']


def test_integration_with_existing_code():
    """Test that the new debug utilities work with existing code patterns."""
    # This simulates how the new utilities would be used in existing files
    
    import logging
    from src.qwen3_vl.utils import conditional_debug
    
    logger = logging.getLogger(__name__)
    
    # Test that conditional_debug doesn't call logger.debug when debug mode is off
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']
    
    with patch.object(logger, 'debug') as mock_debug:
        conditional_debug(logger, "This should not be logged")
        mock_debug.assert_not_called()
    
    # Test that conditional_debug calls logger.debug when debug mode is on
    os.environ['QWEN3_VL_DEBUG'] = '1'
    
    with patch.object(logger, 'debug') as mock_debug:
        conditional_debug(logger, "This should be logged")
        mock_debug.assert_called_once_with("This should be logged")
    
    # Clean up
    if 'QWEN3_VL_DEBUG' in os.environ:
        del os.environ['QWEN3_VL_DEBUG']


if __name__ == "__main__":
    test_conditional_debug()
    test_get_conditional_logger()
    test_debug_if_enabled()
    test_integration_with_existing_code()
    print("All tests passed!")