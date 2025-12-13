"""
Conditional debug logging utilities for production code.

This module provides utilities to conditionally perform debug logging
based on whether debug mode is enabled, helping to reduce performance
impact in production environments.
"""

import logging
from typing import Optional, Any
from src.qwen3_vl.utils.general_utils import is_debug_mode, get_debug_logger


def conditional_debug(logger_instance: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """
    Conditionally log a debug message if debug mode is enabled.
    
    Args:
        logger_instance: The logger instance to use
        message: The message to log
        *args: Additional arguments for string formatting
        **kwargs: Additional keyword arguments for the logger
    """
    if is_debug_mode():
        logger_instance.debug(message, *args, **kwargs)


def get_conditional_logger(name: str) -> tuple[Optional[logging.Logger], bool]:
    """
    Get a logger and a flag indicating whether debug logging is enabled.
    
    Args:
        name: Name for the logger
        
    Returns:
        A tuple of (logger, is_enabled) where logger is the logger instance
        (or None if debug mode is disabled) and is_enabled indicates if
        debug logging is enabled
    """
    if is_debug_mode():
        return logging.getLogger(name), True
    else:
        return None, False


def debug_if_enabled(message: str, *args: Any, logger_name: str = __name__, **kwargs: Any) -> None:
    """
    Log a debug message if debug mode is enabled.
    
    Args:
        message: The message to log
        *args: Additional arguments for string formatting
        logger_name: Name of the logger to use (default: current module name)
        **kwargs: Additional keyword arguments for the logger
    """
    if is_debug_mode():
        logger = logging.getLogger(logger_name)
        logger.debug(message, *args, **kwargs)


# Backwards compatibility alias
log_debug_if_enabled = debug_if_enabled