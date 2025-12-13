"""
Utility functions for the Flexible Model System.
"""

import logging
import time
from typing import Any, Callable, Dict, List


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def time_it(func: Callable) -> Callable:
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    flat_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flat_list.extend(flatten_list(sublist))
        else:
            flat_list.append(sublist)
    return flat_list


def safe_get(dictionary: Dict, key: str, default: Any = None):
    """Safely get a value from a dictionary."""
    try:
        return dictionary[key]
    except KeyError:
        return default


def validate_type(obj: Any, expected_type: type) -> bool:
    """Validate that an object is of the expected type."""
    return isinstance(obj, expected_type)


def batch_iterate(data: List[Any], batch_size: int):
    """Iterate over data in batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]