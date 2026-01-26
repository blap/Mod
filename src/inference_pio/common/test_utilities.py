"""
Common Test Utilities for Inference-PIO

This module consolidates common utilities used across the test and benchmark systems
to eliminate code duplication and ensure consistent behavior.
"""

import hashlib
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


def generate_cache_key(test_path: str, test_args: str = "", extra_salt: str = "") -> str:
    """
    Generate a unique cache key for a test based on its path, arguments, and platform info.

    Args:
        test_path: Path to the test file/function
        test_args: Arguments passed to the test
        extra_salt: Additional salt to differentiate similar tests

    Returns:
        Unique cache key
    """
    key_data = f"{test_path}_{test_args}_{platform.python_version()}_{extra_salt}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def is_cache_valid(timestamp_str: str, max_age_hours: float = 24.0) -> bool:
    """
    Check if a cached item is still valid based on its timestamp.

    Args:
        timestamp_str: ISO format timestamp string
        max_age_hours: Maximum age in hours before cache expires

    Returns:
        True if cache is still valid, False otherwise
    """
    try:
        cache_time = datetime.fromisoformat(timestamp_str)
        current_time = datetime.now()
        
        # Check if cache is still valid (not too old)
        return (current_time - cache_time).total_seconds() < max_age_hours * 3600
    except (ValueError, TypeError):
        return False


def save_json_file(data: Any, file_path: Path, indent: int = 2) -> bool:
    """
    Safely save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except IOError as e:
        print(f"Warning: Could not save file {file_path}: {e}")
        return False


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load data from a JSON file.

    Args:
        file_path: Path to load the file from

    Returns:
        Loaded data if successful, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return None


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()


def measure_execution_time(func, *args, **kwargs) -> tuple:
    """
    Measure execution time of a function.

    Args:
        func: Function to measure
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        raise e


def ensure_directory_exists(directory: Path) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path to ensure

    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def normalize_path_separators(path: str) -> str:
    """
    Normalize path separators to the current platform's convention.

    Args:
        path: Path string to normalize

    Returns:
        Normalized path string
    """
    import os
    return path.replace('/', os.sep).replace('\\', os.sep)


def extract_model_name_from_path(file_path: str) -> str:
    """
    Extract the model name from a file path.

    Args:
        file_path: Path to the test/benchmark file

    Returns:
        Model name string
    """
    import os
    # Normalize path separators
    normalized_path = normalize_path_separators(file_path)
    path_parts = normalized_path.split(os.sep)

    # Look for 'models' directory and extract the next part as model name
    for i, part in enumerate(path_parts):
        if part == 'models' and i + 1 < len(path_parts):
            # The next part should be the model name
            model_part = path_parts[i + 1]
            # Handle the case where the model name has hyphens or other separators
            # Convert to a cleaner format if needed
            return model_part.replace('-', '_').replace('.', '_')

    # If not in models directory, check for plugin_system
    if 'plugin_system' in normalized_path:
        return 'plugin_system'

    return 'general'


def calculate_statistics(values: list) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {}
    
    import statistics
    
    stats = {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values),
        'latest': values[-1]
    }
    
    return stats


def format_bytes(bytes_value: float) -> str:
    """
    Format bytes value into human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Human-readable string with appropriate units
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(mins)}m {int(secs)}s"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import re
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    return sanitized