"""
Shared Test Utilities Module

This module contains common utilities used across the test suite to eliminate
code duplication and ensure consistent behavior.
"""

import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock

import pytest
import torch


def generate_cache_key(
    test_path: str, test_args: str = "", extra_salt: str = ""
) -> str:
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
        with open(file_path, "w", encoding="utf-8") as f:
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
        with open(file_path, "r", encoding="utf-8") as f:
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

    return path.replace("/", os.sep).replace("\\", os.sep)


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
        if part == "models" and i + 1 < len(path_parts):
            # The next part should be the model name
            model_part = path_parts[i + 1]
            # Handle the case where the model name has hyphens or other separators
            # Convert to a cleaner format if needed
            return model_part.replace("-", "_").replace(".", "_")

    # If not in models directory, check for plugin_system
    if "plugin_system" in normalized_path:
        return "plugin_system"

    return "general"


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
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "latest": values[-1],
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
    for unit in ["B", "KB", "MB", "GB", "TB"]:
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
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove control characters
    sanitized = "".join(char for char in sanitized if ord(char) >= 32)
    return sanitized


def create_temp_directory(prefix: str = "test_", suffix: str = "") -> Path:
    """
    Create a temporary directory for testing purposes.

    Args:
        prefix: Prefix for the temporary directory name
        suffix: Suffix for the temporary directory name

    Returns:
        Path object to the temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
    return temp_dir


def cleanup_temp_directory(temp_dir: Path) -> bool:
    """
    Clean up a temporary directory created for testing.

    Args:
        temp_dir: Path to the temporary directory to clean up

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        if temp_dir.exists() and temp_dir.is_dir():
            shutil.rmtree(temp_dir)
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
        return False


def create_mock_model(input_dim: int = 10, output_dim: int = 1) -> torch.nn.Module:
    """
    Create a simple mock PyTorch model for testing purposes.

    Args:
        input_dim: Input dimension of the model
        output_dim: Output dimension of the model

    Returns:
        A simple PyTorch model
    """

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    return SimpleModel(input_dim, output_dim)


def create_mock_plugin_instance(plugin_class, **kwargs):
    """
    Create a mock instance of a plugin for testing purposes.

    Args:
        plugin_class: The plugin class to instantiate
        **kwargs: Arguments to pass to the plugin constructor

    Returns:
        An instance of the plugin class
    """
    # Create a mock plugin instance with default values if not provided
    try:
        return plugin_class(**kwargs)
    except TypeError:
        # If the plugin doesn't accept kwargs, try creating without them
        return plugin_class()


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """
    Assert that a tensor has the expected shape.

    Args:
        tensor: The tensor to check
        expected_shape: The expected shape as a tuple
    """
    assert (
        tensor.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_values_close(
    tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-6
):
    """
    Assert that two tensors have close values within a tolerance.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        tolerance: Tolerance for comparison
    """
    assert torch.allclose(
        tensor1, tensor2, atol=tolerance
    ), f"Tensors are not close within tolerance {tolerance}"


def create_sample_text_data(num_samples: int = 5, max_length: int = 20) -> List[str]:
    """
    Create sample text data for testing.

    Args:
        num_samples: Number of text samples to create
        max_length: Maximum length of each text sample

    Returns:
        List of sample text strings
    """
    import random
    import string

    texts = []
    for i in range(num_samples):
        length = random.randint(5, max_length)
        text = "".join(random.choices(string.ascii_lowercase + " ", k=length)).strip()
        texts.append(f"sample_text_{i}: {text}")

    return texts


def create_sample_tensor_data(
    batch_size: int = 4, seq_len: int = 10, hidden_size: int = 128
) -> torch.Tensor:
    """
    Create sample tensor data for testing.

    Args:
        batch_size: Size of the batch
        seq_len: Sequence length
        hidden_size: Hidden size dimension

    Returns:
        Sample tensor data
    """
    return torch.randn(batch_size, seq_len, hidden_size)


def compare_dicts(dict1: Dict, dict2: Dict, ignore_keys: List[str] = None) -> bool:
    """
    Compare two dictionaries, optionally ignoring certain keys.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        ignore_keys: Keys to ignore during comparison

    Returns:
        True if dictionaries are equal (excluding ignored keys), False otherwise
    """
    if ignore_keys is None:
        ignore_keys = []

    filtered_dict1 = {k: v for k, v in dict1.items() if k not in ignore_keys}
    filtered_dict2 = {k: v for k, v in dict2.items() if k not in ignore_keys}

    return filtered_dict1 == filtered_dict2

# Assertion wrappers
def assert_equal(a, b, msg=None):
    assert a == b, msg or f"{a} != {b}"

def assert_true(a, msg=None):
    assert a, msg or f"{a} is not True"

def assert_false(a, msg=None):
    assert not a, msg or f"{a} is not False"

def assert_is_instance(obj, cls, msg=None):
    assert isinstance(obj, cls), msg or f"{obj} is not instance of {cls}"

def assert_is_none(obj, msg=None):
    assert obj is None, msg or f"{obj} is not None"

def assert_is_not_none(obj, msg=None):
    assert obj is not None, msg or f"{obj} is None"

def assert_in(member, container, msg=None):
    assert member in container, msg or f"{member} not in {container}"

def assert_not_in(member, container, msg=None):
    assert member not in container, msg or f"{member} in {container}"

def assert_greater(a, b, msg=None):
    assert a > b, msg or f"{a} <= {b}"

def assert_less(a, b, msg=None):
    assert a < b, msg or f"{a} >= {b}"

def assert_not_equal(a, b, msg=None):
    assert a != b, msg or f"{a} == {b}"

def assert_raises(exception_class, callable_obj=None, *args, **kwargs):
    if callable_obj is None:
        import pytest
        return pytest.raises(exception_class)
    try:
        callable_obj(*args, **kwargs)
    except exception_class:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exception_class}, but got {type(e)}")
    raise AssertionError(f"Expected {exception_class}, but no exception was raised")

def run_tests(test_functions):
    for test_func in test_functions:
        try:
            test_func()
            print(f"PASS: {test_func.__name__}")
        except Exception as e:
            print(f"FAIL: {test_func.__name__} - {e}")
            import traceback
            traceback.print_exc()
