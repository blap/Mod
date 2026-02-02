"""
Robust Error Handling and Validation for Tests and Benchmarks

This module provides comprehensive error handling and validation utilities
for tests and benchmarks in the Inference-PIO project.
"""

import functools
import logging
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Optional, Type, Union


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class BenchmarkError(Exception):
    """Custom exception for benchmark-specific errors."""

    pass


class TestError(Exception):
    """Custom exception for test-specific errors."""

    pass


def validate_input(
    value: Any, expected_type: Union[Type, tuple], param_name: str = "value"
):
    """
    Validate that an input matches the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        param_name: Name of the parameter for error messages

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Parameter '{param_name}' must be of type {expected_type}, "
            f"got {type(value).__name__}"
        )


def validate_positive_number(value: Union[int, float], param_name: str = "value"):
    """
    Validate that a number is positive.

    Args:
        value: Number to validate
        param_name: Name of the parameter for error messages

    Raises:
        ValidationError: If validation fails
    """
    validate_input(value, (int, float), param_name)
    if value <= 0:
        raise ValidationError(f"Parameter '{param_name}' must be positive, got {value}")


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    param_name: str = "value",
):
    """
    Validate that a number is within a specified range.

    Args:
        value: Number to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Name of the parameter for error messages

    Raises:
        ValidationError: If validation fails
    """
    validate_input(value, (int, float), param_name)
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"Parameter '{param_name}' must be between {min_val} and {max_val}, "
            f"got {value}"
        )


def safe_execute(
    func: Callable, *args, **kwargs
) -> tuple[bool, Any, Optional[Exception]]:
    """
    Safely execute a function and return success status, result, and exception.

    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (success: bool, result: Any, exception: Optional[Exception])
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        return False, None, e


@contextmanager
def error_handler(
    operation_name: str = "operation",
    raise_on_error: bool = True,
    default_return: Any = None,
):
    """
    Context manager for handling errors gracefully.

    Args:
        operation_name: Name of the operation for logging
        raise_on_error: Whether to re-raise exceptions
        default_return: Default value to return if error occurs and not raising
    """
    try:
        yield
    except Exception as e:
        logging.error(f"Error in {operation_name}: {str(e)}")
        logging.debug(f"Traceback for {operation_name}:\n{traceback.format_exc()}")

        if raise_on_error:
            raise
        else:
            return default_return


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay}s..."
                        )
                        import time

                        time.sleep(delay)
                    else:
                        logging.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Last error: {str(e)}"
                        )

            raise last_exception

        return wrapper

    return decorator


def validate_model_plugin(plugin, required_methods: Optional[list] = None):
    """
    Validate that a model plugin has required attributes and methods.

    Args:
        plugin: Plugin instance to validate
        required_methods: List of required method names (uses defaults if None)

    Raises:
        ValidationError: If validation fails
    """
    if required_methods is None:
        required_methods = [
            "initialize",
            "load_model",
            "infer",
            "cleanup",
            "supports_config",
            "tokenize",
            "detokenize",
            "generate_text",
        ]

    # Check if plugin is None
    if plugin is None:
        raise ValidationError("Plugin cannot be None")

    # Check required methods
    for method_name in required_methods:
        if not hasattr(plugin, method_name):
            raise ValidationError(f"Plugin missing required method: {method_name}")

        method = getattr(plugin, method_name)
        if not callable(method):
            raise ValidationError(f"Plugin attribute '{method_name}' is not callable")

    # Check required attributes
    required_attrs = ["metadata", "is_loaded", "is_active"]
    for attr_name in required_attrs:
        if not hasattr(plugin, attr_name):
            raise ValidationError(f"Plugin missing required attribute: {attr_name}")


def validate_benchmark_result(result: dict, required_keys: Optional[list] = None):
    """
    Validate that a benchmark result has required keys and proper structure.

    Args:
        result: Benchmark result dictionary
        required_keys: List of required keys (uses defaults if None)

    Raises:
        ValidationError: If validation fails
    """
    if required_keys is None:
        required_keys = ["value", "unit", "model_name", "benchmark_name"]

    if not isinstance(result, dict):
        raise ValidationError("Benchmark result must be a dictionary")

    for key in required_keys:
        if key not in result:
            raise ValidationError(f"Benchmark result missing required key: {key}")

    # Validate value is a number
    value = result.get("value")
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"Benchmark result value must be a number, got {type(value)}"
        )


def validate_tensor_dimensions(tensor, expected_dims: int, param_name: str = "tensor"):
    """
    Validate that a tensor has the expected number of dimensions.

    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        param_name: Name of the parameter for error messages

    Raises:
        ValidationError: If validation fails
    """
    try:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"Parameter '{param_name}' must be a torch.Tensor")

        if tensor.dim() != expected_dims:
            raise ValidationError(
                f"Parameter '{param_name}' must have {expected_dims} dimensions, "
                f"got {tensor.dim()}"
            )
    except ImportError:
        # If torch is not available, skip torch-specific validation
        warnings.warn(
            "PyTorch not available, skipping tensor validation", ImportWarning
        )


def handle_plugin_initialization(plugin, **init_kwargs):
    """
    Safely initialize a plugin with proper error handling.

    Args:
        plugin: Plugin instance to initialize
        **init_kwargs: Initialization arguments

    Returns:
        Tuple of (success: bool, plugin: plugin or None, error: str or None)
    """
    try:
        validate_model_plugin(plugin)

        success = plugin.initialize(**init_kwargs)
        if not success:
            return False, plugin, "Plugin initialization returned False"

        return True, plugin, None
    except ValidationError as e:
        return False, plugin, f"Validation error during initialization: {str(e)}"
    except Exception as e:
        return False, plugin, f"Error during initialization: {str(e)}"


def handle_model_loading(plugin, config=None):
    """
    Safely load a model with proper error handling.

    Args:
        plugin: Plugin instance
        config: Model configuration (optional)

    Returns:
        Tuple of (success: bool, model: model or None, error: str or None)
    """
    try:
        if not plugin.is_loaded:
            model = plugin.load_model(config)
        else:
            model = plugin._model if hasattr(plugin, "_model") else None

        if model is None:
            return False, None, "Model loading returned None"

        return True, model, None
    except Exception as e:
        return False, None, f"Error during model loading: {str(e)}"


def validate_and_clean_text(text: str, param_name: str = "text") -> str:
    """
    Validate and clean text input.

    Args:
        text: Text to validate and clean
        param_name: Name of the parameter for error messages

    Returns:
        Cleaned text string

    Raises:
        ValidationError: If validation fails
    """
    validate_input(text, str, param_name)

    # Clean the text
    cleaned = text.strip()
    if not cleaned:
        raise ValidationError(
            f"Parameter '{param_name}' cannot be empty or whitespace only"
        )

    # Check for reasonable length (adjust as needed)
    if len(cleaned) > 100000:  # 100k characters max
        raise ValidationError(
            f"Parameter '{param_name}' is too long (>100k characters)"
        )

    return cleaned


def validate_device(device: str) -> str:
    """
    Validate device string.

    Args:
        device: Device string to validate

    Returns:
        Validated device string

    Raises:
        ValidationError: If validation fails
    """
    validate_input(device, str, "device")

    valid_devices = ["cpu", "cuda", "mps"]  # Add more as needed
    if device not in valid_devices:
        # Check for cuda with index (e.g., "cuda:0", "cuda:1", etc.)
        if not device.startswith("cuda:") or not device.split(":")[1].isdigit():
            raise ValidationError(
                f"Invalid device '{device}'. Must be one of {valid_devices} or cuda:X format"
            )

    return device


# Example usage functions
def example_safe_benchmark_operation(plugin, input_data):
    """
    Example of a safe benchmark operation with proper error handling.
    """
    with error_handler("benchmark_operation", raise_on_error=False):
        # Validate inputs
        validate_model_plugin(plugin)
        validate_input(input_data, (str, list), "input_data")

        # Perform operation safely
        success, result, error = safe_execute(plugin.infer, input_data)

        if not success:
            logging.error(f"Benchmark operation failed: {error}")
            return None

        return result


def example_validated_test_helper(expected_methods=None):
    """
    Example of a validated test helper function.
    """
    if expected_methods is None:
        expected_methods = ["test_method_1", "test_method_2"]

    validate_input(expected_methods, list, "expected_methods")

    for method in expected_methods:
        validate_input(method, str, "method in expected_methods")

    return True


__all__ = [
    "ValidationError",
    "BenchmarkError",
    "TestError",
    "validate_input",
    "validate_positive_number",
    "validate_range",
    "safe_execute",
    "error_handler",
    "retry_on_failure",
    "validate_model_plugin",
    "validate_benchmark_result",
    "validate_tensor_dimensions",
    "handle_plugin_initialization",
    "handle_model_loading",
    "validate_and_clean_text",
    "validate_device",
    "example_safe_benchmark_operation",
    "example_validated_test_helper",
]
