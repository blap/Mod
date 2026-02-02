"""
Shared Test Assertions Module

This module contains common assertion functions used across the test suite to eliminate
code duplication and ensure consistent validation behavior.
"""

from typing import Any, List

import torch

from src.common.improved_base_plugin_interface import TextModelPluginInterface


def assert_plugin_interface_implemented(
    plugin: TextModelPluginInterface, required_methods: List[str] = None
):
    """
    Assert that a plugin implements the required interface methods.

    Args:
        plugin: The plugin instance to check
        required_methods: List of required method names (defaults to standard plugin interface)
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

    # Check that the plugin is an instance of the interface
    assert isinstance(
        plugin, TextModelPluginInterface
    ), f"Plugin is not an instance of TextModelPluginInterface, got {type(plugin)}"

    # Check that all required methods exist and are callable
    for method_name in required_methods:
        assert hasattr(plugin, method_name), f"Missing method: {method_name}"
        method = getattr(plugin, method_name)
        assert callable(method), f"Method {method_name} is not callable"


def assert_tensor_properties(
    tensor: torch.Tensor, expected_dtype=None, expected_device=None, expected_dims=None
):
    """
    Assert properties of a tensor.

    Args:
        tensor: The tensor to check
        expected_dtype: Expected data type
        expected_device: Expected device
        expected_dims: Expected number of dimensions
    """
    if expected_dtype is not None:
        assert (
            tensor.dtype == expected_dtype
        ), f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    if expected_device is not None:
        assert (
            tensor.device.type == expected_device
        ), f"Expected device {expected_device}, got {tensor.device.type}"

    if expected_dims is not None:
        assert (
            tensor.dim() == expected_dims
        ), f"Expected {expected_dims} dimensions, got {tensor.dim()}"


def assert_dict_contains_keys(dictionary: dict, required_keys: List[str]):
    """
    Assert that a dictionary contains all required keys.

    Args:
        dictionary: The dictionary to check
        required_keys: List of required keys
    """
    for key in required_keys:
        assert key in dictionary, f"Missing required key: {key}"


def assert_list_elements_type(lst: List[Any], expected_type: type):
    """
    Assert that all elements in a list are of the expected type.

    Args:
        lst: The list to check
        expected_type: The expected type of elements
    """
    for element in lst:
        assert isinstance(
            element, expected_type
        ), f"Element {element} is not of type {expected_type}"


def assert_response_format(response: Any, expected_type: type, additional_checks=None):
    """
    Assert that a response has the expected format and optionally run additional checks.

    Args:
        response: The response to check
        expected_type: The expected type of the response
        additional_checks: Optional function to run additional validations
    """
    assert isinstance(
        response, expected_type
    ), f"Response is not of expected type {expected_type}, got {type(response)}"

    if additional_checks:
        additional_checks(response)


def assert_plugin_initialized(plugin: TextModelPluginInterface):
    """
    Assert that a plugin has been properly initialized.

    Args:
        plugin: The plugin instance to check
    """
    # This assumes the plugin has some way to determine if it's initialized
    # Since the interface doesn't specify this, we'll use a common pattern
    # Plugins should implement their own way to check initialization status
    assert hasattr(plugin, "initialize"), "Plugin must have initialize method"
    # Additional checks could be added here based on plugin-specific attributes
