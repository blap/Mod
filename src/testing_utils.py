"""
Testing Utilities Module for Mod Project

This module provides common utilities for testing different models/plugins
in the Mod project. Each model can use this module independently.
"""

import os
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch


def create_temp_test_config() -> Dict[str, Any]:
    """
    Create a temporary test configuration with common test settings.

    Returns:
        Dictionary with test configuration
    """
    temp_dir = tempfile.mkdtemp()
    return {
        "device": "cpu",
        "batch_size": 1,
        "test_mode": True,
        "temp_dir": temp_dir,
        "use_mock_model": False,  # Use real models for testing
    }


def cleanup_temp_config(config: Dict[str, Any]) -> None:
    """
    Clean up temporary configuration resources.

    Args:
        config: Configuration dictionary to clean up
    """
    if "temp_dir" in config and os.path.exists(config["temp_dir"]):
        import shutil
        shutil.rmtree(config["temp_dir"], ignore_errors=True)


def create_mock_model(input_dim: int = 10, output_dim: int = 1):
    """
    Create a simple mock PyTorch model for testing purposes.

    Args:
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model

    Returns:
        Simple PyTorch model instance
    """
    import torch

    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    return SimpleModel(input_dim, output_dim)


def assert_tensor_shape(tensor: Any, expected_shape: tuple) -> bool:
    """
    Assert that a tensor has the expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape

    Returns:
        True if shapes match, False otherwise
    """
    if hasattr(tensor, 'shape'):
        return tuple(tensor.shape) == expected_shape
    return False


def assert_tensor_values_close(tensor1: Any, tensor2: Any, tolerance: float = 1e-6) -> bool:
    """
    Assert that two tensors have close values within a tolerance.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        tolerance: Tolerance for comparison

    Returns:
        True if tensors are close, False otherwise
    """
    if hasattr(tensor1, 'allclose') and hasattr(tensor2, 'allclose'):
        import torch
        return torch.allclose(tensor1, tensor2, atol=tolerance)
    return False


def safe_execute(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely execute a function and return result or error information.

    Args:
        func: Function to execute
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Dictionary with 'result' and 'error' keys
    """
    try:
        result = func(*args, **kwargs)
        return {'result': result, 'error': None}
    except Exception as e:
        return {'result': None, 'error': str(e)}


def create_test_model_instance(plugin_class, **init_kwargs):
    """
    Create and initialize a model plugin instance for testing.

    Args:
        plugin_class: Class of the plugin to instantiate
        **init_kwargs: Initialization arguments

    Returns:
        Initialized plugin instance
    """
    # Create plugin instance
    plugin = plugin_class()

    # Apply default test configuration
    default_config = {
        "device": "cpu",  # Use CPU for tests to ensure consistency
        "use_mock_model": False,  # Explicitly use real model
    }
    default_config.update(init_kwargs)

    # Initialize the plugin
    success = plugin.initialize(**default_config)

    # If initialization fails, still return the plugin for interface testing
    # but log the issue
    if not success:
        import logging
        logging.warning(
            f"Plugin initialization failed for {plugin_class.__name__}, continuing with uninitialized plugin for interface tests"
        )

    return plugin


def verify_plugin_interface(plugin, required_methods: Optional[list] = None) -> bool:
    """
    Verify that a plugin implements the required interface methods.

    Args:
        plugin: The plugin instance to verify
        required_methods: List of required method names (uses defaults if None)

    Returns:
        True if all required methods are present, False otherwise
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

    for method_name in required_methods:
        if not hasattr(plugin, method_name):
            print(f"Missing required method: {method_name}")
            return False

        method = getattr(plugin, method_name)
        if not callable(method):
            print(f"Method {method_name} is not callable")
            return False

    return True


def run_basic_functionality_tests(
    plugin, test_inputs: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run basic functionality tests on a plugin instance.

    Args:
        plugin: The plugin instance to test
        test_inputs: List of test inputs to use (uses defaults if None)

    Returns:
        Dictionary with test results
    """
    if test_inputs is None:
        test_inputs = ["Hello, world!", "How are you?", "What is your name?"]

    results = {
        "initialize_success": False,
        "basic_inference_success": False,
        "tokenization_success": False,
        "generation_success": False,
        "errors": [],
    }

    try:
        # Test initialization
        init_result = plugin.initialize()
        results["initialize_success"] = init_result is True

        if results["initialize_success"]:
            # Test basic inference
            try:
                if hasattr(plugin, "infer"):
                    test_input = test_inputs[0] if test_inputs else "Test input"
                    inference_result = plugin.infer(test_input)
                    results["basic_inference_success"] = inference_result is not None
            except Exception as e:
                results["errors"].append(f"Inference test failed: {e}")

            # Test tokenization
            try:
                if hasattr(plugin, "tokenize") and hasattr(plugin, "detokenize"):
                    test_text = "Test tokenization"
                    tokens = plugin.tokenize(test_text)
                    detokenized = plugin.detokenize(tokens)
                    results["tokenization_success"] = detokenized is not None
            except Exception as e:
                results["errors"].append(f"Tokenization test failed: {e}")

            # Test text generation
            try:
                if hasattr(plugin, "generate_text"):
                    test_prompt = "Once upon a time"
                    generated = plugin.generate_text(test_prompt, max_new_tokens=10)
                    results["generation_success"] = generated is not None
            except Exception as e:
                results["errors"].append(f"Generation test failed: {e}")

        return results
    except Exception as e:
        results["errors"].append(f"Basic functionality test failed: {e}")
        return results