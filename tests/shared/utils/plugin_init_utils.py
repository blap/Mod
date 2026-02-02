"""
Plugin Initialization Utilities Module

This module contains common utilities for initializing plugins across the test suite
to eliminate code duplication and ensure consistent behavior.
"""

from typing import Any, Dict, Optional, Type

from src.common.improved_base_plugin_interface import TextModelPluginInterface
from src.testing_utils import (
    create_test_model_instance,
    verify_plugin_interface as verify_plugin_interface_shared,
    run_basic_functionality_tests as run_basic_functionality_tests_shared
)


def initialize_plugin_for_test(
    plugin_class: Type[TextModelPluginInterface],
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> TextModelPluginInterface:
    """
    Initialize a plugin instance with test configuration.

    Args:
        plugin_class: The plugin class to instantiate
        config: Configuration dictionary for the plugin
        **kwargs: Additional keyword arguments for initialization

    Returns:
        Initialized plugin instance
    """
    # Use the shared utility function
    return create_test_model_instance(plugin_class, **kwargs)


def create_and_initialize_plugin(
    plugin_class: Type[TextModelPluginInterface],
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> TextModelPluginInterface:
    """
    Create and initialize a plugin instance in one step.

    Args:
        plugin_class: The plugin class to instantiate
        config: Configuration dictionary for the plugin
        **kwargs: Additional keyword arguments for initialization

    Returns:
        Initialized plugin instance
    """
    return initialize_plugin_for_test(plugin_class, config, **kwargs)


def cleanup_plugin(plugin: TextModelPluginInterface) -> bool:
    """
    Safely clean up a plugin instance.

    Args:
        plugin: The plugin instance to clean up

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        return plugin.cleanup()
    except Exception as e:
        print(f"Warning: Error during plugin cleanup: {e}")
        return False


def verify_plugin_interface(
    plugin: TextModelPluginInterface, required_methods: Optional[list] = None
) -> bool:
    """
    Verify that a plugin implements the required interface methods.

    Args:
        plugin: The plugin instance to verify
        required_methods: List of required method names (uses defaults if None)

    Returns:
        True if all required methods are present, False otherwise
    """
    # Use the shared utility function
    return verify_plugin_interface_shared(plugin, required_methods)


def run_basic_functionality_tests(
    plugin: TextModelPluginInterface, test_inputs: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run basic functionality tests on a plugin instance.

    Args:
        plugin: The plugin instance to test
        test_inputs: List of test inputs to use (uses defaults if None)

    Returns:
        Dictionary with test results
    """
    # Use the shared utility function
    return run_basic_functionality_tests_shared(plugin, test_inputs)
