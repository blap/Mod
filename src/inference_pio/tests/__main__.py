"""
Test Runner for Inference-PIO Self-Contained Plugins

This module provides a test runner to execute all tests for the self-contained plugins
using the custom test utilities instead of unittest.
"""

import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ..test_discovery import (
    discover_all_tests,
    discover_tests_for_model,
    discover_tests_for_plugin_system,
    run_discovered_tests
)


def run_all_tests():
    """
    Run all tests for the Inference-PIO self-contained plugins.
    """
    test_functions = discover_all_tests()
    return run_discovered_tests(test_functions, verbose=True)


def run_specific_test_suite(suite_name):
    """
    Run a specific test suite.

    Args:
        suite_name: Name of the test suite to run (e.g., 'glm_4_7_flash', 'qwen3_coder_30b', etc.)
    """
    if suite_name == "glm_4_7_flash":
        test_functions = discover_tests_for_model("glm_4_7_flash")
    elif suite_name == "qwen3_coder_30b":
        test_functions = discover_tests_for_model("qwen3_coder_30b")
    elif suite_name == "qwen3_vl_2b":
        test_functions = discover_tests_for_model("qwen3_vl_2b")
    elif suite_name == "qwen3_4b_instruct_2507":
        test_functions = discover_tests_for_model("qwen3_4b_instruct_2507")
    elif suite_name == "plugin_system":
        test_functions = discover_tests_for_plugin_system()
    else:
        raise ValueError(f"Unknown test suite: {suite_name}")

    return run_discovered_tests(test_functions, verbose=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run tests for Inference-PIO self-contained plugins')
    parser.add_argument('--suite', type=str, help='Run a specific test suite (glm_4_7_flash, qwen3_coder_30b, qwen3_vl_2b, qwen3_4b_instruct_2507, plugin_system)')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    success = True

    if args.suite:
        # Run specific test suite
        success = run_specific_test_suite(args.suite)
    elif args.all:
        # Run all tests
        success = run_all_tests()
    else:
        # Run all tests by default
        success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)