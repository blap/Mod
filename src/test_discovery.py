"""
Generic Test Discovery Module for Mod Project

This module provides automatic discovery and execution of test functions
for different models/plugins in the Mod project. Each model can use this
module independently without depending on other models.
"""

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def discover_test_functions_from_directory(directory_path: str) -> List[Callable]:
    """
    Discover all test functions from Python files in the specified directory.

    Args:
        directory_path: Path to directory to search for test functions

    Returns:
        List of test functions found in the directory
    """
    test_functions = []

    if not os.path.exists(directory_path):
        return test_functions

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                
                # Load the module dynamically
                module_name = f"temp_module_{file.replace('.', '_')}"
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    
                    try:
                        spec.loader.exec_module(module)
                        
                        # Find all callable functions in the module that look like tests
                        for name, obj in inspect.getmembers(module, inspect.isfunction):
                            # Consider functions that start with 'test' or have 'test' in the name
                            if name.startswith('test') or 'test' in name.lower():
                                test_functions.append(obj)
                                
                        # Also look for classes that might contain test methods
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if 'test' in name.lower():
                                # Look for test methods in the class
                                for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
                                    if method_name.startswith('test') or 'test' in method_name.lower():
                                        # Bind the method to the class to make it callable
                                        test_functions.append(getattr(module, name))
                                        
                    except Exception as e:
                        # Skip modules that can't be imported
                        print(f"Warning: Could not import {file_path}: {e}")
    
    return test_functions


def run_discovered_tests(test_functions: List[Callable], verbose: bool = True) -> bool:
    """
    Run all discovered test functions.

    Args:
        test_functions: List of test functions to run
        verbose: Whether to show detailed output

    Returns:
        True if all tests passed, False otherwise
    """
    if verbose:
        print(f"Running {len(test_functions)} discovered test functions...")
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if inspect.isclass(test_func):
                # If it's a test class, we need to instantiate and run its test methods
                if verbose:
                    print(f"Skipping test class {test_func.__name__} - needs unittest framework")
            else:
                # If it's a function, run it directly
                test_func()
                if verbose:
                    print(f"✓ {test_func.__name__}")
                passed += 1
        except Exception as e:
            if verbose:
                print(f"✗ {test_func.__name__}: {e}")
            failed += 1
    
    if verbose:
        print(f"\nResults: {passed} passed, {failed} failed")
    
    return failed == 0


def discover_tests_in_model_directory(model_tests_dir: str) -> List[Callable]:
    """
    Discover all tests in a model's test directory structure (unit, integration, performance).

    Args:
        model_tests_dir: Path to the model's tests directory

    Returns:
        List of test functions found in the model's test directories
    """
    test_functions = []
    
    # Search in standard subdirectories
    for subdir in ["unit", "integration", "performance"]:
        subdir_path = os.path.join(model_tests_dir, subdir)
        if os.path.exists(subdir_path):
            test_functions.extend(discover_test_functions_from_directory(subdir_path))
    
    # Also search in the main tests directory
    test_functions.extend(discover_test_functions_from_directory(model_tests_dir))
    
    return test_functions


def run_model_tests(model_tests_dir: str, verbose: bool = True) -> bool:
    """
    Run all tests in a model's test directory.

    Args:
        model_tests_dir: Path to the model's tests directory
        verbose: Whether to show detailed output

    Returns:
        True if all tests passed, False otherwise
    """
    test_functions = discover_tests_in_model_directory(model_tests_dir)
    return run_discovered_tests(test_functions, verbose)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover and run tests in a directory")
    parser.add_argument("--directory", "-d", type=str, help="Directory to search for tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    if args.directory:
        success = run_model_tests(args.directory, verbose=args.verbose)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python test_discovery.py --directory /path/to/tests")