"""
Test Discovery System for Inference-PIO

This module provides a comprehensive test discovery mechanism that can find and run
test functions in the standardized test structure (unit/, integration/, performance/ subdirectories).
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import List, Callable, Dict, Any
try:
    from .test_utils import run_tests
except ImportError:
    # Handle direct import
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(current_dir, '..')
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from test_utils import run_tests
    except ImportError:
        # If still can't import, try absolute import
        abs_path = os.path.join(current_dir, 'test_utils.py')  # Same directory as test_discovery.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_utils", abs_path)
        test_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_utils)
        run_tests = test_utils.run_tests


def discover_test_functions_from_file(file_path: str) -> List[Callable]:
    """
    Discover test functions from a single Python file.

    Args:
        file_path: Path to the Python file to scan

    Returns:
        List of test functions found in the file
    """
    test_functions = []

    # Import the module dynamically
    spec = importlib.util.spec_from_file_location("__test_module__", file_path)
    if spec is None or spec.loader is None:
        return test_functions

    module = importlib.util.module_from_spec(spec)

    # Add the directory to sys.path temporarily to handle imports
    file_dir = os.path.dirname(file_path)
    original_path = sys.path[:]
    sys.path.insert(0, file_dir)

    try:
        spec.loader.exec_module(module)

        # Find all functions that start with 'test_'
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('test_'):
                test_functions.append(obj)

        # Also check for methods in classes that start with 'test_'
        for name, cls in inspect.getmembers(module, inspect.isclass):
            # Get all methods of the class that start with 'test_'
            for method_name in dir(cls):
                if method_name.startswith('test_'):
                    method_obj = getattr(cls, method_name)
                    if inspect.isfunction(method_obj) or inspect.ismethod(method_obj) or inspect.isbuiltin(method_obj):
                        # Create a wrapper that instantiates the class when called
                        def make_wrapper(cls_type, method_name_str):
                            def wrapper(*args, **kwargs):
                                instance = cls_type()
                                method = getattr(instance, method_name_str)
                                return method(*args, **kwargs)
                            return wrapper

                        wrapped_method = make_wrapper(cls, method_name)
                        wrapped_method.__name__ = f"{cls.__name__}.{method_name}"
                        test_functions.append(wrapped_method)

    except Exception as e:
        print(f"Warning: Could not import {file_path}: {e}")
    finally:
        # Restore original path
        sys.path[:] = original_path

    return test_functions


def discover_test_functions_from_directory(directory_path: str) -> List[Callable]:
    """
    Discover test functions from all Python files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory to scan
        
    Returns:
        List of test functions found in the directory
    """
    test_functions = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                test_functions.extend(discover_test_functions_from_file(file_path))
                
    return test_functions


def discover_tests_for_model(model_name: str) -> List[Callable]:
    """
    Discover all tests for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'qwen3_vl_2b', 'glm_4_7_flash')
        
    Returns:
        List of test functions for the model
    """
    model_tests_path = os.path.join(
        os.path.dirname(__file__), 
        'models', 
        model_name, 
        'tests'
    )
    
    if not os.path.exists(model_tests_path):
        print(f"Warning: Test directory does not exist for model {model_name}")
        return []
    
    # Discover tests from all subdirectories (unit, integration, performance)
    all_test_functions = []
    
    for subdir in ['unit', 'integration', 'performance']:
        subdir_path = os.path.join(model_tests_path, subdir)
        if os.path.exists(subdir_path):
            all_test_functions.extend(discover_test_functions_from_directory(subdir_path))
    
    # Also check the main tests directory
    all_test_functions.extend(discover_test_functions_from_directory(model_tests_path))
    
    return all_test_functions


def discover_tests_for_plugin_system() -> List[Callable]:
    """
    Discover all tests for the plugin system.
    
    Returns:
        List of test functions for the plugin system
    """
    plugin_tests_path = os.path.join(
        os.path.dirname(__file__), 
        'plugin_system', 
        'tests'
    )
    
    if not os.path.exists(plugin_tests_path):
        print("Warning: Plugin system test directory does not exist")
        return []
    
    # Discover tests from all subdirectories (unit, integration, performance)
    all_test_functions = []
    
    for subdir in ['unit', 'integration', 'performance']:
        subdir_path = os.path.join(plugin_tests_path, subdir)
        if os.path.exists(subdir_path):
            all_test_functions.extend(discover_test_functions_from_directory(subdir_path))
    
    # Also check the main tests directory if it exists
    all_test_functions.extend(discover_test_functions_from_directory(plugin_tests_path))
    
    return all_test_functions


def discover_all_tests() -> List[Callable]:
    """
    Discover all tests in the project.
    
    Returns:
        List of all test functions in the project
    """
    all_test_functions = []
    
    # Add tests from the main tests directory
    main_tests_path = os.path.join(os.path.dirname(__file__), 'tests')
    all_test_functions.extend(discover_test_functions_from_directory(main_tests_path))
    
    # Add tests from each model
    models_path = os.path.join(os.path.dirname(__file__), 'models')
    if os.path.exists(models_path):
        for model_name in os.listdir(models_path):
            model_path = os.path.join(models_path, model_name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'tests')):
                all_test_functions.extend(discover_tests_for_model(model_name))
    
    # Add tests from plugin system
    all_test_functions.extend(discover_tests_for_plugin_system())
    
    return all_test_functions


def run_discovered_tests(test_functions: List[Callable], verbose: bool = True) -> bool:
    """
    Run discovered test functions using the custom test utilities.
    
    Args:
        test_functions: List of test functions to run
        verbose: Whether to show detailed output
        
    Returns:
        True if all tests passed, False otherwise
    """
    if verbose:
        print(f"Discovered {len(test_functions)} test functions")
    
    if not test_functions:
        print("No test functions found!")
        return True
    
    return run_tests(test_functions)


def run_model_tests(model_name: str, verbose: bool = True) -> bool:
    """
    Run all tests for a specific model.
    
    Args:
        model_name: Name of the model to test
        verbose: Whether to show detailed output
        
    Returns:
        True if all tests passed, False otherwise
    """
    test_functions = discover_tests_for_model(model_name)
    return run_discovered_tests(test_functions, verbose)


def run_plugin_system_tests(verbose: bool = True) -> bool:
    """
    Run all tests for the plugin system.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        True if all tests passed, False otherwise
    """
    test_functions = discover_tests_for_plugin_system()
    return run_discovered_tests(test_functions, verbose)


def run_all_project_tests(verbose: bool = True) -> bool:
    """
    Run all tests in the project.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        True if all tests passed, False otherwise
    """
    test_functions = discover_all_tests()
    return run_discovered_tests(test_functions, verbose)


def get_test_summary() -> Dict[str, Any]:
    """
    Get a summary of all tests in the project.
    
    Returns:
        Dictionary with test summary information
    """
    summary = {
        'total_tests': 0,
        'models': {},
        'plugin_system': 0,
        'main_tests': 0
    }
    
    # Count main tests
    main_tests_path = os.path.join(os.path.dirname(__file__), 'tests')
    main_test_functions = discover_test_functions_from_directory(main_tests_path)
    summary['main_tests'] = len(main_test_functions)
    
    # Count tests for each model
    models_path = os.path.join(os.path.dirname(__file__), 'models')
    if os.path.exists(models_path):
        for model_name in os.listdir(models_path):
            model_path = os.path.join(models_path, model_name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'tests')):
                model_test_functions = discover_tests_for_model(model_name)
                summary['models'][model_name] = len(model_test_functions)
    
    # Count plugin system tests
    plugin_test_functions = discover_tests_for_plugin_system()
    summary['plugin_system'] = len(plugin_test_functions)
    
    # Calculate total
    summary['total_tests'] = (
        summary['main_tests'] + 
        sum(summary['models'].values()) + 
        summary['plugin_system']
    )
    
    return summary