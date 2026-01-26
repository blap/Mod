"""
Test Discovery for GLM 4 7 Model

This module provides test discovery for the GLM 4 7 model tests.
"""

import sys
import os

# Add the src directory to the path so we can import the modules
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..', '..', '..', '..')
sys.path.insert(0, src_dir)

from src.inference_pio.test_discovery import discover_test_functions_from_directory, run_discovered_tests


def discover_tests():
    """
    Discover all tests for the GLM 4 7 model.
    
    Returns:
        List of test functions found in the model's test directories
    """
    test_functions = []
    
    # Define the base path for this model's tests
    base_path = os.path.dirname(__file__)
    
    # Discover tests from all subdirectories (unit, integration, performance)
    for subdir in ['unit', 'integration', 'performance']:
        subdir_path = os.path.join(base_path, subdir)
        if os.path.exists(subdir_path):
            test_functions.extend(discover_test_functions_from_directory(subdir_path))
    
    # Also check the main tests directory
    test_functions.extend(discover_test_functions_from_directory(base_path))
    
    return test_functions


def run_tests(verbose=True):
    """
    Run all discovered tests for the GLM 4 7 model.
    
    Args:
        verbose: Whether to show detailed output
        
    Returns:
        True if all tests passed, False otherwise
    """
    test_functions = discover_tests()
    return run_discovered_tests(test_functions, verbose)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for GLM 4 7 model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    success = run_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)