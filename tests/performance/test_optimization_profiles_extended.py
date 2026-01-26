"""
Summary test file that imports and runs tests from the new split modules.
This maintains backward compatibility while using the new modular structure.
"""

import unittest
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import test suites from the new split modules
from tests.unit.common.optimization.test_general_profiles import run_tests as run_general_profile_tests
from tests.unit.common.optimization.test_model_profiles import run_tests as run_model_profile_tests

def run_all_optimization_profile_tests():
    """Run all optimization profile tests from the split modules."""
    print("Running all optimization profile tests from split modules...")
    
    all_passed = True
    
    # Run general profile tests
    if not run_general_profile_tests():
        all_passed = False
        
    # Run model-specific profile tests
    if not run_model_profile_tests():
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_all_optimization_profile_tests()
    if success:
        print("\n✓ All optimization profile tests passed!")
    else:
        print("\n✗ Some optimization profile tests failed!")
        sys.exit(1)