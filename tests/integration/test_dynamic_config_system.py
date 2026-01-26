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
from tests.unit.common.config.test_config_management import run_tests as run_config_management_tests
from tests.unit.common.config.test_dynamic_config import run_tests as run_dynamic_config_tests
from tests.unit.common.config.test_model_specific_config import run_tests as run_model_specific_config_tests

def run_all_config_tests():
    """Run all config-related tests from the split modules."""
    print("Running all config tests from split modules...")
    
    all_passed = True
    
    # Run config management tests
    if not run_config_management_tests():
        all_passed = False
        
    # Run dynamic config tests
    if not run_dynamic_config_tests():
        all_passed = False
        
    # Run model-specific config tests
    if not run_model_specific_config_tests():
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_all_config_tests()
    if success:
        print("\n✓ All config tests passed!")
    else:
        print("\n✗ Some config tests failed!")
        sys.exit(1)