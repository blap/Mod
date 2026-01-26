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
from tests.unit.common.optimization.specific.test_quantization_optimization import run_tests as run_quantization_tests
from tests.unit.common.optimization.specific.test_attention_optimization import run_tests as run_attention_tests
from tests.unit.common.optimization.specific.test_model_specific_optimization import run_tests as run_model_specific_tests

def run_all_glm47_optimization_tests():
    """Run all GLM-47 optimization tests from the split modules."""
    print("Running all GLM-47 optimization tests from split modules...")
    
    all_passed = True
    
    # Run quantization tests
    if not run_quantization_tests():
        all_passed = False
        
    # Run attention tests
    if not run_attention_tests():
        all_passed = False
        
    # Run model-specific tests
    if not run_model_specific_tests():
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_all_glm47_optimization_tests()
    if success:
        print("\n✓ All GLM-47 optimization tests passed!")
    else:
        print("\n✗ Some GLM-47 optimization tests failed!")
        sys.exit(1)