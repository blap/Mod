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
from tests.unit.common.security.test_security_contexts import run_tests as run_security_context_tests
from tests.unit.common.security.test_resource_management import run_tests as run_resource_management_tests

def run_all_security_isolation_tests():
    """Run all security isolation tests from the split modules."""
    print("Running all security isolation tests from split modules...")
    
    all_passed = True
    
    # Run security context tests
    if not run_security_context_tests():
        all_passed = False
        
    # Run resource management tests
    if not run_resource_management_tests():
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_all_security_isolation_tests()
    if success:
        print("\n✓ All security isolation tests passed!")
    else:
        print("\n✗ Some security isolation tests failed!")
        sys.exit(1)