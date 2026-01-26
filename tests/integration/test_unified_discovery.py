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
from tests.unit.common.discovery.test_discovery_logic import TestDiscoveryLogic
from tests.unit.common.discovery.test_item_filtering import TestItemFiltering
from tests.unit.common.discovery.test_integration import TestUnifiedTestDiscoveryIntegration, TestUtilityFunctions

# Create a test suite that includes all the new split test modules
def create_unified_discovery_test_suite():
    """Create a test suite from the split discovery test modules."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests from each split module
    suite.addTests(loader.loadTestsFromTestCase(TestDiscoveryLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestItemFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedTestDiscoveryIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    return suite


if __name__ == '__main__':
    # Create the test suite
    suite = create_unified_discovery_test_suite()
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)