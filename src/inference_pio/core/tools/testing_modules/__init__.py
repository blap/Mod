"""
Main Testing Orchestrator for Mod Project

This module provides a unified interface to run all types of tests
across the Mod project. Each test module is independent but can be
orchestrated together through this central module.
"""

import sys
import os
from typing import List, Type
import unittest
from enum import Enum

# Add the testing modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import the individual testing modules
from testing_modules.functional_testing import (
    FunctionalTestBase, ModelFunctionalTest, PluginFunctionalTest, 
    run_functional_tests, functional_test_suite
)
from testing_modules.performance_testing import (
    PerformanceTestBase, ModelPerformanceTest, PluginPerformanceTest,
    run_performance_tests, performance_test_suite
)
from testing_modules.integration_testing import (
    IntegrationTestBase, ModelIntegrationTest, PipelineIntegrationTest, PluginIntegrationTest,
    run_integration_tests, integration_test_suite
)
from testing_modules.regression_testing import (
    RegressionTestBase, ModelRegressionTest, FeatureRegressionTest, SystemRegressionTest,
    run_regression_tests, regression_test_suite
)
from testing_modules.unit_testing import (
    UnitTestBase, ModelUnitTest, PluginUnitTest, ComponentUnitTest,
    run_unit_tests, unit_test_suite
)

class TestType(Enum):
    """Enumeration of different test types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


class TestOrchestrator:
    """
    Main orchestrator for running different types of tests.
    Provides methods to run individual test types or combinations.
    """
    
    def __init__(self):
        self.test_results = {}
    
    def run_tests_by_type(self, test_type: TestType, verbosity: int = 2):
        """
        Run tests of a specific type.
        
        Args:
            test_type: Type of tests to run
            verbosity: Verbosity level for test output
            
        Returns:
            TestResult object with results
        """
        if test_type == TestType.UNIT:
            # For unit tests, we'd normally pass specific test classes
            # Since we're running the suite here, we'll just run the discovery
            suite = unit_test_suite()
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            self.test_results[test_type.value] = result
            return result
            
        elif test_type == TestType.INTEGRATION:
            suite = integration_test_suite()
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            self.test_results[test_type.value] = result
            return result
            
        elif test_type == TestType.FUNCTIONAL:
            suite = functional_test_suite()
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            self.test_results[test_type.value] = result
            return result
            
        elif test_type == TestType.PERFORMANCE:
            suite = performance_test_suite()
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            self.test_results[test_type.value] = result
            return result
            
        elif test_type == TestType.REGRESSION:
            suite = regression_test_suite()
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            self.test_results[test_type.value] = result
            return result
    
    def run_all_tests(self, verbosity: int = 2):
        """
        Run all types of tests.
        
        Args:
            verbosity: Verbosity level for test output
            
        Returns:
            Dictionary mapping test types to their results
        """
        results = {}
        
        for test_type in TestType:
            print(f"\n{'='*50}")
            print(f"Running {test_type.value.upper()} Tests")
            print(f"{'='*50}")
            results[test_type.value] = self.run_tests_by_type(test_type, verbosity)
        
        self.test_results.update(results)
        return results
    
    def run_selected_tests(self, test_types: List[TestType], verbosity: int = 2):
        """
        Run selected types of tests.
        
        Args:
            test_types: List of test types to run
            verbosity: Verbosity level for test output
            
        Returns:
            Dictionary mapping test types to their results
        """
        results = {}
        
        for test_type in test_types:
            print(f"\n{'='*50}")
            print(f"Running {test_type.value.upper()} Tests")
            print(f"{'='*50}")
            results[test_type.value] = self.run_tests_by_type(test_type, verbosity)
        
        self.test_results.update(results)
        return results
    
    def get_test_summary(self):
        """
        Get a summary of all test results.
        
        Returns:
            Dictionary with test summary information
        """
        summary = {}
        
        for test_type, result in self.test_results.items():
            if hasattr(result, 'testsRun'):
                summary[test_type] = {
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors),
                    'success': result.wasSuccessful()
                }
        
        return summary
    
    def print_test_summary(self):
        """Print a formatted summary of all test results."""
        summary = self.get_test_summary()
        
        print(f"\n{'='*60}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for test_type, stats in summary.items():
            print(f"{test_type.upper()}:")
            print(f"  Tests run: {stats['tests_run']}")
            print(f"  Failures: {stats['failures']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Success: {'Yes' if stats['success'] else 'No'}")
            print()
            
            total_tests += stats['tests_run']
            total_failures += stats['failures']
            total_errors += stats['errors']
        
        print(f"TOTAL: {total_tests} tests run, {total_failures} failures, {total_errors} errors")
        print(f"Overall success: {'Yes' if all(s['success'] for s in summary.values()) else 'No'}")
        print(f"{'='*60}")


def run_mod_tests(test_types: List[TestType] = None, verbosity: int = 2):
    """
    Convenience function to run Mod project tests.
    
    Args:
        test_types: List of test types to run (runs all if None)
        verbosity: Verbosity level for test output
        
    Returns:
        TestOrchestrator instance with results
    """
    orchestrator = TestOrchestrator()
    
    if test_types is None:
        # Run all test types
        results = orchestrator.run_all_tests(verbosity)
    else:
        # Run selected test types
        results = orchestrator.run_selected_tests(test_types, verbosity)
    
    # Print summary
    orchestrator.print_test_summary()
    
    return orchestrator


# Example usage
if __name__ == "__main__":
    print("Mod Project Testing Orchestrator")
    print("="*40)
    
    # Example: Run all tests
    # test_types = [TestType.UNIT, TestType.INTEGRATION, TestType.FUNCTIONAL, TestType.PERFORMANCE, TestType.REGRESSION]
    # orchestrator = run_mod_tests(test_types)
    
    # Or run all tests by default
    orchestrator = run_mod_tests()
    
    print("\nTesting completed!")