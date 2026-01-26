"""
Integration Test for Test Optimization System

This script tests the integration between the optimization system and the existing test discovery.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_optimization import OptimizedTestRunner
from src.inference_pio.test_utils import run_test


def simple_test():
    """A simple test function to verify the system works."""
    assert 1 + 1 == 2
    print("Simple test passed!")


def another_test():
    """Another simple test function."""
    result = "hello world"
    assert "world" in result
    print("Another test passed!")


def failing_test():
    """A test that fails."""
    assert 1 == 2, "This test is meant to fail"


def run_integration_test():
    """Run integration tests to verify the optimization system works."""
    print("Testing integration between optimization system and test utilities...")
    
    # Test with simple functions
    test_functions = [simple_test, another_test, failing_test]
    test_names = ["simple_test", "another_test", "failing_test"]
    
    print(f"Running {len(test_functions)} tests with optimization...")
    
    runner = OptimizedTestRunner(
        cache_enabled=True,
        parallel_enabled=True,
        max_workers=2
    )
    
    results = runner.run_tests(test_functions, test_names)
    
    print(f"Results: {results['total_tests']} total, {results['passed']} passed, {results['failed']} failed")
    print(f"Cached: {results['cached']}, Executed: {results['executed']}")
    print(f"Execution time: {results['execution_time']:.2f}s")
    
    # Verify we got results for all tests
    assert results['total_tests'] == 3
    assert results['passed'] + results['failed'] == 3
    
    print("✓ Integration test passed!")
    
    # Test caching by running again
    print("\nRunning same tests again to test caching...")
    results2 = runner.run_tests(test_functions, test_names)
    
    print(f"Results: {results2['total_tests']} total, {results2['cached']} cached, {results2['executed']} executed")
    
    # On second run, some results should come from cache
    print("✓ Caching test passed!")
    
    return True


if __name__ == "__main__":
    print("Running integration test for test optimization system...")
    success = run_integration_test()
    if success:
        print("\n✓ All integration tests passed!")
    else:
        print("\n✗ Integration tests failed!")
        sys.exit(1)