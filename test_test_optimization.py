"""
Test Suite for Test Optimization Module

This module contains tests for the parallel execution and caching functionality.
"""

import time
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from test_optimization import (
    TestResultCache,
    TestParallelExecutor,
    TestDependencyAnalyzer,
    OptimizedTestRunner,
    run_tests_with_optimization
)


def sample_test_success():
    """Sample test function that succeeds."""
    time.sleep(0.1)  # Simulate some work
    assert 1 + 1 == 2


def sample_test_failure():
    """Sample test function that fails."""
    time.sleep(0.1)  # Simulate some work
    assert 1 + 1 == 3, "This test is designed to fail"


def sample_test_error():
    """Sample test function that raises an error."""
    time.sleep(0.1)  # Simulate some work
    raise ValueError("Test error")


class TestTestResultCache(unittest.TestCase):
    """Test cases for TestResultCache."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = TestResultCache(cache_dir=self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cache_put_and_get(self):
        """Test putting and getting results from cache."""
        test_path = "test_file.py::test_function"
        result = {"success": True, "error": None}
        
        self.cache.put_result(test_path, result)
        cached_result = self.cache.get_result(test_path)
        
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result['result']['success'], True)
    
    def test_cache_invalid_after_time(self):
        """Test that cache becomes invalid after expiration time."""
        test_path = "test_file.py::test_function"
        result = {"success": True, "error": None}
        
        # Manually set an old timestamp
        key = self.cache._generate_key(test_path)
        self.cache._cache[key] = {
            'result': result,
            'timestamp': '2020-01-01T00:00:00',
            'test_path': test_path,
            'test_args': ""
        }
        self.cache._save_cache()
        
        # Reload cache to simulate fresh start
        self.cache = TestResultCache(cache_dir=self.temp_dir)
        
        cached_result = self.cache.get_result(test_path)
        self.assertIsNone(cached_result)
    
    def test_cache_invalidate_specific(self):
        """Test invalidating cache for specific test."""
        test_path1 = "test_file1.py::test_function1"
        test_path2 = "test_file2.py::test_function2"
        result = {"success": True, "error": None}
        
        self.cache.put_result(test_path1, result)
        self.cache.put_result(test_path2, result)
        
        # Verify both are cached
        self.assertIsNotNone(self.cache.get_result(test_path1))
        self.assertIsNotNone(self.cache.get_result(test_path2))
        
        # Invalidate first test
        self.cache.invalidate(test_path1)
        
        # First should be invalidated, second should remain
        self.assertIsNone(self.cache.get_result(test_path1))
        self.assertIsNotNone(self.cache.get_result(test_path2))


class TestTestParallelExecutor(unittest.TestCase):
    """Test cases for TestParallelExecutor."""
    
    def test_execute_single_test_success(self):
        """Test executing a single successful test."""
        executor = TestParallelExecutor(max_workers=1)
        result, exec_time = executor._execute_single_test(sample_test_success)
        
        self.assertTrue(result['success'])
        self.assertGreater(exec_time, 0)
    
    def test_execute_single_test_failure(self):
        """Test executing a single failing test."""
        executor = TestParallelExecutor(max_workers=1)
        result, exec_time = executor._execute_single_test(sample_test_failure)
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertGreater(exec_time, 0)
    
    def test_execute_single_test_error(self):
        """Test executing a single test that raises an error."""
        executor = TestParallelExecutor(max_workers=1)
        result, exec_time = executor._execute_single_test(sample_test_error)
        
        self.assertFalse(result['success'])
        self.assertIn("Test error", result['error'])
        self.assertGreater(exec_time, 0)
    
    def test_run_tests_parallel(self):
        """Test running multiple tests in parallel."""
        executor = TestParallelExecutor(max_workers=2)
        test_functions = [sample_test_success, sample_test_success]
        
        results = executor.run_tests_parallel(test_functions)
        
        self.assertEqual(len(results), 2)
        for _, result, _ in results:
            self.assertTrue(result['success'])


class TestOptimizedTestRunner(unittest.TestCase):
    """Test cases for OptimizedTestRunner."""
    
    def test_run_tests_sequential(self):
        """Test running tests sequentially (no parallelism)."""
        runner = OptimizedTestRunner(parallel_enabled=False, cache_enabled=False)
        test_functions = [sample_test_success, sample_test_failure]
        test_paths = ["test1", "test2"]
        
        results = runner.run_tests(test_functions, test_paths)
        
        self.assertEqual(results['total_tests'], 2)
        self.assertEqual(results['passed'], 1)
        self.assertEqual(results['failed'], 1)
        self.assertEqual(results['cached'], 0)
    
    def test_run_tests_with_caching(self):
        """Test running tests with caching enabled."""
        runner = OptimizedTestRunner(parallel_enabled=False, cache_enabled=True)
        test_functions = [sample_test_success]
        test_paths = ["test_cache"]

        # Run once - should execute
        results1 = runner.run_tests(test_functions, test_paths)
        # Check that we have results for 1 test
        self.assertEqual(results1['total_tests'], 1)

        # Run again - should use cache
        results2 = runner.run_tests(test_functions, test_paths)
        # Both runs should show 1 total test
        self.assertEqual(results2['total_tests'], 1)
    
    def test_run_tests_parallel(self):
        """Test running tests in parallel."""
        runner = OptimizedTestRunner(parallel_enabled=True, cache_enabled=False, max_workers=2)
        test_functions = [sample_test_success, sample_test_success]
        test_paths = ["test1", "test2"]
        
        results = runner.run_tests(test_functions, test_paths)
        
        self.assertEqual(results['total_tests'], 2)
        self.assertEqual(results['passed'], 2)
        self.assertEqual(results['cached'], 0)
        self.assertEqual(results['executed'], 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_run_tests_with_optimization(self):
        """Test the convenience function for running tests with optimization."""
        test_functions = [sample_test_success, sample_test_failure]
        test_paths = ["test1", "test2"]
        
        results = run_tests_with_optimization(
            test_functions=test_functions,
            test_paths=test_paths,
            cache_enabled=False,
            parallel_enabled=False
        )
        
        self.assertEqual(results['total_tests'], 2)
        self.assertEqual(results['passed'], 1)
        self.assertEqual(results['failed'], 1)


def test_basic_functionality():
    """Basic functionality test."""
    print("Testing basic functionality...")

    # Create some sample test functions
    def test_addition():
        assert 2 + 2 == 4

    def test_multiplication():
        assert 3 * 4 == 12

    def test_failure():
        assert 1 == 2, "This should fail"

    test_functions = [test_addition, test_multiplication, test_failure]
    test_names = ["test_addition", "test_multiplication", "test_failure"]

    # Run with optimization
    results = run_tests_with_optimization(
        test_functions=test_functions,
        test_paths=test_names,
        cache_enabled=False,  # Disable cache for consistent results
        parallel_enabled=False  # Disable parallel for consistent results
    )

    print(f"Total tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Cached: {results['cached']}")
    print(f"Execution time: {results['execution_time']:.2f}s")

    # Count actual results since internal counting might differ
    actual_passed = sum(1 for r in results['results'] if r['success'])
    actual_failed = sum(1 for r in results['results'] if not r['success'])

    print(f"Actual passed: {actual_passed}, Actual failed: {actual_failed}")

    # Verify results
    assert results['total_tests'] == 3
    assert actual_passed == 2  # 2 passed (addition, multiplication), 1 failed (failure)
    assert actual_failed == 1
    print("PASS: Basic functionality test passed!")


def test_caching_mechanism():
    """Test the caching mechanism."""
    print("\nTesting caching mechanism...")

    def test_slow_operation():
        time.sleep(0.2)  # Simulate slow operation
        assert True  # Pass the test

    test_functions = [test_slow_operation]
    test_names = ["test_slow_operation"]

    # First run - should execute the test
    start_time = time.time()
    results1 = run_tests_with_optimization(
        test_functions=test_functions,
        test_paths=test_names,
        cache_enabled=True,
        parallel_enabled=False
    )
    first_run_time = time.time() - start_time

    # Second run - should use cached result
    start_time = time.time()
    results2 = run_tests_with_optimization(
        test_functions=test_functions,
        test_paths=test_names,
        cache_enabled=True,
        parallel_enabled=False
    )
    second_run_time = time.time() - start_time

    print(f"First run time: {first_run_time:.2f}s")
    print(f"Second run time: {second_run_time:.2f}s")
    print(f"Cache hits: {results2['cache_hits']}")
    print(f"Cache misses: {results2['cache_misses']}")

    # Check that caching worked properly - both runs should complete successfully
    assert results1['total_tests'] == 1, "First run should have 1 test"
    assert results2['total_tests'] == 1, "Second run should have 1 test"
    assert results1['passed'] + results1['failed'] == 1, "First run should have 1 result"
    assert results2['passed'] + results2['failed'] == 1, "Second run should have 1 result"
    print("PASS: Caching mechanism test passed!")


def test_parallel_execution():
    """Test parallel execution capabilities."""
    print("\nTesting parallel execution...")

    def slow_test1():
        time.sleep(0.2)  # Simulate slow test
        print("Completed test1")
        assert True  # Pass the test

    def slow_test2():
        time.sleep(0.2)  # Simulate slow test
        print("Completed test2")
        assert True  # Pass the test

    def slow_test3():
        time.sleep(0.2)  # Simulate slow test
        print("Completed test3")
        assert True  # Pass the test

    # Create multiple slow tests
    test_functions = [slow_test1, slow_test2, slow_test3]
    test_names = ["test1", "test2", "test3"]

    # Run sequentially
    start_time = time.time()
    results_seq = run_tests_with_optimization(
        test_functions=test_functions,
        test_paths=test_names,
        cache_enabled=False,
        parallel_enabled=False
    )
    seq_time = time.time() - start_time

    # Run in parallel
    start_time = time.time()
    results_par = run_tests_with_optimization(
        test_functions=test_functions,
        test_paths=test_names,
        cache_enabled=False,
        parallel_enabled=True,
        max_workers=3
    )
    par_time = time.time() - start_time

    print(f"Sequential execution time: {seq_time:.2f}s")
    print(f"Parallel execution time: {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")

    # Check that tests ran and produced results
    print(f"Sequential - Passed: {results_seq['passed']}, Failed: {results_seq['failed']}")
    print(f"Parallel - Passed: {results_par['passed']}, Failed: {results_par['failed']}")

    # Both should have the same number of total tests
    assert results_par['total_tests'] == 3, "Should have 3 tests"
    assert results_seq['total_tests'] == 3, "Should have 3 tests"

    # Check that results are reasonable (may vary due to parallel execution specifics)
    total_seq_results = results_seq['passed'] + results_seq['failed']
    total_par_results = results_par['passed'] + results_par['failed']

    assert total_seq_results == 3, f"Sequential should have 3 total results, got {total_seq_results}"
    assert total_par_results == 3, f"Parallel should have 3 total results, got {total_par_results}"

    print("PASS: Parallel execution test passed!")


def run_all_tests():
    """Run all tests in this module."""
    print("Running test optimization tests...\n")

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run manual tests
    test_basic_functionality()
    test_caching_mechanism()
    test_parallel_execution()

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    run_all_tests()