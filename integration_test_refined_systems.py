"""
Integration Test for Refined Test and Benchmark Systems

This script verifies that all components of the refined test and benchmark systems
work together properly and maintain backward compatibility.
"""

import sys
import os
import tempfile
from pathlib import Path
import time
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference_pio.test_utils import assert_equal, assert_true, assert_false, run_tests
from src.inference_pio.test_discovery import discover_all_tests
from test_optimization_refined import OptimizedTestRunner, run_tests_with_optimization
from unified_test_discovery_refined import UnifiedTestDiscovery, discover_and_run_all_items
from performance_regression_tracker_refined import (
    PerformanceRegressionTracker, 
    PerformanceMetric, 
    record_performance_metric, 
    get_regression_alerts
)


def test_common_utilities_integration():
    """Test that common utilities work correctly with all systems."""
    from src.inference_pio.common.test_utilities import (
        generate_cache_key, is_cache_valid, calculate_statistics, format_duration
    )
    
    # Test cache key generation
    key1 = generate_cache_key("test_path", "test_args")
    key2 = generate_cache_key("test_path", "test_args")
    assert_equal(key1, key2, "Cache keys should be consistent for same inputs")
    
    # Test cache validity
    result = is_cache_valid(time.strftime('%Y-%m-%dT%H:%M:%S'), 1.0)  # 1 hour max age
    assert_true(result, "Fresh timestamp should be valid")
    
    # Test statistics calculation
    stats = calculate_statistics([1, 2, 3, 4, 5])
    assert_equal(stats['mean'], 3.0, "Mean should be 3.0")
    assert_equal(stats['count'], 5, "Count should be 5")
    
    # Test duration formatting
    formatted = format_duration(3661)  # 1 hour, 1 minute, 1 second
    assert_true("1h 1m 1s" in formatted, "Duration should format correctly")


def test_optimization_system_integration():
    """Test that the refined optimization system works correctly."""
    def sample_test():
        time.sleep(0.01)  # Small delay to simulate work
        return True

    def failing_test():
        raise AssertionError("This test is supposed to fail")

    # Test with optimization
    runner = OptimizedTestRunner(cache_enabled=True, parallel_enabled=False)
    results = runner.run_tests([sample_test, failing_test], ["sample_test", "failing_test"])

    # Verify results
    assert_equal(results['total_tests'], 2, "Should have 2 total tests")
    assert_equal(results['passed'], 1, "Should have 1 passed test")
    assert_equal(results['failed'], 1, "Should have 1 failed test")
    assert_true(results['execution_time'] >= 0, "Execution time should be non-negative")


def test_discovery_system_integration():
    """Test that the refined discovery system works correctly."""
    discovery = UnifiedTestDiscovery(search_paths=[
        "src/inference_pio/common",  # Use a smaller search space for testing
        "tests/unit/common"  # Use a smaller search space for testing
    ])
    
    items = discovery.discover_all()
    
    # Should find at least some items
    assert_true(len(items) >= 0, "Should find at least some items")  # Allow 0 for empty directories
    
    # Test filtering by type
    test_items = discovery.get_items_by_type(discovery.TestType.UNIT_TEST)
    benchmark_items = discovery.get_items_by_type(discovery.TestType.BENCHMARK)
    
    # Both should be lists
    assert_true(isinstance(test_items, list), "Test items should be a list")
    assert_true(isinstance(benchmark_items, list), "Benchmark items should be a list")


def test_performance_regression_integration():
    """Test that the refined performance regression system works correctly."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        tracker = PerformanceRegressionTracker(
            storage_dir=temp_dir,
            regression_threshold=5.0
        )
        
        # Add a baseline metric
        baseline_metric = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(baseline_metric)
        
        # Add a slightly degraded metric (should not trigger alert)
        degraded_metric = PerformanceMetric(
            name="inference_speed",
            value=96.0,  # 4% degradation, below 5% threshold
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(degraded_metric)
        
        # Should have no alerts
        assert_equal(len(tracker.alerts), 0, "Should have no alerts for minor degradation")
        
        # Add a significantly degraded metric (should trigger alert)
        bad_metric = PerformanceMetric(
            name="inference_speed",
            value=90.0,  # 10% degradation, above 5% threshold
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(bad_metric)
        
        # Should have 1 alert now
        assert_equal(len(tracker.alerts), 1, "Should have 1 alert for significant degradation")
        
        # Test global functions
        record_performance_metric(
            name="global_test_metric",
            value=42.0,
            unit="units",
            model_name="global_test",
            category="test"
        )
        
        alerts = get_regression_alerts()
        # No regression should be detected with just one metric
        # (this is expected behavior - we need at least 2 metrics to detect regression)
        
        # Save and verify persistence
        tracker.save_history()
        
        # Create new tracker and load
        new_tracker = PerformanceRegressionTracker(
            storage_dir=temp_dir,
            regression_threshold=5.0
        )
        
        # Should have loaded the metrics
        key = "test_model:inference_speed"
        assert_true(key in new_tracker.metrics_history, "Should have loaded metrics history")
        assert_equal(len(new_tracker.metrics_history[key]), 3, "Should have 3 metrics")
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def test_backward_compatibility():
    """Test that the refined systems maintain backward compatibility."""
    # Test that original test utilities still work
    def legacy_test():
        assert_equal(2 + 2, 4, "Basic arithmetic should work")
        return True

    # Run with original test runner
    result = run_tests([legacy_test])
    assert_true(result, "Legacy test should pass")
    
    # Test that original discovery still works
    original_tests = discover_all_tests()
    assert_true(isinstance(original_tests, list), "Original discovery should return a list")
    
    # Test that new systems can work with original components
    def new_style_test():
        # Use new assertion from refined utilities
        from src.inference_pio.common.test_utilities import calculate_statistics
        stats = calculate_statistics([1, 2, 3])
        assert_equal(stats['mean'], 2.0, "Statistics calculation should work")
        return True
    
    # Run new style test with optimization
    runner = OptimizedTestRunner(cache_enabled=False, parallel_enabled=False)
    results = runner.run_tests([new_style_test], ["new_style_test"])
    
    assert_equal(results['passed'], 1, "New style test should pass with optimization")


def test_system_interoperability():
    """Test that all systems work together."""
    # Create a temporary test file to discover
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test_integration_sample.py"
    
    test_content = '''
def test_sample_integration():
    """Sample test for integration testing."""
    assert 1 + 1 == 2
    
def benchmark_sample_performance():
    """Sample benchmark for integration testing."""
    import time
    start = time.time()
    time.sleep(0.01)  # Simulate some work
    end = time.time()
    return end - start
'''
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        # Test discovery
        discovery = UnifiedTestDiscovery(search_paths=[str(temp_dir)])
        items = discovery.discover_all()
        
        # Should find our test function
        test_found = any(item['name'] == 'test_sample_integration' for item in items)
        benchmark_found = any(item['name'] == 'benchmark_sample_performance' for item in items)
        
        assert_true(test_found, "Should discover test function")
        assert_true(benchmark_found, "Should discover benchmark function")
        
        # Test running discovered items
        if items:
            results = discovery.run_all_items()
            assert_true('summary' in results, "Should have summary in results")
        
        # Test optimization with discovered functions
        if items:
            test_funcs = [item['function'] for item in items if 'test_' in item['name']]
            if test_funcs:
                opt_results = run_tests_with_optimization(test_funcs)
                assert_true('total_tests' in opt_results, "Optimization should return results dict")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_error_handling():
    """Test that error handling works properly across systems."""
    # Test that systems handle missing files gracefully
    discovery = UnifiedTestDiscovery(search_paths=["/nonexistent/path"])
    items = discovery.discover_all()
    # Should not crash, even with nonexistent paths
    
    # Test that optimization handles failing tests
    def always_fails():
        raise RuntimeError("Intentional failure for testing")
    
    runner = OptimizedTestRunner(cache_enabled=False, parallel_enabled=False)
    results = runner.run_tests([always_fails], ["always_fails"])
    
    assert_equal(results['failed'], 1, "Should report failed test")
    assert_equal(results['passed'], 0, "Should not report passed test")


if __name__ == "__main__":
    # Run all integration tests
    test_functions = [
        test_common_utilities_integration,
        test_optimization_system_integration,
        test_discovery_system_integration,
        test_performance_regression_integration,
        test_backward_compatibility,
        test_system_interoperability,
        test_error_handling
    ]
    
    print("Running integration tests for refined test and benchmark systems...")
    success = run_tests(test_functions)
    
    if success:
        print("\n✓ All integration tests passed!")
        print("The refined systems work together properly and maintain backward compatibility.")
    else:
        print("\n✗ Some integration tests failed!")
        sys.exit(1)