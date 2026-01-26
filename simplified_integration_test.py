"""
Simplified Integration Test for Refined Test and Benchmark Systems

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

# Import only the core components we need to test
from src.inference_pio.common.test_utilities import (
    generate_cache_key, is_cache_valid, calculate_statistics, format_duration
)
from test_optimization_refined import OptimizedTestRunner, run_tests_with_optimization
from unified_test_discovery_refined import UnifiedTestDiscovery
from performance_regression_tracker_refined import (
    PerformanceRegressionTracker, 
    PerformanceMetric, 
    record_performance_metric, 
    get_regression_alerts
)


def test_common_utilities_integration():
    """Test that common utilities work correctly with all systems."""
    # Test cache key generation
    key1 = generate_cache_key("test_path", "test_args")
    key2 = generate_cache_key("test_path", "test_args")
    assert key1 == key2, "Cache keys should be consistent for same inputs"
    
    # Test cache validity
    result = is_cache_valid(time.strftime('%Y-%m-%dT%H:%M:%S'), 1.0)  # 1 hour max age
    assert result, "Fresh timestamp should be valid"
    
    # Test statistics calculation
    stats = calculate_statistics([1, 2, 3, 4, 5])
    assert stats['mean'] == 3.0, "Mean should be 3.0"
    assert stats['count'] == 5, "Count should be 5"
    
    # Test duration formatting
    formatted = format_duration(3661)  # 1 hour, 1 minute, 1 second
    assert "1h 1m 1s" in formatted, "Duration should format correctly"
    
    print("✓ Common utilities integration test passed")


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
    assert results['total_tests'] == 2, "Should have 2 total tests"
    assert results['passed'] == 1, "Should have 1 passed test"
    assert results['failed'] == 1, "Should have 1 failed test"
    assert results['execution_time'] >= 0, "Execution time should be non-negative"
    
    print("✓ Optimization system integration test passed")


def test_discovery_system_integration():
    """Test that the refined discovery system works correctly."""
    # Create a temporary test file to discover
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test_sample.py"
    
    test_content = '''
def test_sample_function():
    """Sample test function."""
    assert 1 + 1 == 2
    
def benchmark_sample_function():
    """Sample benchmark function."""
    import time
    start = time.time()
    time.sleep(0.01)  # Simulate some work
    end = time.time()
    return end - start
'''
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        discovery = UnifiedTestDiscovery(search_paths=[str(temp_dir)])
        items = discovery.discover_all()
        
        # Should find our test and benchmark functions
        test_found = any(item['name'] == 'test_sample_function' for item in items)
        benchmark_found = any(item['name'] == 'benchmark_sample_function' for item in items)
        
        assert test_found, "Should discover test function"
        assert benchmark_found, "Should discover benchmark function"
        
        print("✓ Discovery system integration test passed")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


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
        assert len(tracker.alerts) == 0, "Should have no alerts for minor degradation"
        
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
        assert len(tracker.alerts) == 1, "Should have 1 alert for significant degradation"
        
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
        assert key in new_tracker.metrics_history, "Should have loaded metrics history"
        assert len(new_tracker.metrics_history[key]) == 3, "Should have 3 metrics"
        
        print("✓ Performance regression integration test passed")
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


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
        
        assert test_found, "Should discover test function"
        assert benchmark_found, "Should discover benchmark function"
        
        # Test running discovered items
        if items:
            # Just verify we can access the functions without error
            for item in items:
                assert callable(item['function']), f"Function {item['name']} should be callable"
        
        # Test optimization with discovered functions
        if items:
            test_funcs = [item['function'] for item in items if 'test_' in item['name']]
            if test_funcs:
                opt_results = run_tests_with_optimization(test_funcs)
                assert 'total_tests' in opt_results, "Optimization should return results dict"
        
        print("✓ System interoperability test passed")
        
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
    
    assert results['failed'] == 1, "Should report failed test"
    assert results['passed'] == 0, "Should not report passed test"
    
    print("✓ Error handling test passed")


if __name__ == "__main__":
    print("Running simplified integration tests for refined test and benchmark systems...")
    
    # Run all integration tests
    test_common_utilities_integration()
    test_optimization_system_integration()
    test_discovery_system_integration()
    test_performance_regression_integration()
    test_system_interoperability()
    test_error_handling()
    
    print("\n✓ All integration tests passed!")
    print("The refined systems work together properly and maintain backward compatibility.")