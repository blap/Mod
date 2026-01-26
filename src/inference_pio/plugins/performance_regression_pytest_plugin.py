"""
Pytest Plugin for Performance Regression Testing

This plugin integrates performance regression testing with pytest,
allowing performance tests to run alongside unit and integration tests.
"""

import pytest
import time
from typing import Dict, Any, Optional
from pathlib import Path

from src.inference_pio.common.performance_regression_tracker import (
    PerformanceRegressionTracker,
    record_performance_metric,
    get_regression_alerts,
    save_regression_data
)


def pytest_addoption(parser):
    """Add command-line options for performance regression testing."""
    group = parser.getgroup("performance-regression")
    group.addoption(
        "--performance-regression",
        action="store_true",
        help="Enable performance regression testing"
    )
    group.addoption(
        "--perf-threshold",
        type=float,
        default=5.0,
        help="Performance regression threshold percentage (default: 5.0)"
    )
    group.addoption(
        "--perf-storage-dir",
        default="performance_history",
        help="Directory to store performance history (default: performance_history)"
    )
    group.addoption(
        "--perf-fail-on-regression",
        action="store_true",
        help="Fail tests on performance regression detection"
    )


@pytest.fixture(scope="session")
def performance_tracker(request):
    """Session-scoped fixture for the performance regression tracker."""
    threshold = request.config.getoption("--perf-threshold")
    storage_dir = request.config.getoption("--perf-storage-dir")
    
    tracker = PerformanceRegressionTracker(
        storage_dir=storage_dir,
        regression_threshold=threshold
    )
    
    yield tracker
    
    # Save history at the end of the session
    tracker.save_history()


@pytest.fixture
def perf_test_case(performance_tracker):
    """Fixture to provide performance testing utilities to test functions."""
    class PerfTestCase:
        def __init__(self, tracker):
            self.tracker = tracker
        
        def record_metric(self, name: str, value: float, unit: str, 
                         model_name: str, category: str, 
                         metadata: Optional[Dict[str, Any]] = None):
            """Record a performance metric."""
            record_performance_metric(name, value, unit, model_name, category, metadata)
        
        def measure_function_performance(self, func, *args, **kwargs):
            """Measure the performance of a function."""
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Record the timing
            self.record_metric(
                name=f"{func.__name__}_execution_time",
                value=execution_time,
                unit="seconds",
                model_name="general",
                category="performance",
                metadata={"function": func.__name__, "args_count": len(args)}
            )
            
            return result, execution_time
    
    return PerfTestCase(performance_tracker)


def pytest_configure(config):
    """Configure the plugin."""
    if config.getoption("--performance-regression"):
        # Register markers
        config.addinivalue_line(
            "markers", 
            "performance: mark test as a performance test"
        )
        config.addinivalue_line(
            "markers", 
            "regression: mark test as a regression test"
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Handle test setup for performance regression tests."""
    if item.config.getoption("--performance-regression"):
        # Apply performance marker to tests that use perf_test_case fixture
        if "perf_test_case" in item.fixturenames:
            item.add_marker(pytest.mark.performance)


@pytest.hookimpl(trylast=True)
def pytest_runtest_makereport(item, call):
    """Handle test reporting for performance regression tests."""
    if call.when == "teardown" and item.config.getoption("--performance-regression"):
        # Check for regression alerts after each test
        alerts = get_regression_alerts()
        
        if alerts and item.config.getoption("--perf-fail-on-regression"):
            # Get alerts related to this test session
            # In a real implementation, we'd filter by test-specific metrics
            if alerts:
                # Fail the test if there are regression alerts
                item.add_report_section("call", "performance-regression", 
                                      "Performance regression detected")
                
                # Log the alerts
                for alert in alerts[-3:]:  # Last 3 alerts
                    item.add_report_section("call", "regression-alert", alert.message)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add performance regression summary to terminal output."""
    if config.getoption("--performance-regression"):
        alerts = get_regression_alerts()
        
        if alerts:
            terminalreporter.write_sep("=", "PERFORMANCE REGRESSION ALERTS")
            
            critical_alerts = [a for a in alerts if a.severity.value == "critical"]
            warning_alerts = [a for a in alerts if a.severity.value == "warning"]
            
            if critical_alerts:
                terminalreporter.write_line(f"CRITICAL REGRESSIONS: {len(critical_alerts)}", red=True)
                for alert in critical_alerts[-5:]:  # Show last 5
                    terminalreporter.write_line(f"  {alert.message}", red=True)
            
            if warning_alerts:
                terminalreporter.write_line(f"WARNING REGRESSIONS: {len(warning_alerts)}", yellow=True)
                for alert in warning_alerts[-5:]:  # Show last 5
                    terminalreporter.write_line(f"  {alert.message}", yellow=True)
            
            terminalreporter.write_line(f"TOTAL REGRESSIONS: {len(alerts)}")
        else:
            terminalreporter.write_sep("=", "NO PERFORMANCE REGRESSIONS DETECTED", green=True)


# Example test functions that could use this plugin
def example_performance_test(perf_test_case):
    """Example of how to write a performance test using the plugin."""
    import time
    
    def sample_function():
        # Simulate some work
        time.sleep(0.1)
        return "result"
    
    # Measure performance
    result, execution_time = perf_test_case.measure_function_performance(sample_function)
    
    # Verify result
    assert result == "result"
    
    # The execution time is automatically recorded
    print(f"Function executed in {execution_time:.3f}s")


def example_model_performance_test(perf_test_case):
    """Example of testing model performance."""
    # This would typically load a model and test its performance
    # For demonstration, we'll simulate a model inference
    import random
    
    def model_inference(input_size):
        # Simulate model inference time based on input size
        time.sleep(0.01 * input_size)  # 10ms per input token
        return {"output": "generated text", "tokens": input_size}
    
    # Test with different input sizes
    for input_size in [10, 50, 100]:
        result, execution_time = perf_test_case.measure_function_performance(
            model_inference, input_size
        )
        
        # Record tokens per second
        tokens_per_sec = input_size / execution_time if execution_time > 0 else float('inf')
        perf_test_case.record_metric(
            name=f"model_throughput_{input_size}tokens",
            value=tokens_per_sec,
            unit="tokens/sec",
            model_name="example_model",
            category="performance",
            metadata={"input_size": input_size}
        )