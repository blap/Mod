"""
Updated Integration Tests for Performance Regression System

This module tests the performance regression tracking system to ensure
it properly detects and reports performance regressions with enhanced validation.
"""

import tempfile
import shutil
from pathlib import Path
import time

from tests.utils.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, 
    assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, 
    assert_is_instance, assert_raises, run_tests, assert_length, assert_dict_contains,
    assert_list_contains, assert_is_subclass, assert_has_attr, assert_callable,
    assert_iterable, assert_not_is_instance, assert_between, assert_greater_equal,
    assert_less_equal, assert_is, assert_is_not, assert_almost_equal
)

from src.inference_pio.common.performance_regression_tracker import (
    PerformanceRegressionTracker,
    PerformanceMetric,
    RegressionAlert,
    RegressionSeverity
)


def test_basic_metric_tracking():
    """Test basic metric tracking functionality."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add a few metrics
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )

        tracker.add_metric(metric1)

        # Verify the metric was added
        assert_in("test_model:inference_speed", tracker.metrics_history)
        assert_equal(len(tracker.metrics_history["test_model:inference_speed"]), 1)
        assert_equal(tracker.metrics_history["test_model:inference_speed"][0].value, 100.0)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_regression_detection_performance_metric():
    """Test regression detection for performance metrics (higher is better)."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add initial high-performance metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,  # Good performance: 100 tokens/sec
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)

        # Add degraded performance metric (>5% worse)
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=94.0,  # Degraded: ~6% worse
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric2)

        # Verify regression was detected
        assert_equal(len(tracker.alerts), 1)
        alert = tracker.alerts[0]
        assert_equal(alert.severity, RegressionSeverity.WARNING)
        assert_in("Performance regression detected", alert.message)
        assert_almost_equal(alert.previous_value, 100.0, places=1)
        assert_almost_equal(alert.current_value, 94.0, places=1)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_regression_detection_time_metric():
    """Test regression detection for time metrics (lower is better)."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add initial low-time metric
        metric1 = PerformanceMetric(
            name="inference_time",
            value=0.1,  # Fast: 0.1 seconds
            unit="seconds",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)

        # Add degraded time metric (>5% worse)
        metric2 = PerformanceMetric(
            name="inference_time",
            value=0.11,  # Slower: 10% worse
            unit="seconds",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric2)

        # Verify regression was detected
        assert_equal(len(tracker.alerts), 1)
        alert = tracker.alerts[0]
        assert_equal(alert.severity, RegressionSeverity.WARNING)
        assert_in("Performance regression detected", alert.message)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_no_regression_below_threshold():
    """Test that regressions below threshold are not detected."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add initial metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)

        # Add slightly degraded metric (<5% worse)
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=98.0,  # Only 2% worse, below threshold
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric2)

        # Verify no regression was detected
        assert_equal(len(tracker.alerts), 0)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_improvement_not_flagged_as_regression():
    """Test that performance improvements are not flagged as regressions."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add initial metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)

        # Add improved metric
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=110.0,  # Better performance
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric2)

        # Verify no regression was detected
        assert_equal(len(tracker.alerts), 0)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_critical_regression_detection():
    """Test detection of critical regressions (>2x threshold)."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add initial metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)

        # Add severely degraded metric (>10% worse, which is 2x the 5% threshold)
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=85.0,  # 15% worse, >2x threshold
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric2)

        # Verify critical regression was detected
        assert_equal(len(tracker.alerts), 1)
        alert = tracker.alerts[0]
        assert_equal(alert.severity, RegressionSeverity.CRITICAL)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_memory_regression_detection():
    """Test regression detection for memory usage (lower is better)."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add initial low memory usage
        metric1 = PerformanceMetric(
            name="memory_usage",
            value=500.0,  # 500 MB
            unit="MB",
            timestamp=time.time(),
            model_name="test_model",
            category="memory"
        )
        tracker.add_metric(metric1)

        # Add increased memory usage (>5% worse)
        metric2 = PerformanceMetric(
            name="memory_usage",
            value=550.0,  # 10% increase, >5% threshold
            unit="MB",
            timestamp=time.time(),
            model_name="test_model",
            category="memory"
        )
        tracker.add_metric(metric2)

        # Verify regression was detected
        assert_equal(len(tracker.alerts), 1)
        alert = tracker.alerts[0]
        assert_equal(alert.severity, RegressionSeverity.WARNING)
        assert_in("Performance regression detected", alert.message)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_historical_stats_calculation():
    """Test calculation of historical statistics."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add several metrics
        for i, value in enumerate([100, 105, 95, 110, 90]):
            metric = PerformanceMetric(
                name="inference_speed",
                value=value,
                unit="tokens/sec",
                timestamp=time.time() + i,
                model_name="test_model",
                category="performance"
            )
            tracker.add_metric(metric)

        # Get historical stats
        stats = tracker.get_historical_stats("test_model:inference_speed")

        # Verify stats were calculated
        assert_in('mean', stats)
        assert_in('median', stats)
        assert_in('stdev', stats)
        assert_in('min', stats)
        assert_in('max', stats)
        assert_in('count', stats)
        assert_in('latest', stats)

        # Verify specific values
        assert_almost_equal(stats['mean'], 100.0, places=1)
        assert_equal(stats['min'], 90)
        assert_equal(stats['max'], 110)
        assert_equal(stats['count'], 5)
        assert_equal(stats['latest'], 90)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_save_and_load_history():
    """Test saving and loading of historical data."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add some metrics
        for i, value in enumerate([100, 105, 95]):
            metric = PerformanceMetric(
                name="inference_speed",
                value=value,
                unit="tokens/sec",
                timestamp=time.time() + i,
                model_name="test_model",
                category="performance"
            )
            tracker.add_metric(metric)

        # Save history
        tracker.save_history()

        # Create new tracker and load history
        new_tracker = PerformanceRegressionTracker(
            storage_dir=temp_dir,
            regression_threshold=5.0
        )

        # Verify history was loaded
        assert_in("test_model:inference_speed", new_tracker.metrics_history)
        assert_equal(len(new_tracker.metrics_history["test_model:inference_speed"]), 3)
        assert_equal(new_tracker.metrics_history["test_model:inference_speed"][0].value, 100)
        assert_equal(new_tracker.metrics_history["test_model:inference_speed"][2].value, 95)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_report_generation():
    """Test generation of performance reports."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add some metrics and alerts
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)

        # Generate report
        report_path = tracker.generate_report(output_dir=temp_dir / "reports")

        # Verify report was created
        assert_true(Path(report_path).exists())

        # Read and verify content
        with open(report_path, 'r') as f:
            content = f.read()

        assert_in("# Performance Report", content)
        assert_in("test_model:inference_speed", content)
        assert_in("Total metrics tracked: 1", content)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_csv_export():
    """Test CSV export functionality."""
    # Create a temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )

    try:
        # Add some metrics
        for i, value in enumerate([100, 105, 95]):
            metric = PerformanceMetric(
                name="inference_speed",
                value=value,
                unit="tokens/sec",
                timestamp=time.time() + i,
                model_name="test_model",
                category="performance"
            )
            tracker.add_metric(metric)

        # Export to CSV
        csv_path = tracker.export_csv(output_dir=temp_dir / "exports")

        # Verify CSV was created
        assert_true(Path(csv_path).exists())

        # Read and verify content
        with open(csv_path, 'r') as f:
            content = f.read()

        assert_in("Model Name,Metric Name,Value,Unit,Category,Timestamp,Date Time", content)
        assert_in("test_model,inference_speed,95.0,tokens/sec,performance,", content)
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_global_tracker_access():
    """Test access to the global tracker."""
    from src.inference_pio.common.performance_regression_tracker import (
        get_default_tracker,
        record_performance_metric,
        get_regression_alerts,
        save_regression_data
    )

    # Get the default tracker
    tracker = get_default_tracker()

    # Verify it's accessible and functional
    assert_is_instance(tracker, PerformanceRegressionTracker)

    # Record a metric using the global function
    record_performance_metric(
        name="test_metric",
        value=42.0,
        unit="units",
        model_name="test_model",
        category="test"
    )

    # Verify the metric was recorded
    alerts = get_regression_alerts()
    # No regression should be detected with just one metric
    assert_equal(len(alerts), 0)

    # Save the data
    save_regression_data()


def test_tracker_initialization_with_different_thresholds():
    """Test tracker initialization with different regression thresholds."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test with low threshold
        tracker_low = PerformanceRegressionTracker(
            storage_dir=temp_dir,
            regression_threshold=1.0  # 1% threshold
        )
        assert_equal(tracker_low.regression_threshold, 1.0)

        # Test with high threshold
        tracker_high = PerformanceRegressionTracker(
            storage_dir=temp_dir,
            regression_threshold=10.0  # 10% threshold
        )
        assert_equal(tracker_high.regression_threshold, 10.0)
    finally:
        shutil.rmtree(temp_dir)


def test_metric_creation_with_all_fields():
    """Test creating metrics with all possible fields."""
    timestamp = time.time()
    metric = PerformanceMetric(
        name="comprehensive_test",
        value=123.45,
        unit="custom_unit",
        timestamp=timestamp,
        model_name="test_model",
        category="custom_category",
        description="A comprehensive test metric",
        tags=["tag1", "tag2"]
    )
    
    assert_equal(metric.name, "comprehensive_test")
    assert_equal(metric.value, 123.45)
    assert_equal(metric.unit, "custom_unit")
    assert_equal(metric.timestamp, timestamp)
    assert_equal(metric.model_name, "test_model")
    assert_equal(metric.category, "custom_category")
    assert_equal(metric.description, "A comprehensive test metric")
    assert_equal(metric.tags, ["tag1", "tag2"])


def test_tracker_add_multiple_metrics_same_name():
    """Test adding multiple metrics with the same name."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0
    )
    
    try:
        # Add multiple metrics with the same name
        for i in range(5):
            metric = PerformanceMetric(
                name="test_metric",
                value=100 + i,
                unit="units",
                timestamp=time.time(),
                model_name="test_model",
                category="performance"
            )
            tracker.add_metric(metric)
        
        # Should have 5 metrics stored
        assert_equal(len(tracker.metrics_history["test_model:test_metric"]), 5)
    finally:
        shutil.rmtree(temp_dir)


def test_alert_message_content():
    """Test that alert messages contain appropriate information."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0
    )
    
    try:
        # Create a regression scenario
        metric1 = PerformanceMetric(
            name="regression_test",
            value=100.0,
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)
        
        metric2 = PerformanceMetric(
            name="regression_test",
            value=90.0,  # 10% degradation
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric2)
        
        # Check alert message content
        assert_equal(len(tracker.alerts), 1)
        alert = tracker.alerts[0]
        assert_in("regression_test", alert.message)
        assert_in("test_model", alert.message)
        assert_in("10.0%", alert.message)  # Approximate percentage
    finally:
        shutil.rmtree(temp_dir)


def test_tracker_clear_metrics():
    """Test clearing all metrics from the tracker."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0
    )
    
    try:
        # Add some metrics
        metric = PerformanceMetric(
            name="clear_test",
            value=100.0,
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric)
        
        # Verify metric was added
        assert_in("test_model:clear_test", tracker.metrics_history)
        
        # Clear metrics
        tracker.clear_metrics()
        
        # Verify metrics were cleared
        assert_equal(len(tracker.metrics_history), 0)
    finally:
        shutil.rmtree(temp_dir)


def test_tracker_get_metrics_by_model():
    """Test getting metrics filtered by model."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0
    )
    
    try:
        # Add metrics for different models
        metric1 = PerformanceMetric(
            name="metric1",
            value=100.0,
            unit="units",
            timestamp=time.time(),
            model_name="model_a",
            category="performance"
        )
        tracker.add_metric(metric1)
        
        metric2 = PerformanceMetric(
            name="metric2",
            value=200.0,
            unit="units",
            timestamp=time.time(),
            model_name="model_b",
            category="performance"
        )
        tracker.add_metric(metric2)
        
        # Get metrics for model_a
        model_a_metrics = tracker.get_metrics_by_model("model_a")
        assert_equal(len(model_a_metrics), 1)
        assert_equal(model_a_metrics[0].model_name, "model_a")
        
        # Get metrics for model_b
        model_b_metrics = tracker.get_metrics_by_model("model_b")
        assert_equal(len(model_b_metrics), 1)
        assert_equal(model_b_metrics[0].model_name, "model_b")
    finally:
        shutil.rmtree(temp_dir)


def test_tracker_get_metrics_by_category():
    """Test getting metrics filtered by category."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0
    )
    
    try:
        # Add metrics for different categories
        metric1 = PerformanceMetric(
            name="perf_metric",
            value=100.0,
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric1)
        
        metric2 = PerformanceMetric(
            name="mem_metric",
            value=200.0,
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="memory"
        )
        tracker.add_metric(metric2)
        
        # Get performance metrics
        perf_metrics = tracker.get_metrics_by_category("performance")
        assert_equal(len(perf_metrics), 1)
        assert_equal(perf_metrics[0].category, "performance")
        
        # Get memory metrics
        mem_metrics = tracker.get_metrics_by_category("memory")
        assert_equal(len(mem_metrics), 1)
        assert_equal(mem_metrics[0].category, "memory")
    finally:
        shutil.rmtree(temp_dir)


def test_regression_severity_levels():
    """Test different regression severity levels."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0  # 5% threshold
    )
    
    try:
        # Add baseline metric
        baseline = PerformanceMetric(
            name="severity_test",
            value=100.0,
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(baseline)
        
        # Test warning level (just above threshold)
        warning_metric = PerformanceMetric(
            name="severity_test",
            value=94.0,  # 6% degradation (just above 5% threshold)
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(warning_metric)
        
        # Should have a warning
        assert_equal(len(tracker.alerts), 1)
        assert_equal(tracker.alerts[0].severity, RegressionSeverity.WARNING)
        
        # Clear alerts for next test
        tracker.alerts.clear()
        
        # Add another baseline
        tracker.add_metric(baseline)
        
        # Test critical level (well above threshold)
        critical_metric = PerformanceMetric(
            name="severity_test",
            value=80.0,  # 20% degradation (well above 5% threshold)
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(critical_metric)
        
        # Should have a critical alert
        assert_equal(len(tracker.alerts), 1)
        assert_equal(tracker.alerts[0].severity, RegressionSeverity.CRITICAL)
    finally:
        shutil.rmtree(temp_dir)


def test_tracker_storage_directory_creation():
    """Test that storage directory is created if it doesn't exist."""
    temp_base = Path(tempfile.mkdtemp())
    storage_dir = temp_base / "nonexistent" / "storage"
    
    try:
        # Create tracker with non-existent directory
        tracker = PerformanceRegressionTracker(
            storage_dir=storage_dir,
            regression_threshold=5.0
        )
        
        # Directory should be created
        assert_true(storage_dir.exists())
        
        # Add a metric to trigger save
        metric = PerformanceMetric(
            name="storage_test",
            value=100.0,
            unit="units",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        tracker.add_metric(metric)
        
        # Save history
        tracker.save_history()
        
        # Verify data files were created
        assert_true((storage_dir / "metrics_history.json").exists())
    finally:
        shutil.rmtree(temp_base)


def test_metric_equality_comparison():
    """Test equality comparison between metrics."""
    timestamp = time.time()
    
    metric1 = PerformanceMetric(
        name="equality_test",
        value=100.0,
        unit="units",
        timestamp=timestamp,
        model_name="test_model",
        category="performance"
    )
    
    metric2 = PerformanceMetric(
        name="equality_test",
        value=100.0,
        unit="units",
        timestamp=timestamp,
        model_name="test_model",
        category="performance"
    )
    
    # Metrics with same properties should be considered equal
    assert_equal(metric1.name, metric2.name)
    assert_equal(metric1.value, metric2.value)
    assert_equal(metric1.unit, metric2.unit)
    assert_equal(metric1.timestamp, metric2.timestamp)
    assert_equal(metric1.model_name, metric2.model_name)
    assert_equal(metric1.category, metric2.category)


def test_tracker_multiple_model_support():
    """Test tracker support for multiple models."""
    temp_dir = Path(tempfile.mkdtemp())
    tracker = PerformanceRegressionTracker(
        storage_dir=temp_dir,
        regression_threshold=5.0
    )
    
    try:
        # Add metrics for multiple models
        models = ["model_a", "model_b", "model_c"]
        for i, model in enumerate(models):
            metric = PerformanceMetric(
                name=f"metric_{i}",
                value=100 + i * 10,
                unit="units",
                timestamp=time.time(),
                model_name=model,
                category="performance"
            )
            tracker.add_metric(metric)
        
        # Verify all models are represented
        all_metrics = []
        for key, metrics in tracker.metrics_history.items():
            all_metrics.extend(metrics)
        
        assert_equal(len(all_metrics), 3)
        
        # Check that each model has its own metric
        model_names = [m.model_name for m in all_metrics]
        for model in models:
            assert_in(model, model_names)
    finally:
        shutil.rmtree(temp_dir)


def run_performance_regression_tests():
    """Run all performance regression tests."""
    test_functions = [
        test_basic_metric_tracking,
        test_regression_detection_performance_metric,
        test_regression_detection_time_metric,
        test_no_regression_below_threshold,
        test_improvement_not_flagged_as_regression,
        test_critical_regression_detection,
        test_memory_regression_detection,
        test_historical_stats_calculation,
        test_save_and_load_history,
        test_report_generation,
        test_csv_export,
        test_global_tracker_access,
        test_tracker_initialization_with_different_thresholds,
        test_metric_creation_with_all_fields,
        test_tracker_add_multiple_metrics_same_name,
        test_alert_message_content,
        test_tracker_clear_metrics,
        test_tracker_get_metrics_by_model,
        test_tracker_get_metrics_by_category,
        test_regression_severity_levels,
        test_tracker_storage_directory_creation,
        test_metric_equality_comparison,
        test_tracker_multiple_model_support
    ]

    print("Running updated performance regression tests...")
    success = run_tests(test_functions)
    return success


if __name__ == "__main__":
    run_performance_regression_tests()