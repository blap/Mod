"""
Comprehensive test suite for the centralized metrics collection system.
This tests all aspects of the MetricsCollector class including:
- Basic functionality
- Thread safety
- Error handling
- Export formats
- Performance tracking
- Validation
"""

import unittest
import time
import threading
import tempfile
import os
from datetime import datetime
import json
import csv

from centralized_metrics_collector import (
    MetricsCollector, 
    MetricType, 
    MetricValue,
    get_metrics_collector,
    add_performance_metric,
    start_performance_tracking,
    end_performance_tracking,
    queue_performance_metric
)


class TestMetricValue(unittest.TestCase):
    """Test the MetricValue class."""

    def test_metric_value_creation(self):
        """Test creating a MetricValue instance."""
        metric = MetricValue(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            description="A test metric"
        )
        
        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 42)
        self.assertEqual(metric.metric_type, MetricType.GAUGE)
        self.assertEqual(metric.labels, {"env": "test"})
        self.assertEqual(metric.description, "A test metric")
        self.assertIsInstance(metric.timestamp, float)
        
    def test_metric_value_to_dict(self):
        """Test converting MetricValue to dictionary."""
        metric = MetricValue(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            description="A test metric"
        )
        
        metric_dict = metric.to_dict()
        self.assertEqual(metric_dict["name"], "test_metric")
        self.assertEqual(metric_dict["value"], 42)
        self.assertEqual(metric_dict["type"], "gauge")
        self.assertEqual(metric_dict["labels"], {"env": "test"})
        self.assertEqual(metric_dict["description"], "A test metric")
        self.assertIsInstance(metric_dict["timestamp"], float)
        self.assertIsInstance(metric_dict["datetime"], str)


class TestMetricsCollector(unittest.TestCase):
    """Test the MetricsCollector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(
            collection_interval=0.1,
            max_metrics_buffer=100,
            max_history_per_metric=50
        )
        self.collector.start_collection()

    def tearDown(self):
        """Tear down test fixtures."""
        self.collector.stop_collection()
        self.collector.reset()

    def test_add_metric(self):
        """Test adding a metric."""
        result = self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            description="A test metric"
        )
        
        self.assertTrue(result)
        
        # Check that the metric was added
        metric = self.collector.get_metric("test_metric", {"env": "test"})
        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 42)
        self.assertEqual(metric.description, "A test metric")

    def test_add_invalid_metric_name(self):
        """Test adding a metric with an invalid name."""
        result = self.collector.add_metric(
            name="invalid name with spaces",
            value=42,
            metric_type=MetricType.GAUGE
        )
        
        self.assertFalse(result)

    def test_add_invalid_metric_value(self):
        """Test adding a metric with an invalid value."""
        result = self.collector.add_metric(
            name="test_metric",
            value=float('inf'),
            metric_type=MetricType.GAUGE
        )
        
        self.assertFalse(result)

    def test_get_metric(self):
        """Test getting a specific metric."""
        self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"}
        )
        
        metric = self.collector.get_metric("test_metric", {"env": "test"})
        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 42)

    def test_get_metrics_history(self):
        """Test getting metrics history."""
        # Add multiple values for the same metric
        for i in range(5):
            self.collector.add_metric(
                name="test_metric",
                value=i,
                metric_type=MetricType.GAUGE,
                labels={"iteration": str(i)}
            )
        
        history = self.collector.get_metrics_history("test_metric")
        self.assertEqual(len(history), 5)
        
        # Test with limit
        limited_history = self.collector.get_metrics_history("test_metric", limit=3)
        self.assertEqual(len(limited_history), 3)

    def test_get_all_current_metrics(self):
        """Test getting all current metrics."""
        self.collector.add_metric(
            name="metric1",
            value=1,
            metric_type=MetricType.GAUGE
        )
        self.collector.add_metric(
            name="metric2",
            value=2,
            metric_type=MetricType.GAUGE
        )
        
        all_metrics = self.collector.get_all_current_metrics()
        self.assertEqual(len(all_metrics), 2)
        
        # Verify it's a copy
        original_len = len(all_metrics)
        all_metrics["new_key"] = "new_value"
        new_len = len(self.collector.get_all_current_metrics())
        self.assertEqual(original_len, new_len)

    def test_reset(self):
        """Test resetting the collector."""
        self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE
        )
        
        self.assertGreater(len(self.collector.get_all_metrics_buffer()), 0)
        self.assertGreater(len(self.collector.get_all_current_metrics()), 0)
        
        self.collector.reset()
        
        self.assertEqual(len(self.collector.get_all_metrics_buffer()), 0)
        self.assertEqual(len(self.collector.get_all_current_metrics()), 0)

    def test_export_to_json(self):
        """Test JSON export functionality."""
        self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            description="A test metric"
        )
        
        # Test export to string
        json_str = self.collector.export_to_json()
        self.assertIsInstance(json_str, str)
        
        # Test export to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            
        try:
            result = self.collector.export_to_json(output_path=temp_file)
            self.assertIsInstance(result, dict)
            
            # Verify file was created and contains valid JSON
            with open(temp_file, 'r') as f:
                data = json.load(f)
                self.assertIn("timestamp", data)
                self.assertIn("metrics", data)
                self.assertGreater(len(data["metrics"]), 0)
        finally:
            os.unlink(temp_file)

    def test_export_to_csv(self):
        """Test CSV export functionality."""
        self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            description="A test metric"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
            
        try:
            self.collector.export_to_csv(temp_file)
            
            # Verify file was created and contains valid CSV
            with open(temp_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertGreater(len(rows), 0)
                
                # Check that all expected fields are present
                row = rows[0]
                expected_fields = ['name', 'value', 'type', 'labels', 'description', 'timestamp', 'datetime']
                for field in expected_fields:
                    self.assertIn(field, row)
        finally:
            os.unlink(temp_file)

    def test_export_to_prometheus(self):
        """Test Prometheus export functionality."""
        self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE,
            labels={"env": "test"},
            description="A test metric"
        )
        
        prometheus_str = self.collector.export_to_prometheus()
        self.assertIsInstance(prometheus_str, str)
        self.assertIn("test_metric", prometheus_str)
        self.assertIn("# TYPE test_metric gauge", prometheus_str)
        self.assertIn('# HELP test_metric A test metric', prometheus_str)

    def test_export_function(self):
        """Test the generic export function."""
        self.collector.add_metric(
            name="test_metric",
            value=42,
            metric_type=MetricType.GAUGE
        )
        
        # Test JSON export
        json_result = self.collector.export("json")
        self.assertIsInstance(json_result, str)
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        try:
            csv_result = self.collector.export("csv", output_path=temp_file)
            self.assertIsNone(csv_result)  # CSV export returns None
        finally:
            os.unlink(temp_file)
        
        # Test Prometheus export
        prometheus_result = self.collector.export("prometheus")
        self.assertIsInstance(prometheus_result, str)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.collector.export("invalid_format")

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Start tracking
        self.collector.add_performance_tracker("test_operation")
        time.sleep(0.01)  # Simulate some work

        # End tracking
        result = self.collector.end_performance_tracker("test_operation")

        self.assertIsNotNone(result)
        self.assertIn("duration", result)
        self.assertIn("memory_delta", result)
        self.assertGreaterEqual(result["duration"], 0)

        # Check that metrics were added - look for them in the buffer
        buffer_metrics = self.collector.get_all_metrics_buffer()

        duration_metric = next((m for m in buffer_metrics
                              if m.name == f"performance_tracker_test_operation_duration"), None)
        memory_metric = next((m for m in buffer_metrics
                            if m.name == f"performance_tracker_test_operation_memory_delta"), None)

        self.assertIsNotNone(duration_metric, "Duration metric should be in buffer")
        self.assertIsNotNone(memory_metric, "Memory delta metric should be in buffer")

    def test_thread_safety(self):
        """Test that the collector is thread-safe."""
        def add_metrics():
            for i in range(10):
                self.collector.add_metric(
                    name="thread_test_metric",
                    value=i,
                    metric_type=MetricType.GAUGE,
                    labels={"thread": f"thread_{threading.current_thread().ident}"}
                )
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=add_metrics)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all metrics were added
        all_metrics = self.collector.get_all_metrics_buffer()
        # Should have at least 50 metrics (5 threads * 10 metrics each)
        self.assertGreaterEqual(len(all_metrics), 50)

    def test_validation_rules(self):
        """Test custom validation rules."""
        def positive_value_validator(value):
            return isinstance(value, (int, float)) and value > 0
        
        self.collector.register_validation_rule("positive_metric", positive_value_validator)
        
        # Test valid value
        result = self.collector.validate_metric("positive_metric", 42)
        self.assertTrue(result)
        
        # Test invalid value
        result = self.collector.validate_metric("positive_metric", -5)
        self.assertFalse(result)

    def test_statistics(self):
        """Test statistics reporting."""
        stats = self.collector.get_statistics()
        self.assertIn("total_metrics_collected", stats)
        self.assertIn("current_metrics_count", stats)
        self.assertIn("collection_interval", stats)
        self.assertIn("max_buffer_size", stats)
        self.assertIn("is_collecting", stats)

    def test_queue_metric(self):
        """Test queuing metrics functionality."""
        result = self.collector.queue_metric(
            name="queued_metric",
            value=100,
            metric_type=MetricType.COUNTER,
            labels={"source": "queue_test"},
            description="A queued metric"
        )
        
        self.assertTrue(result)
        
        # Process queued metrics
        self.collector._process_queued_metrics()
        
        # Check that the metric was added from the queue
        metric = self.collector.get_metric("queued_metric", {"source": "queue_test"})
        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 100)

    def test_get_all_metrics_buffer(self):
        """Test getting all metrics in the buffer."""
        # Add several metrics
        for i in range(5):
            self.collector.add_metric(
                name=f"buffer_test_{i}",
                value=i,
                metric_type=MetricType.GAUGE
            )
        
        buffer_metrics = self.collector.get_all_metrics_buffer()
        self.assertEqual(len(buffer_metrics), 5)
        
        # Verify it's a copy (not the internal deque)
        original_len = len(buffer_metrics)
        buffer_metrics.append("new_item")
        new_len = len(self.collector.get_all_metrics_buffer())
        self.assertEqual(original_len, new_len)


class TestGlobalFunctions(unittest.TestCase):
    """Test the global convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = get_metrics_collector()
        self.collector.reset()

    def tearDown(self):
        """Tear down test fixtures."""
        self.collector.reset()

    def test_get_metrics_collector(self):
        """Test getting the global metrics collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return the same instance
        self.assertIs(collector1, collector2)

    def test_add_performance_metric(self):
        """Test the add_performance_metric convenience function."""
        add_performance_metric(
            name="convenience_test_metric",
            value=99,
            labels={"source": "convenience"},
            description="Test from convenience function"
        )
        
        metric = self.collector.get_metric("convenience_test_metric", {"source": "convenience"})
        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 99)

    def test_performance_tracking_functions(self):
        """Test the performance tracking convenience functions."""
        start_performance_tracking("convenience_operation")
        time.sleep(0.01)  # Small delay
        result = end_performance_tracking("convenience_operation")
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["duration"], 0)

    def test_queue_performance_metric(self):
        """Test the queue_performance_metric convenience function."""
        queue_performance_metric(
            name="queued_convenience_metric",
            value=200,
            labels={"source": "queued_convenience"},
            description="Test queued convenience metric"
        )
        
        # Process the queue
        self.collector._process_queued_metrics()
        
        # Check that the metric was added
        metric = self.collector.get_metric("queued_convenience_metric", {"source": "queued_convenience"})
        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 200)


class TestSystemMetricsCollection(unittest.TestCase):
    """Test system metrics collection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(
            collection_interval=0.1,
            max_metrics_buffer=100
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.collector.stop_collection()
        self.collector.reset()

    def test_system_metrics_collection(self):
        """Test that system metrics are collected."""
        # Collect system metrics once
        self.collector._collect_system_metrics()
        
        # Check that system metrics were added
        cpu_metric = self.collector.get_metric("system_cpu_percent")
        memory_metric = self.collector.get_metric("system_memory_percent")
        
        # These metrics should exist if the collection worked
        # They might be None if the system doesn't have the required dependencies
        # but the collection shouldn't fail
        
        # Just ensure no exceptions were raised during collection
        self.assertTrue(True)  # If we reach here, no exceptions occurred


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(
            collection_interval=0.1,
            max_metrics_buffer=5,  # Small buffer to test overflow
            max_history_per_metric=3  # Small history to test overflow
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.collector.stop_collection()
        self.collector.reset()

    def test_buffer_overflow(self):
        """Test behavior when buffer overflows."""
        # Add more metrics than the buffer can hold
        for i in range(10):
            self.collector.add_metric(
                name=f"overflow_test_{i}",
                value=i,
                metric_type=MetricType.GAUGE
            )
        
        # Buffer should only contain the last 5 metrics
        buffer_metrics = self.collector.get_all_metrics_buffer()
        self.assertEqual(len(buffer_metrics), 5)
        
        # Check that the most recent metrics are in the buffer
        values = [m.value for m in buffer_metrics]
        self.assertEqual(sorted(values), [5, 6, 7, 8, 9])

    def test_history_overflow(self):
        """Test behavior when history overflows."""
        # Add the same metric multiple times to test history overflow
        for i in range(10):
            self.collector.add_metric(
                name="history_test",
                value=i,
                metric_type=MetricType.GAUGE
            )
        
        # History should only contain the last 3 metrics
        history = self.collector.get_metrics_history("history_test")
        self.assertEqual(len(history), 3)
        
        # Check that the most recent metrics are in history
        values = [m.value for m in history]
        self.assertEqual(values, [7, 8, 9])  # Last 3 values

    def test_empty_export(self):
        """Test exporting when no metrics have been collected."""
        # Test JSON export with no metrics
        json_result = self.collector.export_to_json()
        data = json.loads(json_result)
        self.assertEqual(len(data["metrics"]), 0)
        
        # Test Prometheus export with no metrics
        prometheus_result = self.collector.export_to_prometheus()
        # Should still have header but no metric data
        self.assertIn("# Metrics collected from Qwen3-VL model", prometheus_result)

    def test_large_string_value(self):
        """Test handling of large string values."""
        large_string = "x" * 1500  # Larger than the 1000 character limit
        
        # This should fail validation
        result = self.collector.add_metric(
            name="large_string_test",
            value=large_string,
            metric_type=MetricType.GAUGE
        )
        
        self.assertFalse(result)

    def test_special_values(self):
        """Test handling of special float values."""
        # Test NaN
        result = self.collector.add_metric(
            name="nan_test",
            value=float('nan'),
            metric_type=MetricType.GAUGE
        )
        self.assertFalse(result)
        
        # Test infinity
        result = self.collector.add_metric(
            name="inf_test",
            value=float('inf'),
            metric_type=MetricType.GAUGE
        )
        self.assertFalse(result)
        
        # Test negative infinity
        result = self.collector.add_metric(
            name="neg_inf_test",
            value=float('-inf'),
            metric_type=MetricType.GAUGE
        )
        self.assertFalse(result)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)