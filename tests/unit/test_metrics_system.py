"""
Comprehensive tests for the timing utilities and centralized metrics collector system.

This module tests all aspects of the performance metrics system including:
- Timing utilities
- Centralized metrics collector
- Integration with memory pooling and CPU optimization systems
"""

import unittest
import time
import threading
from unittest.mock import patch, MagicMock
import numpy as np

# Import the modules to test
from timing_utilities import (
    Timer, time_function, time_block, track_resource_usage,
    PerformanceMonitor, get_global_monitor, benchmark_function, measure_memory_usage
)
from centralized_metrics_collector import (
    CentralizedMetricsCollector, MetricType, MetricsAggregator, MetricsFilter,
    get_metrics_collector, record_metric, record_timing, record_counter
)
from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType
from advanced_cpu_optimizations_intel_i5_10210u import IntelOptimizedPipeline, AdvancedCPUOptimizationConfig


class TestTimer(unittest.TestCase):
    """Test the Timer class."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        timer = Timer("test_operation")
        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = timer.stop()
        
        self.assertGreater(elapsed, 0.01)  # Should be at least 10ms
        self.assertLess(elapsed, 0.05)    # Should not be more than 50ms (allowing for some overhead)
    
    def test_timer_time_it(self):
        """Test the time_it method."""
        def sample_function(x, y):
            time.sleep(0.01)
            return x + y
        
        timer = Timer("sample_operation")
        result, timing_result = timer.time_it(sample_function, 5, 10)
        
        self.assertEqual(result, 15)
        self.assertGreater(timing_result.execution_time, 0.01)
        self.assertEqual(timing_result.operation_name, "sample_operation")
    
    def test_timer_reset(self):
        """Test timer reset functionality."""
        timer = Timer("test_operation")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        timer.reset()
        
        self.assertIsNone(timer.start_time)
        self.assertIsNone(timer.end_time)
        self.assertIsNone(timer.elapsed_time)


class TestTimeFunctionDecorator(unittest.TestCase):
    """Test the time_function decorator."""
    
    def test_time_function_basic(self):
        """Test basic functionality of time_function decorator."""
        @time_function("test_function")
        def sample_function():
            time.sleep(0.01)
            return "result"
        
        result = sample_function()
        self.assertEqual(result, "result")
    
    def test_time_function_with_metrics_collection(self):
        """Test time_function with metrics collection."""
        @time_function("test_function_with_metrics", collect_metrics=True)
        def sample_function():
            time.sleep(0.01)
            return "result"
        
        # Clear any existing metrics
        collector = get_metrics_collector()
        collector.clear_metrics()
        
        result = sample_function()
        self.assertEqual(result, "result")
        
        # Check that metrics were recorded
        metrics = collector.get_latest_metrics(5)
        timing_metrics = [m for m in metrics if "test_function_with_metrics" in m.name]
        self.assertGreater(len(timing_metrics), 0)


class TestTimeBlock(unittest.TestCase):
    """Test the time_block context manager."""
    
    def test_time_block_basic(self):
        """Test basic functionality of time_block."""
        with time_block("test_block", collect_metrics=False) as timer:
            time.sleep(0.01)
        
        # Timer should have recorded the time
        self.assertGreater(timer.elapsed_time, 0.01)
    
    def test_time_block_with_metrics(self):
        """Test time_block with metrics collection."""
        collector = get_metrics_collector()
        collector.clear_metrics()
        
        with time_block("test_block_with_metrics", collect_metrics=True):
            time.sleep(0.01)
        
        # Check that metrics were recorded
        metrics = collector.get_latest_metrics(5)
        timing_metrics = [m for m in metrics if "test_block_with_metrics" in m.name]
        self.assertGreater(len(timing_metrics), 0)


class TestPerformanceMonitor(unittest.TestCase):
    """Test the PerformanceMonitor class."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_record_timing(self):
        """Test recording timing information."""
        self.monitor.record_timing("operation1", 0.1)
        self.monitor.record_timing("operation1", 0.2)
        self.monitor.record_timing("operation1", 0.15)
        
        stats = self.monitor.get_stats("operation1")
        self.assertEqual(stats['count'], 3)
        self.assertAlmostEqual(stats['avg_time'], 0.15, places=2)
        self.assertEqual(stats['min_time'], 0.1)
        self.assertEqual(stats['max_time'], 0.2)
    
    def test_get_stats_all(self):
        """Test getting stats for all operations."""
        self.monitor.record_timing("op1", 0.1)
        self.monitor.record_timing("op2", 0.2)
        
        all_stats = self.monitor.get_stats()
        self.assertIn("op1", all_stats)
        self.assertIn("op2", all_stats)
    
    def test_get_trend(self):
        """Test getting trend information."""
        # Record some early values
        for _ in range(5):
            self.monitor.record_timing("trend_test", 0.1)
        
        # Record some later values that are higher
        for _ in range(5):
            self.monitor.record_timing("trend_test", 0.3)
        
        trend = self.monitor.get_trend("trend_test", 10)
        self.assertEqual(trend['trend'], 'increasing')


class TestCentralizedMetricsCollector(unittest.TestCase):
    """Test the CentralizedMetricsCollector class."""
    
    def setUp(self):
        self.collector = get_metrics_collector()
        self.collector.clear_metrics()
    
    def test_singleton_pattern(self):
        """Test that CentralizedMetricsCollector follows singleton pattern."""
        collector1 = CentralizedMetricsCollector.get_instance()
        collector2 = CentralizedMetricsCollector.get_instance()
        self.assertIs(collector1, collector2)
    
    def test_record_metric(self):
        """Test recording a metric."""
        self.collector.record_metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.COUNTER,
            source="test_source",
            tags={"test": "value"}
        )
        
        metrics = self.collector.get_latest_metrics(1)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].name, "test_metric")
        self.assertEqual(metrics[0].value, 42.0)
        self.assertEqual(metrics[0].metric_type, MetricType.COUNTER)
        self.assertEqual(metrics[0].source, "test_source")
        self.assertEqual(metrics[0].tags["test"], "value")
    
    def test_record_timing(self):
        """Test recording timing metric."""
        self.collector.record_timing("test_operation", 0.1, "test_source")
        
        metrics = self.collector.get_latest_metrics(1)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].name, "test_operation_time")
        self.assertEqual(metrics[0].metric_type, MetricType.TIME)
    
    def test_record_counter(self):
        """Test recording counter metric."""
        self.collector.record_counter("test_counter", 5.0, "test_source")
        
        metrics = self.collector.get_latest_metrics(1)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].name, "test_counter")
        self.assertEqual(metrics[0].metric_type, MetricType.COUNTER)
        self.assertEqual(metrics[0].value, 5.0)
    
    def test_get_metric_stats(self):
        """Test getting metric statistics."""
        for i in range(10):
            self.collector.record_metric(f"test_stats", float(i), MetricType.COUNTER)
        
        stats = self.collector.get_metric_stats("test_stats")
        self.assertEqual(stats['count'], 10)
        self.assertEqual(stats['average'], 4.5)  # Average of 0-9 is 4.5
        self.assertEqual(stats['min'], 0.0)
        self.assertEqual(stats['max'], 9.0)
    
    def test_export_functions(self):
        """Test export functions."""
        self.collector.record_metric("export_test", 1.0, MetricType.COUNTER)
        
        # Test JSON export
        self.collector.export_to_json("test_metrics.json")
        
        # Test CSV export
        self.collector.export_to_csv("test_metrics.csv")
        
        # Test Prometheus export
        self.collector.export_to_prometheus("test_metrics.prom")
        
        # Clean up test files
        import os
        for file in ["test_metrics.json", "test_metrics.csv", "test_metrics.prom"]:
            if os.path.exists(file):
                os.remove(file)
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        self.collector.record_metric("test_clear", 1.0, MetricType.COUNTER)
        self.assertEqual(len(self.collector.get_metrics()), 1)
        
        self.collector.clear_metrics()
        self.assertEqual(len(self.collector.get_metrics()), 0)


class TestMetricsAggregator(unittest.TestCase):
    """Test the MetricsAggregator class."""
    
    def setUp(self):
        self.aggregator = MetricsAggregator(window_size=5)
    
    def test_add_metric(self):
        """Test adding metrics to aggregator."""
        for i in range(10):
            self.aggregator.add_metric("test_metric", float(i))
        
        # Should only keep the last 5 values due to window size
        self.assertEqual(len(self.aggregator.metrics["test_metric"]), 5)
        
        stats = self.aggregator.get_stats("test_metric")
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['min'], 5.0)  # Last 5 values: 5, 6, 7, 8, 9
        self.assertEqual(stats['max'], 9.0)
    
    def test_get_recent_values(self):
        """Test getting recent values."""
        for i in range(10):
            self.aggregator.add_metric("recent_test", float(i))
        
        recent = self.aggregator.get_recent_values("recent_test", 3)
        self.assertEqual(recent, [7.0, 8.0, 9.0])  # Last 3 values


class TestMetricsFilter(unittest.TestCase):
    """Test the MetricsFilter class."""
    
    def setUp(self):
        self.collector = get_metrics_collector()
        self.collector.clear_metrics()
        
        # Add some test metrics
        self.collector.record_metric("cpu_usage", 80.0, MetricType.PERCENTAGE, "system")
        self.collector.record_metric("memory_usage", 2048.0, MetricType.MEMORY, "system")
        self.collector.record_metric("request_count", 100.0, MetricType.COUNTER, "api")
    
    def test_name_filter(self):
        """Test filtering by metric name."""
        metrics = self.collector.get_metrics()
        filter_obj = MetricsFilter().add_name_filter("cpu*")
        filtered = filter_obj.apply(metrics)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "cpu_usage")
    
    def test_type_filter(self):
        """Test filtering by metric type."""
        metrics = self.collector.get_metrics()
        filter_obj = MetricsFilter().add_type_filter(MetricType.MEMORY)
        filtered = filter_obj.apply(metrics)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "memory_usage")
    
    def test_source_filter(self):
        """Test filtering by source."""
        metrics = self.collector.get_metrics()
        filter_obj = MetricsFilter().add_source_filter("api")
        filtered = filter_obj.apply(metrics)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "request_count")


class TestIntegrationWithMemoryPoolingSystem(unittest.TestCase):
    """Test integration of metrics with memory pooling system."""
    
    def setUp(self):
        self.collector = get_metrics_collector()
        self.collector.clear_metrics()
        self.memory_system = AdvancedMemoryPoolingSystem()
    
    def test_memory_pooling_metrics(self):
        """Test that memory pooling operations record metrics."""
        # Perform some allocation and deallocation operations
        block = self.memory_system.allocate(TensorType.KV_CACHE, 1024*1024, "test_tensor")
        self.assertIsNotNone(block)
        
        success = self.memory_system.deallocate(TensorType.KV_CACHE, "test_tensor")
        self.assertTrue(success)
        
        # Check that metrics were recorded
        metrics = self.collector.get_metrics()
        self.assertGreater(len(metrics), 0)
        
        # Check for specific metric types
        timing_metrics = [m for m in metrics if m.metric_type == MetricType.TIME]
        self.assertGreater(len(timing_metrics), 0)
        
        memory_metrics = [m for m in metrics if m.metric_type == MetricType.MEMORY]
        self.assertGreater(len(memory_metrics), 0)
    
    def test_get_performance_metrics(self):
        """Test the get_performance_metrics method."""
        # Perform some operations
        for i in range(5):
            block = self.memory_system.allocate(TensorType.KV_CACHE, 1024*100, f"test_tensor_{i}")
            if block:
                self.memory_system.deallocate(TensorType.KV_CACHE, f"test_tensor_{i}")
        
        # Get performance metrics
        perf_metrics = self.memory_system.get_performance_metrics()
        
        # Check that metrics exist
        self.assertIn('system_metrics', perf_metrics)
        self.assertIn('pool_metrics', perf_metrics)
        self.assertIn('buddy_allocator_metrics', perf_metrics)
        
        # Check system metrics
        sys_metrics = perf_metrics['system_metrics']
        self.assertGreaterEqual(sys_metrics['system_allocation_count'], 5)
        self.assertGreaterEqual(sys_metrics['system_deallocation_count'], 5)


class TestIntegrationWithCPUOptimizations(unittest.TestCase):
    """Test integration of metrics with CPU optimization system."""
    
    def setUp(self):
        self.collector = get_metrics_collector()
        self.collector.clear_metrics()
        
        # Create a minimal config for testing
        config = AdvancedCPUOptimizationConfig()
        
        # Create a mock model for testing
        class MockModel:
            def __init__(self):
                self.parameters = lambda: [MagicMock()]
            
            def generate(self, **kwargs):
                return MagicMock()
        
        mock_model = MockModel()
        self.pipeline = IntelOptimizedPipeline(mock_model, config)
    
    def test_cpu_optimization_metrics(self):
        """Test that CPU optimization operations record metrics."""
        # Perform some preprocessing operations
        texts = ["test text"] * 3
        try:
            # This may fail due to missing tokenizer, but should still record metrics
            result = self.pipeline.preprocess_and_infer(texts)
        except Exception:
            # Even if the operation fails, metrics should be recorded
            pass
        
        # Check that metrics were recorded
        metrics = self.collector.get_metrics()
        self.assertGreater(len(metrics), 0)
        
        # Look for pipeline-specific metrics
        pipeline_metrics = [m for m in metrics if "pipeline" in m.name.lower()]
        self.assertGreater(len(pipeline_metrics), 0)


class TestBenchmarkFunction(unittest.TestCase):
    """Test the benchmark_function utility."""
    
    def test_benchmark_function(self):
        """Test benchmarking a function."""
        def sample_function(x):
            time.sleep(0.01)  # Sleep for 10ms
            return x * 2
        
        stats = benchmark_function(sample_function, 5, iterations=3)
        
        self.assertEqual(stats['count'], 3)
        self.assertGreater(stats['avg_time'], 0.01)
        self.assertGreater(stats['min_time'], 0.01)
        self.assertGreater(stats['max_time'], 0.01)


class TestMeasureMemoryUsage(unittest.TestCase):
    """Test the measure_memory_usage utility."""
    
    def test_measure_memory_usage(self):
        """Test measuring memory usage of a function."""
        def memory_intensive_function():
            # Create a large list to consume memory
            data = [0] * 100000
            time.sleep(0.01)
            return len(data)
        
        result, timing_result, memory_delta = measure_memory_usage(memory_intensive_function)
        
        self.assertEqual(result, 100000)
        self.assertGreater(timing_result.execution_time, 0.01)
        # Memory delta might be positive or negative depending on system, just check it's recorded


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for metrics."""
    
    def setUp(self):
        self.collector = get_metrics_collector()
        self.collector.clear_metrics()
    
    def test_convenience_functions(self):
        """Test convenience functions for recording metrics."""
        record_metric("convenience_test", 42.0, MetricType.COUNTER, "test_source")
        record_timing("timing_test", 0.1, "test_source")
        record_counter("counter_test", 5.0, "test_source")
        
        metrics = self.collector.get_metrics()
        self.assertEqual(len(metrics), 3)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of the metrics system."""
    
    def test_thread_safe_metrics_collection(self):
        """Test that metrics collection is thread-safe."""
        collector = get_metrics_collector()
        collector.clear_metrics()
        
        def record_metrics(thread_id):
            for i in range(10):
                collector.record_metric(f"thread_{thread_id}_metric", float(i), MetricType.COUNTER)
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_metrics, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check that all metrics were recorded properly
        metrics = collector.get_metrics()
        self.assertEqual(len(metrics), 50)  # 5 threads * 10 metrics each


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)