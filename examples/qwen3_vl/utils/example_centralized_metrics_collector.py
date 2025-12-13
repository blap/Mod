"""
Comprehensive example demonstrating the usage of the centralized metrics collection system.
This example shows how to use the metrics collector in various scenarios.
"""

import time
import torch
import threading
from src.qwen3_vl.utils.centralized_metrics_collector import (
    MetricsCollector,
    MetricType,
    get_metrics_collector,
    add_performance_metric,
    start_performance_tracking,
    end_performance_tracking,
    queue_performance_metric
)


def example_basic_usage():
    """Example of basic metrics collection."""
    print("=== Basic Metrics Collection Example ===")
    
    # Get the global collector
    collector = get_metrics_collector()
    
    # Add some basic metrics
    collector.add_metric(
        name="model_accuracy",
        value=0.95,
        metric_type=MetricType.GAUGE,
        labels={"model": "qwen3-vl", "dataset": "validation"},
        description="Model accuracy on validation dataset"
    )
    
    collector.add_metric(
        name="model_inference_count",
        value=100,
        metric_type=MetricType.COUNTER,
        labels={"model": "qwen3-vl"},
        description="Number of inference operations performed"
    )
    
    # Retrieve a specific metric
    accuracy_metric = collector.get_metric("model_accuracy", {"model": "qwen3-vl", "dataset": "validation"})
    print(f"Retrieved accuracy: {accuracy_metric.value if accuracy_metric else 'Not found'}")
    
    # Export to JSON
    json_export = collector.export_to_json(include_history=True)
    print(f"JSON export (first 200 chars): {json_export[:200]}...")
    
    print()


def example_performance_tracking():
    """Example of performance tracking."""
    print("=== Performance Tracking Example ===")
    
    # Start tracking a specific operation
    start_performance_tracking("model_inference")
    
    # Simulate some work
    time.sleep(0.1)  # Simulate model inference
    
    # End tracking and get results
    perf_result = end_performance_tracking("model_inference")
    
    if perf_result:
        print(f"Operation duration: {perf_result['duration']:.4f}s")
        print(f"Memory delta: {perf_result['memory_delta']} bytes")
        print(f"Start memory: {perf_result['start_memory']} bytes")
        print(f"End memory: {perf_result['end_memory']} bytes")
    
    print()


def example_system_metrics():
    """Example of system metrics collection."""
    print("=== System Metrics Collection Example ===")
    
    collector = get_metrics_collector()
    
    # Collect system metrics manually
    collector._collect_system_metrics()
    
    # Get some system metrics
    cpu_metric = collector.get_metric("system_cpu_percent")
    memory_metric = collector.get_metric("system_memory_percent")
    
    if cpu_metric:
        print(f"CPU Usage: {cpu_metric.value}%")
    if memory_metric:
        print(f"Memory Usage: {memory_metric.value}%")
    
    print()


def example_export_formats():
    """Example of different export formats."""
    print("=== Export Formats Example ===")
    
    collector = get_metrics_collector()
    
    # Add some metrics for export
    collector.add_metric(
        name="example_metric",
        value=42.5,
        metric_type=MetricType.GAUGE,
        labels={"category": "test", "env": "production"},
        description="An example metric for export"
    )
    
    # Export to JSON
    print("JSON Export:")
    json_data = collector.export_to_json()
    print(json_data)
    print()
    
    # Export to Prometheus format
    print("Prometheus Export:")
    prometheus_data = collector.export_to_prometheus()
    print(prometheus_data)
    print()
    
    # Export to CSV (to a temporary file)
    collector.export_to_csv("example_metrics.csv")
    print("CSV export completed to 'example_metrics.csv'")
    print()


def example_threaded_usage():
    """Example of using the metrics collector in a multi-threaded environment."""
    print("=== Threaded Usage Example ===")
    
    collector = get_metrics_collector()
    
    def worker_thread(thread_id):
        """Simulate work in a thread and collect metrics."""
        for i in range(5):
            # Add a metric from this thread
            collector.add_metric(
                name="thread_work_count",
                value=i + 1,
                metric_type=MetricType.COUNTER,
                labels={"thread_id": str(thread_id), "iteration": str(i)},
                description=f"Work count for thread {thread_id}"
            )
            
            time.sleep(0.01)  # Simulate work
    
    # Start multiple threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_thread, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print(f"Total metrics collected from threads: {len(collector.get_all_metrics_buffer())}")
    print()


def example_validation():
    """Example of custom validation rules."""
    print("=== Validation Example ===")
    
    collector = get_metrics_collector()
    
    # Register a custom validation rule
    def positive_value_validator(value):
        return isinstance(value, (int, float)) and value > 0
    
    collector.register_validation_rule("positive_metric", positive_value_validator)
    
    # Test validation
    is_valid = collector.validate_metric("positive_metric", 42)
    print(f"Is 42 valid for positive_metric? {is_valid}")
    
    is_valid = collector.validate_metric("positive_metric", -5)
    print(f"Is -5 valid for positive_metric? {is_valid}")
    
    # Try to add a metric with validation
    result = collector.add_metric(
        name="positive_metric",
        value=10,
        metric_type=MetricType.GAUGE
    )
    print(f"Adding positive metric succeeded: {result}")
    
    result = collector.add_metric(
        name="positive_metric",
        value=-5,
        metric_type=MetricType.GAUGE
    )
    print(f"Adding negative metric succeeded: {result}")
    
    print()


def example_queue_metrics():
    """Example of queuing metrics for high-frequency scenarios."""
    print("=== Queued Metrics Example ===")
    
    collector = get_metrics_collector()
    
    # Queue some metrics (useful for high-frequency metrics)
    for i in range(10):
        success = collector.queue_metric(
            name="high_freq_metric",
            value=i,
            metric_type=MetricType.COUNTER,
            labels={"source": "high_frequency", "iteration": str(i)},
            description="High frequency metric example"
        )
        if not success:
            print(f"Failed to queue metric {i}")
    
    # Process queued metrics
    collector._process_queued_metrics()
    
    # Check that metrics were added
    history = collector.get_metrics_history("high_freq_metric", limit=5)
    print(f"Retrieved {len(history)} recent high frequency metrics")
    for metric in history:
        print(f"  Value: {metric.value}, Labels: {metric.labels}")
    
    print()


def example_convenience_functions():
    """Example of using convenience functions."""
    print("=== Convenience Functions Example ===")
    
    # Use convenience functions
    add_performance_metric(
        name="convenience_metric",
        value=99.5,
        labels={"source": "convenience"},
        description="Added via convenience function"
    )
    
    # Start and end performance tracking using convenience functions
    start_performance_tracking("convenience_operation")
    time.sleep(0.05)  # Simulate work
    result = end_performance_tracking("convenience_operation")
    
    if result:
        print(f"Convenience operation duration: {result['duration']:.4f}s")
    
    # Queue a metric using convenience function
    queue_performance_metric(
        name="queued_convenience_metric",
        value=200,
        labels={"source": "queued_convenience"},
        description="Queued via convenience function"
    )
    
    # Process the queued metric
    collector = get_metrics_collector()
    collector._process_queued_metrics()
    
    print("Convenience functions example completed")
    print()


def example_statistics():
    """Example of getting statistics about the collector."""
    print("=== Collector Statistics Example ===")
    
    collector = get_metrics_collector()
    
    # Add a few metrics
    for i in range(5):
        collector.add_metric(
            name=f"stat_test_metric_{i}",
            value=i * 10,
            metric_type=MetricType.GAUGE
        )
    
    # Get statistics
    stats = collector.get_statistics()
    print("Collector Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()


def main():
    """Run all examples."""
    print("Centralized Metrics Collection System - Comprehensive Examples")
    print("=" * 60)
    
    # Start the collector
    collector = get_metrics_collector()
    collector.start_collection()
    
    try:
        example_basic_usage()
        example_performance_tracking()
        example_system_metrics()
        example_export_formats()
        example_threaded_usage()
        example_validation()
        example_queue_metrics()
        example_convenience_functions()
        example_statistics()
        
        print("All examples completed successfully!")
        
    finally:
        # Stop the collector
        collector.stop_collection()
        
        # Show final statistics
        stats = collector.get_statistics()
        print(f"\nFinal statistics:")
        print(f"  Total metrics collected: {stats['total_metrics_collected']}")
        print(f"  Current metrics count: {stats['current_metrics_count']}")
        print(f"  Is collecting: {stats['is_collecting']}")


if __name__ == "__main__":
    main()