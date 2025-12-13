# Centralized Metrics Collection System Documentation

## Overview

The Centralized Metrics Collection System is a comprehensive solution for collecting, aggregating, and exporting performance, memory, and operational metrics from all components of the Qwen3-VL model. This system replaces scattered metrics collection approaches with a unified, standardized system that provides real-time monitoring capabilities.

## Features

- **Centralized Collection**: Gather metrics from all model components in one place
- **Standardized Format**: Consistent metric naming and structure following Prometheus conventions
- **Real-time Aggregation**: Collect and process metrics in real-time
- **Multiple Export Formats**: Export to JSON, CSV, and Prometheus formats
- **Performance Tracking**: Built-in performance tracking for operations
- **Validation and Error Handling**: Robust validation and error handling
- **Hardware Monitoring**: Automatic collection of CPU, GPU, memory, and process metrics
- **Thread Safety**: Safe to use from multiple threads simultaneously
- **Queue Support**: High-frequency metrics can be queued to avoid blocking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Metrics Collector                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ System      │  │ Custom      │  │ Performance │        │
│  │ Metrics     │  │ Metrics     │  │ Tracking    │        │
│  │ Collector   │  │ Collector   │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ CPU Metrics     │  │ Memory Metrics  │  │ GPU Metrics │ │
│  │ Process Metrics │  │ Custom Metrics  │  │ ...         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ JSON        │  │ CSV         │  │ Prometheus  │        │
│  │ Export      │  │ Export      │  │ Export      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Installation and Setup

The metrics collector is included as part of the Qwen3-VL codebase. To use it:

```python
from centralized_metrics_collector import get_metrics_collector

# Get the global collector instance
collector = get_metrics_collector()

# Start collection (optional, can be run in background)
collector.start_collection()
```

## Usage Examples

### Basic Metric Collection

```python
from centralized_metrics_collector import get_metrics_collector, MetricType

collector = get_metrics_collector()

# Add a simple metric
collector.add_metric(
    name="model_inference_latency",
    value=0.125,
    metric_type=MetricType.GAUGE,
    labels={"model": "qwen3-vl", "input_size": "large"},
    description="Model inference latency in seconds"
)

# Get current value of a metric
metric = collector.get_metric("model_inference_latency", {"model": "qwen3-vl", "input_size": "large"})
print(f"Latency: {metric.value}s")
```

### Performance Tracking

```python
from centralized_metrics_collector import start_performance_tracking, end_performance_tracking

# Start tracking performance
start_performance_tracking("model_inference")

# ... perform model inference ...

# End tracking and get results
perf_result = end_performance_tracking("model_inference")
print(f"Duration: {perf_result['duration']:.4f}s")
print(f"Memory delta: {perf_result['memory_delta']} bytes")
```

### Exporting Metrics

```python
# Export to JSON
json_data = collector.export_to_json()

# Export to CSV file
collector.export_to_csv("metrics.csv")

# Export to Prometheus format
prometheus_data = collector.export_to_prometheus()

# Generic export method
json_data = collector.export("json")
csv_data = collector.export("csv", output_path="metrics.csv")
prometheus_data = collector.export("prometheus")
```

### System Metrics Collection

The collector automatically gathers system metrics including:

- CPU usage and frequency
- Memory usage (total, available, percentage)
- GPU metrics (if available): memory allocation, utilization, temperature, power
- Process metrics: memory RSS/VMS, CPU usage, thread count

### High-Frequency Metrics with Queue

For high-frequency metrics that could block your application, use the queue:

```python
# Queue metrics to avoid blocking
success = collector.queue_metric(
    name="request_count",
    value=1,
    metric_type=MetricType.COUNTER,
    labels={"endpoint": "/api/predict"},
    description="Number of requests to prediction endpoint"
)

# The collector processes queued metrics in the background
```

## Metric Types

The system supports the following metric types:

- **Counter**: Monotonically increasing values (e.g., request count)
- **Gauge**: Instantaneous values that can go up and down (e.g., temperature)
- **Histogram**: Samples observations and counts them in configurable buckets
- **Summary**: Samples observations and calculates percentiles

## Naming Conventions

Metric names should follow Prometheus naming conventions:

- Start with a letter
- Contain only letters, numbers, and underscores
- Use snake_case for multi-word names
- Include subsystem prefix (e.g., `system_cpu_percent`, `model_inference_latency`)

## Validation and Error Handling

The system includes robust validation and error handling:

```python
# Register custom validation rules
def positive_value_validator(value):
    return isinstance(value, (int, float)) and value > 0

collector.register_validation_rule("positive_metric", positive_value_validator)

# Validate metric values
is_valid = collector.validate_metric("positive_metric", 42)
```

## Integration with Existing Components

The metrics collector can be integrated with existing performance monitoring:

```python
# Example integration with existing performance tracking
def monitor_model_performance(model, inputs):
    collector = get_metrics_collector()

    # Start performance tracking
    start_performance_tracking("model_inference")

    # Run model inference
    start_time = time.time()
    output = model(inputs)
    end_time = time.time()

    # Add custom metrics
    collector.add_metric(
        name="model_inference_time",
        value=end_time - start_time,
        metric_type=MetricType.GAUGE,
        labels={"model": "qwen3-vl"},
        description="Model inference time in seconds"
    )

    # End performance tracking
    perf_result = end_performance_tracking("model_inference")

    return output
```

## Export Formats

### JSON Format

```json
{
  "timestamp": "2023-10-01T12:00:00",
  "metrics": [
    {
      "name": "model_inference_latency",
      "value": 0.125,
      "type": "gauge",
      "labels": {"model": "qwen3-vl", "input_size": "large"},
      "description": "Model inference latency in seconds",
      "timestamp": 1696161600.0,
      "datetime": "2023-10-01T12:00:00"
    }
  ],
  "history": {
    "model_inference_latency": [...]
  }
}
```

### Prometheus Format

```
# TYPE model_inference_latency gauge
# HELP model_inference_latency Model inference latency in seconds
model_inference_latency{model="qwen3-vl",input_size="large"} 0.125 1696161600000
```

### CSV Format

```
name,value,type,labels,description,timestamp,datetime
model_inference_latency,0.125,gauge,"{'model': 'qwen3-vl', 'input_size': 'large'}",Model inference latency in seconds,1696161600.0,2023-10-01T12:00:00
```

## Configuration

The metrics collector can be configured with the following parameters:

```python
collector = MetricsCollector(
    collection_interval=1.0,          # Seconds between system metric collections
    max_metrics_buffer=10000,         # Maximum metrics to keep in memory
    max_history_per_metric=1000,      # Maximum history per individual metric
    enable_prometheus_export=True,    # Enable Prometheus export functionality
    validation_enabled=True           # Enable metric validation
)
```

## Best Practices

1. **Use Descriptive Names**: Choose clear, descriptive metric names that follow conventions
2. **Add Labels**: Use labels to provide additional dimensions for metrics
3. **Include Descriptions**: Always provide meaningful descriptions for metrics
4. **Monitor Resource Usage**: Be mindful of the resource overhead of metrics collection
5. **Export Regularly**: Set up regular export of metrics for analysis
6. **Validate Data**: Use validation rules to ensure data quality
7. **Use Queues for High-Frequency Metrics**: For metrics collected at high frequency, use the queue mechanism to avoid blocking your application
8. **Clean Up**: Always call `stop_collection()` when you're done with the collector

## Performance Considerations

- The metrics collection runs in a background thread to minimize impact
- Metrics are buffered in memory with configurable size limits
- System metrics collection has minimal overhead
- Consider using the queue mechanism for high-frequency metrics to reduce blocking
- Monitor the queue size to ensure it's not filling up faster than it's being processed

## Troubleshooting

### Common Issues

1. **Invalid Metric Names**: Ensure metric names follow naming conventions
2. **Memory Usage**: Monitor buffer size and adjust `max_metrics_buffer` if needed
3. **Thread Safety**: The collector is thread-safe for concurrent access
4. **GPU Metrics**: GPU metrics require PyTorch CUDA and may require additional libraries like pynvml

### Error Handling

The system provides comprehensive error handling:

- Invalid metric names/values are rejected with clear error messages
- System metrics collection continues even if individual collectors fail
- Background collection continues even if individual collection cycles fail
- Queue mechanism prevents blocking when metrics are generated faster than they can be processed

## API Reference

### MetricsCollector Class

#### `__init__(collection_interval=1.0, max_metrics_buffer=10000, enable_prometheus_export=True, max_history_per_metric=1000, validation_enabled=True)`

Initialize the metrics collector.

#### `add_metric(name, value, metric_type, labels=None, description="", timestamp=None)`

Add a metric to the collector. Returns True if successful.

#### `queue_metric(name, value, metric_type, labels=None, description="", timestamp=None)`

Queue a metric for later processing. Useful for high-frequency metrics. Returns True if successfully queued.

#### `get_metric(name, labels=None)`

Get the current value of a specific metric.

#### `get_metrics_history(name, limit=None)`

Get historical values for a specific metric.

#### `get_all_current_metrics()`

Get all current metric values.

#### `get_all_metrics_buffer()`

Get all metrics in the buffer.

#### `export_to_json(output_path=None, include_history=True)`

Export metrics to JSON format.

#### `export_to_csv(output_path, include_history=False)`

Export metrics to CSV format.

#### `export_to_prometheus()`

Export metrics in Prometheus text format.

#### `export(format_type, output_path=None, **kwargs)`

Generic export method for different formats.

#### `start_collection()`

Start the metrics collection thread.

#### `stop_collection()`

Stop the metrics collection thread.

#### `add_performance_tracker(name, start_time=None, start_memory=None)`

Start tracking performance for a specific operation.

#### `end_performance_tracker(name)`

End performance tracking and return results.

#### `reset()`

Reset all collected metrics.

#### `get_statistics()`

Get statistics about the collector's state.

### Global Functions

#### `get_metrics_collector()`

Get the global metrics collector instance.

#### `add_performance_metric(name, value, labels=None, description="")`

Convenience function to add a performance metric.

#### `start_performance_tracking(name)`

Start tracking performance for a specific operation.

#### `end_performance_tracking(name)`

End performance tracking and return results.

#### `queue_performance_metric(name, value, labels=None, description="")`

Convenience function to queue a performance metric.

## Integration Examples

### With Model Training

```python
def train_with_metrics(model, data_loader, optimizer, criterion):
    collector = get_metrics_collector()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(data_loader):
            batch_start_time = time.time()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Add batch metrics
            batch_time = time.time() - batch_start_time
            collector.add_metric(
                name="training_batch_time",
                value=batch_time,
                metric_type=MetricType.GAUGE,
                labels={"epoch": str(epoch), "batch": str(batch_idx)}
            )

        # Add epoch metrics
        epoch_time = time.time() - epoch_start_time
        collector.add_metric(
            name="training_epoch_time",
            value=epoch_time,
            metric_type=MetricType.GAUGE,
            labels={"epoch": str(epoch)}
        )
```

### With Memory Management

```python
def memory_management_with_metrics():
    collector = get_metrics_collector()

    # Monitor memory before operation
    start_memory = torch.cuda.memory_allocated()

    # Perform memory-intensive operation
    # ... operation code ...

    # Monitor memory after operation
    end_memory = torch.cuda.memory_allocated()
    memory_delta = end_memory - start_memory

    collector.add_metric(
        name="memory_operation_delta",
        value=memory_delta,
        metric_type=MetricType.GAUGE,
        labels={"operation": "memory_intensive_task"}
    )
```

## Testing

The system includes comprehensive tests in `test_centralized_metrics_collector.py`:

```bash
python -m pytest test_centralized_metrics_collector.py -v
```

## Conclusion

The Centralized Metrics Collection System provides a robust, scalable solution for monitoring all aspects of the Qwen3-VL model. With standardized metrics, multiple export formats, and comprehensive system monitoring, it enables detailed performance analysis and optimization across all components of the model.