"""
Example demonstrating the usage of the centralized metrics collection system
with the Qwen3-VL model components.
"""

import time
import torch
import torch.nn as nn
from src.qwen3_vl.utils.centralized_metrics_collector import (
    get_metrics_collector,
    MetricType,
    start_performance_tracking,
    end_performance_tracking
)


class ExampleModel(nn.Module):
    """Example model to demonstrate metrics collection."""
    
    def __init__(self, input_size=768, hidden_size=1024, output_size=768):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


def simulate_model_inference():
    """Simulate model inference with metrics collection."""
    print("=== Simulating Model Inference with Metrics Collection ===")
    
    # Get the metrics collector
    collector = get_metrics_collector()
    
    # Create example model and input
    model = ExampleModel()
    input_tensor = torch.randn(1, 768)  # Batch size 1, input size 768
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Start performance tracking for the entire inference
    start_performance_tracking("model_inference")
    
    # Warm up the model (important for accurate timing)
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Start detailed timing
    inference_start_time = time.time()
    
    # Simulate model inference
    with torch.no_grad():
        output = model(input_tensor)
    
    inference_end_time = time.time()
    
    # End performance tracking
    perf_result = end_performance_tracking("model_inference")
    
    # Add custom metrics
    collector.add_metric(
        name="model_inference_time",
        value=inference_end_time - inference_start_time,
        metric_type=MetricType.GAUGE,
        labels={"model": "example_model", "input_size": "768"},
        description="Model inference time in seconds"
    )
    
    collector.add_metric(
        name="model_output_size",
        value=output.numel(),
        metric_type=MetricType.GAUGE,
        labels={"model": "example_model"},
        description="Number of elements in model output"
    )
    
    print(f"Inference time: {inference_end_time - inference_start_time:.4f}s")
    print(f"Performance tracking result: {perf_result}")
    
    return output


def simulate_memory_operations():
    """Simulate memory operations with metrics collection."""
    print("\n=== Simulating Memory Operations with Metrics Collection ===")
    
    collector = get_metrics_collector()
    
    # Simulate memory allocation
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Create a large tensor
    large_tensor = torch.randn(1000, 1000)  # ~38MB tensor
    
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else large_tensor.numel() * 4  # 4 bytes per float
    
    memory_allocated = end_memory - start_memory
    
    collector.add_metric(
        name="memory_allocation_size",
        value=memory_allocated,
        metric_type=MetricType.GAUGE,
        labels={"operation": "tensor_creation", "size": "1000x1000"},
        description="Memory allocated for tensor creation in bytes"
    )
    
    print(f"Memory allocated for tensor: {memory_allocated / (1024**2):.2f} MB")
    
    # Simulate memory deallocation
    del large_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Memory deallocation completed")


def simulate_batch_processing():
    """Simulate batch processing with metrics collection."""
    print("\n=== Simulating Batch Processing with Metrics Collection ===")
    
    collector = get_metrics_collector()
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        start_performance_tracking(f"batch_processing_{batch_size}")
        
        # Process batch
        inputs = torch.randn(batch_size, 768)
        model = ExampleModel()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        end_time = time.time()
        
        perf_result = end_performance_tracking(f"batch_processing_{batch_size}")
        
        # Add metrics for this batch size
        collector.add_metric(
            name="batch_processing_time",
            value=end_time - start_time,
            metric_type=MetricType.GAUGE,
            labels={"batch_size": str(batch_size), "operation": "forward_pass"},
            description="Time taken for batch processing"
        )
        
        collector.add_metric(
            name="throughput_samples_per_second",
            value=batch_size / (end_time - start_time),
            metric_type=MetricType.GAUGE,
            labels={"batch_size": str(batch_size)},
            description="Throughput in samples per second"
        )
        
        print(f"Batch size {batch_size}: {end_time - start_time:.4f}s, "
              f"Throughput: {batch_size / (end_time - start_time):.2f} samples/s")


def demonstrate_export_formats():
    """Demonstrate different export formats."""
    print("\n=== Demonstrating Export Formats ===")
    
    collector = get_metrics_collector()
    
    # Add some metrics for demonstration
    collector.add_metric("demo_counter", 42, MetricType.COUNTER, description="A demo counter")
    collector.add_metric("demo_gauge", 123.45, MetricType.GAUGE, description="A demo gauge")
    
    # Export to JSON
    print("JSON Export (first 300 characters):")
    json_data = collector.export_to_json()
    print(json_data[:300] + "..." if len(json_data) > 300 else json_data)
    
    # Export to Prometheus format
    print("\nPrometheus Export (first 300 characters):")
    prometheus_data = collector.export_to_prometheus()
    print(prometheus_data[:300] + "..." if len(prometheus_data) > 300 else prometheus_data)
    
    # Export to CSV (to a temporary file)
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        collector.export_to_csv(temp_path)
        print(f"\nCSV Export saved to: {temp_path}")
        
        # Read and display first few lines
        with open(temp_path, 'r') as f:
            lines = f.readlines()
            print("CSV Content (first 5 lines):")
            for i, line in enumerate(lines[:5]):
                print(f"  {line.strip()}")
    finally:
        os.unlink(temp_path)


def demonstrate_validation():
    """Demonstrate metrics validation."""
    print("\n=== Demonstrating Metrics Validation ===")
    
    collector = get_metrics_collector()
    
    # Register a validation rule for positive values
    def positive_value_validator(value):
        return isinstance(value, (int, float)) and value > 0
    
    collector.register_validation_rule("positive_metric", positive_value_validator)
    
    # Add valid metric
    collector.add_metric("positive_metric", 42, MetricType.GAUGE)
    is_valid = collector.validate_metric("positive_metric", 42)
    print(f"Valid metric validation: {is_valid}")
    
    # Try to add invalid metric
    try:
        collector.add_metric("positive_metric", -5, MetricType.GAUGE)
        is_valid_negative = collector.validate_metric("positive_metric", -5)
        print(f"Invalid metric validation: {is_valid_negative}")
    except ValueError as e:
        print(f"Validation caught invalid value: {e}")


def demonstrate_system_metrics():
    """Demonstrate automatic system metrics collection."""
    print("\n=== Demonstrating System Metrics Collection ===")
    
    collector = get_metrics_collector()
    
    # Collect system metrics
    collector._collect_system_metrics()
    
    # Show some collected metrics
    cpu_metric = collector.get_metric("system_cpu_percent")
    memory_metric = collector.get_metric("system_memory_percent")
    
    if cpu_metric:
        print(f"CPU Usage: {cpu_metric.value}%")
    if memory_metric:
        print(f"Memory Usage: {memory_metric.value}%")
    
    # Show all current system metrics
    all_metrics = collector.get_all_current_metrics()
    system_metrics = {k: v for k, v in all_metrics.items() if k.startswith('system_') or k.startswith('gpu_') or k.startswith('process_')}
    
    print(f"Total system metrics collected: {len(system_metrics)}")
    
    # Show first few system metrics
    for i, (name, metric) in enumerate(list(system_metrics.items())[:5]):
        print(f"  {name}: {metric.value} ({metric.labels})")


def main():
    """Main function demonstrating the metrics collection system."""
    print("Centralized Metrics Collection System - Example Usage")
    print("=" * 60)
    
    # Start metrics collection
    collector = get_metrics_collector()
    collector.start_collection()
    
    try:
        # Simulate various operations
        simulate_model_inference()
        simulate_memory_operations()
        simulate_batch_processing()
        demonstrate_system_metrics()
        demonstrate_validation()
        demonstrate_export_formats()
        
        # Show statistics
        stats = collector.get_statistics()
        print(f"\n=== Collector Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show total metrics collected
        print(f"\nTotal metrics in buffer: {len(collector.get_all_metrics_buffer())}")
        print(f"Current metrics count: {len(collector.get_all_current_metrics())}")
        
    finally:
        # Stop collection
        collector.stop_collection()
        print("\nMetrics collection stopped.")


if __name__ == "__main__":
    main()