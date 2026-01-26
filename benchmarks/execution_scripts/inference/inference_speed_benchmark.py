"""
Focused benchmark module for inference speed measurements.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any
import json
import csv
from datetime import datetime


def benchmark_inference_speed(model, input_data, num_iterations=100, warmup_iterations=10):
    """
    Benchmark the inference speed of a model.
    
    Args:
        model: The model to benchmark
        input_data: Input data for the model
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary with benchmark results
    """
    # Warmup iterations
    for i in range(warmup_iterations):
        with torch.no_grad():
            _ = model(input_data)
    
    # Actual benchmark
    start_time = time.time()
    for i in range(num_iterations):
        with torch.no_grad():
            _ = model(input_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    
    results = {
        'benchmark_type': 'inference_speed',
        'num_iterations': num_iterations,
        'warmup_iterations': warmup_iterations,
        'total_time_seconds': total_time,
        'avg_time_per_iteration_seconds': avg_time,
        'throughput_iterations_per_second': throughput,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def benchmark_inference_speed_with_various_inputs(model, input_sizes: List[int], sequence_lengths: List[int]):
    """
    Benchmark inference speed with various input configurations.
    
    Args:
        model: The model to benchmark
        input_sizes: List of input sizes to test
        sequence_lengths: List of sequence lengths to test
    
    Returns:
        List of benchmark results for different configurations
    """
    results = []
    
    for size in input_sizes:
        for seq_len in sequence_lengths:
            # Create input data with specified size and sequence length
            input_data = torch.randn(size, seq_len)
            
            result = benchmark_inference_speed(model, input_data, num_iterations=50, warmup_iterations=5)
            result['input_size'] = size
            result['sequence_length'] = seq_len
            
            results.append(result)
    
    return results


def run_inference_benchmarks(model, config: Dict[str, Any]):
    """
    Run a series of inference benchmarks.
    
    Args:
        model: The model to benchmark
        config: Configuration dictionary with benchmark parameters
    
    Returns:
        Dictionary with all benchmark results
    """
    print("Running inference speed benchmarks...")
    
    # Get configuration parameters
    input_sizes = config.get('input_sizes', [1, 4, 8])
    sequence_lengths = config.get('sequence_lengths', [64, 128, 256])
    num_iterations = config.get('num_iterations', 100)
    warmup_iterations = config.get('warmup_iterations', 10)
    
    # Run benchmarks with various inputs
    detailed_results = benchmark_inference_speed_with_various_inputs(
        model, input_sizes, sequence_lengths
    )
    
    # Calculate aggregate statistics
    avg_times = [r['avg_time_per_iteration_seconds'] for r in detailed_results]
    throughputs = [r['throughput_iterations_per_second'] for r in detailed_results]
    
    aggregate_results = {
        'benchmark_type': 'inference_speed_aggregate',
        'num_configurations_tested': len(detailed_results),
        'avg_avg_time_per_iteration': np.mean(avg_times),
        'std_avg_time_per_iteration': np.std(avg_times),
        'min_avg_time_per_iteration': np.min(avg_times),
        'max_avg_time_per_iteration': np.max(avg_times),
        'avg_throughput': np.mean(throughputs),
        'std_throughput': np.std(throughputs),
        'min_throughput': np.min(throughputs),
        'max_throughput': np.max(throughputs),
        'timestamp': datetime.now().isoformat(),
        'detailed_results': detailed_results
    }
    
    return aggregate_results


def save_inference_benchmark_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Save inference benchmark results to files.
    
    Args:
        results: Results dictionary to save
        output_dir: Directory to save results to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_filename = os.path.join(output_dir, f"inference_benchmark_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV summary
    csv_filename = os.path.join(output_dir, f"inference_benchmark_summary_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Benchmark Type', 'Num Configurations Tested', 'Avg Avg Time', 
            'Std Avg Time', 'Min Avg Time', 'Max Avg Time',
            'Avg Throughput', 'Std Throughput', 'Min Throughput', 
            'Max Throughput', 'Timestamp'
        ])
        
        # Write summary row
        writer.writerow([
            results.get('benchmark_type', ''),
            results.get('num_configurations_tested', 0),
            results.get('avg_avg_time_per_iteration', 0),
            results.get('std_avg_time_per_iteration', 0),
            results.get('min_avg_time_per_iteration', 0),
            results.get('max_avg_time_per_iteration', 0),
            results.get('avg_throughput', 0),
            results.get('std_throughput', 0),
            results.get('min_throughput', 0),
            results.get('max_throughput', 0),
            results.get('timestamp', '')
        ])
    
    print(f"Inference benchmark results saved to: {json_filename} and {csv_filename}")


if __name__ == "__main__":
    # Example usage
    print("Example usage of inference speed benchmark module")
    
    # Create a dummy model for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128)
        
        def forward(self, x):
            return self.linear(x)
    
    dummy_model = DummyModel()
    
    # Configuration for benchmarks
    config = {
        'input_sizes': [1, 4],
        'sequence_lengths': [64, 128],
        'num_iterations': 50,
        'warmup_iterations': 5
    }
    
    # Run benchmarks
    results = run_inference_benchmarks(dummy_model, config)
    
    # Save results
    save_inference_benchmark_results(results)
    
    print("Inference benchmark example completed.")