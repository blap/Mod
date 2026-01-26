"""
Focused benchmark module for memory usage measurements.
"""

import psutil
import torch
import gc
from typing import Dict, Any
import json
import csv
from datetime import datetime
import os


def get_memory_usage():
    """
    Get current memory usage of the process.
    
    Returns:
        Dictionary with memory usage information
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # If CUDA is available, also get GPU memory
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated()
    
    return {
        'ram_bytes': memory_info.rss,
        'ram_mb': memory_info.rss / (1024 * 1024),
        'gpu_memory_bytes': gpu_memory,
        'gpu_memory_mb': gpu_memory / (1024 * 1024) if gpu_memory > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }


def benchmark_memory_usage(model, input_data, num_iterations=100):
    """
    Benchmark the memory usage of a model during inference.
    
    Args:
        model: The model to benchmark
        input_data: Input data for the model
        num_iterations: Number of iterations to run
    
    Returns:
        Dictionary with benchmark results
    """
    # Initial memory usage
    initial_memory = get_memory_usage()
    
    # Run inference iterations
    for i in range(num_iterations):
        with torch.no_grad():
            _ = model(input_data)
        
        # Clear cache periodically to prevent accumulation
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final memory usage
    final_memory = get_memory_usage()
    
    # Calculate differences
    memory_diff = {
        'ram_increase_bytes': final_memory['ram_bytes'] - initial_memory['ram_bytes'],
        'ram_increase_mb': final_memory['ram_mb'] - initial_memory['ram_mb'],
        'gpu_memory_increase_bytes': final_memory['gpu_memory_bytes'] - initial_memory['gpu_memory_bytes'],
        'gpu_memory_increase_mb': final_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb']
    }
    
    results = {
        'benchmark_type': 'memory_usage',
        'num_iterations': num_iterations,
        'initial_memory': initial_memory,
        'final_memory': final_memory,
        'memory_difference': memory_diff,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def benchmark_peak_memory_usage(model, input_data, num_iterations=100):
    """
    Benchmark the peak memory usage of a model during inference.
    
    Args:
        model: The model to benchmark
        input_data: Input data for the model
        num_iterations: Number of iterations to run
    
    Returns:
        Dictionary with benchmark results
    """
    # Track peak memory usage
    peak_ram = get_memory_usage()['ram_mb']
    peak_gpu_memory = get_memory_usage()['gpu_memory_mb']
    
    for i in range(num_iterations):
        with torch.no_grad():
            _ = model(input_data)
        
        # Check memory usage after each iteration
        current_memory = get_memory_usage()
        peak_ram = max(peak_ram, current_memory['ram_mb'])
        peak_gpu_memory = max(peak_gpu_memory, current_memory['gpu_memory_mb'])
        
        # Clear cache periodically
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    results = {
        'benchmark_type': 'peak_memory_usage',
        'num_iterations': num_iterations,
        'peak_ram_mb': peak_ram,
        'peak_gpu_memory_mb': peak_gpu_memory,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def benchmark_memory_usage_with_various_inputs(model, input_sizes: list, sequence_lengths: list):
    """
    Benchmark memory usage with various input configurations.
    
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
            
            # Run memory benchmark
            result = benchmark_memory_usage(model, input_data, num_iterations=50)
            result['input_size'] = size
            result['sequence_length'] = seq_len
            
            results.append(result)
    
    return results


def run_memory_benchmarks(model, config: Dict[str, Any]):
    """
    Run a series of memory benchmarks.
    
    Args:
        model: The model to benchmark
        config: Configuration dictionary with benchmark parameters
    
    Returns:
        Dictionary with all benchmark results
    """
    print("Running memory usage benchmarks...")
    
    # Get configuration parameters
    input_sizes = config.get('input_sizes', [1, 4, 8])
    sequence_lengths = config.get('sequence_lengths', [64, 128, 256])
    num_iterations = config.get('num_iterations', 100)
    
    # Run detailed memory benchmarks
    detailed_results = benchmark_memory_usage_with_various_inputs(
        model, input_sizes, sequence_lengths
    )
    
    # Calculate aggregate statistics
    ram_increases = [r['memory_difference']['ram_increase_mb'] for r in detailed_results]
    gpu_memory_increases = [r['memory_difference']['gpu_memory_increase_mb'] for r in detailed_results]
    
    aggregate_results = {
        'benchmark_type': 'memory_usage_aggregate',
        'num_configurations_tested': len(detailed_results),
        'avg_ram_increase_mb': sum(ram_increases) / len(ram_increases) if ram_increases else 0,
        'max_ram_increase_mb': max(ram_increases) if ram_increases else 0,
        'min_ram_increase_mb': min(ram_increases) if ram_increases else 0,
        'avg_gpu_memory_increase_mb': sum(gpu_memory_increases) / len(gpu_memory_increases) if gpu_memory_increases else 0,
        'max_gpu_memory_increase_mb': max(gpu_memory_increases) if gpu_memory_increases else 0,
        'min_gpu_memory_increase_mb': min(gpu_memory_increases) if gpu_memory_increases else 0,
        'timestamp': datetime.now().isoformat(),
        'detailed_results': detailed_results
    }
    
    return aggregate_results


def save_memory_benchmark_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Save memory benchmark results to files.
    
    Args:
        results: Results dictionary to save
        output_dir: Directory to save results to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_filename = os.path.join(output_dir, f"memory_benchmark_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV summary
    csv_filename = os.path.join(output_dir, f"memory_benchmark_summary_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Benchmark Type', 'Num Configurations Tested', 'Avg RAM Increase MB', 
            'Max RAM Increase MB', 'Min RAM Increase MB', 'Avg GPU Memory Increase MB',
            'Max GPU Memory Increase MB', 'Min GPU Memory Increase MB', 'Timestamp'
        ])
        
        # Write summary row
        writer.writerow([
            results.get('benchmark_type', ''),
            results.get('num_configurations_tested', 0),
            results.get('avg_ram_increase_mb', 0),
            results.get('max_ram_increase_mb', 0),
            results.get('min_ram_increase_mb', 0),
            results.get('avg_gpu_memory_increase_mb', 0),
            results.get('max_gpu_memory_increase_mb', 0),
            results.get('min_gpu_memory_increase_mb', 0),
            results.get('timestamp', '')
        ])
    
    print(f"Memory benchmark results saved to: {json_filename} and {csv_filename}")


if __name__ == "__main__":
    # Example usage
    print("Example usage of memory usage benchmark module")
    
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
        'num_iterations': 50
    }
    
    # Run benchmarks
    results = run_memory_benchmarks(dummy_model, config)
    
    # Save results
    save_memory_benchmark_results(results)
    
    print("Memory benchmark example completed.")