"""
Focused benchmark module for optimization impact measurements.
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Callable
import json
import csv
from datetime import datetime


def benchmark_optimization_impact(
    original_model: torch.nn.Module, 
    optimized_model: torch.nn.Module, 
    input_data, 
    num_iterations: int = 100,
    warmup_iterations: int = 10
):
    """
    Benchmark the impact of optimizations by comparing original vs optimized models.
    
    Args:
        original_model: The original unoptimized model
        optimized_model: The optimized model
        input_data: Input data for both models
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary with benchmark results comparing both models
    """
    # Warmup for original model
    for i in range(warmup_iterations):
        with torch.no_grad():
            _ = original_model(input_data)
    
    # Benchmark original model
    start_time = time.time()
    for i in range(num_iterations):
        with torch.no_grad():
            _ = original_model(input_data)
    original_time = time.time() - start_time
    
    # Warmup for optimized model
    for i in range(warmup_iterations):
        with torch.no_grad():
            _ = optimized_model(input_data)
    
    # Benchmark optimized model
    start_time = time.time()
    for i in range(num_iterations):
        with torch.no_grad():
            _ = optimized_model(input_data)
    optimized_time = time.time() - start_time
    
    # Calculate improvement
    time_improvement = original_time - optimized_time
    improvement_percentage = (time_improvement / original_time) * 100 if original_time > 0 else 0
    
    results = {
        'benchmark_type': 'optimization_impact',
        'num_iterations': num_iterations,
        'warmup_iterations': warmup_iterations,
        'original_model_time_seconds': original_time,
        'optimized_model_time_seconds': optimized_time,
        'time_improvement_seconds': time_improvement,
        'improvement_percentage': improvement_percentage,
        'original_model_throughput_ips': num_iterations / original_time if original_time > 0 else 0,
        'optimized_model_throughput_ips': num_iterations / optimized_time if optimized_time > 0 else 0,
        'throughput_improvement_factor': (num_iterations / optimized_time) / (num_iterations / original_time) if original_time > 0 and optimized_time > 0 else 1.0,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def benchmark_optimization_memory_impact(
    original_model: torch.nn.Module, 
    optimized_model: torch.nn.Module, 
    input_data
):
    """
    Benchmark the memory impact of optimizations.
    
    Args:
        original_model: The original unoptimized model
        optimized_model: The optimized model
        input_data: Input data for both models
    
    Returns:
        Dictionary with memory benchmark results comparing both models
    """
    import psutil
    import os
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
        
        return {
            'ram_bytes': memory_info.rss,
            'ram_mb': memory_info.rss / (1024 * 1024),
            'gpu_memory_bytes': gpu_memory,
            'gpu_memory_mb': gpu_memory / (1024 * 1024) if gpu_memory > 0 else 0
        }
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    
    # Measure memory after loading original model
    _ = original_model(input_data)
    original_memory = get_memory_usage()
    
    # Measure memory after loading optimized model
    _ = optimized_model(input_data)
    optimized_memory = get_memory_usage()
    
    # Calculate differences from initial
    original_ram_diff = original_memory['ram_mb'] - initial_memory['ram_mb']
    optimized_ram_diff = optimized_memory['ram_mb'] - initial_memory['ram_mb']
    original_gpu_diff = original_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb']
    optimized_gpu_diff = optimized_memory['gpu_memory_mb'] - initial_memory['gpu_memory_mb']
    
    # Calculate improvements
    ram_improvement = original_ram_diff - optimized_ram_diff
    gpu_improvement = original_gpu_diff - optimized_gpu_diff
    ram_improvement_pct = (ram_improvement / original_ram_diff) * 100 if original_ram_diff != 0 else 0
    gpu_improvement_pct = (gpu_improvement / original_gpu_diff) * 100 if original_gpu_diff != 0 else 0
    
    results = {
        'benchmark_type': 'optimization_memory_impact',
        'original_model_ram_increase_mb': original_ram_diff,
        'optimized_model_ram_increase_mb': optimized_ram_diff,
        'ram_improvement_mb': ram_improvement,
        'ram_improvement_percentage': ram_improvement_pct,
        'original_model_gpu_increase_mb': original_gpu_diff,
        'optimized_model_gpu_increase_mb': optimized_gpu_diff,
        'gpu_improvement_mb': gpu_improvement,
        'gpu_improvement_percentage': gpu_improvement_pct,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def run_optimization_impact_benchmarks(
    original_model: torch.nn.Module, 
    optimized_model: torch.nn.Module, 
    config: Dict[str, Any]
):
    """
    Run a series of optimization impact benchmarks.
    
    Args:
        original_model: The original unoptimized model
        optimized_model: The optimized model
        config: Configuration dictionary with benchmark parameters
    
    Returns:
        Dictionary with all benchmark results
    """
    print("Running optimization impact benchmarks...")
    
    # Get configuration parameters
    input_sizes = config.get('input_sizes', [1, 4, 8])
    sequence_lengths = config.get('sequence_lengths', [64, 128, 256])
    num_iterations = config.get('num_iterations', 100)
    warmup_iterations = config.get('warmup_iterations', 10)
    
    detailed_results = []
    
    for size in input_sizes:
        for seq_len in sequence_lengths:
            # Create input data with specified size and sequence length
            input_data = torch.randn(size, seq_len)
            
            # Run performance benchmark
            perf_result = benchmark_optimization_impact(
                original_model, optimized_model, input_data, 
                num_iterations=num_iterations, warmup_iterations=warmup_iterations
            )
            perf_result['input_size'] = size
            perf_result['sequence_length'] = seq_len
            
            # Run memory benchmark
            mem_result = benchmark_optimization_memory_impact(
                original_model, optimized_model, input_data
            )
            mem_result['input_size'] = size
            mem_result['sequence_length'] = seq_len
            
            detailed_results.append({
                'performance': perf_result,
                'memory': mem_result
            })
    
    # Calculate aggregate statistics
    perf_results = [r['performance'] for r in detailed_results]
    time_improvements = [r['improvement_percentage'] for r in perf_results]
    throughput_factors = [r['throughput_improvement_factor'] for r in perf_results]
    
    mem_results = [r['memory'] for r in detailed_results]
    ram_improvements = [r['ram_improvement_percentage'] for r in mem_results]
    gpu_improvements = [r['gpu_improvement_percentage'] for r in mem_results]
    
    aggregate_results = {
        'benchmark_type': 'optimization_impact_aggregate',
        'num_configurations_tested': len(detailed_results),
        'avg_performance_improvement_percentage': np.mean(time_improvements),
        'std_performance_improvement_percentage': np.std(time_improvements),
        'min_performance_improvement_percentage': np.min(time_improvements),
        'max_performance_improvement_percentage': np.max(time_improvements),
        'avg_throughput_improvement_factor': np.mean(throughput_factors),
        'std_throughput_improvement_factor': np.std(throughput_factors),
        'avg_ram_improvement_percentage': np.mean(ram_improvements),
        'avg_gpu_improvement_percentage': np.mean(gpu_improvements),
        'timestamp': datetime.now().isoformat(),
        'detailed_results': detailed_results
    }
    
    return aggregate_results


def save_optimization_impact_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Save optimization impact benchmark results to files.
    
    Args:
        results: Results dictionary to save
        output_dir: Directory to save results to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_filename = os.path.join(output_dir, f"optimization_impact_benchmark_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV summary
    csv_filename = os.path.join(output_dir, f"optimization_impact_benchmark_summary_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Benchmark Type', 'Num Configurations Tested', 'Avg Perf Improvement %', 
            'Std Perf Improvement %', 'Min Perf Improvement %', 'Max Perf Improvement %',
            'Avg Throughput Improvement Factor', 'Std Throughput Improvement Factor',
            'Avg RAM Improvement %', 'Avg GPU Improvement %', 'Timestamp'
        ])
        
        # Write summary row
        writer.writerow([
            results.get('benchmark_type', ''),
            results.get('num_configurations_tested', 0),
            results.get('avg_performance_improvement_percentage', 0),
            results.get('std_performance_improvement_percentage', 0),
            results.get('min_performance_improvement_percentage', 0),
            results.get('max_performance_improvement_percentage', 0),
            results.get('avg_throughput_improvement_factor', 0),
            results.get('std_throughput_improvement_factor', 0),
            results.get('avg_ram_improvement_percentage', 0),
            results.get('avg_gpu_improvement_percentage', 0),
            results.get('timestamp', '')
        ])
    
    print(f"Optimization impact benchmark results saved to: {json_filename} and {csv_filename}")


if __name__ == "__main__":
    # Example usage
    print("Example usage of optimization impact benchmark module")
    
    # Create dummy models for demonstration
    class OriginalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128)
        
        def forward(self, x):
            return self.linear(x)
    
    class OptimizedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128)
        
        def forward(self, x):
            return self.linear(x)
    
    original_model = OriginalModel()
    optimized_model = OptimizedModel()
    
    # Configuration for benchmarks
    config = {
        'input_sizes': [1, 4],
        'sequence_lengths': [64, 128],
        'num_iterations': 50,
        'warmup_iterations': 5
    }
    
    # Run benchmarks
    results = run_optimization_impact_benchmarks(original_model, optimized_model, config)
    
    # Save results
    save_optimization_impact_results(results)
    
    print("Optimization impact benchmark example completed.")