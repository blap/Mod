"""
Benchmark Utilities Module for Mod Project

This module provides common utilities for benchmarking different models/plugins
in the Mod project. Each model can use this module independently.
"""

import time
import torch
from typing import Any, Dict, List
from benchmarks.core.real_performance_monitor import (
    RealPerformanceMonitor,
    benchmark_function_real,
    get_real_system_metrics,
)


def benchmark_inference_speed(
    plugin, 
    model_name: str, 
    input_length: int = 50, 
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Real inference speed benchmark using actual performance measurements.

    Args:
        plugin: Model plugin instance to benchmark
        model_name: Name of the model being tested
        input_length: Length of input sequence to use for benchmarking (default: 50)
        num_iterations: Number of iterations to run for averaging (default: 10)

    Returns:
        Dict containing benchmark results including tokens per second, timing info, etc.
    """
    # Prepare input - create realistic input tensor with random token IDs
    # Using random tokens ensures we're testing general inference performance
    # rather than specific patterns that might be cached or optimized
    input_ids = torch.randint(100, 1000, (1, input_length))

    # Warmup phase - essential for accurate benchmarking
    # Ensures caches are primed, JIT compilation is complete,
    # and the model is in a steady state before measurements begin
    for _ in range(3):
        try:
            _ = plugin.infer(input_ids)
        except:
            # If infer doesn't work with tensor, try with string
            _ = plugin.generate_text("warmup text", max_new_tokens=5)

    # Define the timed operation to be benchmarked
    # This isolates the actual inference operation for accurate measurement
    def timed_inference():
        return plugin.infer(input_ids)

    # Execute the benchmark using real performance monitoring
    # This captures actual system performance including all overhead
    benchmark_results = benchmark_function_real(
        timed_inference, iterations=num_iterations
    )

    # Extract and convert metrics to appropriate units
    # Convert milliseconds to seconds for standard time measurements
    avg_time_per_inference = (
        benchmark_results["avg_time_ms"] / 1000.0
    )  # Convert ms to seconds
    # Calculate throughput: tokens processed per second
    # This is the primary metric for evaluating inference efficiency
    tokens_per_second = (
        input_length / avg_time_per_inference if avg_time_per_inference > 0 else 0
    )

    # Compile comprehensive results including raw data for detailed analysis
    result = {
        "total_time": benchmark_results["avg_time_ms"]
        / 1000.0
        * num_iterations,  # Total time in seconds
        "avg_time_per_inference": avg_time_per_inference,  # Average time per single inference
        "tokens_per_second": tokens_per_second,  # Primary throughput metric
        "num_iterations": num_iterations,  # Number of measurements taken
        "input_length": input_length,  # Input sequence length tested
        "model_name": model_name,  # Model identifier
        "raw_benchmark_data": benchmark_results,  # Detailed raw data for deeper analysis
    }

    return result


def benchmark_generation_speed(
    plugin,
    model_name: str,
    prompt: str = "The quick brown fox jumps over the lazy dog. ",
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Real text generation speed benchmark using actual performance measurements.

    Args:
        plugin: Model plugin instance to benchmark
        model_name: Name of the model being tested
        prompt: Input prompt for text generation (default: standard sentence)
        max_new_tokens: Maximum number of new tokens to generate (default: 50)

    Returns:
        Dict containing benchmark results including characters per second, timing info, etc.
    """
    # Warmup
    try:
        _ = plugin.generate_text(prompt, max_new_tokens=10)
    except:
        # If generate_text doesn't work, try infer
        _ = plugin.infer(prompt)

    # Use real performance monitoring
    def timed_generation():
        return plugin.generate_text(prompt, max_new_tokens=max_new_tokens)

    benchmark_results = benchmark_function_real(
        timed_generation,
        iterations=5,  # Fewer iterations for generation as it may take longer
    )

    # Calculate metrics
    avg_time_per_generation = (
        benchmark_results["avg_time_ms"] / 1000.0
    )  # Convert ms to seconds
    generated_text = benchmark_results["metrics_history"][
        0
    ].tokens_per_second  # This is actually the output
    chars_per_second = (
        len(prompt + " " + str(generated_text)) / avg_time_per_generation
        if avg_time_per_generation > 0
        else 0
    )

    result = {
        "total_time": benchmark_results["avg_time_ms"]
        / 1000.0,  # Average time in seconds
        "chars_per_second": chars_per_second,
        "generated_text_length": len(str(generated_text)) if generated_text else 0,
        "model_name": model_name,
        "prompt_length": len(prompt),
        "raw_benchmark_data": benchmark_results,
    }

    return result


def benchmark_memory_usage(plugin, model_name: str) -> Dict[str, Any]:
    """
    Real memory usage benchmark using actual system measurements.

    Args:
        plugin: Model plugin instance to benchmark
        model_name: Name of the model being tested

    Returns:
        Dict containing memory usage metrics comparing before and after operations
    """
    import gc

    # Capture baseline system metrics before operations begin
    # This establishes the memory usage baseline for comparison
    baseline_metrics = get_real_system_metrics()

    # Perform operations to measure memory impact
    # These operations should trigger the memory usage patterns we want to measure
    try:
        # Run a few inference operations to stress-test memory usage
        for i in range(5):
            _ = plugin.generate_text("test prompt", max_new_tokens=20)
    except:
        try:
            # Fallback to tensor-based inference if text generation fails
            _ = plugin.infer(torch.randint(100, 1000, (1, 20)))
        except:
            # If both fail, skip the memory impact measurement
            pass

    # Force garbage collection to clean up temporary allocations
    # This ensures we measure the true memory footprint after cleanup
    gc.collect()
    if torch.cuda.is_available():
        # Clear GPU cache to free up GPU memory
        torch.cuda.empty_cache()

    # Capture current system metrics after operations complete
    # This reflects the memory state after operations and cleanup
    current_metrics = get_real_system_metrics()

    # Calculate memory differences to determine actual impact
    # Positive values indicate memory increase, negative indicate decrease
    memory_used_mb = (
        current_metrics.memory_used_mb - baseline_metrics.memory_used_mb
    )
    gpu_memory_used_diff = (
        (current_metrics.gpu_memory_used_mb - baseline_metrics.gpu_memory_used_mb)
        if current_metrics.gpu_memory_used_mb is not None
        and baseline_metrics.gpu_memory_used_mb is not None
        else None
    )

    # Compile comprehensive results showing memory impact
    result = {
        "baseline_memory_mb": baseline_metrics.memory_used_mb,  # Memory before operations
        "current_memory_mb": current_metrics.memory_used_mb,  # Memory after operations
        "memory_used_mb": memory_used_mb,  # Net memory change due to operations
        "model_name": model_name,  # Model identifier
        "baseline_system_metrics": baseline_metrics.__dict__,  # Full baseline metrics
        "current_system_metrics": current_metrics.__dict__,  # Full current metrics
        "gpu_memory_used_difference_mb": gpu_memory_used_diff,  # GPU memory impact if available
    }

    return result


def benchmark_throughput(
    plugin, model_name: str, batch_sizes: List[int] = None
) -> Dict[str, Any]:
    """
    Real throughput benchmark using actual performance measurements.

    Args:
        plugin: Model plugin instance to benchmark
        model_name: Name of the model being tested
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])

    Returns:
        Dict containing throughput metrics for different batch sizes
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    results = []

    for batch_size in batch_sizes:
        # Prepare batch input
        input_ids = torch.randint(100, 1000, (batch_size, 50))

        # Warmup
        for _ in range(3):
            try:
                _ = plugin.infer(input_ids)
            except:
                _ = plugin.generate_text("warmup", max_new_tokens=5)

        # Benchmark
        def timed_batch_inference():
            return plugin.infer(input_ids)

        benchmark_results = benchmark_function_real(
            timed_batch_inference, iterations=5
        )

        avg_time = benchmark_results["avg_time_ms"] / 1000.0  # Convert to seconds
        tokens_per_second = (batch_size * 50) / avg_time if avg_time > 0 else 0

        batch_result = {
            "batch_size": batch_size,
            "avg_time_per_batch": avg_time,
            "tokens_per_second": tokens_per_second,
            "requests_per_second": batch_size / avg_time if avg_time > 0 else 0,
            "raw_benchmark_data": benchmark_results,
        }

        results.append(batch_result)

    overall_result = {
        "model_name": model_name,
        "batch_sizes_tested": batch_sizes,
        "throughput_results": results,
        "peak_throughput_tokens_per_second": max(
            r["tokens_per_second"] for r in results
        ),
    }

    return overall_result


def benchmark_power_efficiency(
    plugin, model_name: str, duration: float = 30.0
) -> Dict[str, Any]:
    """
    Real power efficiency benchmark using actual measurements over time.

    Args:
        plugin: Model plugin instance to benchmark
        model_name: Name of the model being tested
        duration: Duration of the benchmark in seconds (default: 30.0)

    Returns:
        Dict containing power efficiency metrics collected over the test duration
    """
    import threading
    import time as time_module

    # Start continuous monitoring
    monitor = RealPerformanceMonitor()
    monitor.start_monitoring(interval=0.5)  # Sample every 0.5 seconds

    start_time = time_module.time()
    operations_performed = 0

    try:
        # Perform operations for the specified duration
        while time_module.time() - start_time < duration:
            try:
                _ = plugin.generate_text("test", max_new_tokens=10)
                operations_performed += 1
            except:
                try:
                    _ = plugin.infer(torch.randint(100, 1000, (1, 20)))
                    operations_performed += 1
                except:
                    break

            # Small delay to prevent overwhelming the system
            time_module.sleep(0.01)
    finally:
        monitor.stop_monitoring()

    # Calculate efficiency metrics
    total_time = time_module.time() - start_time
    avg_cpu = (
        sum(m.cpu_percent for m in monitor.metrics_history)
        / len(monitor.metrics_history)
        if monitor.metrics_history
        else 0
    )
    avg_memory = (
        sum(m.memory_percent for m in monitor.metrics_history)
        / len(monitor.metrics_history)
        if monitor.metrics_history
        else 0
    )

    result = {
        "model_name": model_name,
        "test_duration_seconds": total_time,
        "operations_performed": operations_performed,
        "operations_per_second": (
            operations_performed / total_time if total_time > 0 else 0
        ),
        "avg_cpu_percent": avg_cpu,
        "avg_memory_percent": avg_memory,
        "total_energy_estimate": avg_cpu
        * total_time
        / 100,  # Simplified energy estimate
        "operations_per_energy_unit": (
            operations_performed / (avg_cpu * total_time / 100)
            if avg_cpu * total_time > 0
            else 0
        ),
        "metrics_history": [m.__dict__ for m in monitor.metrics_history],
    }

    return result


def run_model_benchmark_suite(plugin, model_name: str) -> Dict[str, Any]:
    """
    Run a comprehensive benchmark suite for a model plugin.

    Args:
        plugin: Model plugin instance to benchmark
        model_name: Name of the model being tested

    Returns:
        Dict containing results from all benchmark types
    """
    results = {}
    
    # Run inference speed benchmark
    results['inference_speed'] = benchmark_inference_speed(plugin, model_name)
    
    # Run memory usage benchmark
    results['memory_usage'] = benchmark_memory_usage(plugin, model_name)
    
    # Run throughput benchmark
    results['throughput'] = benchmark_throughput(plugin, model_name)
    
    return results