#!/usr/bin/env python3
"""
Focused Real Performance Benchmark: GLM-4.7-Flash with Real Parameters
This script measures performance of the GLM-4.7-Flash model using its real parameters
and architecture by running real inference tasks.
"""

import time
import torch
import psutil
import os
from pathlib import Path
import json

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def measure_performance(model_func, input_data, num_iterations=3):
    """Measure performance of a model function."""
    # Warmup
    for _ in range(1):
        _ = model_func(input_data)
    
    # Memory measurement
    start_memory = get_memory_usage()
    
    # Timing run
    start_time = time.time()
    for i in range(num_iterations):
        result = model_func(input_data)
    end_time = time.time()
    
    end_memory = get_memory_usage()
    
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_iterations
    tokens_per_second = len(input_data) / avg_time_per_inference if avg_time_per_inference > 0 else float('inf')
    memory_used = end_memory - start_memory
    
    return {
        'total_time': total_time,
        'avg_time_per_inference': avg_time_per_inference,
        'tokens_per_second': tokens_per_second,
        'memory_used': memory_used,
        'num_iterations': num_iterations
    }

def create_glm_4_7_flash_real():
    """Create a real version of GLM-4.7-Flash model with actual parameters."""
    # Import the plugin
    from src.inference_pio.models.glm_4_7.plugin import create_glm_4_7_flash_plugin

    plugin = create_glm_4_7_flash_plugin()
    
    # Initialize with minimal optimizations
    success = plugin.initialize(
        device="cpu",
        torch_compile_mode=None,  # Disable torch.compile
        gradient_checkpointing=False,
        use_flash_attention_2=False,
        use_sparse_attention=False,
        use_kv_cache_compression=False,
        use_prefix_caching=False,
        max_new_tokens=16
    )
    
    if not success:
        return None, None
    
    model = plugin.load_model()
    return plugin, model

def create_glm_4_7_flash_optimized():
    """Create an optimized version of GLM-4.7-Flash model with real parameters."""
    # Import the plugin
    from src.inference_pio.models.glm_4_7.plugin import create_glm_4_7_flash_plugin

    plugin = create_glm_4_7_flash_plugin()
    
    # Initialize with all optimizations
    success = plugin.initialize(
        device="cpu",
        torch_compile_mode="reduce-overhead",  # Enable torch.compile
        gradient_checkpointing=True,
        use_flash_attention_2=True,
        use_sparse_attention=True,
        use_kv_cache_compression=True,
        use_prefix_caching=True,
        enable_predictive_management=True,
        enable_kernel_fusion=True,
        enable_distributed_simulation=True,
        enable_tensor_compression=True,
        enable_sharding=True,
        enable_disk_offloading=True,
        enable_model_surgery=True,
        enable_disk_pipeline=True,
        enable_activation_offloading=True,
        max_new_tokens=16
    )
    
    if not success:
        return None, None
    
    model = plugin.load_model()
    return plugin, model

def benchmark_model_pair(unoptimized_func, optimized_func, model_name, input_data):
    """Benchmark both unoptimized and optimized versions of a model."""
    print(f"\nBenchmarking {model_name}...")
    
    # Benchmark unoptimized version
    print(f"  Testing unoptimized version...")
    start_time = time.time()
    try:
        unopt_plugin, unopt_model = unoptimized_func()
        elapsed = time.time() - start_time
        print(f"    Model loaded in {elapsed:.2f}s")
        
        if unopt_plugin is None:
            print(f"    Failed to initialize unoptimized {model_name}")
            unopt_results = None
        else:
            unopt_results = measure_performance(
                lambda x: unopt_plugin.generate_text(x, max_new_tokens=8), 
                input_data
            )
            # Cleanup
            if hasattr(unopt_plugin, 'cleanup'):
                unopt_plugin.cleanup()
            print(f"    Unoptimized benchmark completed")
    except Exception as e:
        print(f"    Error with unoptimized {model_name}: {e}")
        unopt_results = None
    
    # Benchmark optimized version
    print(f"  Testing optimized version...")
    start_time = time.time()
    try:
        opt_plugin, opt_model = optimized_func()
        elapsed = time.time() - start_time
        print(f"    Model loaded in {elapsed:.2f}s")
        
        if opt_plugin is None:
            print(f"    Failed to initialize optimized {model_name}")
            opt_results = None
        else:
            opt_results = measure_performance(
                lambda x: opt_plugin.generate_text(x, max_new_tokens=8), 
                input_data
            )
            # Cleanup
            if hasattr(opt_plugin, 'cleanup'):
                opt_plugin.cleanup()
            print(f"    Optimized benchmark completed")
    except Exception as e:
        print(f"    Error with optimized {model_name}: {e}")
        opt_results = None
    
    return unopt_results, opt_results

def calculate_improvement(unopt_result, opt_result):
    """Calculate percentage improvement from unoptimized to optimized."""
    if unopt_result is None or opt_result is None:
        return None
    
    # Calculate improvements
    time_improvement = ((unopt_result['avg_time_per_inference'] - opt_result['avg_time_per_inference']) 
                        / unopt_result['avg_time_per_inference'] * 100)
    speed_improvement = ((opt_result['tokens_per_second'] - unopt_result['tokens_per_second']) 
                         / unopt_result['tokens_per_second'] * 100)
    memory_improvement = ((unopt_result['memory_used'] - opt_result['memory_used']) 
                          / unopt_result['memory_used'] * 100)
    
    return {
        'time_improvement_pct': time_improvement,
        'speed_improvement_pct': speed_improvement,
        'memory_improvement_pct': memory_improvement
    }

def run_glm_4_7_flash_benchmark():
    """Run benchmark for GLM-4.7-Flash with real parameters."""
    print("="*70)
    print("REAL PERFORMANCE BENCHMARK: GLM-4.7-Flash WITH REAL PARAMETERS")
    print("="*70)

    # Sample input data
    input_data = "The future of artificial intelligence"

    model_name = "GLM-4.7-Flash"
    # For real benchmark, we'll just test the optimized version with real parameters
    # since the "unoptimized" version would be the same model without optimizations
    opt_plugin, opt_model = create_glm_4_7_flash_optimized()
    opt_results = benchmark_model(opt_plugin, input_data, iterations=5)

    # Create a simulated "unoptimized" version for comparison purposes
    unopt_plugin, unopt_model = create_glm_4_7_flash_real()
    unopt_results = benchmark_model(unopt_plugin, input_data, iterations=5)
    
    improvement = calculate_improvement(unopt_results, opt_results)
    
    print(f"\n{model_name} Results:")
    print(f"Unoptimized:")
    if unopt_results:
        print(f"  Avg time per inference: {unopt_results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {unopt_results['tokens_per_second']:.2f}")
        print(f"  Memory used: {unopt_results['memory_used']:.2f}MB")
    else:
        print("  Failed to run")
    
    print(f"Optimized:")
    if opt_results:
        print(f"  Avg time per inference: {opt_results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {opt_results['tokens_per_second']:.2f}")
        print(f"  Memory used: {opt_results['memory_used']:.2f}MB")
    else:
        print("  Failed to run")
    
    if improvement:
        print(f"\n{model_name} Performance Improvements:")
        print(f"  Time Reduction: {improvement['time_improvement_pct']:.2f}%")
        print(f"  Speed Improvement: {improvement['speed_improvement_pct']:.2f}%")
        print(f"  Memory Improvement: {improvement['memory_improvement_pct']:.2f}%")
    else:
        print(f"\nCould not calculate improvements")
    
    # Save results
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'model': model_name,
        'unoptimized': unopt_results,
        'optimized': opt_results,
        'improvement': improvement
    }
    
    results_file = results_dir / f"real_{model_name.lower().replace('-', '_')}_optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = run_glm_4_7_flash_benchmark()

    print("\n" + "="*70)
    print("GLM-4.7-Flash REAL PERFORMANCE BENCHMARK WITH REAL PARAMETERS COMPLETED")
    print("="*70)
    print("Performance measured using the actual GLM-4.7-Flash model parameters and architecture.")
    print("Results saved in the 'benchmark_results' directory.")