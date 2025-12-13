"""
Performance Benchmarking Test for Qwen3-VL-2B-Instruct Architecture
This test benchmarks performance against the original baseline to validate improvements.
"""
import sys
import os
import torch
import time
import gc
import numpy as np
from typing import Dict, Any, Tuple
import psutil
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager, MemoryConfig
from kv_cache_optimizer import KVCacheConfig, OptimizedKVCacheManager
from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def create_baseline_model():
    """Create a baseline model without optimizations for comparison"""
    config = Qwen3VLConfig()
    config.use_sparsity = False
    config.use_gradient_checkpointing = False
    config.hidden_size = 512  # Use smaller size for testing
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_hidden_layers = 8
    config.vocab_size = 1000
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model, config


def create_optimized_model():
    """Create an optimized model with all optimizations enabled"""
    config = Qwen3VLConfig()
    config.use_sparsity = True
    config.sparsity_ratio = 0.5
    config.exit_threshold = 0.75
    config.use_gradient_checkpointing = True
    config.hidden_size = 512  # Use smaller size for testing
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_hidden_layers = 8
    config.vocab_size = 1000
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Integrate memory manager
    memory_manager = MemoryManager(MemoryConfig(memory_pool_size=2**25))  # 32MB pool
    # Integrate KV cache optimizer
    kv_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=64,
        use_sliding_window=True,
        sliding_window_size=512,
        use_hybrid=True
    )
    kv_cache_manager = OptimizedKVCacheManager(kv_config, memory_manager)
    
    # Note: In a real implementation, these would be integrated during model construction
    model.memory_manager = memory_manager
    model.kv_cache_manager = kv_cache_manager
    
    return model, config


def benchmark_inference_performance():
    """Benchmark inference performance"""
    print("Benchmarking inference performance...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Define test scenarios
    test_scenarios = [
        {"batch_size": 1, "seq_len": 16, "name": "Small batch, short sequence"},
        {"batch_size": 2, "seq_len": 32, "name": "Medium batch, medium sequence"},
        {"batch_size": 1, "seq_len": 64, "name": "Small batch, long sequence"},
        {"batch_size": 4, "seq_len": 16, "name": "Large batch, short sequence"},
    ]
    
    results = {
        "baseline_times": [],
        "optimized_times": [],
        "scenarios": [],
        "improvements": []
    }
    
    for scenario in test_scenarios:
        batch_size = scenario["batch_size"]
        seq_len = scenario["seq_len"]
        name = scenario["name"]
        
        print(f"  Testing: {name} (batch={batch_size}, seq_len={seq_len})")
        
        # Create inputs for this scenario
        input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
        
        # Warm up both models
        with torch.no_grad():
            for _ in range(3):
                _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
                _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Benchmark baseline model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        baseline_times = []
        for _ in range(10):  # Run 10 times for average
            start_time = time.time()
            with torch.no_grad():
                _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
            baseline_times.append(time.time() - start_time)
        
        avg_baseline_time = np.mean(baseline_times)
        
        # Benchmark optimized model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        optimized_times = []
        for _ in range(10):  # Run 10 times for average
            start_time = time.time()
            with torch.no_grad():
                _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
            optimized_times.append(time.time() - start_time)
        
        avg_optimized_time = np.mean(optimized_times)
        
        # Calculate improvement
        improvement = (avg_baseline_time - avg_optimized_time) / avg_baseline_time * 100 if avg_baseline_time > 0 else 0
        
        results["baseline_times"].append(avg_baseline_time)
        results["optimized_times"].append(avg_optimized_time)
        results["scenarios"].append(name)
        results["improvements"].append(improvement)
        
        print(f"    Baseline: {avg_baseline_time:.4f}s")
        print(f"    Optimized: {avg_optimized_time:.4f}s")
        print(f"    Improvement: {improvement:.2f}%")
    
    return results


def benchmark_memory_efficiency():
    """Benchmark memory efficiency improvements"""
    print("\nBenchmarking memory efficiency...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    # Measure baseline memory usage
    baseline_memory_before = psutil.virtual_memory().percent
    
    with torch.no_grad():
        _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
    
    baseline_memory_after = psutil.virtual_memory().percent
    baseline_memory_used = baseline_memory_after - baseline_memory_before
    
    # Measure optimized memory usage
    optimized_memory_before = psutil.virtual_memory().percent
    
    with torch.no_grad():
        _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
    
    optimized_memory_after = psutil.virtual_memory().percent
    optimized_memory_used = optimized_memory_after - optimized_memory_before
    
    # Calculate memory efficiency improvement
    if baseline_memory_used != 0:
        memory_improvement = (baseline_memory_used - optimized_memory_used) / baseline_memory_used * 100
    else:
        memory_improvement = 0
    
    print(f"  Baseline memory usage: {baseline_memory_used:.2f}%")
    print(f"  Optimized memory usage: {optimized_memory_used:.2f}%")
    print(f"  Memory efficiency improvement: {memory_improvement:.2f}%")
    
    # If using CUDA, also measure GPU memory
    if torch.cuda.is_available():
        print(f"  GPU memory - Baseline allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  GPU memory - Baseline reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
        # Clear cache and run optimized model
        torch.cuda.empty_cache()
        with torch.no_grad():
            _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        
        print(f"  GPU memory - Optimized allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  GPU memory - Optimized reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    return memory_improvement


def benchmark_throughput():
    """Benchmark throughput improvements"""
    print("\nBenchmarking throughput...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
            _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Measure baseline throughput
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    baseline_start = time.time()
    baseline_processed = 0
    time_limit = 10.0  # Run for 10 seconds
    
    while time.time() - baseline_start < time_limit:
        with torch.no_grad():
            _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
        baseline_processed += 1
    
    baseline_time = time.time() - baseline_start
    baseline_throughput = baseline_processed / baseline_time
    
    # Measure optimized throughput
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    optimized_start = time.time()
    optimized_processed = 0
    
    while time.time() - optimized_start < time_limit:
        with torch.no_grad():
            _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        optimized_processed += 1
    
    optimized_time = time.time() - optimized_start
    optimized_throughput = optimized_processed / optimized_time
    
    # Calculate throughput improvement
    throughput_improvement = (optimized_throughput - baseline_throughput) / baseline_throughput * 100
    
    print(f"  Baseline throughput: {baseline_throughput:.2f} samples/sec")
    print(f"  Optimized throughput: {optimized_throughput:.2f} samples/sec")
    print(f"  Throughput improvement: {throughput_improvement:.2f}%")
    
    return throughput_improvement


def plot_benchmark_results(results):
    """Plot benchmark results"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot inference times
        x = np.arange(len(results["scenarios"]))
        width = 0.35
        
        ax1.bar(x - width/2, results["baseline_times"], width, label='Baseline', alpha=0.8)
        ax1.bar(x + width/2, results["optimized_times"], width, label='Optimized', alpha=0.8)
        
        ax1.set_xlabel('Test Scenario')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Inference Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.split('(')[0] for s in results["scenarios"]], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot improvements
        ax2.bar(results["scenarios"], results["improvements"], alpha=0.8, color='green')
        ax2.set_xlabel('Test Scenario')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Performance Improvement by Scenario')
        ax2.set_xticklabels([s.split('(')[0] for s in results["scenarios"]], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at 30% improvement
        ax2.axhline(y=30, color='red', linestyle='--', label='Target: 30%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("  Benchmark results plot saved as 'benchmark_results.png'")
    except ImportError:
        print("  Matplotlib not available, skipping plot generation")


def run_performance_benchmarking():
    """Run all performance benchmarking tests"""
    print("=" * 80)
    print("PERFORMANCE BENCHMARKING TEST FOR QWEN3-VL-2B-INSTRUCT ARCHITECTURE")
    print("=" * 80)
    
    print("Creating models for benchmarking...")
    
    # Run inference performance benchmark
    inference_results = benchmark_inference_performance()
    
    # Run memory efficiency benchmark
    memory_improvement = benchmark_memory_efficiency()
    
    # Run throughput benchmark
    throughput_improvement = benchmark_throughput()
    
    # Calculate overall improvement
    avg_improvement = np.mean(inference_results["improvements"])
    
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKING SUMMARY")
    print("=" * 80)
    
    print(f"Average inference improvement: {avg_improvement:.2f}%")
    print(f"Memory efficiency improvement: {memory_improvement:.2f}%")
    print(f"Throughput improvement: {throughput_improvement:.2f}%")
    
    # Check if targets are met
    meets_inference_target = avg_improvement >= 30
    meets_memory_target = memory_improvement >= 20  # More conservative target for memory
    meets_throughput_target = throughput_improvement >= 30
    
    print(f"\nTarget Achievement:")
    print(f"  Inference improvement (30%+): {'✓' if meets_inference_target else '✗'}")
    print(f"  Memory efficiency improvement (20%+): {'✓' if meets_memory_target else '✗'}")
    print(f"  Throughput improvement (30%+): {'✓' if meets_throughput_target else '✗'}")
    
    overall_success = meets_inference_target and meets_throughput_target
    
    print(f"\nOverall Performance Target: {'✓ ACHIEVED' if overall_success else '✗ NOT ACHIEVED'}")
    
    # Plot results
    plot_benchmark_results(inference_results)
    
    return overall_success


if __name__ == "__main__":
    success = run_performance_benchmarking()
    
    print(f"\n{'='*80}")
    print("PERFORMANCE BENCHMARKING STATUS:", "PASSED" if success else "FAILED")
    print(f"{'='*80}")
    
    sys.exit(0 if success else 1)