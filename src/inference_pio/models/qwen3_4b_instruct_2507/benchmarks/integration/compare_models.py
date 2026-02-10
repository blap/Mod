"""
General benchmark script for both Qwen3 models
"""
import time
import sys
import os
import psutil

# Adicionando o caminho para os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin


def benchmark_model(plugin, model_name, prompt, max_new_tokens=50):
    """Benchmark a single model."""
    print(f"\nBenchmarking {model_name}...")
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    response = plugin.generate_text(prompt, max_new_tokens=max_new_tokens)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    result = {
        "model_name": model_name,
        "elapsed_time_seconds": elapsed_time,
        "memory_used_mb": memory_used,
        "response_length": len(response) if response else 0,
        "tokens_per_second": max_new_tokens / elapsed_time if elapsed_time > 0 else 0
    }
    
    print(f"  Elapsed Time: {result['elapsed_time_seconds']:.2f}s")
    print(f"  Memory Used: {result['memory_used_mb']:.2f} MB")
    print(f"  Response Length: {result['response_length']} chars")
    print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")
    
    return result


def run_comparison_benchmarks():
    """Run benchmarks for both models and compare."""
    print("Initializing Qwen3 Models for Benchmarking...")
    
    # Initialize plugins
    qwen3_4b_plugin = Qwen3_4B_Instruct_2507_Plugin()
    qwen3_4b_plugin.initialize()
    
    qwen3_0_6b_plugin = Qwen3_0_6B_Plugin()
    qwen3_0_6b_plugin.initialize()
    
    # Benchmark parameters
    prompt = "Explain the concept of artificial intelligence in simple terms."
    
    print("\nRunning Comparison Benchmarks:")
    print("="*70)
    
    # Benchmark Qwen3-4B
    result_4b = benchmark_model(
        qwen3_4b_plugin, 
        "Qwen3-4B-Instruct-2507", 
        prompt, 
        max_new_tokens=100
    )
    
    # Benchmark Qwen3-0.6B
    result_0_6b = benchmark_model(
        qwen3_0_6b_plugin, 
        "Qwen3-0.6B", 
        prompt, 
        max_new_tokens=100
    )
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS:")
    print("="*70)
    print(f"{'Metric':<20} {'Qwen3-4B':<15} {'Qwen3-0.6B':<15} {'Winner':<10}")
    print("-"*70)
    
    # Compare inference time
    time_winner = "Qwen3-4B" if result_4b['elapsed_time_seconds'] < result_0_6b['elapsed_time_seconds'] else "Qwen3-0.6B"
    print(f"{'Time (sec)':<20} {result_4b['elapsed_time_seconds']:<15.2f} {result_0_6b['elapsed_time_seconds']:<15.2f} {time_winner:<10}")
    
    # Compare memory usage
    mem_winner = "Qwen3-0.6B" if result_0_6b['memory_used_mb'] < result_4b['memory_used_mb'] else "Qwen3-4B"
    print(f"{'Memory (MB)':<20} {result_4b['memory_used_mb']:<15.2f} {result_0_6b['memory_used_mb']:<15.2f} {mem_winner:<10}")
    
    # Compare throughput
    throughput_winner = "Qwen3-4B" if result_4b['tokens_per_second'] > result_0_6b['tokens_per_second'] else "Qwen3-0.6B"
    print(f"{'Throughput (tok/s)':<20} {result_4b['tokens_per_second']:<15.2f} {result_0_6b['tokens_per_second']:<15.2f} {throughput_winner:<10}")
    
    print("\n" + "="*70)
    print("Benchmark comparison completed!")


if __name__ == "__main__":
    run_comparison_benchmarks()