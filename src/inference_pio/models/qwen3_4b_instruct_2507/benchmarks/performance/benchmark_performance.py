"""
Performance benchmarks for Qwen3-4B-Instruct-2507 Model
"""
import time
import sys
import os
import psutil

# Adicionando o caminho para o módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin


def benchmark_inference_time(plugin, prompt, max_new_tokens=50):
    """Benchmark the inference time of the model."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    response = plugin.generate_text(prompt, max_new_tokens=max_new_tokens)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    return {
        "elapsed_time_seconds": elapsed_time,
        "memory_used_mb": memory_used,
        "response_length": len(response) if response else 0,
        "tokens_per_second": max_new_tokens / elapsed_time if elapsed_time > 0 else 0
    }


def run_benchmarks():
    """Run all benchmarks."""
    print("Initializing Qwen3-4B-Instruct-2507 Plugin...")
    plugin = Qwen3_4B_Instruct_2507_Plugin()
    plugin.initialize()
    
    # Benchmark parameters
    prompts = [
        "Hello, how are you?",
        "Tell me about artificial intelligence.",
        "Write a short poem about technology."
    ]
    
    print("\nRunning Performance Benchmarks for Qwen3-4B-Instruct-2507:")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nBenchmark {i}: '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'")
        result = benchmark_inference_time(plugin, prompt, max_new_tokens=50)
        
        print(f"  Elapsed Time: {result['elapsed_time_seconds']:.2f}s")
        print(f"  Memory Used: {result['memory_used_mb']:.2f} MB")
        print(f"  Response Length: {result['response_length']} chars")
        print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")
    
    print("\n" + "="*60)
    print("Benchmark completed!")


if __name__ == "__main__":
    run_benchmarks()