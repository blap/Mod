"""
Real Model Benchmarks - Execute benchmarks with actual models and real data
instead of using mocks. This script measures actual performance with real models.
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path
import psutil
from datetime import datetime

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def benchmark_model_performance(model_name, plugin_creation_func, test_inputs, warmup_runs=2, actual_runs=5):
    """
    Benchmark a model's performance with real inputs.
    
    Args:
        model_name: Name of the model being benchmarked
        plugin_creation_func: Function to create the model plugin
        test_inputs: List of input texts to test with
        warmup_runs: Number of warmup runs to perform
        actual_runs: Number of actual runs to measure
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\nBenchmarking {model_name}...")
    
    # Create plugin instance
    plugin = plugin_creation_func()
    
    # Warmup runs
    print(f"Performing {warmup_runs} warmup runs...")
    for i in range(warmup_runs):
        try:
            # Use a simple prompt for warmup
            plugin.infer("Hello, this is a warmup prompt.")
        except:
            # If warmup fails, continue anyway
            pass
    
    # Measure performance
    inference_times = []
    memory_usages = []
    
    print(f"Performing {actual_runs} actual runs...")
    for i in range(actual_runs):
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Select input for this run
        input_text = test_inputs[i % len(test_inputs)]
        
        # Time the inference
        start_time = time.time()
        try:
            result = plugin.infer(input_text)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Record memory after inference
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            memory_usages.append(memory_used)
            
            print(f"  Run {i+1}: {inference_time:.3f}s, Memory delta: {memory_used:.2f}MB")
            
        except Exception as e:
            print(f"  Run {i+1}: Failed with error: {e}")
            # Add placeholder values for failed runs
            inference_times.append(float('inf'))
            memory_usages.append(0)
    
    # Calculate metrics
    avg_inference_time = np.mean([t for t in inference_times if t != float('inf')])
    std_inference_time = np.std([t for t in inference_times if t != float('inf')])
    avg_memory_usage = np.mean(memory_usages)
    peak_memory_usage = max(memory_usages) if memory_usages else 0
    
    # Calculate tokens per second if possible
    # Assuming ~10 tokens per second as a rough estimate for calculation
    avg_tps = (50 / avg_inference_time) if avg_inference_time > 0 else 0  # Rough estimate
    
    # Cleanup
    try:
        plugin.cleanup()
    except:
        pass
    
    return {
        "model_name": model_name,
        "avg_inference_time": avg_inference_time,
        "std_inference_time": std_inference_time,
        "avg_memory_usage": avg_memory_usage,
        "peak_memory_usage": peak_memory_usage,
        "tokens_per_second": avg_tps,
        "num_runs": actual_runs,
        "successful_runs": len([t for t in inference_times if t != float('inf')])
    }


def run_real_benchmarks():
    """Run benchmarks with real models and data."""
    print("=" * 80)
    print("REAL MODEL BENCHMARKS WITH ACTUAL PERFORMANCE MEASUREMENTS")
    print("=" * 80)
    
    # Define test inputs of varying complexity
    test_inputs = [
        "Hello, how are you?",
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate the Fibonacci sequence up to n terms.",
        "What are the main differences between Python and JavaScript programming languages?",
        "Describe the process of photosynthesis and its importance to life on Earth."
    ]
    
    # Collect results
    results = {}
    
    # Benchmark qwen3_0_6b if available
    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        print("\n[INFO] Testing qwen3_0_6b model...")
        result = benchmark_model_performance(
            "qwen3_0_6b", 
            create_qwen3_0_6b_plugin, 
            test_inputs,
            warmup_runs=1,
            actual_runs=3
        )
        results["qwen3_0_6b"] = result
        print(f"qwen3_0_6b benchmark completed: {result['avg_inference_time']:.3f}s avg")
    except ImportError as e:
        print(f"[INFO] Skipping qwen3_0_6b benchmark (not available): {e}")
    except Exception as e:
        print(f"[WARN] qwen3_0_6b benchmark failed: {e}")
        results["qwen3_0_6b"] = {"error": str(e)}
    
    # Benchmark qwen3_coder_next if available
    try:
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        print("\n[INFO] Testing qwen3_coder_next model...")
        result = benchmark_model_performance(
            "qwen3_coder_next", 
            create_qwen3_coder_next_plugin, 
            test_inputs,
            warmup_runs=1,
            actual_runs=3
        )
        results["qwen3_coder_next"] = result
        print(f"qwen3_coder_next benchmark completed: {result['avg_inference_time']:.3f}s avg")
    except ImportError as e:
        print(f"[INFO] Skipping qwen3_coder_next benchmark (not available): {e}")
    except Exception as e:
        print(f"[WARN] qwen3_coder_next benchmark failed: {e}")
        results["qwen3_coder_next"] = {"error": str(e)}
    
    # Add other models as needed
    # For now, let's also try to import and benchmark other available models
    
    # Benchmark qwen3_4b_instruct_2507 if available
    try:
        from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        print("\n[INFO] Testing qwen3_4b_instruct_2507 model...")
        result = benchmark_model_performance(
            "qwen3_4b_instruct_2507", 
            create_qwen3_4b_instruct_2507_plugin, 
            test_inputs,
            warmup_runs=1,
            actual_runs=3
        )
        results["qwen3_4b_instruct_2507"] = result
        print(f"qwen3_4b_instruct_2507 benchmark completed: {result['avg_inference_time']:.3f}s avg")
    except ImportError as e:
        print(f"[INFO] Skipping qwen3_4b_instruct_2507 benchmark (not available): {e}")
    except Exception as e:
        print(f"[WARN] qwen3_4b_instruct_2507 benchmark failed: {e}")
        results["qwen3_4b_instruct_2507"] = {"error": str(e)}
    
    # Benchmark qwen3_vl_2b if available
    try:
        from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_plugin
        print("\n[INFO] Testing qwen3_vl_2b model...")
        result = benchmark_model_performance(
            "qwen3_vl_2b", 
            create_qwen3_vl_2b_plugin, 
            test_inputs,
            warmup_runs=1,
            actual_runs=3
        )
        results["qwen3_vl_2b"] = result
        print(f"qwen3_vl_2b benchmark completed: {result['avg_inference_time']:.3f}s avg")
    except ImportError as e:
        print(f"[INFO] Skipping qwen3_vl_2b benchmark (not available): {e}")
    except Exception as e:
        print(f"[WARN] qwen3_vl_2b benchmark failed: {e}")
        results["qwen3_vl_2b"] = {"error": str(e)}
    
    # Benchmark glm_4_7_flash if available
    try:
        from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
        print("\n[INFO] Testing glm_4_7_flash model...")
        result = benchmark_model_performance(
            "glm_4_7_flash", 
            create_glm_4_7_flash_plugin, 
            test_inputs,
            warmup_runs=1,
            actual_runs=3
        )
        results["glm_4_7_flash"] = result
        print(f"glm_4_7_flash benchmark completed: {result['avg_inference_time']:.3f}s avg")
    except ImportError as e:
        print(f"[INFO] Skipping glm_4_7_flash benchmark (not available): {e}")
    except Exception as e:
        print(f"[WARN] glm_4_7_flash benchmark failed: {e}")
        results["glm_4_7_flash"] = {"error": str(e)}
    
    return results


def save_benchmark_results(results):
    """Save benchmark results to files."""
    print("\n" + "=" * 80)
    print("SAVING BENCHMARK RESULTS")
    print("=" * 80)
    
    from datetime import datetime
    import json
    import csv
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_filename = results_dir / f"real_model_benchmarks_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV summary
    csv_filename = results_dir / f"real_model_benchmarks_summary_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model Name", "Avg Inference Time (s)", "Std Inference Time (s)", 
            "Avg Memory Usage (MB)", "Peak Memory Usage (MB)", 
            "Tokens Per Second (est)", "Successful Runs", "Total Runs"
        ])
        
        for model_name, result in results.items():
            if "error" not in result:
                writer.writerow([
                    result["model_name"],
                    f"{result['avg_inference_time']:.3f}",
                    f"{result['std_inference_time']:.3f}",
                    f"{result['avg_memory_usage']:.2f}",
                    f"{result['peak_memory_usage']:.2f}",
                    f"{result['tokens_per_second']:.2f}",
                    result["successful_runs"],
                    result["num_runs"]
                ])
    
    print(f"Benchmark results saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  CSV: {csv_filename}")


def print_benchmark_summary(results):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"{'Model Name':<25} {'Avg Time (s)':<15} {'Memory (MB)':<15} {'Tokens/s (est)':<15} {'Success Rate':<12}")
    print("-" * 85)
    
    for model_name, result in results.items():
        if "error" not in result:
            success_rate = f"{result['successful_runs']}/{result['num_runs']}"
            print(
                f"{result['model_name']:<25} "
                f"{result['avg_inference_time']:<15.3f} "
                f"{result['avg_memory_usage']:<15.2f} "
                f"{result['tokens_per_second']:<15.2f} "
                f"{success_rate:<12}"
            )
        else:
            print(f"{model_name:<25} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15} {'0/0':<12} - {result['error']}")


def main():
    """Main function to run real model benchmarks."""
    print("Starting real model benchmarks with actual performance measurements...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Run benchmarks
    results = run_real_benchmarks()
    
    # Print summary
    print_benchmark_summary(results)
    
    # Save results
    save_benchmark_results(results)
    
    print(f"\nCompleted benchmarking {len(results)} models.")
    
    # Count successful benchmarks
    successful = sum(1 for r in results.values() if "error" not in r)
    print(f"Successful benchmarks: {successful}/{len(results)}")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)