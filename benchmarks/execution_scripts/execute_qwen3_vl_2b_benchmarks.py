#!/usr/bin/env python3
"""
Execute real benchmarks for qwen3_vl_2b model with actual performance data collection
"""

import os
import sys
import time
import json
from pathlib import Path

def execute_qwen3_vl_2b_benchmarks():
    """
    Execute real benchmarks for qwen3_vl_2b model with actual performance data
    """
    print("="*70)
    print("EXECUTING REAL BENCHMARKS FOR QWEN3_VL_2B MODEL")
    print("="*70)
    
    # Define the model and benchmark categories
    model = 'qwen3_vl_2b'
    categories = [
        'benchmark_accuracy',
        'benchmark_inference_speed', 
        'benchmark_memory_usage',
        'benchmark_optimization_impact',
        'benchmark_power_efficiency',
        'benchmark_throughput'
    ]
    
    results = {}
    
    print(f"Model: {model}")
    print(f"Categories: {categories}")
    print(f"Total benchmark runs: {len(categories)}")
    print()
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    print(f"[1/1] Processing model: {model}")
    
    model_results = {}
    
    for category_idx, category in enumerate(categories):
        print(f"  [{category_idx+1}/{len(categories)}] Running {category}...")
        
        # Construct benchmark file path
        benchmark_path = f"src/inference_pio/models/{model}/benchmarks/{category}.py"
        
        if not os.path.exists(benchmark_path):
            print(f"    Warning: Benchmark file does not exist: {benchmark_path}")
            model_results[category] = {"status": "missing", "error": "File not found"}
            continue
        
        print(f"    Executing: python {benchmark_path}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Import and execute the benchmark directly
            sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
            
            # Import the specific plugin to make sure it works
            from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
            
            # Dynamically import the benchmark module
            import importlib.util
            spec = importlib.util.spec_from_file_location(category, benchmark_path)
            benchmark_module = importlib.util.module_from_spec(spec)
            
            # Capture stdout to collect actual benchmark results
            import io
            from contextlib import redirect_stdout
            
            # Create a StringIO buffer to capture output
            output_buffer = io.StringIO()
            
            # Execute the benchmark module
            with redirect_stdout(output_buffer):
                spec.loader.exec_module(benchmark_module)
            
            # Record end time
            end_time = time.time()
            
            # Store results
            execution_time = end_time - start_time
            output = output_buffer.getvalue()
            
            model_results[category] = {
                "status": "completed",
                "execution_time": execution_time,
                "output_preview": output[:1000] + "..." if len(output) > 1000 else output,
                "output_length": len(output)
            }
            
            print(f"    [SUCCESS] Completed in {execution_time:.2f}s")
            
            # Save detailed output to file
            output_file = results_dir / f"{model}_{category}_output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Model: {model}\n")
                f.write(f"Category: {category}\n")
                f.write(f"Execution time: {execution_time:.2f}s\n")
                f.write("="*60 + "\n")
                f.write(output)
            
        except ImportError as e:
            execution_time = time.time() - start_time
            error_msg = f"Import error: {str(e)}"
            print(f"    [ERROR] {error_msg}")
            model_results[category] = {
                "status": "error",
                "execution_time": execution_time,
                "error": error_msg
            }
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Runtime error: {str(e)}"
            print(f"    [ERROR] {error_msg}")
            model_results[category] = {
                "status": "error", 
                "execution_time": execution_time,
                "error": error_msg
            }
    
    results[model] = model_results
    
    # Save results
    results_file = results_dir / f"{model}_results.json"
    with open(results_file, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    # Print summary
    print()
    print("="*70)
    print("QWEN3_VL_2B BENCHMARK EXECUTION SUMMARY")
    print("="*70)
    
    total_runs = len(categories)
    completed_runs = sum(1 for r in model_results.values() if r['status'] == 'completed')
    error_runs = sum(1 for r in model_results.values() if r['status'] == 'error')
    total_time = sum(r['execution_time'] for r in model_results.values())
    
    print(f"\n{model}:")
    for category, result in model_results.items():
        if result['status'] == 'completed':
            print(f"  [SUCCESS] {category}: {result['execution_time']:.2f}s (output: {result['output_length']} chars)")
        else:
            print(f"  [FAILED] {category}: {result.get('error', 'Unknown error')}")
    
    print(f"  Model {model}: {completed_runs}/{total_runs} completed, {error_runs} errors, {total_time:.2f}s total")
    
    print(f"\nOverall: {completed_runs}/{total_runs} completed, {error_runs} errors")
    print(f"Total execution time: {total_time:.2f}s")
    
    return results

if __name__ == "__main__":
    results = execute_qwen3_vl_2b_benchmarks()
    
    print("\n" + "="*70)
    print("QWEN3_VL_2B BENCHMARK COMPLETED")
    print("="*70)
    print("Benchmarks executed with real model and collected actual performance data.")
    print("Results saved in the 'benchmark_results' directory.")