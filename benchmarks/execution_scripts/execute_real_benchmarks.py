#!/usr/bin/env python3
"""
Execute real benchmarks for all models with actual performance data collection
"""

import os
import sys
import time
import json
from pathlib import Path

def execute_real_benchmarks():
    """
    Execute real benchmarks for all models and collect actual performance data
    """
    print("="*60)
    print("EXECUTING REAL BENCHMARKS FOR ALL MODELS")
    print("="*60)
    
    # Define models and benchmark categories
    models = [
        'glm_4_7',
        'qwen3_4b_instruct_2507', 
        'qwen3_coder_30b',
        'qwen3_vl_2b'
    ]
    
    categories = [
        'benchmark_accuracy',
        'benchmark_comparison', 
        'benchmark_inference_speed',
        'benchmark_memory_usage',
        'benchmark_optimization_impact',
        'benchmark_power_efficiency',
        'benchmark_throughput'
    ]
    
    results = {}
    
    print(f"Models: {models}")
    print(f"Categories: {categories}")
    print(f"Total benchmark runs: {len(models) * len(categories)}")
    print()
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    for model_idx, model in enumerate(models):
        print(f"[{model_idx+1}/{len(models)}] Processing model: {model}")
        
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
                
                # Dynamically import the benchmark module
                import importlib.util
                spec = importlib.util.spec_from_file_location(category, benchmark_path)
                benchmark_module = importlib.util.module_from_spec(spec)
                
                # Capture stdout to collect results
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
                    "output_preview": output[:500] + "..." if len(output) > 500 else output,
                    "output_length": len(output)
                }
                
                print(f"    Completed in {execution_time:.2f}s")
                
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
                print(f"    Error: {error_msg}")
                model_results[category] = {
                    "status": "error",
                    "execution_time": execution_time,
                    "error": error_msg
                }
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Runtime error: {str(e)}"
                print(f"    Error: {error_msg}")
                model_results[category] = {
                    "status": "error", 
                    "execution_time": execution_time,
                    "error": error_msg
                }
        
        results[model] = model_results
        
        # Save intermediate results
        intermediate_file = results_dir / f"{model}_results.json"
        with open(intermediate_file, 'w') as f:
            json.dump(model_results, f, indent=2)
    
    # Save final results
    final_results_file = results_dir / "all_benchmark_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("="*60)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*60)
    
    total_runs = 0
    completed_runs = 0
    error_runs = 0
    total_time = 0
    
    for model, model_results in results.items():
        print(f"\n{model}:")
        model_total_time = 0
        model_completed = 0
        model_errors = 0
        
        for category, result in model_results.items():
            total_runs += 1
            if result['status'] == 'completed':
                completed_runs += 1
                model_completed += 1
                exec_time = result['execution_time']
                model_total_time += exec_time
                total_time += exec_time
                print(f"  [PASS] {category}: {exec_time:.2f}s")
            else:
                error_runs += 1
                model_errors += 1
                print(f"  [FAIL] {category}: {result.get('error', 'Unknown error')}")
        
        print(f"  Model {model}: {model_completed} completed, {model_errors} errors, {model_total_time:.2f}s total")
    
    print(f"\nOverall: {completed_runs}/{total_runs} completed, {error_runs} errors")
    print(f"Total execution time: {total_time:.2f}s")
    
    return results

if __name__ == "__main__":
    execute_real_benchmarks()