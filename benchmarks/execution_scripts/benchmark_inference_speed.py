"""
Custom Benchmark Executor for Inference Speed - All Models

This script runs inference speed benchmarks for all models without using
pytest or unittest frameworks.
"""

import sys
import time
import torch
from pathlib import Path
import importlib
import traceback

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def run_inference_speed_benchmark_for_model(model_name: str):
    """
    Run inference speed benchmark for a specific model.
    """
    print(f"\n{'='*60}")
    print(f"Running Inference Speed Benchmark for {model_name}")
    print(f"{'='*60}")
    
    # Import the benchmark module
    module_path = f"inference_pio.models.{model_name}.benchmarks.performance.benchmark_inference_speed"
    try:
        benchmark_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Failed to import {module_path}: {e}")
        return {"error": str(e), "status": "failed"}
    
    # Find the benchmark class
    benchmark_class = None
    for attr_name in dir(benchmark_module):
        attr = getattr(benchmark_module, attr_name)
        if (hasattr(attr, '__bases__') and 
            len(attr.__bases__) > 0 and 
            'Benchmark' in attr_name and 
            attr_name.endswith('InferenceSpeed')):
            benchmark_class = attr
            break
    
    if not benchmark_class:
        print(f"No benchmark class found in {module_path}")
        return {"error": "No benchmark class found", "status": "failed"}
    
    # Create an instance of the benchmark class
    benchmark_instance = benchmark_class()
    
    # Call setUp if it exists
    if hasattr(benchmark_instance, 'setUp'):
        try:
            benchmark_instance.setUp()
        except Exception as e:
            print(f"Setup failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    # Run specific inference speed tests
    results = {
        "model": model_name,
        "category": "inference_speed",
        "timestamp": time.time(),
        "results": [],
        "status": "success"
    }
    
    test_methods = [
        'test_inference_speed_short_input',
        'test_inference_speed_medium_input', 
        'test_inference_speed_long_input',
        'test_generation_speed',
        'test_batch_inference_speed',
        'test_variable_length_inference_speed'
    ]
    
    for method_name in test_methods:
        if hasattr(benchmark_instance, method_name):
            print(f"\nRunning {method_name}...")
            try:
                method = getattr(benchmark_instance, method_name)
                
                # Capture print output
                import io
                import contextlib
                
                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    method()
                
                output = output_buffer.getvalue()
                
                results["results"].append({
                    "test_method": method_name,
                    "status": "passed",
                    "output": output
                })
                
                print(f"✓ {method_name} completed")
                
            except Exception as e:
                error_msg = f"Failed to run {method_name}: {str(e)}"
                print(f"✗ {method_name} failed: {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")
                
                results["results"].append({
                    "test_method": method_name,
                    "status": "failed", 
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                })
                results["status"] = "partial"
        else:
            print(f"Method {method_name} not found in benchmark class")
    
    # Call tearDown if it exists
    if hasattr(benchmark_instance, 'tearDown'):
        try:
            benchmark_instance.tearDown()
        except Exception as e:
            print(f"TearDown failed: {e}")
    
    return results

def run_all_inference_speed_benchmarks():
    """
    Run inference speed benchmarks for all models.
    """
    models = [
        "glm_4_7",
        "qwen3_4b_instruct_2507", 
        "qwen3_coder_30b",
        "qwen3_vl_2b"
    ]
    
    print("Starting Inference Speed Benchmarks for All Models...")
    
    all_results = {}
    for model in models:
        result = run_inference_speed_benchmark_for_model(model)
        all_results[model] = result
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK SUMMARY")
    print("="*60)
    
    for model, result in all_results.items():
        if result.get("status") == "success":
            passed = sum(1 for r in result.get("results", []) if r.get("status") == "passed")
            total = len(result.get("results", []))
            print(f"{model}: {passed}/{total} tests passed")
        else:
            print(f"{model}: FAILED - {result.get('error', 'Unknown error')}")
    
    return all_results

if __name__ == "__main__":
    results = run_all_inference_speed_benchmarks()
    
    # Save results to file
    import json
    with open("inference_speed_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to inference_speed_benchmark_results.json")