"""
Enhanced Custom Benchmark Executor for Accuracy - All Models

This script runs accuracy benchmarks for all models in both original and modified states,
without using pytest or unittest frameworks. It includes comparison functionality.
"""

import sys
import time
import torch
from pathlib import Path
import importlib
import traceback
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from enhanced_benchmark_utils import (
    run_benchmark_for_model_category,
    compare_results,
    save_results_json,
    save_results_csv,
    generate_detailed_report
)


def run_accuracy_benchmark_for_model(model_name: str, state: str = "original") -> Dict[str, Any]:
    """
    Run accuracy benchmark for a specific model in a specific state.
    """
    print(f"\n{'='*60}")
    print(f"Running Accuracy Benchmark for {model_name} ({state.upper()})")
    print(f"{'='*60}")

    # Import the benchmark module
    module_path = f"inference_pio.models.{model_name}.benchmarks.unit.benchmark_accuracy"
    try:
        benchmark_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Failed to import {module_path}: {e}")
        return {"error": str(e), "status": "failed", "model": model_name, "state": state}

    # Find the benchmark class
    benchmark_class = None
    for attr_name in dir(benchmark_module):
        attr = getattr(benchmark_module, attr_name)
        if (hasattr(attr, '__bases__') and
            len(attr.__bases__) > 0 and
            'Benchmark' in attr_name and
            attr_name.endswith('Accuracy')):
            benchmark_class = attr
            break

    if not benchmark_class:
        print(f"No benchmark class found in {module_path}")
        return {"error": "No benchmark class found", "status": "failed", "model": model_name, "state": state}

    # Create an instance of the benchmark class
    benchmark_instance = benchmark_class()

    # Call setUp if it exists
    if hasattr(benchmark_instance, 'setUp'):
        try:
            benchmark_instance.setUp()
        except Exception as e:
            print(f"Setup failed: {e}")
            return {"error": str(e), "status": "failed", "model": model_name, "state": state}

    # Run specific accuracy tests
    results = {
        "model": model_name,
        "category": "accuracy",
        "state": state,
        "timestamp": time.time(),
        "results": [],
        "status": "success"
    }

    test_methods = [
        'test_accuracy_on_standard_tasks',
        'test_accuracy_with_optimizations',
        'test_generative_accuracy',
        'test_classification_accuracy'
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
                    start_time = time.time()
                    method()
                    end_time = time.time()

                output = output_buffer.getvalue()

                results["results"].append({
                    "test_method": method_name,
                    "status": "passed",
                    "output": output,
                    "duration": end_time - start_time
                })

                print(f"✓ {method_name} completed ({end_time - start_time:.2f}s)")

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
            # Try alternative method names that might exist
            alt_methods = [
                'test_accuracy',
                'benchmark_accuracy'
            ]
            found = False
            for alt_method in alt_methods:
                if hasattr(benchmark_instance, alt_method):
                    print(f"\nRunning {alt_method}...")
                    try:
                        method = getattr(benchmark_instance, alt_method)

                        # Capture print output
                        import io
                        import contextlib

                        output_buffer = io.StringIO()
                        with contextlib.redirect_stdout(output_buffer):
                            start_time = time.time()
                            method()
                            end_time = time.time()

                        output = output_buffer.getvalue()

                        results["results"].append({
                            "test_method": alt_method,
                            "status": "passed",
                            "output": output,
                            "duration": end_time - start_time
                        })

                        print(f"✓ {alt_method} completed ({end_time - start_time:.2f}s)")
                        found = True
                        break

                    except Exception as e:
                        error_msg = f"Failed to run {alt_method}: {str(e)}"
                        print(f"✗ {alt_method} failed: {error_msg}")
                        print(f"Traceback: {traceback.format_exc()}")

                        results["results"].append({
                            "test_method": alt_method,
                            "status": "failed",
                            "error": error_msg,
                            "traceback": traceback.format_exc()
                        })
                        results["status"] = "partial"
                        found = True
                        break

            if not found:
                print(f"Method {method_name} not found in benchmark class")

    # Call tearDown if it exists
    if hasattr(benchmark_instance, 'tearDown'):
        try:
            benchmark_instance.tearDown()
        except Exception as e:
            print(f"TearDown failed: {e}")

    return results


def run_all_accuracy_benchmarks_for_state(state: str = "original") -> Dict[str, Any]:
    """
    Run accuracy benchmarks for all models in a specific state.
    """
    models = [
        "glm_4_7",
        "qwen3_4b_instruct_2507",
        "qwen3_coder_30b",
        "qwen3_vl_2b"
    ]

    print(f"Starting Accuracy Benchmarks for All Models ({state.upper()})...")

    all_results = {}
    for model in models:
        result = run_accuracy_benchmark_for_model(model, state)
        all_results[model] = result

    # Print summary
    print("\n" + "="*60)
    print(f"ACCURACY BENCHMARK SUMMARY ({state.upper()})")
    print("="*60)

    for model, result in all_results.items():
        if result.get("status") == "success":
            passed = sum(1 for r in result.get("results", []) if r.get("status") == "passed")
            total = len(result.get("results", []))
            print(f"{model}: {passed}/{total} tests passed")
        else:
            print(f"{model}: FAILED - {result.get('error', 'Unknown error')}")

    return {
        "state": state,
        "timestamp": time.time(),
        "models": models,
        "category": "accuracy",
        "results": all_results
    }


def run_comprehensive_accuracy_benchmarks():
    """
    Run accuracy benchmarks for all models in both original and modified states,
    then compare the results.
    """
    print("Running comprehensive accuracy benchmarks for original and modified models...")
    
    # Run benchmarks for original state
    print("\nRunning benchmarks for ORIGINAL state...")
    original_results = run_all_accuracy_benchmarks_for_state("original")
    
    # Run benchmarks for modified state
    print("\nRunning benchmarks for MODIFIED state...")
    modified_results = run_all_accuracy_benchmarks_for_state("modified")
    
    # Compare results
    comparison_results = compare_results(
        {"summary": {"models": original_results["models"], "duration": 0}, "details": original_results["results"]},
        {"summary": {"models": modified_results["models"], "duration": 0}, "details": modified_results["results"]}
    )
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("benchmark_results") / "accuracy"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_results_json(original_results, str(results_dir / f"accuracy_benchmark_results_original_{timestamp}.json"))
    save_results_json(modified_results, str(results_dir / f"accuracy_benchmark_results_modified_{timestamp}.json"))
    save_results_json(comparison_results, str(results_dir / f"accuracy_benchmark_comparison_{timestamp}.json"))
    
    # Save as CSV
    save_results_csv(original_results, str(results_dir / f"accuracy_benchmark_results_original_{timestamp}.csv"))
    save_results_csv(modified_results, str(results_dir / f"accuracy_benchmark_results_modified_{timestamp}.csv"))
    
    # Generate reports
    generate_detailed_report(original_results, str(results_dir / f"accuracy_benchmark_original_report_{timestamp}.md"))
    generate_detailed_report(modified_results, str(results_dir / f"accuracy_benchmark_modified_report_{timestamp}.md"))
    generate_detailed_report(comparison_results, str(results_dir / f"accuracy_benchmark_comparison_report_{timestamp}.md"))
    
    print(f"\nAccuracy benchmark results saved to {results_dir}/ directory")
    
    return {
        "original": original_results,
        "modified": modified_results,
        "comparison": comparison_results
    }


if __name__ == "__main__":
    results = run_comprehensive_accuracy_benchmarks()
    
    print("\n" + "="*60)
    print("ACCURACY BENCHMARK EXECUTION COMPLETE")
    print("="*60)
    
    # Print summary of comparison
    comparison = results["comparison"]
    for model_name, model_comp in comparison["model_comparisons"].items():
        print(f"\n{model_name.upper()}:")
        for category_name, cat_comp in model_comp["category_comparisons"].items():
            dur_impr = cat_comp["metrics_comparison"].get("duration_improvement_pct", 0)
            print(f"  {category_name.upper()}: {dur_impr:+.2f}%")