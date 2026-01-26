"""
Enhanced Utility functions for running comprehensive benchmarks across original and modified models.

This module provides reusable functionality for executing benchmarks that compare
original and modified versions of models, with proper result storage, comparison,
and reporting capabilities.
"""

import sys
import time
import json
import csv
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Callable
import io
import contextlib
import os
from datetime import datetime
import shutil
import subprocess
import gc
import torch
import psutil


def setup_environment():
    """Add the src directory to the Python path."""
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def discover_benchmark_class(module, category: str):
    """
    Discover the benchmark class in a module based on the category.

    Args:
        module: The imported benchmark module
        category: The benchmark category (e.g., 'accuracy', 'inference_speed')

    Returns:
        The benchmark class if found, None otherwise
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (hasattr(attr, '__bases__') and
            len(attr.__bases__) > 0 and
            'Benchmark' in attr_name and
            attr_name.lower().endswith(category.replace('_', '').lower())):
            return attr
    return None


def discover_test_methods(cls) -> List[str]:
    """
    Discover test methods in a test class.

    Args:
        cls: The benchmark class

    Returns:
        List of test method names
    """
    methods = []
    for attr_name in dir(cls):
        if attr_name.startswith('test_') and callable(getattr(cls, attr_name)):
            methods.append(attr_name)
    return methods


def run_benchmark_method(instance, method_name: str) -> Dict[str, Any]:
    """
    Run a single benchmark method and capture its output.

    Args:
        instance: The benchmark instance
        method_name: Name of the method to run

    Returns:
        Dictionary with results of the method execution
    """
    print(f"Running {method_name}...")
    try:
        method = getattr(instance, method_name)

        # Capture print output
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            start_time = time.time()
            method()
            end_time = time.time()

        output = output_buffer.getvalue()

        result = {
            "test_method": method_name,
            "status": "passed",
            "output": output,
            "duration": end_time - start_time
        }

        print(f"✓ {method_name} completed ({result['duration']:.2f}s)")
        return result

    except Exception as e:
        error_msg = f"Failed to run {method_name}: {str(e)}"
        print(f"✗ {method_name} failed: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")

        return {
            "test_method": method_name,
            "status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def run_benchmark_for_model_category(model_name: str, category: str, state: str = "original") -> Dict[str, Any]:
    """
    Run a specific benchmark category for a specific model in a specific state.

    Args:
        model_name: Name of the model to benchmark
        category: Benchmark category to run
        state: Model state ('original' or 'modified')

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Running {category.upper()} benchmarks for {model_name} ({state.upper()})")
    print(f"{'='*60}")

    # Import the benchmark module
    # Determine the correct subdirectory based on benchmark category
    if category in ['inference_speed', 'memory_usage', 'throughput', 'power_efficiency',
                   'optimization_impact', 'inference_speed_comparison']:
        subdir = 'performance'
    elif category in ['accuracy']:
        subdir = 'unit'
    elif category in ['comparison', 'async_multimodal_processing', 'intelligent_multimodal_caching']:
        subdir = 'integration'
    else:
        # Default to performance for most cases
        subdir = 'performance'

    module_path = f"inference_pio.models.{model_name}.benchmarks.{subdir}.benchmark_{category}"
    try:
        benchmark_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Failed to import {module_path}: {e}")
        return {"error": str(e), "status": "failed", "model": model_name, "category": category, "state": state}

    # Find the benchmark class
    benchmark_class = discover_benchmark_class(benchmark_module, category)
    if not benchmark_class:
        print(f"No benchmark class found in {module_path}")
        # Try alternative naming patterns
        for attr_name in dir(benchmark_module):
            attr = getattr(benchmark_module, attr_name)
            if (hasattr(attr, '__bases__') and
                len(attr.__bases__) > 0 and
                'Benchmark' in attr_name):
                benchmark_class = attr
                break

    if not benchmark_class:
        print(f"No benchmark class found in {module_path} after trying alternatives")
        return {"error": "No benchmark class found", "status": "failed", "model": model_name, "category": category, "state": state}

    # Create an instance of the benchmark class
    benchmark_instance = benchmark_class()

    # Call setUp if it exists
    if hasattr(benchmark_instance, 'setUp'):
        try:
            print("Setting up benchmark...")
            benchmark_instance.setUp()
            print("Setup completed")
        except Exception as e:
            print(f"Setup failed: {e}")
            return {"error": str(e), "status": "failed", "model": model_name, "category": category, "state": state}

    # Run the benchmark class methods
    results = {
        "model": model_name,
        "category": category,
        "state": state,
        "timestamp": time.time(),
        "results": [],
        "status": "success"
    }

    # Discover and run test methods
    test_methods = discover_test_methods(benchmark_class)

    if not test_methods:
        print(f"No test methods found in {benchmark_class.__name__}")
        # Try some common method names
        common_methods = [
            f'test_{category.replace("_", "")}',
            f'benchmark_{category}',
            f'test_{category}_basic',
            f'test_{category}_performance'
        ]
        for method_name in common_methods:
            if hasattr(benchmark_instance, method_name):
                test_methods.append(method_name)

    if not test_methods:
        print("No test methods found after trying common patterns")
        results["status"] = "no_tests_found"
        results["warning"] = "No test methods found to execute"
    else:
        for method_name in test_methods:
            result = run_benchmark_method(benchmark_instance, method_name)
            results["results"].append(result)

            # Update overall status if any test fails
            if result["status"] == "failed":
                results["status"] = "partial"

    # Call tearDown if it exists
    if hasattr(benchmark_instance, 'tearDown'):
        try:
            print("Tearing down benchmark...")
            benchmark_instance.tearDown()
            print("Teardown completed")
        except Exception as e:
            print(f"TearDown failed: {e}")

    return results


def run_all_benchmarks_for_state(models: List[str], categories: List[str], state: str) -> Dict[str, Any]:
    """
    Run all benchmarks for all models across all categories in a specific state.

    Args:
        models: List of model names to benchmark
        categories: List of benchmark categories to run
        state: Model state ('original' or 'modified')

    Returns:
        Dictionary with comprehensive benchmark results for the state
    """
    print(f"Starting comprehensive benchmark execution for {state.upper()} state...")
    print(f"Models: {models}")
    print(f"Categories: {categories}")

    setup_environment()

    overall_results = {
        "state": state,
        "summary": {
            "total_models": len(models),
            "total_categories": len(categories),
            "total_benchmarks": len(models) * len(categories),
            "models": models,
            "categories": categories,
            "start_time": time.time(),
            "end_time": None,
            "duration": None
        },
        "details": {}
    }

    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Processing model: {model}")
        model_results = {}

        for j, category in enumerate(categories):
            print(f"  [{j+1}/{len(categories)}] Running {category} benchmarks...")

            benchmark_result = run_benchmark_for_model_category(model, category, state)
            model_results[category] = benchmark_result

        overall_results["details"][model] = model_results

    overall_results["summary"]["end_time"] = time.time()
    overall_results["summary"]["duration"] = (
        overall_results["summary"]["end_time"] -
        overall_results["summary"]["start_time"]
    )

    return overall_results


def calculate_percentage_difference(original_value: float, modified_value: float) -> float:
    """
    Calculate percentage difference between original and modified values.
    
    Args:
        original_value: Value from original model
        modified_value: Value from modified model
    
    Returns:
        Percentage difference (positive if modified is better, negative if worse)
    """
    if original_value == 0:
        if modified_value == 0:
            return 0.0
        else:
            return float('inf') if modified_value > 0 else float('-inf')
    
    return ((modified_value - original_value) / abs(original_value)) * 100


def compare_results(original_results: Dict[str, Any], modified_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare original and modified benchmark results to calculate differences.
    
    Args:
        original_results: Results from original model
        modified_results: Results from modified model
    
    Returns:
        Dictionary with comparison results
    """
    comparison_results = {
        "comparison_metadata": {
            "comparison_timestamp": time.time(),
            "original_start_time": original_results["summary"]["start_time"],
            "modified_start_time": modified_results["summary"]["start_time"],
            "configuration": {
                "platform": sys.platform,
                "python_version": sys.version,
                "torch_version": torch.__version__ if 'torch' in sys.modules else "N/A"
            }
        },
        "model_comparisons": {}
    }
    
    models = original_results["summary"]["models"]
    
    for model in models:
        model_comparison = {
            "model": model,
            "category_comparisons": {}
        }
        
        original_model_results = original_results["details"][model]
        modified_model_results = modified_results["details"][model]
        
        for category in original_model_results.keys():
            if category in modified_model_results:
                original_cat_results = original_model_results[category]
                modified_cat_results = modified_model_results[category]
                
                category_comparison = {
                    "category": category,
                    "original_results": original_cat_results,
                    "modified_results": modified_cat_results,
                    "metrics_comparison": {}
                }
                
                # Compare basic metrics
                original_duration = original_cat_results.get("summary", {}).get("duration", 0)
                modified_duration = modified_cat_results.get("summary", {}).get("duration", 0)
                
                duration_diff_pct = calculate_percentage_difference(original_duration, modified_duration)
                category_comparison["metrics_comparison"]["duration_improvement_pct"] = duration_diff_pct
                
                # Compare individual test results if available
                original_tests = original_cat_results.get("results", [])
                modified_tests = modified_cat_results.get("results", [])
                
                test_comparisons = []
                for i, orig_test in enumerate(original_tests):
                    if i < len(modified_tests):
                        mod_test = modified_tests[i]
                        
                        test_comparison = {
                            "test_name": orig_test.get("test_method", f"test_{i}"),
                            "original_status": orig_test.get("status"),
                            "modified_status": mod_test.get("status"),
                            "original_duration": orig_test.get("duration", 0),
                            "modified_duration": mod_test.get("duration", 0)
                        }
                        
                        # Calculate duration difference
                        dur_diff_pct = calculate_percentage_difference(
                            test_comparison["original_duration"],
                            test_comparison["modified_duration"]
                        )
                        test_comparison["duration_improvement_pct"] = dur_diff_pct
                        
                        test_comparisons.append(test_comparison)
                
                category_comparison["test_comparisons"] = test_comparisons
                model_comparison["category_comparisons"][category] = category_comparison
        
        comparison_results["model_comparisons"][model] = model_comparison
    
    return comparison_results


def save_results_json(results: Dict[str, Any], filename: str):
    """
    Save benchmark results to a JSON file.

    Args:
        results: Dictionary with benchmark results
        filename: Name of the file to save results to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filename}")


def save_results_csv(results: Dict[str, Any], filename: str):
    """
    Save benchmark results to a CSV file.

    Args:
        results: Dictionary with benchmark results
        filename: Name of the file to save results to
    """
    rows = []
    
    # Extract data for CSV
    if "model_comparisons" in results:
        # This is a comparison result
        for model_name, model_comp in results["model_comparisons"].items():
            for category_name, cat_comp in model_comp["category_comparisons"].items():
                row = {
                    "model": model_name,
                    "category": category_name,
                    "duration_improvement_pct": cat_comp["metrics_comparison"].get("duration_improvement_pct", 0)
                }
                rows.append(row)
    else:
        # This is a regular result
        for model_name, model_results in results.get("details", {}).items():
            for category_name, cat_result in model_results.items():
                row = {
                    "model": model_name,
                    "category": category_name,
                    "state": results.get("state", "unknown"),
                    "status": cat_result.get("status", "unknown"),
                    "duration": cat_result.get("summary", {}).get("duration", 0)
                }
                rows.append(row)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        if rows:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    
    print(f"Results saved to {filename}")


def generate_detailed_report(results: Dict[str, Any], filename: str):
    """
    Generate a detailed markdown report of the benchmark results.

    Args:
        results: Dictionary with benchmark results
        filename: Name of the file to save the report to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Results Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if "comparison_metadata" in results:
            # This is a comparison report
            f.write("## Comparison Analysis\n\n")
            f.write("This report compares the performance of original and modified models.\n\n")
            
            f.write("### Configuration\n")
            config = results["comparison_metadata"]["configuration"]
            for key, value in config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            f.write("### Model-by-Model Comparison\n\n")
            
            for model_name, model_comp in results["model_comparisons"].items():
                f.write(f"#### {model_name.upper()}\n\n")
                
                for category_name, cat_comp in model_comp["category_comparisons"].items():
                    f.write(f"##### {category_name.upper()}\n")
                    
                    dur_impr = cat_comp["metrics_comparison"].get("duration_improvement_pct", 0)
                    f.write(f"- Duration Improvement: {dur_impr:+.2f}%\n")
                    
                    if dur_impr < 0:
                        f.write("- **Performance Degraded**\n")
                    elif dur_impr > 0:
                        f.write("- **Performance Improved**\n")
                    else:
                        f.write("- **No Performance Change**\n")
                    
                    f.write("\n")
        else:
            # This is a regular results report
            summary = results.get("summary", {})
            f.write(f"## Execution Summary\n")
            f.write(f"- Total Duration: {summary.get('duration', 0):.2f} seconds\n")
            f.write(f"- Models Tested: {summary.get('total_models', 0)}\n")
            f.write(f"- Categories Tested: {summary.get('total_categories', 0)}\n")
            f.write(f"- Total Benchmarks: {summary.get('total_benchmarks', 0)}\n")
            f.write(f"- State: {results.get('state', 'unknown')}\n\n")
            
            f.write("## Detailed Results by Model\n\n")
            
            for model_name, model_results in results.get("details", {}).items():
                f.write(f"### {model_name.upper()}\n")
                
                for category_name, cat_result in model_results.items():
                    status = cat_result.get("status", "unknown")
                    f.write(f"- **{category_name.upper()}**: {status}\n")
                
                f.write("\n")


def backup_original_models(models: List[str]) -> Path:
    """Return the path to the current model implementations (no backup needed)."""
    print("Using current models as baseline (no backup needed)...")

    src_dir = Path("src/inference_pio/models")
    print(f"Current models location: {src_dir}")
    return src_dir


def restore_original_models(backup_dir: Path):
    """Restore the original model implementations (no restoration needed)."""
    print("Using current models as baseline (no restoration needed)...")

    # No action needed since we're using the current models
    print("Current models remain unchanged")


def apply_modifications():
    """
    Apply custom code modifications to models.
    This is a placeholder - in a real scenario, this would apply actual modifications.
    """
    print("Applying custom code modifications...")
    
    # In a real implementation, this would apply actual modifications
    # For now, we'll simulate this by creating a temporary marker
    modification_marker = Path("MODIFICATIONS_APPLIED.marker")
    modification_marker.touch()
    
    print("Custom code modifications applied")


def remove_modifications():
    """Remove custom code modifications and restore original state."""
    print("Removing custom code modifications...")
    
    modification_marker = Path("MODIFICATIONS_APPLIED.marker")
    if modification_marker.exists():
        modification_marker.unlink()
    
    print("Custom code modifications removed")


def cleanup_resources():
    """Clean up resources between benchmark runs."""
    print("Cleaning up resources...")
    
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Small delay to allow system to settle
    time.sleep(2)


def check_system_resources(min_memory_gb=8):
    """Check if system has sufficient resources."""
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)
    
    print(f"Available Memory: {available_memory_gb:.2f} GB")
    
    if available_memory_gb < min_memory_gb:
        print(f"WARNING: Low memory condition. Available: {available_memory_gb:.2f} GB, Recommended: {min_memory_gb} GB")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'
    
    return True