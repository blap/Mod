"""
Utility module for running benchmarks across all models.

This module provides functions to run, save, and summarize benchmark results
for various model categories using real performance measurements.
"""

import csv
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.core.real_performance_monitor import (
    RealPerformanceMonitor,
    get_real_system_metrics,
    start_continuous_monitoring,
    stop_continuous_monitoring,
)
from src.inference_pio.utils.benchmarking_utils import *


def run_all_benchmarks(models: List[str], categories: List[str]) -> Dict[str, Any]:
    """
    Run all benchmarks for specified models and categories using real performance measurements.

    Args:
        models: List of model names to benchmark
        categories: List of benchmark categories to run

    Returns:
        Dictionary containing benchmark results
    """
    print("Initializing benchmark utilities with real performance monitoring...")

    # Start system-wide monitoring
    monitor = RealPerformanceMonitor()
    monitor.start_monitoring(interval=2.0)  # Sample every 2 seconds during benchmarking

    results = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "categories": categories,
        "results": {},
        "system_metrics_during_benchmark": [],
    }

    for model in models:
        print(f"\nProcessing model: {model}")
        model_results = {}

        for category in categories:
            print(f"  Running {category} benchmarks...")

            # Determine the correct subdirectory based on benchmark category
            if category in [
                "inference_speed",
                "memory_usage",
                "throughput",
                "power_efficiency",
                "optimization_impact",
                "inference_speed_comparison",
            ]:
                subdir = "performance"
            elif category in ["accuracy"]:
                subdir = "unit"
            elif category in [
                "comparison",
                "async_multimodal_processing",
                "intelligent_multimodal_caching",
            ]:
                subdir = "integration"
            else:
                # Default to performance for most cases
                subdir = "performance"

            try:
                # Construct the module path
                module_path = f"inference_pio.models.{model}.benchmarks.{subdir}.benchmark_{category}"

                # Try to import the benchmark module
                try:
                    benchmark_module = importlib.import_module(module_path)

                    # Look for a function that runs the specific benchmark
                    run_func_name = (
                        f"run_{model.replace('-', '_').replace('.', '_')}_benchmark"
                    )
                    if hasattr(benchmark_module, run_func_name):
                        run_func = getattr(benchmark_module, run_func_name)
                        result = run_func()
                        model_results[category] = result
                    else:
                        # If specific function not found, try generic patterns
                        possible_funcs = [
                            name
                            for name in dir(benchmark_module)
                            if name.startswith("run_") or name.startswith("benchmark_")
                        ]

                        if possible_funcs:
                            run_func = getattr(benchmark_module, possible_funcs[0])
                            result = run_func()
                            model_results[category] = result
                        else:
                            model_results[category] = {
                                "status": "no_benchmark_function_found",
                                "details": f"No benchmark function found in {module_path}",
                                "system_metrics_at_failure": get_real_system_metrics().__dict__,
                            }

                except ImportError as e:
                    print(f"    Warning: Could not import {module_path}: {e}")
                    model_results[category] = {
                        "status": "import_error",
                        "error": str(e),
                        "system_metrics_at_failure": get_real_system_metrics().__dict__,
                    }
                except Exception as e:
                    print(f"    Error running {category} for {model}: {e}")
                    model_results[category] = {
                        "status": "execution_error",
                        "error": str(e),
                        "system_metrics_at_failure": get_real_system_metrics().__dict__,
                    }

            except Exception as e:
                print(f"    Unexpected error with {category} for {model}: {e}")
                model_results[category] = {
                    "status": "unexpected_error",
                    "error": str(e),
                    "system_metrics_at_failure": get_real_system_metrics().__dict__,
                }

        results["results"][model] = model_results

    # Capture final system metrics
    results["system_metrics_during_benchmark"] = [
        m.__dict__ for m in monitor.metrics_history
    ]

    # Stop monitoring
    monitor.stop_monitoring()

    return results


def run_benchmark_for_model_category(model_name: str, category: str) -> Dict[str, Any]:
    """
    Run a specific benchmark category for a specific model using real performance measurements.

    Args:
        model_name: Name of the model to benchmark
        category: Benchmark category to run

    Returns:
        Dictionary containing benchmark results for the specific model/category
    """
    print(f"Running {category} benchmark for {model_name}")

    # Start monitoring for this specific benchmark
    monitor = RealPerformanceMonitor()
    monitor.start_monitoring(interval=1.0)

    # Determine the correct subdirectory based on benchmark category
    if category in [
        "inference_speed",
        "memory_usage",
        "throughput",
        "power_efficiency",
        "optimization_impact",
        "inference_speed_comparison",
    ]:
        subdir = "performance"
    elif category in ["accuracy"]:
        subdir = "unit"
    elif category in [
        "comparison",
        "async_multimodal_processing",
        "intelligent_multimodal_caching",
    ]:
        subdir = "integration"
    else:
        # Default to performance for most cases
        subdir = "performance"

    result = {
        "model": model_name,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "results": [],
        "status": "no_tests_found",
        "system_metrics_during_benchmark": [],
    }

    try:
        # Construct the module path
        module_path = f"inference_pio.models.{model_name}.benchmarks.{subdir}.benchmark_{category}"

        # Try to import the benchmark module
        try:
            benchmark_module = importlib.import_module(module_path)

            # Look for a function that runs the specific benchmark
            run_func_name = (
                f"run_{model_name.replace('-', '_').replace('.', '_')}_benchmark"
            )
            if hasattr(benchmark_module, run_func_name):
                run_func = getattr(benchmark_module, run_func_name)
                benchmark_result = run_func()
                result["results"] = [benchmark_result] if benchmark_result else []
                result["status"] = "success"
            else:
                # If specific function not found, try generic patterns
                possible_funcs = [
                    name
                    for name in dir(benchmark_module)
                    if name.startswith("run_") or name.startswith("benchmark_")
                ]

                if possible_funcs:
                    run_func = getattr(benchmark_module, possible_funcs[0])
                    benchmark_result = run_func()
                    result["results"] = [benchmark_result] if benchmark_result else []
                    result["status"] = "success"
                else:
                    result["status"] = "no_benchmark_function_found"
                    result["error"] = f"No benchmark function found in {module_path}"

        except ImportError as e:
            print(f"Warning: Could not import {module_path}: {e}")
            result["status"] = "import_error"
            result["error"] = str(e)
        except Exception as e:
            print(f"Error running {category} for {model_name}: {e}")
            result["status"] = "execution_error"
            result["error"] = str(e)

    except Exception as e:
        print(f"Unexpected error with {category} for {model_name}: {e}")
        result["status"] = "unexpected_error"
        result["error"] = str(e)

    # Capture system metrics during the benchmark
    result["system_metrics_during_benchmark"] = [
        m.__dict__ for m in monitor.metrics_history
    ]

    # Stop monitoring
    monitor.stop_monitoring()

    return result


def save_results(results: Dict[str, Any], filename: str = None) -> str:
    """
    Save benchmark results to a file with real performance measurements.

    Args:
        results: Dictionary containing benchmark results
        filename: Name of the file to save results to. If None, generates a filename.

    Returns:
        Path to the saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_benchmark_results_{timestamp}.json"

    # Ensure the results directory exists
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Real benchmark results saved to {filepath}")
    return str(filepath)


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of benchmark results with real performance measurements.

    Args:
        results: Dictionary containing benchmark results
    """
    print("\n" + "=" * 70)
    print("REAL PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Models tested: {len(results.get('models', []))}")
    print(f"Categories tested: {len(results.get('categories', []))}")

    # Show system metrics summary
    system_metrics = results.get("system_metrics_during_benchmark", [])
    if system_metrics:
        avg_cpu = (
            sum(m.get("cpu_percent", 0) for m in system_metrics) / len(system_metrics)
            if system_metrics
            else 0
        )
        avg_memory = (
            sum(m.get("memory_percent", 0) for m in system_metrics)
            / len(system_metrics)
            if system_metrics
            else 0
        )
        print(f"Average CPU usage during benchmarks: {avg_cpu:.2f}%")
        print(f"Average memory usage during benchmarks: {avg_memory:.2f}%")

    print("\nDetailed Results:")
    for model, model_results in results.get("results", {}).items():
        print(f"\n  {model}:")
        for category, category_result in model_results.items():
            if isinstance(category_result, dict) and "status" in category_result:
                status = category_result["status"]
                print(f"    {category}: {status}")
                if "error" in category_result:
                    print(f"      Error: {category_result['error'][:100]}...")
            else:
                print(f"    {category}: Completed")

    print("\n" + "=" * 70)


def discover_benchmark_class(benchmark_module: Any, category: str):
    """
    Discover the benchmark class in a module based on the category.

    Args:
        benchmark_module: The imported benchmark module
        category: The benchmark category

    Returns:
        The benchmark class if found, None otherwise
    """
    # Look for classes in the module that match the category
    for name in dir(benchmark_module):
        obj = getattr(benchmark_module, name)
        if (
            hasattr(obj, "__module__")
            and obj.__module__ == benchmark_module.__name__
            and isinstance(obj, type)
        ):
            # Check if the class name relates to the category
            if category.lower() in name.lower() or "benchmark" in name.lower():
                return obj

    # If no specific class found, return the first class in the module
    for name in dir(benchmark_module):
        obj = getattr(benchmark_module, name)
        if (
            hasattr(obj, "__module__")
            and obj.__module__ == benchmark_module.__name__
            and isinstance(obj, type)
        ):
            return obj

    return None


def calculate_percentage_difference(original_value: float, new_value: float) -> float:
    """
    Calculate the percentage difference between two values using real measurements.

    Args:
        original_value: The original/base value
        new_value: The new value to compare against

    Returns:
        Percentage difference (positive if new_value is greater, negative if smaller)
    """
    if original_value == 0:
        if new_value == 0:
            return 0.0
        else:
            # If original is 0 but new is not, return infinity or a large number
            return float("inf") if new_value > 0 else float("-inf")

    return ((new_value - original_value) / abs(original_value)) * 100.0


def setup_environment():
    """
    Setup the environment for running benchmarks with real performance monitoring.
    """
    print("Setting up benchmark environment with real performance monitoring...")
    # Add any necessary setup here
    pass


def apply_modifications():
    """
    Apply any necessary modifications before running benchmarks with real metrics.
    """
    print("Applying benchmark modifications with real performance tracking...")
    # Add any necessary modifications here
    pass


def remove_modifications():
    """
    Remove any modifications after running benchmarks with real metrics.
    """
    print(
        "Removing benchmark modifications and cleaning up real performance tracking..."
    )
    # Add any necessary cleanup here
    pass


if __name__ == "__main__":
    # Example usage
    models = ["glm_4_7", "qwen3_4b_instruct_2507"]
    categories = ["accuracy", "inference_speed"]

    results = run_all_benchmarks(models, categories)
    print_summary(results)
    save_results(results)
