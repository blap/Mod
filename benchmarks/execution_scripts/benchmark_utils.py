"""
Utility functions for running comprehensive benchmarks across all models and categories.

This module provides reusable functionality for executing benchmarks without using
pytest or unittest frameworks, following the DRY principle.
"""

import sys
import time
import json
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Callable
import io
import contextlib


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


def run_benchmark_for_model_category(model_name: str, category: str) -> Dict[str, Any]:
    """
    Run a specific benchmark category for a specific model.
    
    Args:
        model_name: Name of the model to benchmark
        category: Benchmark category to run
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Running {category.upper()} benchmarks for {model_name}")
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
        return {"error": str(e), "status": "failed", "model": model_name, "category": category}

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
        return {"error": "No benchmark class found", "status": "failed", "model": model_name, "category": category}

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
            return {"error": str(e), "status": "failed", "model": model_name, "category": category}

    # Run the benchmark class methods
    results = {
        "model": model_name,
        "category": category,
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


def run_all_benchmarks(models: List[str], categories: List[str]) -> Dict[str, Any]:
    """
    Run all benchmarks for all models across all categories.
    
    Args:
        models: List of model names to benchmark
        categories: List of benchmark categories to run
        
    Returns:
        Dictionary with comprehensive benchmark results
    """
    print("Starting comprehensive benchmark execution...")
    print(f"Models: {models}")
    print(f"Categories: {categories}")

    setup_environment()

    overall_results = {
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
            
            benchmark_result = run_benchmark_for_model_category(model, category)
            model_results[category] = benchmark_result

        overall_results["details"][model] = model_results

    overall_results["summary"]["end_time"] = time.time()
    overall_results["summary"]["duration"] = (
        overall_results["summary"]["end_time"] -
        overall_results["summary"]["start_time"]
    )

    return overall_results


def save_results(results: Dict[str, Any], filename: str = "comprehensive_benchmark_results.json"):
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Dictionary with benchmark results
        filename: Name of the file to save results to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filename}")


def print_summary(results: Dict[str, Any]):
    """
    Print a summary of the benchmark results.
    
    Args:
        results: Dictionary with benchmark results
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE BENCHMARK EXECUTION SUMMARY")
    print("="*70)

    summary = results["summary"]
    print(f"Total Duration: {summary['duration']:.2f} seconds ({summary['duration']/60:.2f} minutes)")
    print(f"Start Time: {time.ctime(summary['start_time'])}")
    print(f"End Time: {time.ctime(summary['end_time'])}")
    print(f"Models Tested: {summary['total_models']}")
    print(f"Categories Tested: {summary['total_categories']}")
    print(f"Total Benchmarks: {summary['total_benchmarks']}")

    print("\nDetailed Results by Model:")
    total_passed = 0
    total_tests = 0
    
    for model, model_results in results["details"].items():
        print(f"\n{model.upper()}:")
        model_passed = 0
        model_total = 0

        for category, category_results in model_results.items():
            if isinstance(category_results, dict) and "results" in category_results:
                test_results = category_results.get("results", [])
                for test_result in test_results:
                    model_total += 1
                    total_tests += 1
                    if test_result.get("status") == "passed":
                        model_passed += 1
                        total_passed += 1
            else:
                # Category benchmark itself failed
                model_total += 1
                total_tests += 1

        success_rate = (model_passed / model_total * 100) if model_total > 0 else 0
        print(f"  Tests: {model_passed}/{model_total} passed ({success_rate:.1f}%)")

        if model_passed < model_total:
            failed_count = model_total - model_passed
            print(f"  Failed: {failed_count} tests")

    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOverall: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1f}%)")

    # Identify problematic areas
    print("\nProblematic Categories by Model:")
    for model, model_results in results["details"].items():
        failed_categories = []
        for category, category_results in model_results.items():
            if isinstance(category_results, dict):
                if category_results.get("status") == "failed":
                    failed_categories.append(f"{category} (benchmark failed)")
                elif "results" in category_results:
                    failed_tests = [r for r in category_results["results"] if r.get("status") == "failed"]
                    if failed_tests:
                        failed_categories.append(f"{category} ({len(failed_tests)} tests failed)")
        
        if failed_categories:
            print(f"  {model}: {', '.join(failed_categories)}")
        else:
            print(f"  {model}: All categories passed")