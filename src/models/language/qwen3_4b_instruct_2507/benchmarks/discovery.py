"""
GLM-4-7 Model Benchmark Discovery Module

This module provides automatic discovery and execution of benchmark functions
specifically for the GLM-4-7 model.
"""

import csv
import importlib
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from benchmarks.core.benchmark_discovery import BenchmarkDiscovery


class GLM47BenchmarkDiscovery(BenchmarkDiscovery):
    """
    Specialized benchmark discovery for GLM-4-7 model benchmarks.
    """

    def __init__(self):
        """
        Initialize the GLM-4-7 benchmark discovery system.
        """
        super().__init__(search_paths=["src/inference_pio/models/glm_4_7/benchmarks"])
        self.model_name = "glm_4_7"


def run_glm_4_7_benchmarks(include_categories: List[str] = None) -> Dict[str, Any]:
    """
    Run all GLM-4-7 benchmarks.

    Args:
        include_categories: List of categories to include. If None, runs all.

    Returns:
        Dictionary with results from GLM-4-7 benchmarks
    """
    discovery = GLM47BenchmarkDiscovery()
    discovery.discover_benchmarks()

    print(
        f"Discovered {len(discovery.discovered_benchmarks)} GLM-4-7 benchmark functions"
    )

    results = discovery.run_model_benchmarks(discovery.model_name)

    # Save results
    discovery.save_results(results, output_dir="benchmark_results/glm_4_7")

    return results


def run_glm_4_7_performance_benchmarks() -> Dict[str, Any]:
    """
    Run only performance benchmarks for GLM-4-7.

    Returns:
        Dictionary with results from GLM-4-7 performance benchmarks
    """
    discovery = GLM47BenchmarkDiscovery()
    discovery.discover_benchmarks()

    # Filter to only performance benchmarks
    performance_benchmarks = discovery.get_benchmarks_by_category("performance")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "glm_4_7",
        "results": {},
        "summary": {},
    }

    print(f"Running {len(performance_benchmarks)} GLM-4-7 performance benchmarks...")

    for benchmark in performance_benchmarks:
        print(f"Running {benchmark['full_name']}...")
        result = discovery.run_benchmark(benchmark["function"])
        results["results"][benchmark["full_name"]] = result

    # Generate summary
    successful_runs = sum(1 for r in results["results"].values() if "error" not in r)
    total_runs = len(results["results"])

    results["summary"] = {
        "total_benchmarks": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": total_runs - successful_runs,
        "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
    }

    print(f"\nGLM-4-7 Performance Benchmark Summary:")
    print(f"  Total: {total_runs}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {total_runs - successful_runs}")
    print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

    # Save results
    discovery.save_results(results, output_dir="benchmark_results/glm_4_7")

    return results


if __name__ == "__main__":
    # Example usage
    results = run_glm_4_7_benchmarks()
    print("\nGLM-4-7 benchmark discovery and execution completed!")
