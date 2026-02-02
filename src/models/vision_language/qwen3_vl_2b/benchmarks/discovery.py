"""
Qwen3-vl-2b Model Benchmark Discovery Module

This module provides automatic discovery and execution of benchmark functions
specifically for the Qwen3-vl-2b model.
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


class Qwen3Vl2bBenchmarkDiscovery(BenchmarkDiscovery):
    """
    Specialized benchmark discovery for Qwen3-vl-2b model benchmarks.
    """

    def __init__(self):
        """
        Initialize the Qwen3-vl-2b benchmark discovery system.
        """
        super().__init__(
            search_paths=["src/inference_pio/models/qwen3_vl_2b/benchmarks"]
        )
        self.model_name = "qwen3_vl_2b"


def run_qwen3_vl_2b_benchmarks(include_categories: List[str] = None) -> Dict[str, Any]:
    """
    Run all Qwen3-vl-2b benchmarks.

    Args:
        include_categories: List of categories to include. If None, runs all.

    Returns:
        Dictionary with results from Qwen3-vl-2b benchmarks
    """
    discovery = Qwen3Vl2bBenchmarkDiscovery()
    discovery.discover_benchmarks()

    print(
        f"Discovered {len(discovery.discovered_benchmarks)} Qwen3-vl-2b benchmark functions"
    )

    results = discovery.run_model_benchmarks(discovery.model_name)

    # Save results
    discovery.save_results(results, output_dir="benchmark_results/qwen3_vl_2b")

    return results


def run_qwen3_vl_2b_performance_benchmarks() -> Dict[str, Any]:
    """
    Run only performance benchmarks for Qwen3-vl-2b.

    Returns:
        Dictionary with results from Qwen3-vl-2b performance benchmarks
    """
    discovery = Qwen3Vl2bBenchmarkDiscovery()
    discovery.discover_benchmarks()

    # Filter to only performance benchmarks
    performance_benchmarks = discovery.get_benchmarks_by_category("performance")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "qwen3_vl_2b",
        "results": {},
        "summary": {},
    }

    print(
        f"Running {len(performance_benchmarks)} Qwen3-vl-2b performance benchmarks..."
    )

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

    print(f"\nQwen3-vl-2b Performance Benchmark Summary:")
    print(f"  Total: {total_runs}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {total_runs - successful_runs}")
    print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

    # Save results
    discovery.save_results(results, output_dir="benchmark_results/qwen3_vl_2b")

    return results


if __name__ == "__main__":
    # Example usage
    results = run_qwen3_vl_2b_benchmarks()
    print("\nQwen3-vl-2b benchmark discovery and execution completed!")
