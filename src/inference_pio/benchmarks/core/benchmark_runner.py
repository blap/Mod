"""
Benchmark Runner Module

This module provides the logic for executing discovered benchmarks.
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, List


class BenchmarkRunner:
    """
    A class to run benchmark functions safely and collect results.
    """

    def __init__(self, benchmarks: List[Dict[str, Any]] = None):
        """
        Initialize the benchmark runner.

        Args:
            benchmarks: List of benchmark dictionaries to run
        """
        self.benchmarks = benchmarks or []

    def run_benchmark(self, benchmark_func: Callable, *args, **kwargs) -> Any:
        """
        Run a single benchmark function safely.

        Args:
            benchmark_func: The benchmark function to run
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the benchmark function
        """
        try:
            return benchmark_func(*args, **kwargs)
        except Exception as e:
            print(f"Error running benchmark {benchmark_func.__name__}: {e}")
            return {"error": str(e)}

    def run_all_benchmarks(
        self, include_categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run all discovered benchmarks.

        Args:
            include_categories: List of categories to include. If None, runs all.

        Returns:
            Dictionary with results from all benchmarks
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "results": {},
            "summary": {},
        }

        benchmarks_to_run = self.benchmarks

        if include_categories:
            benchmarks_to_run = [
                b for b in self.benchmarks if b["category"] in include_categories
            ]

        print(f"Running {len(benchmarks_to_run)} benchmarks...")

        for benchmark in benchmarks_to_run:
            print(f"Running {benchmark['full_name']}...")
            result = self.run_benchmark(benchmark["function"])
            results["results"][benchmark["full_name"]] = result

        # Generate summary
        successful_runs = sum(
            1 for r in results["results"].values() if "error" not in r
        )
        total_runs = len(results["results"])

        results["summary"] = {
            "total_benchmarks": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": total_runs - successful_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
        }

        print(f"\nBenchmark Summary:")
        print(f"  Total: {total_runs}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {total_runs - successful_runs}")
        print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

        return results

    def run_model_benchmarks(self, model_name: str) -> Dict[str, Any]:
        """
        Run all benchmarks for a specific model.

        Args:
            model_name: Name of the model to run benchmarks for

        Returns:
            Dictionary with results from model benchmarks
        """
        model_benchmarks = [b for b in self.benchmarks if b["model_name"] == model_name]

        results = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "results": {},
            "summary": {},
        }

        print(f"Running {len(model_benchmarks)} benchmarks for model {model_name}...")

        for benchmark in model_benchmarks:
            print(f"Running {benchmark['full_name']}...")
            result = self.run_benchmark(benchmark["function"])
            results["results"][benchmark["full_name"]] = result

        # Generate summary
        successful_runs = sum(
            1 for r in results["results"].values() if "error" not in r
        )
        total_runs = len(results["results"])

        results["summary"] = {
            "total_benchmarks": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": total_runs - successful_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
        }

        print(f"\nModel {model_name} Benchmark Summary:")
        print(f"  Total: {total_runs}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {total_runs - successful_runs}")
        print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

        return results

    def save_results(
        self, results: Dict[str, Any], output_dir: str = "benchmark_results"
    ):
        """
        Save benchmark results using the results handler.

        Args:
            results: Results dictionary to save
            output_dir: Directory to save results to
        """
        from .benchmark_results_handler import BenchmarkResultsHandler

        handler = BenchmarkResultsHandler()
        handler.save_results(results, output_dir)
