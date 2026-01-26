"""
Plugin System Benchmark Discovery Module

This module provides automatic discovery and execution of benchmark functions
specifically for the plugin system.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Callable, Any
from datetime import datetime
import json
import csv

from benchmarks.discovery import BenchmarkDiscovery


class PluginSystemBenchmarkDiscovery(BenchmarkDiscovery):
    """
    Specialized benchmark discovery for plugin system benchmarks.
    """

    def __init__(self):
        """
        Initialize the plugin system benchmark discovery system.
        """
        super().__init__(search_paths=[
            "src/inference_pio/plugin_system/benchmarks"
        ])
        self.model_name = "plugin_system"


def run_plugin_system_benchmarks(include_categories: List[str] = None) -> Dict[str, Any]:
    """
    Run all plugin system benchmarks.

    Args:
        include_categories: List of categories to include. If None, runs all.

    Returns:
        Dictionary with results from plugin system benchmarks
    """
    discovery = PluginSystemBenchmarkDiscovery()
    discovery.discover_benchmarks()
    
    print(f"Discovered {len(discovery.discovered_benchmarks)} plugin system benchmark functions")
    
    results = discovery.run_model_benchmarks(discovery.model_name)
    
    # Save results
    discovery.save_results(results, output_dir="benchmark_results/plugin_system")
    
    return results


def run_plugin_system_performance_benchmarks() -> Dict[str, Any]:
    """
    Run only performance benchmarks for plugin system.

    Returns:
        Dictionary with results from plugin system performance benchmarks
    """
    discovery = PluginSystemBenchmarkDiscovery()
    discovery.discover_benchmarks()
    
    # Filter to only performance benchmarks
    performance_benchmarks = discovery.get_benchmarks_by_category('performance')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'plugin_system',
        'results': {},
        'summary': {}
    }
    
    print(f"Running {len(performance_benchmarks)} plugin system performance benchmarks...")
    
    for benchmark in performance_benchmarks:
        print(f"Running {benchmark['full_name']}...")
        result = discovery.run_benchmark(benchmark['function'])
        results['results'][benchmark['full_name']] = result
    
    # Generate summary
    successful_runs = sum(1 for r in results['results'].values() if 'error' not in r)
    total_runs = len(results['results'])
    
    results['summary'] = {
        'total_benchmarks': total_runs,
        'successful_runs': successful_runs,
        'failed_runs': total_runs - successful_runs,
        'success_rate': successful_runs / total_runs if total_runs > 0 else 0
    }
    
    print(f"\nPlugin System Performance Benchmark Summary:")
    print(f"  Total: {total_runs}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {total_runs - successful_runs}")
    print(f"  Success Rate: {results['summary']['success_rate']:.2%}")
    
    # Save results
    discovery.save_results(results, output_dir="benchmark_results/plugin_system")
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run_plugin_system_benchmarks()
    print("\nPlugin system benchmark discovery and execution completed!")