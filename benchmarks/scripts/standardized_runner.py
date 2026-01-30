"""
Standardized Benchmark Runner for All Models

This module provides a unified interface to run benchmarks across all models in the project.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import importlib
import inspect
from datetime import datetime
import json
import csv

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add src directory to path
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from inference_pio.common.benchmark_interface import (
    BenchmarkRunner,
    get_full_suite,
    get_performance_suite,
    get_accuracy_suite
)


class ModelBenchmarkRunner:
    """Class to run benchmarks across multiple models."""

    def __init__(self, model_dirs: List[str] = None):
        """
        Initialize the model benchmark runner.

        Args:
            model_dirs: List of directories containing model plugins. 
                       If None, uses default model directories.
        """
        if model_dirs is None:
            self.model_dirs = [
                "src/inference_pio/models/glm_4_7_flash",
                "src/inference_pio/models/qwen3_4b_instruct_2507", 
                "src/inference_pio/models/qwen3_coder_30b",
                "src/inference_pio/models/qwen3_vl_2b"
            ]
        else:
            self.model_dirs = model_dirs

        self.runner = BenchmarkRunner()

    def discover_models(self) -> List[Dict[str, str]]:
        """
        Discover all available models in the specified directories.

        Returns:
            List of dictionaries containing model information
        """
        models = []

        for model_dir in self.model_dirs:
            dir_path = Path(model_dir)
            if not dir_path.exists():
                print(f"Warning: Model directory does not exist: {model_dir}")
                continue

            # Look for plugin.py files which should contain the model plugin
            plugin_file = dir_path / "plugin.py"
            if plugin_file.exists():
                # Extract model name from directory name
                model_name = dir_path.name.replace('_', '-')
                
                models.append({
                    'name': model_name,
                    'directory': str(dir_path),
                    'plugin_file': str(plugin_file)
                })

        return models

    def load_model_plugin(self, model_info: Dict[str, str]):
        """
        Load the plugin for a specific model.

        Args:
            model_info: Dictionary containing model information

        Returns:
            The loaded plugin instance
        """
        plugin_path = Path(model_info['plugin_file'])
        model_dir = plugin_path.parent

        # Add model directory to Python path temporarily
        sys.path.insert(0, str(model_dir))

        try:
            # Import the plugin module
            spec = importlib.util.spec_from_file_location("model_plugin", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)

            # Find the create_*_plugin function
            create_func_names = [name for name in dir(plugin_module) 
                                if name.startswith('create_') and name.endswith('_plugin')]
            
            if not create_func_names:
                raise ValueError(f"No create_*_plugin function found in {plugin_path}")
            
            create_func_name = create_func_names[0]
            create_func = getattr(plugin_module, create_func_name)

            # Create and return the plugin instance
            plugin = create_func()
            return plugin

        finally:
            # Remove the directory from Python path
            if str(model_dir) in sys.path:
                sys.path.remove(str(model_dir))

    def run_benchmarks_for_model(self, model_info: Dict[str, str], 
                                benchmark_suite: str = 'full') -> List[Any]:
        """
        Run benchmarks for a specific model.

        Args:
            model_info: Dictionary containing model information
            benchmark_suite: Type of benchmark suite to run ('full', 'performance', 'accuracy')

        Returns:
            List of benchmark results
        """
        print(f"\nLoading model: {model_info['name']}")
        
        try:
            # Load the model plugin
            plugin = self.load_model_plugin(model_info)
            
            # Initialize the plugin (using CPU for consistent benchmarks)
            success = plugin.initialize(device="cpu", use_mock_model=True)
            if not success:
                print(f"Warning: Could not initialize {model_info['name']}")
                return []

            print(f"Running {benchmark_suite} benchmarks for {model_info['name']}...")

            # Select the appropriate benchmark suite
            if benchmark_suite == 'performance':
                benchmarks = get_performance_suite(plugin, model_info['name'])
            elif benchmark_suite == 'accuracy':
                benchmarks = get_accuracy_suite(plugin, model_info['name'])
            else:  # full
                benchmarks = get_full_suite(plugin, model_info['name'])

            # Run the benchmarks
            results = self.runner.run_multiple_benchmarks(benchmarks)

            print(f"Completed {len(results)} benchmarks for {model_info['name']}")

            return results

        except Exception as e:
            print(f"Error running benchmarks for {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def run_all_benchmarks(self, benchmark_suite: str = 'full') -> Dict[str, Any]:
        """
        Run benchmarks for all discovered models.

        Args:
            benchmark_suite: Type of benchmark suite to run ('full', 'performance', 'accuracy')

        Returns:
            Dictionary with aggregated results
        """
        print("="*80)
        print(f"RUNNING {benchmark_suite.upper()} BENCHMARKS FOR ALL MODELS")
        print("="*80)

        # Discover all models
        models = self.discover_models()
        print(f"Discovered {len(models)} models")

        all_results = {}

        # Run benchmarks for each model
        for model_info in models:
            model_results = self.run_benchmarks_for_model(model_info, benchmark_suite)
            all_results[model_info['name']] = model_results

        # Print summary
        self.runner.print_summary()

        # Prepare aggregated results
        aggregated_results = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_suite': benchmark_suite,
            'models_processed': len(models),
            'results': all_results,
            'summary': self._generate_summary(all_results)
        }

        return aggregated_results

    def _generate_summary(self, all_results: Dict[str, List]) -> Dict[str, Any]:
        """
        Generate a summary of all benchmark results.

        Args:
            all_results: Dictionary containing results for all models

        Returns:
            Summary dictionary
        """
        summary = {}

        for model_name, results in all_results.items():
            model_summary = {}
            
            for result in results:
                if hasattr(result, 'name') and hasattr(result, 'value'):
                    # Store key metrics for summary
                    if 'inference_speed' in result.name:
                        model_summary[result.name] = {
                            'value': result.value,
                            'unit': result.unit
                        }
                    elif 'memory_usage' in result.name:
                        model_summary[result.name] = {
                            'value': result.value,
                            'unit': result.unit
                        }
                    elif 'accuracy' in result.name:
                        model_summary[result.name] = {
                            'value': result.value,
                            'unit': result.unit
                        }

            summary[model_name] = model_summary

        return summary

    def save_aggregated_results(self, results: Dict[str, Any], 
                              output_dir: str = "benchmark_results",
                              filename_prefix: str = "aggregated_benchmarks") -> Dict[str, str]:
        """
        Save aggregated benchmark results to files.

        Args:
            results: Aggregated results dictionary
            output_dir: Directory to save results
            filename_prefix: Prefix for output files

        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_filename = output_path / f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV summary
        csv_filename = output_path / f"{filename_prefix}_summary_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Model Name', 'Benchmark Suite', 'Models Processed', 
                'Total Benchmarks', 'Timestamp'
            ])

            # Write summary row
            writer.writerow([
                'ALL_MODELS',
                results.get('benchmark_suite', 'unknown'),
                results.get('models_processed', 0),
                sum(len(results_list) for results_list in results.get('results', {}).values()),
                results.get('timestamp', '')
            ])

        print(f"\nAggregated results saved to:")
        print(f"  JSON: {json_filename}")
        print(f"  CSV: {csv_filename}")

        return {
            'json_file': str(json_filename),
            'csv_file': str(csv_filename)
        }


def run_standardized_benchmarks(benchmark_suite: str = 'full'):
    """
    Convenience function to run standardized benchmarks across all models.

    Args:
        benchmark_suite: Type of benchmark suite to run ('full', 'performance', 'accuracy')

    Returns:
        Dictionary with aggregated results
    """
    runner = ModelBenchmarkRunner()
    results = runner.run_all_benchmarks(benchmark_suite)
    
    # Save results
    runner.save_aggregated_results(results)
    
    print(f"\nStandardized benchmarking completed for {benchmark_suite} suite!")
    return results


if __name__ == "__main__":
    # Run full benchmark suite by default
    results = run_standardized_benchmarks(benchmark_suite='full')