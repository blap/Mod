"""
Benchmark Discovery Module

This module provides automatic discovery of benchmark functions
across the project's standardized benchmark structure.
"""

import importlib
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List


class BenchmarkDiscovery:
    """
    A class to discover benchmark functions from standardized
    benchmark directories across the project.
    """

    def __init__(self, search_paths: List[str] = None):
        """
        Initialize the benchmark discovery system.

        Args:
            search_paths: List of paths to search for benchmarks. If None, uses default paths.
        """
        self.search_paths = search_paths or [
            "src/inference_pio/models",
            "src/inference_pio/plugin_system",
            "benchmarks",
        ]
        self.benchmark_functions: Dict[str, Callable] = {}
        self.discovered_benchmarks: List[Dict[str, Any]] = []

    def discover_benchmarks(self) -> List[Dict[str, Any]]:
        """
        Discover all benchmark functions in the search paths.

        Returns:
            List of dictionaries containing benchmark information
        """
        self.discovered_benchmarks = []

        for search_path in self.search_paths:
            path_obj = Path(search_path)
            if not path_obj.exists():
                print(f"Warning: Search path does not exist: {search_path}")
                continue

            # Look for benchmark directories in the standard structure
            for benchmark_dir in path_obj.rglob("benchmarks"):
                if benchmark_dir.is_dir():
                    self._discover_from_directory(benchmark_dir)

        return self.discovered_benchmarks

    def _discover_from_directory(self, benchmark_dir: Path) -> None:
        """
        Discover benchmark functions from a specific benchmark directory.

        Args:
            benchmark_dir: Path to the benchmark directory
        """
        # Look in subdirectories: unit, integration, performance
        subdirs = ["unit", "integration", "performance"]

        for subdir in subdirs:
            subdir_path = benchmark_dir / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                self._scan_directory_for_benchmarks(subdir_path, str(benchmark_dir))

    def _scan_directory_for_benchmarks(self, directory: Path, parent_dir: str) -> None:
        """
        Scan a directory for Python files containing benchmark functions.

        Args:
            directory: Directory to scan
            parent_dir: Parent directory path for context
        """
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("__"):
                continue  # Skip __init__.py and other special files

            module_name = self._get_module_name(py_file, parent_dir)

            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)

                # Add the directory to sys.path temporarily to handle imports
                module_dir = str(py_file.parent)
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)

                spec.loader.exec_module(module)

                # Find benchmark functions in the module
                benchmark_funcs = self._find_benchmark_functions(
                    module, module_name, str(py_file)
                )

                for func_info in benchmark_funcs:
                    self.discovered_benchmarks.append(func_info)

            except ImportError as e:
                print(f"Warning: Could not import {py_file}: {e}")
            except Exception as e:
                print(f"Warning: Error processing {py_file}: {e}")
            finally:
                # Remove the directory from sys.path if we added it
                if module_dir in sys.path:
                    sys.path.remove(module_dir)

    def _get_module_name(self, file_path: Path, parent_dir: str) -> str:
        """
        Generate a module name based on the file path.

        Args:
            file_path: Path to the Python file
            parent_dir: Parent directory for context

        Returns:
            Module name as a string
        """
        # Convert file path to module name
        rel_path = file_path.relative_to(Path("."))
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
        return module_name

    def _find_benchmark_functions(
        self, module: Any, module_name: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Find all benchmark functions in a module.

        Args:
            module: The imported module
            module_name: Name of the module
            file_path: Path to the file

        Returns:
            List of benchmark function information
        """
        benchmark_funcs = []

        for name, obj in inspect.getmembers(module):
            # Check if it's a callable function that starts with 'run_' or 'benchmark_'
            if (inspect.isfunction(obj) or inspect.ismethod(obj)) and (
                name.startswith("run_") or name.startswith("benchmark_")
            ):
                # Skip private functions
                if not name.startswith("_"):
                    func_info = {
                        "name": name,
                        "function": obj,
                        "module_name": module_name,
                        "file_path": file_path,
                        "full_name": f"{module_name}.{name}",
                        "category": self._determine_category(file_path),
                        "model_name": self._extract_model_name(file_path),
                    }
                    benchmark_funcs.append(func_info)

        return benchmark_funcs

    def _determine_category(self, file_path: str) -> str:
        """
        Determine the category of a benchmark based on its path.

        Args:
            file_path: Path to the benchmark file

        Returns:
            Category string ('unit', 'integration', 'performance', or 'other')
        """
        path_lower = file_path.lower()
        if "unit" in path_lower:
            return "unit"
        elif "integration" in path_lower:
            return "integration"
        elif "performance" in path_lower:
            return "performance"
        else:
            return "other"

    def _extract_model_name(self, file_path: str) -> str:
        """
        Extract the model name from the file path.

        Args:
            file_path: Path to the benchmark file

        Returns:
            Model name string
        """
        # Look for model names in the path
        path_parts = file_path.split(os.sep)
        for i, part in enumerate(path_parts):
            if "models" in part and i + 1 < len(path_parts):
                # The next part should be the model name
                model_part = path_parts[i + 1]
                # Handle the case where the model name has hyphens or other separators
                # Convert to a cleaner format if needed
                return model_part.replace("-", "_").replace(".", "_")

        # If not in models directory, check for plugin_system
        if "plugin_system" in file_path:
            return "plugin_system"

        return "general"

    def get_benchmarks_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all benchmarks in a specific category.

        Args:
            category: Category to filter by ('unit', 'integration', 'performance', 'other')

        Returns:
            List of benchmark functions in the category
        """
        return [b for b in self.discovered_benchmarks if b["category"] == category]

    def get_benchmarks_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all benchmarks for a specific model.

        Args:
            model_name: Name of the model to filter by

        Returns:
            List of benchmark functions for the model
        """
        return [b for b in self.discovered_benchmarks if b["model_name"] == model_name]


def discover_and_run_all_benchmarks() -> Dict[str, Any]:
    """
    Convenience function to discover and run all benchmarks in the project.

    Returns:
        Dictionary with results from all benchmarks
    """
    from .benchmark_results_handler import BenchmarkResultsHandler
    from .benchmark_runner import BenchmarkRunner

    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    print(f"Discovered {len(discovery.discovered_benchmarks)} benchmark functions")

    # Run all benchmarks using the runner
    runner = BenchmarkRunner(discovery.discovered_benchmarks)

    # Run all benchmarks
    results = runner.run_all_benchmarks()

    # Save results using the handler
    handler = BenchmarkResultsHandler()
    handler.save_results(results)

    return results
