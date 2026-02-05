"""
Unified Benchmark Discovery System

This module provides a comprehensive discovery system for all benchmarks in the project.
It supports multiple discovery methods and integrates with the standardized benchmark interface.
"""

import csv
import fnmatch
import importlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult


@dataclass
class DiscoveredBenchmark:
    """Represents a discovered benchmark."""

    name: str
    function: Optional[Callable] = None
    benchmark_class: Optional[Type[BaseBenchmark]] = None
    module_name: str = ""
    file_path: str = ""
    full_name: str = ""
    category: str = "other"
    model_name: str = "general"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class UnifiedBenchmarkDiscoverer:
    """
    A unified system for discovering benchmarks across the project.
    Supports both function-based and class-based benchmarks.
    """

    def __init__(
        self,
        search_paths: List[str] = None,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
    ):
        """
        Initialize the unified benchmark discoverer.

        Args:
            search_paths: List of paths to search for benchmarks
            include_patterns: Patterns to include (fnmatch style)
            exclude_patterns: Patterns to exclude (fnmatch style)
        """
        self.search_paths = search_paths or [
            "src/inference_pio/models",
            "src/inference_pio/plugin_system",
            "benchmarks",
            "tests",
        ]
        self.include_patterns = include_patterns or ["*.py"]
        self.exclude_patterns = exclude_patterns or [
            "*_test.py",
            "test_*.py",
            "__pycache__/*",
        ]
        self.discovered_benchmarks: List[DiscoveredBenchmark] = []
        self.benchmark_categories = [
            "unit",
            "integration",
            "performance",
            "accuracy",
            "stress",
            "regression",
        ]

    def _matches_pattern(self, path: str, patterns: List[str]) -> bool:
        """Check if a path matches any of the given patterns."""
        path_str = str(Path(path).as_posix())  # Normalize path separators
        for pattern in patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    def _should_include_path(self, path: str) -> bool:
        """Determine if a path should be included in discovery."""
        # Check exclude patterns first
        if self._matches_pattern(path, self.exclude_patterns):
            return False

        # Then check include patterns
        if self.include_patterns:
            return self._matches_pattern(path, self.include_patterns)

        return True

    def discover_benchmarks(self) -> List[DiscoveredBenchmark]:
        """
        Discover all benchmarks in the search paths.

        Returns:
            List of discovered benchmarks
        """
        self.discovered_benchmarks = []

        for search_path in self.search_paths:
            path_obj = Path(search_path)
            if not path_obj.exists():
                print(f"Warning: Search path does not exist: {search_path}")
                continue

            # Walk through all Python files in the path
            for py_file in path_obj.rglob("*.py"):
                if not self._should_include_path(str(py_file)):
                    continue

                self._discover_from_file(py_file)

        return self.discovered_benchmarks

    def _discover_from_file(self, py_file: Path) -> None:
        """Discover benchmarks from a single Python file."""
        module_name = self._get_module_name(py_file)

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)

            # Add the directory to sys.path temporarily to handle imports
            module_dir = str(py_file.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            spec.loader.exec_module(module)

            # Find function-based benchmarks
            function_benchmarks = self._find_function_benchmarks(
                module, module_name, str(py_file)
            )
            self.discovered_benchmarks.extend(function_benchmarks)

            # Find class-based benchmarks
            class_benchmarks = self._find_class_benchmarks(
                module, module_name, str(py_file)
            )
            self.discovered_benchmarks.extend(class_benchmarks)

        except ImportError as e:
            print(f"Warning: Could not import {py_file}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {py_file}: {e}")
        finally:
            # Remove the directory from sys.path if we added it
            if module_dir in sys.path and module_dir in sys.path:
                sys.path.remove(module_dir)

    def _get_module_name(self, file_path: Path) -> str:
        """Generate a module name based on the file path."""
        # Convert file path to module name
        rel_path = file_path.relative_to(Path("."))
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
        return module_name

    def _find_function_benchmarks(
        self, module: Any, module_name: str, file_path: str
    ) -> List[DiscoveredBenchmark]:
        """Find function-based benchmarks in a module."""
        benchmarks = []

        for name, obj in inspect.getmembers(module):
            # Check if it's a callable function that starts with 'run_' or 'benchmark_'
            if (inspect.isfunction(obj) or inspect.ismethod(obj)) and (
                name.startswith("run_") or name.startswith("benchmark_")
            ):
                # Skip private functions
                if not name.startswith("_"):
                    benchmark = DiscoveredBenchmark(
                        name=name,
                        function=obj,
                        module_name=module_name,
                        file_path=file_path,
                        full_name=f"{module_name}.{name}",
                        category=self._determine_category(file_path),
                        model_name=self._extract_model_name(file_path),
                        tags=self._extract_tags(name),
                    )
                    benchmarks.append(benchmark)

        return benchmarks

    def _find_class_benchmarks(
        self, module: Any, module_name: str, file_path: str
    ) -> List[DiscoveredBenchmark]:
        """Find class-based benchmarks in a module."""
        benchmarks = []

        for name, obj in inspect.getmembers(module):
            # Check if it's a class that inherits from BaseBenchmark
            if (
                inspect.isclass(obj)
                and obj != BaseBenchmark
                and issubclass(obj, BaseBenchmark)
            ):

                # Create an instance to get the benchmark info
                benchmark = DiscoveredBenchmark(
                    name=name,
                    benchmark_class=obj,
                    module_name=module_name,
                    file_path=file_path,
                    full_name=f"{module_name}.{name}",
                    category=self._determine_category(file_path),
                    model_name=self._extract_model_name(file_path),
                    tags=["class-based"],
                )
                benchmarks.append(benchmark)

        return benchmarks

    def _determine_category(self, file_path: str) -> str:
        """Determine the category of a benchmark based on its path."""
        path_lower = file_path.lower()
        for category in self.benchmark_categories:
            if category in path_lower:
                return category
        return "other"

    def _extract_model_name(self, file_path: str) -> str:
        """Extract the model name from the file path."""
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

    def _extract_tags(self, function_name: str) -> List[str]:
        """Extract tags from function name."""
        tags = []
        name_lower = function_name.lower()

        if "speed" in name_lower or "performance" in name_lower:
            tags.append("performance")
        if "accuracy" in name_lower:
            tags.append("accuracy")
        if "memory" in name_lower:
            tags.append("memory")
        if "throughput" in name_lower:
            tags.append("throughput")
        if "latency" in name_lower:
            tags.append("latency")

        return tags

    def get_benchmarks_by_category(self, category: str) -> List[DiscoveredBenchmark]:
        """Get all benchmarks in a specific category."""
        return [b for b in self.discovered_benchmarks if b.category == category]

    def get_benchmarks_by_model(self, model_name: str) -> List[DiscoveredBenchmark]:
        """Get all benchmarks for a specific model."""
        return [b for b in self.discovered_benchmarks if b.model_name == model_name]

    def get_benchmarks_by_tag(self, tag: str) -> List[DiscoveredBenchmark]:
        """Get all benchmarks with a specific tag."""
        return [b for b in self.discovered_benchmarks if tag in b.tags]

    def instantiate_benchmark(
        self, discovered_benchmark: DiscoveredBenchmark, *args, **kwargs
    ):
        """Instantiate a benchmark from a discovered benchmark definition."""
        if discovered_benchmark.benchmark_class:
            # Class-based benchmark
            return discovered_benchmark.benchmark_class(*args, **kwargs)
        elif discovered_benchmark.function:
            # Function-based benchmark
            return discovered_benchmark.function
        else:
            raise ValueError(
                f"No executable found for benchmark: {discovered_benchmark.name}"
            )

    def run_benchmark(
        self, discovered_benchmark: DiscoveredBenchmark, *args, **kwargs
    ) -> Any:
        """Run a single discovered benchmark."""
        try:
            if discovered_benchmark.benchmark_class:
                # Instantiate and run class-based benchmark
                benchmark_instance = self.instantiate_benchmark(
                    discovered_benchmark, *args, **kwargs
                )
                return benchmark_instance.run()
            elif discovered_benchmark.function:
                # Run function-based benchmark
                return discovered_benchmark.function(*args, **kwargs)
            else:
                return {"error": "No executable benchmark found"}
        except Exception as e:
            print(f"Error running benchmark {discovered_benchmark.full_name}: {e}")
            return {"error": str(e)}

    def run_all_benchmarks(
        self, include_categories: List[str] = None, include_models: List[str] = None
    ) -> Dict[str, Any]:
        """Run all discovered benchmarks with optional filtering."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "results": {},
            "summary": {},
        }

        benchmarks_to_run = self.discovered_benchmarks

        # Apply category filter
        if include_categories:
            benchmarks_to_run = [
                b for b in benchmarks_to_run if b.category in include_categories
            ]

        # Apply model filter
        if include_models:
            benchmarks_to_run = [
                b for b in benchmarks_to_run if b.model_name in include_models
            ]

        print(f"Running {len(benchmarks_to_run)} benchmarks...")

        for benchmark in benchmarks_to_run:
            print(f"Running {benchmark.full_name}...")
            result = self.run_benchmark(benchmark)
            results["results"][benchmark.full_name] = result

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

    def save_discovery_results(
        self,
        output_dir: str = "benchmark_results",
        filename: str = "discovered_benchmarks.json",
    ) -> str:
        """Save discovery results to a file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Prepare discovery data
        discovery_data = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self.discovered_benchmarks),
            "benchmarks": [
                {
                    "name": b.name,
                    "module_name": b.module_name,
                    "file_path": b.file_path,
                    "full_name": b.full_name,
                    "category": b.category,
                    "model_name": b.model_name,
                    "tags": b.tags,
                    "has_function": b.function is not None,
                    "has_class": b.benchmark_class is not None,
                }
                for b in self.discovered_benchmarks
            ],
            "category_counts": {
                cat: len(self.get_benchmarks_by_category(cat))
                for cat in self.benchmark_categories + ["other"]
            },
            "model_counts": {
                model: len(self.get_benchmarks_by_model(model))
                for model in set(b.model_name for b in self.discovered_benchmarks)
            },
        }

        filepath = output_path / filename
        with open(filepath, "w") as f:
            json.dump(discovery_data, f, indent=2, default=str)

        print(f"Discovery results saved to: {filepath}")
        return str(filepath)

    def print_discovery_summary(self) -> None:
        """Print a summary of discovered benchmarks."""
        print("\n" + "=" * 80)
        print("UNIFIED BENCHMARK DISCOVERY SUMMARY")
        print("=" * 80)
        print(f"Total Benchmarks Discovered: {len(self.discovered_benchmarks)}")

        # Category breakdown
        print(f"\nBy Category:")
        for cat in self.benchmark_categories + ["other"]:
            count = len(self.get_benchmarks_by_category(cat))
            if count > 0:
                print(f"  {cat}: {count}")

        # Model breakdown
        print(f"\nBy Model:")
        models = set(b.model_name for b in self.discovered_benchmarks)
        for model in sorted(models):
            count = len(self.get_benchmarks_by_model(model))
            print(f"  {model}: {count}")

        # Tag breakdown
        all_tags = set(tag for b in self.discovered_benchmarks for tag in b.tags)
        if all_tags:
            print(f"\nBy Tag:")
            for tag in sorted(all_tags):
                count = len(self.get_benchmarks_by_tag(tag))
                print(f"  {tag}: {count}")


def discover_and_run_all_benchmarks() -> Dict[str, Any]:
    """
    Convenience function to discover and run all benchmarks in the project.

    Returns:
        Dictionary with results from all benchmarks
    """
    discoverer = UnifiedBenchmarkDiscoverer()
    discovered_benchmarks = discoverer.discover_benchmarks()

    print(f"Discovered {len(discovered_benchmarks)} benchmark functions/classes")

    # Print discovery summary
    discoverer.print_discovery_summary()

    # Run all benchmarks
    results = discoverer.run_all_benchmarks()

    # Save discovery results
    discoverer.save_discovery_results()

    return results


# Backward compatibility wrapper for the old discovery system
class BenchmarkDiscovery(UnifiedBenchmarkDiscoverer):
    """Backward compatible wrapper for the old discovery system."""

    raise NotImplementedError("Method not implemented")


if __name__ == "__main__":
    # Example usage
    results = discover_and_run_all_benchmarks()
    print("\nUnified benchmark discovery and execution completed!")
