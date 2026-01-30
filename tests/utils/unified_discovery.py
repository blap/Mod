"""
Unified Test Discovery System for Inference-PIO

This module provides a comprehensive test and benchmark discovery mechanism that can find and run
both traditional tests and benchmark functions in the standardized test and benchmark structures.
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import List, Callable, Dict, Any, Union
from enum import Enum
import re
import fnmatch


class TestType(Enum):
    """Enumeration for different types of discovered items."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"


class UnifiedTestDiscovery:
    """
    A unified class to discover, collect, and run both tests and benchmarks
    from standardized test and benchmark directories across the project.
    """

    def __init__(self, search_paths: List[str] = None):
        """
        Initialize the unified discovery system.

        Args:
            search_paths: List of paths to search for tests and benchmarks. If None, uses default paths.
        """
        self.search_paths = search_paths or [
            "src/inference_pio/tests",
            "src/inference_pio/models",
            "src/inference_pio/plugin_system",
            "tests",
            "benchmarks",
            "standardized_tests",
            "standardized_dev_tests"
        ]
        self.discovered_items = []
        self.test_functions = []
        self.benchmark_functions = []

    def discover_all(self) -> List[Dict[str, Any]]:
        """
        Discover all test and benchmark functions in the search paths.

        Returns:
            List of dictionaries containing discovered item information
        """
        self.discovered_items = []
        self.test_functions = []
        self.benchmark_functions = []

        for search_path in self.search_paths:
            path_obj = Path(search_path)
            if not path_obj.exists():
                print(f"Warning: Search path does not exist: {search_path}")
                continue

            # Look for test and benchmark directories in the standard structure
            for test_dir in path_obj.rglob("tests"):
                if test_dir.is_dir():
                    self._discover_from_test_directory(test_dir)

            for benchmark_dir in path_obj.rglob("benchmarks"):
                if benchmark_dir.is_dir():
                    self._discover_from_benchmark_directory(benchmark_dir)

        # Separate test and benchmark functions
        for item in self.discovered_items:
            if item['type'] in [TestType.UNIT_TEST, TestType.INTEGRATION_TEST, TestType.PERFORMANCE_TEST]:
                self.test_functions.append(item)
            elif item['type'] == TestType.BENCHMARK:
                self.benchmark_functions.append(item)

        return self.discovered_items

    def _discover_from_test_directory(self, test_dir: Path):
        """
        Discover test functions from a specific test directory.

        Args:
            test_dir: Path to the test directory
        """
        # Look in subdirectories: unit, integration, performance
        subdirs = ["unit", "integration", "performance"]

        for subdir in subdirs:
            subdir_path = test_dir / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                self._scan_directory_for_tests(subdir_path, str(test_dir), subdir)

    def _discover_from_benchmark_directory(self, benchmark_dir: Path):
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
                self._scan_directory_for_benchmarks(subdir_path, str(benchmark_dir), subdir)

    def _scan_directory_for_tests(self, directory: Path, parent_dir: str, category: str):
        """
        Scan a directory for Python files containing test functions.

        Args:
            directory: Directory to scan
            parent_dir: Parent directory for context
            category: Category of tests (unit, integration, performance)
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

                # Find test functions in the module
                test_funcs = self._find_test_functions(module, module_name, str(py_file), category)

                for func_info in test_funcs:
                    self.discovered_items.append(func_info)

            except ImportError as e:
                print(f"Warning: Could not import {py_file}: {e}")
            except Exception as e:
                print(f"Warning: Error processing {py_file}: {e}")
            finally:
                # Remove the directory from sys.path if we added it
                if module_dir in sys.path:
                    sys.path.remove(module_dir)

    def _scan_directory_for_benchmarks(self, directory: Path, parent_dir: str, category: str):
        """
        Scan a directory for Python files containing benchmark functions.

        Args:
            directory: Directory to scan
            parent_dir: Parent directory for context
            category: Category of benchmarks (unit, integration, performance)
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
                benchmark_funcs = self._find_benchmark_functions(module, module_name, str(py_file), category)

                for func_info in benchmark_funcs:
                    self.discovered_items.append(func_info)

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
        rel_path = file_path.relative_to(Path('.'))
        module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
        return module_name

    def _find_test_functions(self, module: Any, module_name: str, file_path: str, category: str) -> List[Dict[str, Any]]:
        """
        Find all test functions in a module.

        Args:
            module: The imported module
            module_name: Name of the module
            file_path: Path to the file
            category: Category of tests (unit, integration, performance)

        Returns:
            List of test function information
        """
        test_funcs = []

        for name, obj in inspect.getmembers(module):
            # Check if it's a callable function that matches test naming patterns
            if (inspect.isfunction(obj) or inspect.ismethod(obj)) and self._is_test_function(name):
                # Skip private functions
                if not name.startswith('_'):
                    func_info = {
                        'name': name,
                        'function': obj,
                        'module_name': module_name,
                        'file_path': file_path,
                        'full_name': f"{module_name}.{name}",
                        'type': self._determine_test_type(category),
                        'category': category,
                        'model_name': self._extract_model_name(file_path)
                    }
                    test_funcs.append(func_info)

            # Also check for test methods in classes
            if inspect.isclass(obj):
                for method_name in dir(obj):
                    if self._is_test_function(method_name):
                        method_obj = getattr(obj, method_name)
                        if inspect.isfunction(method_obj) or inspect.ismethod(method_obj):
                            # Create a wrapper that instantiates the class when called
                            def make_wrapper(cls_type, method_name_str):
                                def wrapper(*args, **kwargs):
                                    instance = cls_type()
                                    method = getattr(instance, method_name_str)
                                    return method(*args, **kwargs)
                                return wrapper

                            wrapped_method = make_wrapper(obj, method_name)
                            wrapped_method.__name__ = f"{obj.__name__}.{method_name}"

                            func_info = {
                                'name': f"{obj.__name__}.{method_name}",
                                'function': wrapped_method,
                                'module_name': module_name,
                                'file_path': file_path,
                                'full_name': f"{module_name}.{obj.__name__}.{method_name}",
                                'type': self._determine_test_type(category),
                                'category': category,
                                'model_name': self._extract_model_name(file_path)
                            }
                            test_funcs.append(func_info)

        return test_funcs

    def _is_test_function(self, name: str) -> bool:
        """
        Check if a function name indicates it's a test function based on naming conventions.

        Args:
            name: Name of the function

        Returns:
            True if the function name suggests it's a test function
        """
        # Convert to lowercase for comparison
        lower_name = name.lower()

        # Standard test naming patterns
        test_patterns = [
            'test_',           # Standard test prefix
            'should_',         # Behavior-driven test naming
            'when_',          # Scenario-based test naming
            'verify_',        # Verification-based test naming
            'validate_',      # Validation-based test naming
            'check_'          # Check-based test naming
        ]

        # Check if the name starts with any of the test patterns
        if any(lower_name.startswith(pattern) for pattern in test_patterns):
            return True

        # Also check if the name is exactly 'test' (single word)
        if lower_name == 'test':
            return True

        # For camelCase, check if it starts with 'test' followed by uppercase letter
        if lower_name.startswith('test') and len(name) > 4 and name[4].isupper():
            return True

        return False

    def _find_benchmark_functions(self, module: Any, module_name: str, file_path: str, category: str) -> List[Dict[str, Any]]:
        """
        Find all benchmark functions in a module.

        Args:
            module: The imported module
            module_name: Name of the module
            file_path: Path to the file
            category: Category of benchmarks (unit, integration, performance)

        Returns:
            List of benchmark function information
        """
        benchmark_funcs = []

        for name, obj in inspect.getmembers(module):
            # Check if it's a callable function that matches benchmark naming patterns
            if (inspect.isfunction(obj) or inspect.ismethod(obj)) and self._is_benchmark_function(name):
                # Skip private functions
                if not name.startswith('_'):
                    func_info = {
                        'name': name,
                        'function': obj,
                        'module_name': module_name,
                        'file_path': file_path,
                        'full_name': f"{module_name}.{name}",
                        'type': TestType.BENCHMARK,
                        'category': category,
                        'model_name': self._extract_model_name(file_path)
                    }
                    benchmark_funcs.append(func_info)

        return benchmark_funcs

    def _is_benchmark_function(self, name: str) -> bool:
        """
        Check if a function name indicates it's a benchmark function based on naming conventions.

        Args:
            name: Name of the function

        Returns:
            True if the function name suggests it's a benchmark function
        """
        # Standard benchmark naming patterns
        benchmark_patterns = [
            'run_',            # Standard run prefix
            'benchmark_',      # Standard benchmark prefix
            'perf_',          # Performance-related prefix
            'measure_',       # Measurement-related prefix
            'profile_',       # Profiling-related prefix
            'time_',          # Timing-related prefix
            'speed_',         # Speed-related prefix
            'stress_',        # Stress testing prefix
            'load_'           # Load testing prefix
        ]

        # Check if the name starts with any of the benchmark patterns
        return any(name.lower().startswith(pattern) for pattern in benchmark_patterns)

    def _determine_test_type(self, category: str) -> TestType:
        """
        Determine the test type based on its category.

        Args:
            category: Category string ('unit', 'integration', 'performance', or 'other')

        Returns:
            TestType enum value
        """
        if category == 'unit':
            return TestType.UNIT_TEST
        elif category == 'integration':
            return TestType.INTEGRATION_TEST
        elif category == 'performance':
            return TestType.PERFORMANCE_TEST
        else:
            return TestType.UNKNOWN

    def _extract_model_name(self, file_path: str) -> str:
        """
        Extract the model name from the file path.

        Args:
            file_path: Path to the test/benchmark file

        Returns:
            Model name string
        """
        # Normalize path separators
        normalized_path = file_path.replace('/', os.sep).replace('\\', os.sep)
        path_parts = normalized_path.split(os.sep)

        # Look for 'models' directory and extract the next part as model name
        for i, part in enumerate(path_parts):
            if part == 'models' and i + 1 < len(path_parts):
                # The next part should be the model name
                model_part = path_parts[i + 1]
                # Handle the case where the model name has hyphens or other separators
                # Convert to a cleaner format if needed
                return model_part.replace('-', '_').replace('.', '_')

        # If not in models directory, check for plugin_system
        if 'plugin_system' in normalized_path:
            return 'plugin_system'

        return 'general'

    def get_items_by_type(self, item_type: TestType) -> List[Dict[str, Any]]:
        """
        Get all items of a specific type.

        Args:
            item_type: Type to filter by (TestType.UNIT_TEST, TestType.BENCHMARK, etc.)

        Returns:
            List of items of the specified type
        """
        return [item for item in self.discovered_items if item['type'] == item_type]

    def get_items_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all items in a specific category.

        Args:
            category: Category to filter by ('unit', 'integration', 'performance', 'other')

        Returns:
            List of items in the category
        """
        return [item for item in self.discovered_items if item['category'] == category]

    def get_items_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all items for a specific model.

        Args:
            model_name: Name of the model to filter by

        Returns:
            List of items for the model
        """
        return [item for item in self.discovered_items if item['model_name'] == model_name]

    def run_item(self, item_func: Callable, *args, **kwargs) -> Any:
        """
        Run a single test or benchmark function safely.

        Args:
            item_func: The function to run
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function
        """
        try:
            return item_func(*args, **kwargs)
        except Exception as e:
            print(f"Error running {item_func.__name__}: {e}")
            return {'error': str(e)}

    def run_all_items(self, include_types: List[TestType] = None, include_categories: List[str] = None) -> Dict[str, Any]:
        """
        Run all discovered items.

        Args:
            include_types: List of types to include. If None, runs all types.
            include_categories: List of categories to include. If None, runs all categories.

        Returns:
            Dictionary with results from all items
        """
        from datetime import datetime
        import json

        results = {
            'timestamp': datetime.now().isoformat(),
            'results': {},
            'summary': {}
        }

        items_to_run = self.discovered_items

        if include_types:
            items_to_run = [
                item for item in self.discovered_items
                if item['type'] in include_types
            ]

        if include_categories:
            items_to_run = [
                item for item in items_to_run
                if item['category'] in include_categories
            ]

        print(f"Running {len(items_to_run)} items...")

        for item in items_to_run:
            print(f"Running {item['full_name']}...")
            result = self.run_item(item['function'])
            results['results'][item['full_name']] = result

        # Generate summary
        successful_runs = sum(1 for r in results['results'].values() if 'error' not in r)
        total_runs = len(results['results'])

        results['summary'] = {
            'total_items': total_runs,
            'successful_runs': successful_runs,
            'failed_runs': total_runs - successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0
        }

        print(f"\nSummary:")
        print(f"  Total: {total_runs}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {total_runs - successful_runs}")
        print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

        return results

    def run_tests_only(self) -> Dict[str, Any]:
        """
        Run only the discovered tests (not benchmarks).

        Returns:
            Dictionary with results from tests
        """
        test_types = [TestType.UNIT_TEST, TestType.INTEGRATION_TEST, TestType.PERFORMANCE_TEST]
        return self.run_all_items(include_types=test_types)

    def run_benchmarks_only(self) -> Dict[str, Any]:
        """
        Run only the discovered benchmarks (not tests).

        Returns:
            Dictionary with results from benchmarks
        """
        benchmark_types = [TestType.BENCHMARK]
        return self.run_all_items(include_types=benchmark_types)

    def run_model_items(self, model_name: str) -> Dict[str, Any]:
        """
        Run all items for a specific model.

        Args:
            model_name: Name of the model to run items for

        Returns:
            Dictionary with results from model items
        """
        from datetime import datetime

        model_items = self.get_items_by_model(model_name)

        results = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'results': {},
            'summary': {}
        }

        print(f"Running {len(model_items)} items for model {model_name}...")

        for item in model_items:
            print(f"Running {item['full_name']}...")
            result = self.run_item(item['function'])
            results['results'][item['full_name']] = result

        # Generate summary
        successful_runs = sum(1 for r in results['results'].values() if 'error' not in r)
        total_runs = len(results['results'])

        results['summary'] = {
            'total_items': total_runs,
            'successful_runs': successful_runs,
            'failed_runs': total_runs - successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0
        }

        print(f"\nModel {model_name} Item Summary:")
        print(f"  Total: {total_runs}")
        print(f"  Successful: {successful_runs}")
        print(f"  Failed: {total_runs - successful_runs}")
        print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "discovery_results") -> None:
        """
        Save discovery results to JSON file.

        Args:
            results: Results dictionary to save
            output_dir: Directory to save results to
        """
        from datetime import datetime
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_filename = output_path / f"discovery_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {json_filename}")


def discover_and_run_all_items():
    """
    Convenience function to discover and run all tests and benchmarks in the project.

    Returns:
        Dictionary with results from all items
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()

    print(f"Discovered {len(discovery.discovered_items)} items:")
    print(f"  Tests: {len(discovery.test_functions)}")
    print(f"  Benchmarks: {len(discovery.benchmark_functions)}")

    # Run all items
    results = discovery.run_all_items()

    # Save results
    discovery.save_results(results)

    return results


def discover_and_run_tests_only():
    """
    Convenience function to discover and run only tests (not benchmarks).

    Returns:
        Dictionary with results from tests
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()

    print(f"Discovered {len(discovery.test_functions)} test functions")

    # Run only tests
    results = discovery.run_tests_only()

    # Save results
    discovery.save_results(results, output_dir="test_results")

    return results


def discover_and_run_benchmarks_only():
    """
    Convenience function to discover and run only benchmarks (not tests).

    Returns:
        Dictionary with results from benchmarks
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()

    print(f"Discovered {len(discovery.benchmark_functions)} benchmark functions")

    # Run only benchmarks
    results = discovery.run_benchmarks_only()

    # Save results
    discovery.save_results(results, output_dir="benchmark_results")

    return results


def discover_tests_for_model(model_name: str) -> List[Dict[str, Any]]:
    """
    Discover all tests for a specific model.

    Args:
        model_name: Name of the model to discover tests for

    Returns:
        List of test functions for the model
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()

    # Filter to only tests for the specified model
    model_tests = []
    for item in discovery.test_functions:
        if item['model_name'] == model_name:
            model_tests.append(item)

    return model_tests


def discover_benchmarks_for_model(model_name: str) -> List[Dict[str, Any]]:
    """
    Discover all benchmarks for a specific model.

    Args:
        model_name: Name of the model to discover benchmarks for

    Returns:
        List of benchmark functions for the model
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()

    # Filter to only benchmarks for the specified model
    model_benchmarks = []
    for item in discovery.benchmark_functions:
        if item['model_name'] == model_name:
            model_benchmarks.append(item)

    return model_benchmarks


def run_tests_for_model(model_name: str) -> Dict[str, Any]:
    """
    Run all tests for a specific model.

    Args:
        model_name: Name of the model to run tests for

    Returns:
        Dictionary with results from model tests
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()
    
    # Get tests for the specific model
    model_tests = discovery.get_items_by_model(model_name)
    # Filter to only include test types
    model_test_functions = [item for item in model_tests if item['type'] in [
        TestType.UNIT_TEST, TestType.INTEGRATION_TEST, TestType.PERFORMANCE_TEST
    ]]

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'results': {},
        'summary': {}
    }

    print(f"Running {len(model_test_functions)} tests for model {model_name}...")

    for item in model_test_functions:
        print(f"Running {item['full_name']}...")
        result = discovery.run_item(item['function'])
        results['results'][item['full_name']] = result

    # Generate summary
    successful_runs = sum(1 for r in results['results'].values() if 'error' not in r)
    total_runs = len(results['results'])

    results['summary'] = {
        'total_tests': total_runs,
        'successful_runs': successful_runs,
        'failed_runs': total_runs - successful_runs,
        'success_rate': successful_runs / total_runs if total_runs > 0 else 0
    }

    print(f"\nModel {model_name} Test Summary:")
    print(f"  Total: {total_runs}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {total_runs - successful_runs}")
    print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

    return results


def run_benchmarks_for_model(model_name: str) -> Dict[str, Any]:
    """
    Run all benchmarks for a specific model.

    Args:
        model_name: Name of the model to run benchmarks for

    Returns:
        Dictionary with results from model benchmarks
    """
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()
    
    # Get benchmarks for the specific model
    model_benchmarks = discovery.get_items_by_model(model_name)
    # Filter to only include benchmark types
    model_benchmark_functions = [item for item in model_benchmarks if item['type'] == TestType.BENCHMARK]

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'results': {},
        'summary': {}
    }

    print(f"Running {len(model_benchmark_functions)} benchmarks for model {model_name}...")

    for item in model_benchmark_functions:
        print(f"Running {item['full_name']}...")
        result = discovery.run_item(item['function'])
        results['results'][item['full_name']] = result

    # Generate summary
    successful_runs = sum(1 for r in results['results'].values() if 'error' not in r)
    total_runs = len(results['results'])

    results['summary'] = {
        'total_benchmarks': total_runs,
        'successful_runs': successful_runs,
        'failed_runs': total_runs - successful_runs,
        'success_rate': successful_runs / total_runs if total_runs > 0 else 0
    }

    print(f"\nModel {model_name} Benchmark Summary:")
    print(f"  Total: {total_runs}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {total_runs - successful_runs}")
    print(f"  Success Rate: {results['summary']['success_rate']:.2%}")

    return results


def get_discovery_summary() -> Dict[str, Any]:
    """
    Get a summary of all discovered items in the project.

    Returns:
        Dictionary with discovery summary information
    """
    from datetime import datetime
    
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_items': len(discovery.discovered_items),
        'total_tests': len(discovery.test_functions),
        'total_benchmarks': len(discovery.benchmark_functions),
        'by_type': {},
        'by_category': {},
        'by_model': {},
        'search_paths': discovery.search_paths
    }

    # Count by type
    for item in discovery.discovered_items:
        item_type = item['type'].value
        summary['by_type'][item_type] = summary['by_type'].get(item_type, 0) + 1

    # Count by category
    for item in discovery.discovered_items:
        category = item['category']
        summary['by_category'][category] = summary['by_category'].get(category, 0) + 1

    # Count by model
    for item in discovery.discovered_items:
        model = item['model_name']
        summary['by_model'][model] = summary['by_model'].get(model, 0) + 1

    return summary


if __name__ == "__main__":
    # Example usage
    print("Starting unified test and benchmark discovery...")
    results = discover_and_run_all_items()
    print("\nDiscovery and execution completed!")