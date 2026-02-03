"""
Test Optimization Module for Parallel Execution and Caching

This module implements systems to run tests in parallel where possible and cache results
to improve execution speed while maintaining test integrity and reliability.
"""

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestResultCache:
    """
    A cache for storing and retrieving test results to avoid redundant execution.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the test result cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to .test_cache in project root.
        """
        if cache_dir is None:
            cache_dir = os.path.join(project_root, ".test_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "test_results.json"

        # Load existing cache
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")

    def _generate_key(self, test_path: str, test_args: str = "") -> str:
        """
        Generate a unique key for a test based on its path and arguments.

        Args:
            test_path: Path to the test file/function
            test_args: Arguments passed to the test

        Returns:
            Unique cache key
        """
        key_data = f"{test_path}_{test_args}_{platform.python_version()}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get_result(
        self, test_path: str, test_args: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for a test.

        Args:
            test_path: Path to the test file/function
            test_args: Arguments passed to the test

        Returns:
            Cached result or None if not found/cached result is invalid
        """
        key = self._generate_key(test_path, test_args)

        if key in self._cache:
            cached_data = self._cache[key]

            # Check if cache is still valid (not too old)
            cache_time = datetime.fromisoformat(cached_data.get("timestamp", ""))
            current_time = datetime.now()

            # Cache is valid for 24 hours
            if (current_time - cache_time).total_seconds() < 24 * 3600:
                return cached_data

        return None

    def put_result(self, test_path: str, result: Dict[str, Any], test_args: str = ""):
        """
        Store test result in cache.

        Args:
            test_path: Path to the test file/function
            result: Test result to cache
            test_args: Arguments passed to the test
        """
        key = self._generate_key(test_path, test_args)

        self._cache[key] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "test_path": test_path,
            "test_args": test_args,
        }

        self._save_cache()

    def invalidate(self, test_path: str = None):
        """
        Invalidate cache for a specific test or all tests.

        Args:
            test_path: Specific test path to invalidate, or None for all
        """
        if test_path is None:
            self._cache.clear()
        else:
            # Remove all keys related to this test path
            keys_to_remove = []
            for key, value in self._cache.items():
                if value.get("test_path", "").startswith(test_path):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

        self._save_cache()


class TestParallelExecutor:
    """
    Execute tests in parallel using multiple processes or threads.
    """

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = True):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of workers. Defaults to CPU count.
            use_processes: Whether to use processes (True) or threads (False)
        """
        if max_workers is None:
            max_workers = min(
                mp.cpu_count(), 8
            )  # Cap at 8 to avoid resource exhaustion

        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor_class = (
            ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        )

    def run_tests_parallel(
        self, test_functions: List[Callable]
    ) -> List[Tuple[Callable, Any, float]]:
        """
        Run multiple test functions in parallel.

        Args:
            test_functions: List of test functions to run

        Returns:
            List of tuples (test_function, result, execution_time)
        """
        results = []

        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all tests to the executor
            future_to_test = {
                executor.submit(self._execute_single_test, test_func): test_func
                for test_func in test_functions
            }

            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_func = future_to_test[future]
                try:
                    result, exec_time = future.result()
                    results.append((test_func, result, exec_time))
                except Exception as e:
                    results.append(
                        (test_func, {"error": str(e), "success": False}, 0.0)
                    )

        # Sort results to maintain order
        results.sort(key=lambda x: test_functions.index(x[0]))
        return results

    def _execute_single_test(self, test_func: Callable) -> Tuple[Dict[str, Any], float]:
        """
        Execute a single test function and measure execution time.

        Args:
            test_func: Test function to execute

        Returns:
            Tuple of (result dict, execution time in seconds)
        """
        start_time = time.time()
        try:
            test_func()
            result = {"success": True, "error": None}
        except Exception as e:
            result = {"success": False, "error": str(e)}
        finally:
            execution_time = time.time() - start_time

        return result, execution_time


class TestDependencyAnalyzer:
    """
    Analyze test dependencies to determine safe parallelization groups.
    """

    def __init__(self):
        self.dependencies = {}
        self.resources_used = {}  # Track resources used by each test

    def analyze_dependencies(self, test_files: List[str]) -> Dict[str, List[str]]:
        """
        Analyze dependencies between test files to determine safe parallelization.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary mapping test files to their dependencies
        """
        dependencies = {}

        for test_file in test_files:
            # Simple heuristic: tests that modify shared resources should be grouped
            deps = self._analyze_single_file(test_file)
            dependencies[test_file] = deps

        return dependencies

    def _analyze_single_file(self, test_file: str) -> List[str]:
        """
        Analyze a single test file for dependencies.

        Args:
            test_file: Path to test file

        Returns:
            List of dependent files/resources
        """
        deps = []

        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Look for common patterns that indicate dependencies
                if "torch.cuda" in content:
                    deps.append("gpu")
                if "os.environ" in content:
                    deps.append("environment")
                if "tempfile" in content or "tmp" in content:
                    deps.append("filesystem")
                if (
                    "network" in content.lower()
                    or "socket" in content.lower()
                    or "http" in content.lower()
                ):
                    deps.append("network")

        except Exception:
            pass  # Ignore files that can't be read

        return deps


class OptimizedTestRunner:
    """
    Main test runner that combines parallel execution and caching.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        parallel_enabled: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the optimized test runner.

        Args:
            cache_enabled: Whether to enable result caching
            parallel_enabled: Whether to enable parallel execution
            max_workers: Maximum number of parallel workers
        """
        self.cache_enabled = cache_enabled
        self.parallel_enabled = parallel_enabled
        self.cache = TestResultCache() if cache_enabled else None
        self.executor = (
            TestParallelExecutor(max_workers=max_workers) if parallel_enabled else None
        )
        self.dependency_analyzer = TestDependencyAnalyzer()

    def run_tests(
        self, test_functions: List[Callable], test_paths: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run tests with optimization (parallel execution and caching).

        Args:
            test_functions: List of test functions to run
            test_paths: Optional paths corresponding to test functions

        Returns:
            Dictionary with test results and statistics
        """
        start_time = time.time()

        if test_paths is None:
            test_paths = [getattr(tf, "__name__", str(tf)) for tf in test_functions]

        # Prepare results structure
        results = {
            "total_tests": len(test_functions),
            "passed": 0,
            "failed": 0,
            "cached": 0,
            "executed": 0,
            "results": [],
            "execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Separate cached and uncached tests
        uncached_tests = []
        uncached_indices = []

        for i, (test_func, test_path) in enumerate(zip(test_functions, test_paths)):
            if self.cache_enabled:
                cached_result = self.cache.get_result(test_path)
                if cached_result is not None:
                    # Use cached result
                    result_data = cached_result["result"]
                    results["results"].append(
                        {
                            "test": test_path,
                            "success": result_data["success"],
                            "error": result_data.get("error"),
                            "cached": True,
                            "execution_time": 0.0,
                        }
                    )
                    results["cached"] += 1
                    results["cache_hits"] += 1

                    if result_data["success"]:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                else:
                    # Need to execute this test
                    uncached_tests.append(test_func)
                    uncached_indices.append(i)
                    results["cache_misses"] += 1
            else:
                # No caching, all tests need execution
                uncached_tests.append(test_func)
                uncached_indices.append(i)

        # Execute uncached tests
        if uncached_tests:
            if self.parallel_enabled and len(uncached_tests) > 1:
                # Run in parallel
                parallel_results = self.executor.run_tests_parallel(uncached_tests)

                for idx, (test_func, result, exec_time) in zip(
                    uncached_indices, parallel_results
                ):
                    test_path = test_paths[idx]

                    # Store result
                    result_entry = {
                        "test": test_path,
                        "success": result["success"],
                        "error": result.get("error"),
                        "cached": False,
                        "execution_time": exec_time,
                    }

                    results["results"].append(result_entry)
                    results["executed"] += 1

                    # Update counters
                    if result["success"]:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1

                    # Cache the result if caching is enabled
                    if self.cache_enabled:
                        self.cache.put_result(test_path, result)
            else:
                # Run sequentially
                for idx, test_func in zip(uncached_indices, uncached_tests):
                    test_path = test_paths[idx]

                    # Execute test
                    start_test = time.time()
                    try:
                        test_func()
                        result = {"success": True, "error": None}
                        exec_time = time.time() - start_test
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                        exec_time = time.time() - start_test

                    # Store result
                    result_entry = {
                        "test": test_path,
                        "success": result["success"],
                        "error": result["error"],
                        "cached": False,
                        "execution_time": exec_time,
                    }

                    results["results"].append(result_entry)
                    results["executed"] += 1

                    # Update counters
                    if result["success"]:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1

                    # Cache the result if caching is enabled
                    if self.cache_enabled:
                        self.cache.put_result(test_path, result)

        results["execution_time"] = time.time() - start_time
        return results

    def run_test_files(self, test_files: List[str]) -> Dict[str, Any]:
        """
        Run tests from multiple test files with optimization.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary with test results and statistics
        """
        # Import test functions from files
        test_functions = []
        test_paths = []

        for test_file in test_files:
            functions = self._discover_test_functions_from_file(test_file)
            for func in functions:
                test_functions.append(func)
                test_paths.append(f"{test_file}::{func.__name__}")

        return self.run_tests(test_functions, test_paths)

    def _discover_test_functions_from_file(self, file_path: str) -> List[Callable]:
        """
        Discover test functions from a single Python file.

        Args:
            file_path: Path to the Python file to scan

        Returns:
            List of test functions found in the file
        """
        import importlib.util
        import inspect
        import sys

        test_functions = []

        # Import the module dynamically
        spec = importlib.util.spec_from_file_location("__test_module__", file_path)
        if spec is None or spec.loader is None:
            return test_functions

        module = importlib.util.module_from_spec(spec)

        # Add the directory to sys.path temporarily to handle imports
        file_dir = os.path.dirname(file_path)
        original_path = sys.path[:]
        sys.path.insert(0, file_dir)

        try:
            spec.loader.exec_module(module)

            # Find all functions that start with 'test_'
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("test_"):
                    test_functions.append(obj)

        except Exception as e:
            print(f"Warning: Could not import {file_path}: {e}")
        finally:
            # Restore original path
            sys.path[:] = original_path

        return test_functions

    def invalidate_cache(self, test_path: str = None):
        """
        Invalidate the test result cache.

        Args:
            test_path: Specific test path to invalidate, or None for all
        """
        if self.cache_enabled:
            self.cache.invalidate(test_path)


def run_tests_with_optimization(
    test_functions: List[Callable],
    test_paths: List[str] = None,
    cache_enabled: bool = True,
    parallel_enabled: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run tests with optimization.

    Args:
        test_functions: List of test functions to run
        test_paths: Optional paths corresponding to test functions
        cache_enabled: Whether to enable result caching
        parallel_enabled: Whether to enable parallel execution
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with test results and statistics
    """
    runner = OptimizedTestRunner(
        cache_enabled=cache_enabled,
        parallel_enabled=parallel_enabled,
        max_workers=max_workers,
    )

    return runner.run_tests(test_functions, test_paths)


def run_test_directory(
    directory_path: str,
    cache_enabled: bool = True,
    parallel_enabled: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run all tests in a directory with optimization.

    Args:
        directory_path: Path to directory containing test files
        cache_enabled: Whether to enable result caching
        parallel_enabled: Whether to enable parallel execution
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with test results and statistics
    """
    runner = OptimizedTestRunner(
        cache_enabled=cache_enabled,
        parallel_enabled=parallel_enabled,
        max_workers=max_workers,
    )

    # Find all Python test files in the directory
    test_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py") and (file.startswith("test_") or "_test" in file):
                test_files.append(os.path.join(root, file))

    return runner.run_test_files(test_files)
