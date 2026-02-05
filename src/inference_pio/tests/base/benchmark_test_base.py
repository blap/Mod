"""
Base classes for performance benchmark tests in the Mod project.

These base classes provide common functionality and setup for performance
benchmarking of different components.
"""

import os
import statistics
import tempfile
import time
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import GPUtil
import psutil
import pytest
import torch

from src.inference_pio.utils.testing_utils import create_temp_test_config
from src.inference_pio.utils.benchmarking_utils import (
    benchmark_inference_speed,
    benchmark_generation_speed,
    benchmark_memory_usage,
    benchmark_throughput,
    benchmark_power_efficiency
)


class BaseBenchmarkTest(unittest.TestCase, ABC):
    """
    Base class for all performance benchmark tests in the Mod project.

    This class provides common setup, teardown, and utility methods for
    measuring performance characteristics of components.
    """

    def setUp(self):
        """Set up test fixtures before each benchmark test method."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = create_temp_test_config()
        self.benchmark_results = {}

    def tearDown(self):
        """Clean up after each benchmark test method."""
        super().tearDown()
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @abstractmethod
    def run_performance_test(self):
        """
        Abstract method that must be implemented by subclasses.
        Each benchmark test class must define its core performance test.
        """
        raise NotImplementedError("Method not implemented")

    def measure_execution_time(self, func, *args, **kwargs) -> Dict[str, float]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        start_cpu_percent = psutil.cpu_percent(interval=None)
        start_memory = psutil.Process().memory_info().rss

        # Get GPU memory if available
        start_gpu_memory = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            start_gpu_memory = gpus[0].memoryUsed  # Use first GPU

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        end_cpu_percent = psutil.cpu_percent(interval=None)
        end_memory = psutil.Process().memory_info().rss

        end_gpu_memory = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            end_gpu_memory = gpus[0].memoryUsed  # Use first GPU

        execution_time = end_time - start_time
        cpu_used = end_cpu_percent - start_cpu_percent
        memory_used = end_memory - start_memory
        gpu_memory_used = end_gpu_memory - start_gpu_memory

        return {
            "execution_time": execution_time,
            "cpu_used": cpu_used,
            "memory_used": memory_used,
            "gpu_memory_used": gpu_memory_used,
            "result": result,
        }

    def run_multiple_iterations(
        self, func, iterations: int = 10, *args, **kwargs
    ) -> List[Dict[str, float]]:
        """Run a function multiple times and collect performance metrics."""
        results = []
        for _ in range(iterations):
            result = self.measure_execution_time(func, *args, **kwargs)
            results.append(result)
        return results

    def calculate_performance_stats(
        self, results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate performance statistics from multiple runs."""
        if not results:
            return {}

        execution_times = [r["execution_time"] for r in results]
        memory_usages = [r["memory_used"] for r in results]
        cpu_usages = [r["cpu_used"] for r in results]
        gpu_memory_usages = [r["gpu_memory_used"] for r in results]

        stats = {
            "avg_execution_time": statistics.mean(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "std_execution_time": (
                statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            ),
            "avg_memory_used": statistics.mean(memory_usages),
            "avg_cpu_used": statistics.mean(cpu_usages),
            "avg_gpu_memory_used": statistics.mean(gpu_memory_usages),
            "total_runs": len(results),
        }

        return stats


class ModelBenchmarkTest(BaseBenchmarkTest, ABC):
    """
    Base class for performance benchmarking of model plugins.

    Provides additional utilities specific to measuring model performance.
    """

    def setUp(self):
        """Set up test fixtures for model benchmark tests."""
        super().setUp()

    @abstractmethod
    def get_model_plugin_class(self):
        """Return the model plugin class to be benchmarked."""
        raise NotImplementedError("Method not implemented")

    def create_model_instance(self, **kwargs):
        """Create an instance of the model plugin with test configuration."""
        from tests.shared.utils.plugin_init_utils import create_and_initialize_plugin

        plugin_class = self.get_model_plugin_class()

        # Use the centralized utility to create and initialize the plugin
        return create_and_initialize_plugin(plugin_class, **kwargs)

    def benchmark_model_initialization(self, iterations: int = 5) -> Dict[str, float]:
        """Benchmark the model initialization process."""

        def init_func():
            model_plugin = self.create_model_instance()
            success = model_plugin.initialize()
            return success

        results = self.run_multiple_iterations(init_func, iterations)
        return self.calculate_performance_stats(results)

    def benchmark_model_inference(
        self, input_tensor: torch.Tensor, iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark the model inference process."""
        model_plugin = self.create_model_instance()
        model_plugin.initialize()

        def inference_func():
            result = model_plugin.infer(input_tensor)
            return result

        results = self.run_multiple_iterations(inference_func, iterations)
        return self.calculate_performance_stats(results)

    def benchmark_model_loading(
        self, config: Dict[str, Any], iterations: int = 3
    ) -> Dict[str, float]:
        """Benchmark the model loading process."""

        def load_func():
            model_plugin = self.create_model_instance()
            model = model_plugin.load_model(config)
            return model

        results = self.run_multiple_iterations(load_func, iterations)
        return self.calculate_performance_stats(results)


class SystemBenchmarkTest(BaseBenchmarkTest, ABC):
    """
    Base class for performance benchmarking of system components.

    Measures performance of system-level operations and workflows.
    """

    def setUp(self):
        """Set up test fixtures for system benchmark tests."""
        super().setUp()

    def benchmark_system_operation(
        self, operation_func, iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark a system operation."""
        results = self.run_multiple_iterations(operation_func, iterations)
        return self.calculate_performance_stats(results)

    def benchmark_concurrent_operations(
        self, operation_func, num_concurrent: int = 5
    ) -> Dict[str, float]:
        """Benchmark concurrent operations to measure system scalability."""
        import queue
        import threading

        result_queue = queue.Queue()

        def worker():
            result = self.measure_execution_time(operation_func)
            result_queue.put(result)

        threads = []
        start_time = time.perf_counter()

        # Start all threads
        for _ in range(num_concurrent):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.perf_counter()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        total_time = end_time - start_time
        avg_individual_time = sum(r["execution_time"] for r in results) / len(results)

        return {
            "total_execution_time": total_time,
            "avg_individual_time": avg_individual_time,
            "concurrent_factor": (
                total_time / avg_individual_time if avg_individual_time > 0 else 0
            ),
            "num_concurrent": num_concurrent,
            "results": results,
        }
