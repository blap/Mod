"""Base classes and utilities for benchmarks using standardized test hierarchy."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List

import torch

# Import real performance monitoring
from benchmarks.core.real_performance_monitor import (
    RealPerformanceMonitor,
    benchmark_function_real,
    get_real_system_metrics,
)
from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin

# Import the actual model plugins
from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    create_qwen3_4b_instruct_2507_plugin,
)
from src.inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
from tests.base.benchmark_test_base import ModelBenchmarkTest
from src.inference_pio.utils.benchmarking_utils import (
    benchmark_inference_speed,
    benchmark_generation_speed,
    benchmark_memory_usage,
    benchmark_throughput,
    benchmark_power_efficiency
)


class BenchmarkBase(ModelBenchmarkTest):
    """Base class for benchmarks using standardized test hierarchy."""

    def get_model_plugin_class(self):
        """Abstract method implementation - to be overridden by subclasses."""
        # This is a generic base, so we'll raise NotImplementedError
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses must implement get_model_plugin_class")

    def test_required_functionality(self):
        """Implementation of abstract method from base class."""
        # Generic implementation - subclasses should override with specific functionality
        pass

    def initialize_model(self, model_name: str, create_func, device: str = "cpu"):
        """
        Initialize a model with common settings.

        Args:
            model_name: Name of the model to initialize
            create_func: Function to create the model plugin instance
            device: Device to initialize the model on (default: "cpu")

        Returns:
            Initialized plugin instance

        Raises:
            unittest.SkipTest: If model initialization fails
        """
        plugin = create_func()
        success = plugin.initialize(device=device)
        if not success:
            self.skipTest(f"Could not initialize {model_name}")
        self.models[model_name] = plugin
        return plugin

    def run_performance_test(self):
        """Implementation of abstract method from base class."""
        # Generic performance test - subclasses should override with specific tests
        pass

    def benchmark_inference_speed(
        self, plugin, model_name: str, input_length: int = 50, num_iterations: int = 10
    ):
        """
        Real inference speed benchmark using actual performance measurements.

        This method performs a comprehensive speed benchmark by measuring the actual
        time taken for inference operations. It includes warmup phases to ensure
        the model is properly initialized and caches are primed, then measures
        performance across multiple iterations to provide statistically meaningful results.

        The benchmark calculates tokens per second as the primary performance metric,
        which represents the model's throughput capability. This metric is crucial
        for understanding how efficiently the model can process text in production scenarios.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            input_length: Length of input sequence to use for benchmarking (default: 50)
            num_iterations: Number of iterations to run for averaging (default: 10)

        Returns:
            Dict containing benchmark results including tokens per second, timing info, etc.
        """
        # Delegate to the shared utility function
        result = benchmark_inference_speed(plugin, model_name, input_length, num_iterations)

        # Store results for potential aggregation or reporting
        self.benchmark_results[f"{model_name}_speed_{input_length}"] = result
        return result

    def benchmark_generation_speed(
        self,
        plugin,
        model_name: str,
        prompt: str = "The quick brown fox jumps over the lazy dog. ",
        max_new_tokens: int = 50,
    ):
        """
        Real text generation speed benchmark using actual performance measurements.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            prompt: Input prompt for text generation (default: standard sentence)
            max_new_tokens: Maximum number of new tokens to generate (default: 50)

        Returns:
            Dict containing benchmark results including characters per second, timing info, etc.
        """
        # Delegate to the shared utility function
        result = benchmark_generation_speed(plugin, model_name, prompt, max_new_tokens)

        self.benchmark_results[f"{model_name}_generation"] = result
        return result

    def benchmark_memory_usage(self, plugin, model_name: str):
        """
        Real memory usage benchmark using actual system measurements.

        This method measures the actual memory impact of model operations by
        capturing system memory metrics before and after performing inference
        operations. It accounts for both CPU and GPU memory usage to provide
        a comprehensive view of the model's memory footprint.

        The benchmark includes garbage collection to ensure accurate measurements
        by removing temporary allocations that might skew results. It also
        handles both successful and unsuccessful operations to provide robust
        memory impact assessment.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested

        Returns:
            Dict containing memory usage metrics comparing before and after operations
        """
        # Delegate to the shared utility function
        result = benchmark_memory_usage(plugin, model_name)

        # Store results for potential aggregation or reporting
        self.benchmark_results[f"{model_name}_memory"] = result
        return result

    def benchmark_throughput(
        self, plugin, model_name: str, batch_sizes: List[int] = None
    ):
        """
        Real throughput benchmark using actual performance measurements.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])

        Returns:
            Dict containing throughput metrics for different batch sizes
        """
        # Delegate to the shared utility function
        result = benchmark_throughput(plugin, model_name, batch_sizes)

        self.benchmark_results[f"{model_name}_throughput"] = result
        return result

    def benchmark_power_efficiency(
        self, plugin, model_name: str, duration: float = 30.0
    ):
        """
        Real power efficiency benchmark using actual measurements over time.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            duration: Duration of the benchmark in seconds (default: 30.0)

        Returns:
            Dict containing power efficiency metrics collected over the test duration
        """
        # Delegate to the shared utility function
        result = benchmark_power_efficiency(plugin, model_name, duration)

        self.benchmark_results[f"{model_name}_power_efficiency"] = result
        return result
