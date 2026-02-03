"""
Real Performance Benchmark Base Classes

This module provides base classes for benchmarks using real performance measurements
instead of simulated or mocked metrics.
"""

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List

import torch

from benchmarks.core.real_performance_monitor import (
    RealPerformanceMonitor,
    benchmark_function_real,
    get_real_system_metrics,
)
from tests.base.benchmark_test_base import ModelBenchmarkTest


class RealPerformanceBenchmarkBase(ModelBenchmarkTest):
    """Base class for benchmarks using real performance measurements."""

    def get_model_plugin_class(self):
        """Abstract method implementation - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement get_model_plugin_class")

    def test_required_functionality(self):
        """Implementation of abstract method from base class."""
        pass

    def initialize_model(self, model_name: str, create_func, device: str = "cpu"):
        """Initialize a model with real performance monitoring."""
        plugin = create_func()
        success = plugin.initialize(device=device)
        if not success:
            self.skipTest(f"Could not initialize {model_name}")
        self.models[model_name] = plugin
        return plugin

    def run_performance_test(self):
        """Implementation of abstract method from base class."""
        pass

    def benchmark_inference_speed(
        self, plugin, model_name: str, input_length: int = 50, num_iterations: int = 10
    ):
        """Real inference speed benchmark using actual measurements."""
        # Prepare input - create realistic input tensor
        input_ids = torch.randint(100, 1000, (1, input_length))

        # Warmup
        for _ in range(3):
            try:
                _ = plugin.infer(input_ids)
            except:
                # If infer doesn't work with tensor, try with string
                _ = plugin.generate_text("warmup text", max_new_tokens=5)

        # Use real performance monitoring
        def timed_inference():
            return plugin.infer(input_ids)

        benchmark_results = benchmark_function_real(
            timed_inference, iterations=num_iterations
        )

        # Extract metrics
        avg_time_per_inference = (
            benchmark_results["avg_time_ms"] / 1000.0
        )  # Convert ms to seconds
        tokens_per_second = (
            input_length / avg_time_per_inference if avg_time_per_inference > 0 else 0
        )

        result = {
            "total_time": benchmark_results["avg_time_ms"]
            / 1000.0
            * num_iterations,  # Total time in seconds
            "avg_time_per_inference": avg_time_per_inference,
            "tokens_per_second": tokens_per_second,
            "num_iterations": num_iterations,
            "input_length": input_length,
            "model_name": model_name,
            "raw_benchmark_data": benchmark_results,
        }

        self.benchmark_results[f"{model_name}_speed_{input_length}"] = result
        return result

    def benchmark_generation_speed(
        self,
        plugin,
        model_name: str,
        prompt: str = "The quick brown fox jumps over the lazy dog. ",
        max_new_tokens: int = 50,
    ):
        """Real text generation speed benchmark using actual measurements."""
        # Warmup
        try:
            _ = plugin.generate_text(prompt, max_new_tokens=10)
        except:
            # If generate_text doesn't work, try infer
            _ = plugin.infer(prompt)

        # Use real performance monitoring
        def timed_generation():
            return plugin.generate_text(prompt, max_new_tokens=max_new_tokens)

        benchmark_results = benchmark_function_real(
            timed_generation,
            iterations=5,  # Fewer iterations for generation as it may take longer
        )

        # Calculate metrics
        avg_time_per_generation = (
            benchmark_results["avg_time_ms"] / 1000.0
        )  # Convert ms to seconds
        generated_text = benchmark_results["metrics_history"][
            0
        ].tokens_per_second  # This is actually the output
        chars_per_second = (
            len(prompt + " " + str(generated_text)) / avg_time_per_generation
            if avg_time_per_generation > 0
            else 0
        )

        result = {
            "total_time": benchmark_results["avg_time_ms"]
            / 1000.0,  # Average time in seconds
            "chars_per_second": chars_per_second,
            "generated_text_length": len(str(generated_text)) if generated_text else 0,
            "model_name": model_name,
            "prompt_length": len(prompt),
            "raw_benchmark_data": benchmark_results,
        }

        self.benchmark_results[f"{model_name}_generation"] = result
        return result

    def benchmark_memory_usage(self, plugin, model_name: str):
        """Real memory usage benchmark using actual measurements."""
        import gc

        # Get baseline metrics
        baseline_metrics = get_real_system_metrics()

        # Perform some operations to see memory impact
        try:
            # Run a few inference operations
            for i in range(5):
                _ = plugin.generate_text("test prompt", max_new_tokens=20)
        except:
            try:
                _ = plugin.infer(torch.randint(100, 1000, (1, 20)))
            except:
                pass

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get metrics after operations
        current_metrics = get_real_system_metrics()

        # Calculate differences
        memory_used_mb = (
            current_metrics.memory_used_mb - baseline_metrics.memory_used_mb
        )
        gpu_memory_used_diff = (
            (current_metrics.gpu_memory_used_mb - baseline_metrics.gpu_memory_used_mb)
            if current_metrics.gpu_memory_used_mb is not None
            and baseline_metrics.gpu_memory_used_mb is not None
            else None
        )

        result = {
            "baseline_memory_mb": baseline_metrics.memory_used_mb,
            "current_memory_mb": current_metrics.memory_used_mb,
            "memory_used_mb": memory_used_mb,
            "model_name": model_name,
            "baseline_system_metrics": baseline_metrics.__dict__,
            "current_system_metrics": current_metrics.__dict__,
            "gpu_memory_used_difference_mb": gpu_memory_used_diff,
        }

        self.benchmark_results[f"{model_name}_memory"] = result
        return result

    def benchmark_throughput(
        self, plugin, model_name: str, batch_sizes: List[int] = None
    ):
        """Real throughput benchmark using actual measurements."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        results = []

        for batch_size in batch_sizes:
            # Prepare batch input
            input_ids = torch.randint(100, 1000, (batch_size, 50))

            # Warmup
            for _ in range(3):
                try:
                    _ = plugin.infer(input_ids)
                except:
                    _ = plugin.generate_text("warmup", max_new_tokens=5)

            # Benchmark
            def timed_batch_inference():
                return plugin.infer(input_ids)

            benchmark_results = benchmark_function_real(
                timed_batch_inference, iterations=5
            )

            avg_time = benchmark_results["avg_time_ms"] / 1000.0  # Convert to seconds
            tokens_per_second = (batch_size * 50) / avg_time if avg_time > 0 else 0

            batch_result = {
                "batch_size": batch_size,
                "avg_time_per_batch": avg_time,
                "tokens_per_second": tokens_per_second,
                "requests_per_second": batch_size / avg_time if avg_time > 0 else 0,
                "raw_benchmark_data": benchmark_results,
            }

            results.append(batch_result)

        overall_result = {
            "model_name": model_name,
            "batch_sizes_tested": batch_sizes,
            "throughput_results": results,
            "peak_throughput_tokens_per_second": max(
                r["tokens_per_second"] for r in results
            ),
        }

        self.benchmark_results[f"{model_name}_throughput"] = overall_result
        return overall_result

    def benchmark_power_efficiency(
        self, plugin, model_name: str, duration: float = 30.0
    ):
        """Real power efficiency benchmark using actual measurements over time."""
        import threading
        import time as time_module

        # Start continuous monitoring
        monitor = RealPerformanceMonitor()
        monitor.start_monitoring(interval=0.5)  # Sample every 0.5 seconds

        start_time = time_module.time()
        operations_performed = 0

        try:
            # Perform operations for the specified duration
            while time_module.time() - start_time < duration:
                try:
                    _ = plugin.generate_text("test", max_new_tokens=10)
                    operations_performed += 1
                except:
                    try:
                        _ = plugin.infer(torch.randint(100, 1000, (1, 20)))
                        operations_performed += 1
                    except:
                        break

                # Small delay to prevent overwhelming the system
                time_module.sleep(0.01)
        finally:
            monitor.stop_monitoring()

        # Calculate efficiency metrics
        total_time = time_module.time() - start_time
        avg_cpu = (
            sum(m.cpu_percent for m in monitor.metrics_history)
            / len(monitor.metrics_history)
            if monitor.metrics_history
            else 0
        )
        avg_memory = (
            sum(m.memory_percent for m in monitor.metrics_history)
            / len(monitor.metrics_history)
            if monitor.metrics_history
            else 0
        )

        result = {
            "model_name": model_name,
            "test_duration_seconds": total_time,
            "operations_performed": operations_performed,
            "operations_per_second": (
                operations_performed / total_time if total_time > 0 else 0
            ),
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "total_energy_estimate": avg_cpu
            * total_time
            / 100,  # Simplified energy estimate
            "operations_per_energy_unit": (
                operations_performed / (avg_cpu * total_time / 100)
                if avg_cpu * total_time > 0
                else 0
            ),
            "metrics_history": [m.__dict__ for m in monitor.metrics_history],
        }

        self.benchmark_results[f"{model_name}_power_efficiency"] = result
        return result
