"""
Template for creating new benchmark modules in the Mod project.

This module provides a template for creating new types of benchmarks
that can be easily integrated into the benchmarking framework.
Each benchmark module is independent and follows the standardized interface.
"""

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List

import torch

from src.benchmarks.benchmark_base import BenchmarkBase

logger = logging.getLogger(__name__)


class TemplateBenchmarkBase(BenchmarkBase):
    """
    Template base class for new benchmark types.
    
    Extend this class to create new categories of benchmarks.
    """

    def get_model_plugin_class(self):
        """Abstract method implementation - to be overridden by subclasses."""
        raise NotImplementedError("Method not implemented")

    def test_required_functionality(self):
        """Implementation of abstract method from base class."""
        # Generic implementation - subclasses should override with specific functionality
        raise NotImplementedError("Method not implemented")

    def run_performance_test(self):
        """Implementation of abstract method from base class."""
        # Generic performance test - subclasses should override with specific tests
        raise NotImplementedError("Method not implemented")

    def run_template_benchmark(self):
        """
        Template method for running custom benchmarks.
        
        Override this method to implement your specific benchmark logic.
        """
        raise NotImplementedError("Method not implemented")


class CustomAccuracyBenchmark(TemplateBenchmarkBase):
    """
    Template for custom accuracy benchmarks.
    
    Use this class to create benchmarks that measure model accuracy.
    """

    def get_model_plugin_class(self):
        """Return the model plugin class to benchmark."""
        # This should return the actual model plugin class
        # For template purposes, we'll raise NotImplementedError
        raise NotImplementedError("Method not implemented")

    def test_required_functionality(self):
        """Test required functionality for accuracy benchmarking."""
        # TODO: Implement this functionality
        # Example:
        # inputs = ["sample input 1", "sample input 2"]
        # expected_outputs = ["expected output 1", "expected output 2"]
        # 
        # for inp, expected in zip(inputs, expected_outputs):
        #     actual = self.model.infer(inp)
        #     self.assertEqual(actual, expected)
        raise NotImplementedError("Method not implemented")

    def run_performance_test(self):
        """Run performance test specific to accuracy."""
        # TODO: Implement this functionality
        raise NotImplementedError("Method not implemented")

    def run_accuracy_benchmark(self, plugin, model_name: str, test_dataset: List[Dict] = None):
        """
        Run accuracy benchmark on a test dataset.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            test_dataset: List of input-output pairs for accuracy testing

        Returns:
            Dict containing accuracy metrics
        """
        if test_dataset is None:
            # Create a simple test dataset for demonstration
            test_dataset = [
                {"input": "What is 2+2?", "expected": "4"},
                {"input": "Capital of France?", "expected": "Paris"},
            ]

        correct_predictions = 0
        total_predictions = len(test_dataset)

        for item in test_dataset:
            try:
                # Get model prediction
                prediction = plugin.infer(item["input"])
                
                # Compare with expected output (simplified comparison)
                # In practice, you'd want more sophisticated comparison logic
                if str(prediction).lower() in item["expected"].lower() or \
                   item["expected"].lower() in str(prediction).lower():
                    correct_predictions += 1
            except Exception as e:
                logger.warning(f"Error processing test item: {e}")
                continue

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        result = {
            "model_name": model_name,
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "accuracy_percentage": accuracy * 100
        }

        self.benchmark_results[f"{model_name}_accuracy"] = result
        return result


class ResourceUtilizationBenchmark(TemplateBenchmarkBase):
    """
    Template for resource utilization benchmarks.
    
    Use this class to create benchmarks that measure resource usage.
    """

    def get_model_plugin_class(self):
        """Return the model plugin class to benchmark."""
        raise NotImplementedError("Method not implemented")

    def test_required_functionality(self):
        """Test required functionality for resource utilization benchmarking."""
        raise NotImplementedError("Method not implemented")

    def run_performance_test(self):
        """Run performance test specific to resource utilization."""
        raise NotImplementedError("Method not implemented")

    def run_resource_benchmark(
        self, 
        plugin, 
        model_name: str, 
        duration: float = 30.0,
        sampling_interval: float = 1.0
    ):
        """
        Run resource utilization benchmark over a period of time.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            duration: Duration of the benchmark in seconds
            sampling_interval: Interval between resource measurements in seconds

        Returns:
            Dict containing resource utilization metrics
        """
        import psutil
        import threading
        import time as time_module

        # Lists to store resource measurements
        cpu_percentages = []
        memory_usages = []
        gpu_memory_usages = []

        # Flag to control the monitoring loop
        should_stop = threading.Event()

        def monitor_resources():
            """Monitor system resources in a separate thread."""
            while not should_stop.is_set():
                # CPU usage percentage
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_percentages.append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_usages.append(memory.percent)

                # GPU memory usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_memory = sum([gpu.memoryUtil for gpu in gpus]) / len(gpus) * 100
                        gpu_memory_usages.append(gpu_memory)
                except ImportError:
                    # If GPUtil is not available, skip GPU monitoring
                    pass  # GPU monitoring not available, continue with CPU/Memory monitoring

                # Wait for the next sampling interval
                should_stop.wait(timeout=sampling_interval)

        # Start resource monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Perform operations for the specified duration
        start_time = time_module.time()
        operations_count = 0

        try:
            while time_module.time() - start_time < duration:
                # Perform a sample operation with the model
                try:
                    # Try text generation first
                    result = plugin.generate_text("Brief test prompt", max_new_tokens=10)
                except AttributeError:
                    # If generate_text is not available, try infer
                    try:
                        result = plugin.infer("test input")
                    except:
                        # If both fail, use a simple tensor operation
                        result = plugin.infer(torch.randn(1, 10))

                operations_count += 1
                time_module.sleep(0.1)  # Small delay between operations
        finally:
            # Stop the monitoring thread
            should_stop.set()
            monitor_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish

        # Calculate statistics
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        max_memory = max(memory_usages) if memory_usages else 0
        
        avg_gpu_memory = sum(gpu_memory_usages) / len(gpu_memory_usages) if gpu_memory_usages else 0
        max_gpu_memory = max(gpu_memory_usages) if gpu_memory_usages else 0

        result = {
            "model_name": model_name,
            "test_duration_seconds": duration,
            "operations_performed": operations_count,
            "operations_per_second": operations_count / duration if duration > 0 else 0,
            "cpu_stats": {
                "average_percent": avg_cpu,
                "max_percent": max_cpu,
                "samples_count": len(cpu_percentages)
            },
            "memory_stats": {
                "average_percent": avg_memory,
                "max_percent": max_memory,
                "samples_count": len(memory_usages)
            },
            "gpu_memory_stats": {
                "average_percent": avg_gpu_memory,
                "max_percent": max_gpu_memory,
                "samples_count": len(gpu_memory_usages)
            },
            "sampling_interval": sampling_interval
        }

        self.benchmark_results[f"{model_name}_resource_utilization"] = result
        return result


class ScalabilityBenchmark(TemplateBenchmarkBase):
    """
    Template for scalability benchmarks.
    
    Use this class to create benchmarks that measure how performance scales.
    """

    def get_model_plugin_class(self):
        """Return the model plugin class to benchmark."""
        raise NotImplementedError("Method not implemented")

    def test_required_functionality(self):
        """Test required functionality for scalability benchmarking."""
        raise NotImplementedError("Method not implemented")

    def run_performance_test(self):
        """Run performance test specific to scalability."""
        raise NotImplementedError("Method not implemented")

    def run_scalability_benchmark(
        self,
        plugin,
        model_name: str,
        input_sizes: List[int] = None,
        num_iterations: int = 5
    ):
        """
        Run scalability benchmark with varying input sizes.

        Args:
            plugin: Model plugin instance to benchmark
            model_name: Name of the model being tested
            input_sizes: List of input sizes to test
            num_iterations: Number of iterations for each input size

        Returns:
            Dict containing scalability metrics
        """
        if input_sizes is None:
            input_sizes = [10, 50, 100, 200, 500]

        results = []

        for size in input_sizes:
            # Prepare input of the specified size
            input_data = torch.randint(100, 1000, (1, size))

            # Warmup
            for _ in range(3):
                try:
                    _ = plugin.infer(input_data)
                except:
                    # If tensor input doesn't work, try string
                    _ = plugin.generate_text("test", max_new_tokens=min(size, 50))

            # Benchmark with multiple iterations
            times = []
            for _ in range(num_iterations):
                start_time = time.time()
                try:
                    _ = plugin.infer(input_data)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except:
                    try:
                        _ = plugin.generate_text("test", max_new_tokens=min(size, 50))
                        end_time = time.time()
                        times.append(end_time - start_time)
                    except:
                        # If both fail, record a high time as failure indicator
                        times.append(float('inf'))

            # Calculate metrics for this input size
            avg_time = sum(times) / len(times) if times and all(t != float('inf') for t in times) else float('inf')
            min_time = min(times) if times else float('inf')
            max_time = max(times) if times else float('inf')

            size_result = {
                "input_size": size,
                "avg_time_per_inference": avg_time,
                "min_time_per_inference": min_time,
                "max_time_per_inference": max_time,
                "num_iterations": num_iterations,
                "throughput_tokens_per_second": (size / avg_time) if avg_time > 0 and avg_time != float('inf') else 0
            }

            results.append(size_result)

        overall_result = {
            "model_name": model_name,
            "input_sizes_tested": input_sizes,
            "scalability_results": results,
            "num_iterations_per_size": num_iterations
        }

        self.benchmark_results[f"{model_name}_scalability"] = overall_result
        return overall_result


def run_template_benchmarks(benchmark_instances: List[TemplateBenchmarkBase]):
    """
    Run template benchmarks with specified benchmark instances.

    Args:
        benchmark_instances: List of benchmark instances to run

    Returns:
        Combined results from all benchmarks
    """
    results = {}
    for benchmark in benchmark_instances:
        # Run the benchmark - this would typically involve calling specific test methods
        # For this template, we'll just return the stored results
        results.update(benchmark.benchmark_results)
    
    return results


# Example usage and benchmark runner
if __name__ == "__main__":
    # This would typically be called from the main benchmark runner
    # For demonstration purposes, we'll show the structure
    print("Template Benchmark Module loaded successfully")
    print("Available benchmark classes:")
    print("- TemplateBenchmarkBase")
    print("- CustomAccuracyBenchmark")
    print("- ResourceUtilizationBenchmark")
    print("- ScalabilityBenchmark")