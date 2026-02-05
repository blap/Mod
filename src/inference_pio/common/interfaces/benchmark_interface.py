"""
Standardized Benchmark Interface for Model Evaluation

This module defines a common interface for benchmarking models across the project.
All models should implement these interfaces to ensure consistent benchmarking.
"""

import abc
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import torch


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""

    name: str
    value: float
    unit: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    model_name: Optional[str] = None
    category: Optional[str] = None


class ModelPluginProtocol(Protocol):
    """Protocol defining the interface that model plugins should implement."""

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        ...

    def initialize(self, device: str = "cpu", **kwargs) -> bool:
        """Initialize the model plugin."""
        ...

    def load_model(self):
        """Load the model."""
        ...

    def infer(self, input_ids: torch.Tensor):
        """Run inference on the model."""
        ...

    def generate_text(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Generate text using the model."""
        ...

    def cleanup(self):
        """Clean up resources."""
        ...


class BaseBenchmark(abc.ABC):
    """Base abstract class for all benchmarks."""

    def __init__(self, model_plugin: ModelPluginProtocol, model_name: str):
        """
        Initialize the benchmark.

        Args:
            model_plugin: The model plugin instance to benchmark
            model_name: Name of the model being benchmarked
        """
        self.model_plugin = model_plugin
        self.model_name = model_name

    @abc.abstractmethod
    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results."""
        raise NotImplementedError("Method not implemented")

    def warmup(self, num_iterations: int = 3):
        """Perform warmup runs to stabilize measurements."""
        # Create dummy input for warmup
        dummy_input = torch.randint(0, 1000, (1, 20))

        for _ in range(num_iterations):
            try:
                _ = self.model_plugin.infer(dummy_input)
            except:
                # If infer fails, try generate_text as alternative
                try:
                    _ = self.model_plugin.generate_text("warmup", max_new_tokens=5)
                except:
                    # If both fail, skip warmup
                    break

    def prepare_input(self, input_length: int, batch_size: int = 1) -> torch.Tensor:
        """Prepare standardized input for benchmarking."""
        return torch.randint(0, 1000, (batch_size, input_length))


class InferenceSpeedBenchmark(BaseBenchmark):
    """Benchmark for measuring inference speed."""

    def __init__(
        self,
        model_plugin: ModelPluginProtocol,
        model_name: str,
        input_length: int = 50,
        num_iterations: int = 10,
    ):
        super().__init__(model_plugin, model_name)
        self.input_length = input_length
        self.num_iterations = num_iterations

    def run(self) -> BenchmarkResult:
        """Run inference speed benchmark."""
        # Load model if not already loaded
        if not self.model_plugin.is_loaded:
            self.model_plugin.load_model()

        # Prepare input
        input_ids = self.prepare_input(self.input_length)

        # Warmup
        self.warmup()

        # Timing run
        start_time = time.time()
        for i in range(self.num_iterations):
            _ = self.model_plugin.infer(input_ids)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_inference = total_time / self.num_iterations
        tokens_per_second = (
            self.input_length / avg_time_per_inference
            if avg_time_per_inference > 0
            else float("inf")
        )

        metadata = {
            "input_length": self.input_length,
            "num_iterations": self.num_iterations,
            "total_time": total_time,
            "avg_time_per_inference": avg_time_per_inference,
        }

        return BenchmarkResult(
            name=f"inference_speed_{self.input_length}tokens",
            value=tokens_per_second,
            unit="tokens/sec",
            metadata=metadata,
            model_name=self.model_name,
            category="performance",
        )


class MemoryUsageBenchmark(BaseBenchmark):
    """Benchmark for measuring memory usage."""

    def run(self) -> BenchmarkResult:
        """Run memory usage benchmark."""
        import gc

        import psutil

        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load model if not already loaded
        if not self.model_plugin.is_loaded:
            self.model_plugin.load_model()

        # Get memory after loading
        memory_after_load = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after_load - baseline_memory

        # Run inference to trigger full memory usage
        input_ids = self.prepare_input(50)
        try:
            result = self.model_plugin.infer(input_ids)
        except:
            # If infer fails, try generate_text as alternative
            result = self.model_plugin.generate_text("test", max_new_tokens=10)

        # Force garbage collection
        gc.collect()

        # Get memory after inference
        memory_after_inference = process.memory_info().rss / 1024 / 1024  # MB

        metadata = {
            "baseline_memory_mb": baseline_memory,
            "memory_after_load_mb": memory_after_load,
            "memory_increase_mb": memory_increase,
            "memory_after_inference_mb": memory_after_inference,
        }

        return BenchmarkResult(
            name="memory_usage",
            value=memory_increase,
            unit="MB",
            metadata=metadata,
            model_name=self.model_name,
            category="performance",
        )


class AccuracyBenchmark(BaseBenchmark):
    """Benchmark for measuring model accuracy."""

    def run(self) -> BenchmarkResult:
        """Run accuracy benchmark."""
        # Load model if not already loaded
        if not self.model_plugin.is_loaded:
            self.model_plugin.load_model()

        # Test with a simple known fact
        prompt = "The capital of France is"
        expected_answer = "Paris"

        try:
            generated = self.model_plugin.generate_text(prompt, max_new_tokens=10)

            # Check if expected answer appears in generated text
            accuracy_score = (
                1.0 if expected_answer.lower() in generated.lower() else 0.0
            )

            metadata = {
                "prompt": prompt,
                "expected_answer": expected_answer,
                "generated_text": generated,
                "accuracy_score": accuracy_score,
            }

            return BenchmarkResult(
                name="accuracy_score",
                value=accuracy_score,
                unit="ratio",
                metadata=metadata,
                model_name=self.model_name,
                category="accuracy",
            )
        except Exception as e:
            metadata = {"error": str(e), "accuracy_score": 0.0}

            return BenchmarkResult(
                name="accuracy_score",
                value=0.0,
                unit="ratio",
                metadata=metadata,
                model_name=self.model_name,
                category="accuracy",
            )


class BatchProcessingBenchmark(BaseBenchmark):
    """Benchmark for measuring batch processing capabilities."""

    def __init__(
        self,
        model_plugin: ModelPluginProtocol,
        model_name: str,
        batch_sizes: List[int] = None,
    ):
        super().__init__(model_plugin, model_name)
        self.batch_sizes = batch_sizes or [1, 2, 4, 8]

    def run(self) -> BenchmarkResult:
        """Run batch processing benchmark."""
        # Load model if not already loaded
        if not self.model_plugin.is_loaded:
            self.model_plugin.load_model()

        batch_results = {}

        for batch_size in self.batch_sizes:
            input_ids = self.prepare_input(30, batch_size)

            # Warmup
            self.warmup(num_iterations=2)

            start_time = time.time()
            result = self.model_plugin.infer(input_ids)
            processing_time = time.time() - start_time

            batch_results[f"batch_{batch_size}"] = {
                "processing_time": processing_time,
                "throughput": (
                    (batch_size * 30) / processing_time
                    if processing_time > 0
                    else float("inf")
                ),
            }

        # Calculate average throughput
        throughputs = [
            v["throughput"]
            for v in batch_results.values()
            if v["throughput"] != float("inf")
        ]
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0

        metadata = {
            "batch_results": batch_results,
            "batch_sizes_tested": self.batch_sizes,
        }

        return BenchmarkResult(
            name="batch_processing_throughput",
            value=avg_throughput,
            unit="tokens/sec",
            metadata=metadata,
            model_name=self.model_name,
            category="performance",
        )


class ModelLoadingTimeBenchmark(BaseBenchmark):
    """Benchmark for measuring model loading time."""

    def run(self) -> BenchmarkResult:
        """Run model loading time benchmark."""
        # Ensure model is unloaded first
        if self.model_plugin.is_loaded:
            self.model_plugin.cleanup()

        # Measure loading time
        start_time = time.time()
        model = self.model_plugin.load_model()
        end_time = time.time()

        loading_time = end_time - start_time

        metadata = {"loading_time_seconds": loading_time}

        return BenchmarkResult(
            name="model_loading_time",
            value=loading_time,
            unit="seconds",
            metadata=metadata,
            model_name=self.model_name,
            category="performance",
        )


class BenchmarkRunner:
    """Class to run multiple benchmarks and collect results."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_benchmark(self, benchmark: BaseBenchmark) -> BenchmarkResult:
        """Run a single benchmark and return results."""
        try:
            result = benchmark.run()
            self.results.append(result)
            return result
        except Exception as e:
            # Return error result
            error_result = BenchmarkResult(
                name=f"{benchmark.__class__.__name__}_error",
                value=0.0,
                unit="error",
                metadata={"error": str(e)},
                model_name=getattr(benchmark, "model_name", "unknown"),
                category="error",
            )
            self.results.append(error_result)
            return error_result

    def run_multiple_benchmarks(
        self, benchmarks: List[BaseBenchmark]
    ) -> List[BenchmarkResult]:
        """Run multiple benchmarks and return all results."""
        results = []
        for benchmark in benchmarks:
            result = self.run_benchmark(benchmark)
            results.append(result)
        return results

    def save_results(
        self, output_dir: str = "benchmark_results", filename_prefix: str = "benchmark"
    ) -> Dict[str, str]:
        """Save benchmark results to JSON and CSV files."""
        import datetime

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare results data
        results_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "value": r.value,
                    "unit": r.unit,
                    "model_name": r.model_name,
                    "category": r.category,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

        # Save JSON results
        json_filename = output_path / f"{filename_prefix}_results_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save CSV results
        csv_filename = output_path / f"{filename_prefix}_results_{timestamp}.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Model Name",
                    "Benchmark Name",
                    "Value",
                    "Unit",
                    "Category",
                    "Timestamp",
                ]
            )

            # Write data rows
            for result in self.results:
                writer.writerow(
                    [
                        result.model_name,
                        result.name,
                        result.value,
                        result.unit,
                        result.category,
                        results_data["timestamp"],
                    ]
                )

        return {"json_file": str(json_filename), "csv_file": str(csv_filename)}

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        # Group results by model
        model_results: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            model_name = result.model_name or "Unknown"
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)

        for model_name, results in model_results.items():
            print(f"\n{model_name}:")
            for result in results:
                print(f"  {result.name}: {result.value} {result.unit}")
                if result.metadata and "error" in result.metadata:
                    print(f"    ERROR: {result.metadata['error']}")


# Predefined benchmark suites
def get_performance_suite(
    model_plugin: ModelPluginProtocol, model_name: str
) -> List[BaseBenchmark]:
    """Get a suite of performance benchmarks."""
    return [
        InferenceSpeedBenchmark(model_plugin, model_name, input_length=20),
        InferenceSpeedBenchmark(model_plugin, model_name, input_length=50),
        InferenceSpeedBenchmark(model_plugin, model_name, input_length=100),
        MemoryUsageBenchmark(model_plugin, model_name),
        BatchProcessingBenchmark(model_plugin, model_name),
        ModelLoadingTimeBenchmark(model_plugin, model_name),
    ]


def get_accuracy_suite(
    model_plugin: ModelPluginProtocol, model_name: str
) -> List[BaseBenchmark]:
    """Get a suite of accuracy benchmarks."""
    return [AccuracyBenchmark(model_plugin, model_name)]


def get_full_suite(
    model_plugin: ModelPluginProtocol, model_name: str
) -> List[BaseBenchmark]:
    """Get a full suite of benchmarks."""
    return get_performance_suite(model_plugin, model_name) + get_accuracy_suite(
        model_plugin, model_name
    )
