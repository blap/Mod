"""
Real Performance Benchmark for Inference Speed - GLM-4.7-Flash

This module benchmarks the inference speed for the GLM-4.7-Flash model using real performance measurements.
"""

import time
import unittest

import torch

from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.common.real_performance_monitor import (
    benchmark_function_real,
    get_real_system_metrics,
)


class BenchmarkGLM47FlashInferenceSpeed(unittest.TestCase):
    """Benchmark cases for GLM-4.7-Flash inference speed using real performance measurements."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_flash_plugin()
        # Initialize with real model - use CPU for consistent benchmarks
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model = self.plugin.load_model()
        self.assertTrue(self.model is not None)

    def benchmark_inference_speed(self, input_length=50, num_iterations=10):
        """Real benchmark inference speed for given input length."""
        # Prepare input
        input_ids = torch.randint(0, 1000, (1, input_length))

        # Warmup
        for _ in range(3):
            _ = self.plugin.infer(input_ids)

        # Use real performance monitoring
        def timed_inference():
            return self.plugin.infer(input_ids)

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

        return {
            "total_time": benchmark_results["avg_time_ms"]
            / 1000.0
            * num_iterations,  # Total time in seconds
            "avg_time_per_inference": avg_time_per_inference,
            "tokens_per_second": tokens_per_second,
            "num_iterations": num_iterations,
            "input_length": input_length,
            "raw_benchmark_data": benchmark_results,
        }

    def test_inference_speed_short_input(self):
        """Benchmark inference speed with short input sequences using real metrics."""
        results = self.benchmark_inference_speed(input_length=20, num_iterations=5)

        print(f"\nGLM-4.7 Short Input (20 tokens) Inference Speed:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Avg time per inference: {results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {results['tokens_per_second']:.2f}")
        print(
            f"  Raw benchmark data: {results['raw_benchmark_data']['avg_time_ms']:.2f}ms avg"
        )

        # Basic sanity check - should be positive values
        self.assertGreater(results["tokens_per_second"], 0)
        self.assertGreater(results["avg_time_per_inference"], 0)

    def test_inference_speed_medium_input(self):
        """Benchmark inference speed with medium input sequences using real metrics."""
        results = self.benchmark_inference_speed(input_length=50, num_iterations=5)

        print(f"\nGLM-4.7 Medium Input (50 tokens) Inference Speed:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Avg time per inference: {results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {results['tokens_per_second']:.2f}")
        print(
            f"  Raw benchmark data: {results['raw_benchmark_data']['avg_time_ms']:.2f}ms avg"
        )

        # Basic sanity check
        self.assertGreater(results["tokens_per_second"], 0)

    def test_inference_speed_long_input(self):
        """Benchmark inference speed with long input sequences using real metrics."""
        results = self.benchmark_inference_speed(input_length=100, num_iterations=3)

        print(f"\nGLM-4.7 Long Input (100 tokens) Inference Speed:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Avg time per inference: {results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {results['tokens_per_second']:.2f}")
        print(
            f"  Raw benchmark data: {results['raw_benchmark_data']['avg_time_ms']:.2f}ms avg"
        )

        # Basic sanity check
        self.assertGreater(results["tokens_per_second"], 0)

    def test_generation_speed(self):
        """Benchmark text generation speed using real metrics."""
        prompt = "The quick brown fox jumps over the lazy dog. "

        # Warmup
        _ = self.plugin.generate_text(prompt, max_new_tokens=10)

        # Use real performance monitoring
        def timed_generation():
            return self.plugin.generate_text(prompt, max_new_tokens=50)

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

        print(f"\nGLM-4.7 Generation Speed:")
        print(f"  Avg time per generation: {avg_time_per_generation:.4f}s")
        print(
            f"  Generated text length: {len(str(generated_text)) if generated_text else 0} chars"
        )
        print(f"  Characters per second: {chars_per_second:.2f}")
        print(f"  Raw benchmark data: {benchmark_results['avg_time_ms']:.2f}ms avg")

        # Basic sanity check
        self.assertGreater(avg_time_per_generation, 0)
        self.assertIsInstance(str(generated_text) if generated_text else "", str)

    def test_batch_inference_speed(self):
        """Benchmark batch inference speed using real metrics."""
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_ids = torch.randint(0, 1000, (batch_size, 30))

                # Warmup
                for _ in range(2):
                    _ = self.plugin.infer(input_ids)

                # Use real performance monitoring
                def timed_batch_inference():
                    return self.plugin.infer(input_ids)

                benchmark_results = benchmark_function_real(
                    timed_batch_inference, iterations=5
                )

                avg_time_per_batch = (
                    benchmark_results["avg_time_ms"] / 1000.0
                )  # Convert ms to seconds
                tokens_per_second = (
                    (batch_size * 30) / avg_time_per_batch
                    if avg_time_per_batch > 0
                    else 0
                )

                print(f"\nGLM-4.7 Batch Size {batch_size} Inference Speed:")
                print(f"  Avg time per batch: {avg_time_per_batch:.4f}s")
                print(f"  Tokens per second: {tokens_per_second:.2f}")
                print(
                    f"  Raw benchmark data: {benchmark_results['avg_time_ms']:.2f}ms avg"
                )

                # Basic sanity check
                self.assertGreater(tokens_per_second, 0)

    def test_variable_length_inference_speed(self):
        """Benchmark inference speed with variable input lengths using real metrics."""
        lengths = [10, 25, 50, 75, 100]

        results = []
        for length in lengths:
            with self.subTest(length=length):
                result = self.benchmark_inference_speed(
                    input_length=length, num_iterations=3
                )
                results.append(result)

                print(
                    f"  Length {length}: {result['tokens_per_second']:.2f} tokens/sec"
                )

        # Verify all lengths produced valid results
        for result in results:
            self.assertGreater(result["tokens_per_second"], 0)

    def test_system_resources_during_benchmark(self):
        """Test system resource usage during benchmark."""
        # Get initial system metrics
        initial_metrics = get_real_system_metrics()

        # Run a benchmark
        results = self.benchmark_inference_speed(input_length=50, num_iterations=10)

        # Get final system metrics
        final_metrics = get_real_system_metrics()

        print(f"\nSystem Resource Usage During Benchmark:")
        print(f"  Initial CPU: {initial_metrics.cpu_percent:.2f}%")
        print(f"  Final CPU: {final_metrics.cpu_percent:.2f}%")
        print(f"  Initial Memory: {initial_metrics.memory_used_mb:.2f}MB")
        print(f"  Final Memory: {final_metrics.memory_used_mb:.2f}MB")

        # Basic checks
        self.assertIsNotNone(initial_metrics.timestamp)
        self.assertIsNotNone(final_metrics.timestamp)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
