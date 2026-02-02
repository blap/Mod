"""
Standardized Benchmark for Inference Speed - Qwen3-4B-Instruct-2507

This module benchmarks the inference speed for the Qwen3-4B-Instruct-2507 model.
"""

import time
import unittest

import torch

from inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    create_qwen3_4b_instruct_2507_plugin,
)


class BenchmarkQwen34BInstruct2507InferenceSpeed(unittest.TestCase):
    """Benchmark cases for Qwen3-4B-Instruct-2507 inference speed."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_4b_instruct_2507_plugin()
        success = self.plugin.initialize(
            device="cpu"
        )  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model = self.plugin.load_model()
        self.assertTrue(self.model is not None)

    def benchmark_inference_speed(self, input_length=50, num_iterations=10):
        """Benchmark inference speed for given input length."""
        # Prepare input
        input_ids = torch.randint(0, 1000, (1, input_length))

        # Warmup
        for _ in range(3):
            _ = self.plugin.infer(input_ids)

        # Timing run
        start_time = time.time()
        for i in range(num_iterations):
            _ = self.plugin.infer(input_ids)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_iterations
        tokens_per_second = input_length / avg_time_per_inference

        return {
            "total_time": total_time,
            "avg_time_per_inference": avg_time_per_inference,
            "tokens_per_second": tokens_per_second,
            "num_iterations": num_iterations,
            "input_length": input_length,
        }

    def test_inference_speed_short_input(self):
        """Benchmark inference speed with short input sequences."""
        results = self.benchmark_inference_speed(input_length=20, num_iterations=5)

        print(f"\nQwen3-4B-Instruct-2507 Short Input (20 tokens) Inference Speed:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Avg time per inference: {results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {results['tokens_per_second']:.2f}")

        # Basic sanity check - should be positive values
        self.assertGreater(results["tokens_per_second"], 0)
        self.assertGreater(results["avg_time_per_inference"], 0)

    def test_inference_speed_medium_input(self):
        """Benchmark inference speed with medium input sequences."""
        results = self.benchmark_inference_speed(input_length=50, num_iterations=5)

        print(f"\nQwen3-4B-Instruct-2507 Medium Input (50 tokens) Inference Speed:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Avg time per inference: {results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {results['tokens_per_second']:.2f}")

        # Basic sanity check
        self.assertGreater(results["tokens_per_second"], 0)

    def test_inference_speed_long_input(self):
        """Benchmark inference speed with long input sequences."""
        results = self.benchmark_inference_speed(input_length=100, num_iterations=3)

        print(f"\nQwen3-4B-Instruct-2507 Long Input (100 tokens) Inference Speed:")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Avg time per inference: {results['avg_time_per_inference']:.4f}s")
        print(f"  Tokens per second: {results['tokens_per_second']:.2f}")

        # Basic sanity check
        self.assertGreater(results["tokens_per_second"], 0)

    def test_generation_speed(self):
        """Benchmark text generation speed."""
        prompt = "The quick brown fox jumps over the lazy dog. "

        # Warmup
        _ = self.plugin.generate_text(prompt, max_new_tokens=10)

        # Timing run
        start_time = time.time()
        generated_text = self.plugin.generate_text(prompt, max_new_tokens=50)
        end_time = time.time()

        total_time = end_time - start_time
        chars_per_second = (
            len(generated_text) / total_time if len(generated_text) > 0 else 0
        )

        print(f"\nQwen3-4B-Instruct-2507 Generation Speed:")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Generated text length: {len(generated_text)} chars")
        print(f"  Characters per second: {chars_per_second:.2f}")

        # Basic sanity check
        self.assertGreater(total_time, 0)
        self.assertIsInstance(generated_text, str)

    def test_batch_inference_speed(self):
        """Benchmark batch inference speed."""
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_ids = torch.randint(0, 1000, (batch_size, 30))

                # Warmup
                for _ in range(2):
                    _ = self.plugin.infer(input_ids)

                # Timing run
                start_time = time.time()
                for i in range(5):
                    _ = self.plugin.infer(input_ids)
                end_time = time.time()

                total_time = end_time - start_time
                avg_time_per_batch = total_time / 5
                tokens_per_second = (batch_size * 30) / avg_time_per_batch

                print(
                    f"\nQwen3-4B-Instruct-2507 Batch Size {batch_size} Inference Speed:"
                )
                print(f"  Avg time per batch: {avg_time_per_batch:.4f}s")
                print(f"  Tokens per second: {tokens_per_second:.2f}")

                # Basic sanity check
                self.assertGreater(tokens_per_second, 0)

    def test_variable_length_inference_speed(self):
        """Benchmark inference speed with variable input lengths."""
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

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
