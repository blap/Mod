"""
Standardized Performance Benchmark for qwen3_vl_2b

This module benchmarks the performance for the qwen3_vl_2b model using the standardized interface.
"""

import unittest

import torch

from inference_pio.common.benchmark_interface import (
    BatchProcessingBenchmark,
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
)
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b


class Qwen3Vl2bPerformanceBenchmark(unittest.TestCase):
    """Benchmark cases for qwen3_vl_2b performance using standardized interface."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_vl_2b()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model_name = "qwen3_vl_2b"

    def test_inference_speed_with_standard_interface(self):
        """Test inference speed using the standardized benchmark interface."""
        # Test different input lengths
        for input_length in [20, 50, 100]:
            benchmark = InferenceSpeedBenchmark(
                self.plugin, self.model_name, input_length=input_length
            )
            result = benchmark.run()

            print(
                f"{self.model_name} Inference Speed ({input_length} tokens): {result.value} {result.unit}"
            )

            # Basic validation
            self.assertIsNotNone(result.value)
            if result.value != float("inf"):  # Handle infinite values
                self.assertGreater(result.value, 0)

    def test_memory_usage_with_standard_interface(self):
        """Test memory usage using the standardized benchmark interface."""
        benchmark = MemoryUsageBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{self.model_name} Memory Usage: {result.value} {result.unit}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)

    def test_batch_processing_with_standard_interface(self):
        """Test batch processing using the standardized benchmark interface."""
        benchmark = BatchProcessingBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(
            f"{self.model_name} Batch Processing Throughput: {result.value} {result.unit}"
        )

        # Basic validation
        self.assertIsNotNone(result.value)
        if result.value != float("inf"):  # Handle infinite values
            self.assertGreaterEqual(result.value, 0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
