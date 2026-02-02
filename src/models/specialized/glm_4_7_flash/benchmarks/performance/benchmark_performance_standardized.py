"""
Standardized Performance Benchmark for glm_4_7_flash

This module benchmarks the performance for the glm_4_7_flash model using the standardized interface.
"""

import unittest
from typing import TYPE_CHECKING

import torch

from inference_pio.common.benchmark_interface import (
    BatchProcessingBenchmark,
    BenchmarkResult,
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
)
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash

if TYPE_CHECKING:
    from inference_pio.common.benchmark_interface import ModelPluginProtocol


class Glm47FlashPerformanceBenchmark(unittest.TestCase):
    """Benchmark cases for glm_4_7_flash performance using standardized interface."""

    def setUp(self) -> None:
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_flash()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model_name = "glm_4_7_flash"

    def test_inference_speed_with_standard_interface(self) -> None:
        """Test inference speed using the standardized benchmark interface."""
        # Test different input lengths
        for input_length in [20, 50, 100]:
            benchmark = InferenceSpeedBenchmark(
                self.plugin, self.model_name, input_length=input_length
            )
            result: BenchmarkResult = benchmark.run()

            print(
                f"{self.model_name} Inference Speed ({input_length} tokens): {result.value} {result.unit}"
            )

            # Basic validation
            self.assertIsNotNone(result.value)
            if result.value != float("inf"):  # Handle infinite values
                self.assertGreater(result.value, 0)

    def test_memory_usage_with_standard_interface(self) -> None:
        """Test memory usage using the standardized benchmark interface."""
        benchmark = MemoryUsageBenchmark(self.plugin, self.model_name)
        result: BenchmarkResult = benchmark.run()

        print(f"{self.model_name} Memory Usage: {result.value} {result.unit}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)

    def test_batch_processing_with_standard_interface(self) -> None:
        """Test batch processing using the standardized benchmark interface."""
        benchmark = BatchProcessingBenchmark(self.plugin, self.model_name)
        result: BenchmarkResult = benchmark.run()

        print(
            f"{self.model_name} Batch Processing Throughput: {result.value} {result.unit}"
        )

        # Basic validation
        self.assertIsNotNone(result.value)
        if result.value != float("inf"):  # Handle infinite values
            self.assertGreaterEqual(result.value, 0)

    def tearDown(self) -> None:
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
