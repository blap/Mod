"""
Standardized Throughput Benchmark for glm_4_7_flash

This module benchmarks the throughput for the glm_4_7_flash model using the standardized interface.
"""

import unittest

import torch

from inference_pio.common.benchmark_interface import BatchProcessingBenchmark
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash


class Glm47FlashThroughputBenchmark(unittest.TestCase):
    """Benchmark cases for glm_4_7_flash throughput using standardized interface."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_flash()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model_name = "glm_4_7_flash"

    def test_throughput_with_standard_interface(self):
        """Test throughput using the standardized benchmark interface."""
        benchmark = BatchProcessingBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{self.model_name} Batch Processing Throughput: {result.value} {result.unit}")

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
