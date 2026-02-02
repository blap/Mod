"""
Standardized Accuracy Benchmark for qwen3_coder_30b

This module benchmarks the accuracy for the qwen3_coder_30b model using the standardized interface.
"""

import unittest

import torch

from inference_pio.common.benchmark_interface import AccuracyBenchmark
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b


class Qwen3Coder30bAccuracyBenchmark(unittest.TestCase):
    """Benchmark cases for qwen3_coder_30b accuracy using standardized interface."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_coder_30b()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model_name = "qwen3_coder_30b"

    def test_accuracy_with_standard_interface(self):
        """Test accuracy using the standardized benchmark interface."""
        benchmark = AccuracyBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{self.model_name} Accuracy: {result.value} {result.unit}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)  # Accuracy should be non-negative

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
