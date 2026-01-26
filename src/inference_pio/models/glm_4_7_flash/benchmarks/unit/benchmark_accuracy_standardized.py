"""
Standardized Accuracy Benchmark for glm_4_7_flash

This module benchmarks the accuracy for the glm_4_7_flash model using the standardized interface.
"""

import unittest
import torch
from typing import TYPE_CHECKING
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash
from inference_pio.common.benchmark_interface import AccuracyBenchmark, BenchmarkResult


if TYPE_CHECKING:
    from inference_pio.common.benchmark_interface import ModelPluginProtocol


class Glm47FlashAccuracyBenchmark(unittest.TestCase):
    """Benchmark cases for glm_4_7_flash accuracy using standardized interface."""

    def setUp(self) -> None:
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_flash()
        success = self.plugin.initialize(device="cpu", use_mock_model=True)
        self.assertTrue(success)
        self.model_name = "glm_4_7_flash"

    def test_accuracy_with_standard_interface(self) -> None:
        """Test accuracy using the standardized benchmark interface."""
        benchmark = AccuracyBenchmark(self.plugin, self.model_name)
        result: BenchmarkResult = benchmark.run()

        print(f"{self.model_name} Accuracy: {result.value} {result.unit}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)  # Accuracy should be non-negative

    def tearDown(self) -> None:
        """Clean up after each test method."""
        if hasattr(self.plugin, 'cleanup') and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == '__main__':
    unittest.main()
