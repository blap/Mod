"""
Unit benchmarks for Qwen3 0.6B model.

This module contains unit-level benchmarks for the Qwen3 0.6B model,
following the standardized benchmark interface.
"""

import unittest
import torch

from inference_pio.common.interfaces.benchmark_interface import (
    AccuracyBenchmark,
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
    BenchmarkRunner
)
from inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin


class Qwen306BUnitBenchmark(unittest.TestCase):
    """Unit benchmarks for Qwen3 0.6B model."""

    def setUp(self):
        """Set up benchmark fixtures for Qwen3 0.6B."""
        self.plugin = create_qwen3_0_6b_plugin()
        success = self.plugin.initialize(device="cpu")  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model_name = "Qwen3-0.6B"

        # Load model for benchmarking
        model = self.plugin.load_model()
        self.assertTrue(model is not None)

    def test_accuracy_unit(self):
        """Test accuracy for Qwen3 0.6B."""
        benchmark = AccuracyBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"\nQwen3-0.6B Accuracy Result: {result.value} {result.unit}")
        
        # Accuracy should be between 0 and 1 for ratio
        if result.unit == "ratio":
            self.assertGreaterEqual(result.value, 0)
            self.assertLessEqual(result.value, 1.0)
        
        self.assertEqual(result.model_name, self.model_name)
        self.assertEqual(result.category, "accuracy")

    def test_inference_speed_unit(self):
        """Test inference speed for Qwen3 0.6B."""
        benchmark = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=20)
        result = benchmark.run()

        print(f"\nQwen3-0.6B Inference Speed Result: {result.value} {result.unit}")
        
        # Speed should be positive
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_name, self.model_name)
        self.assertEqual(result.category, "performance")

    def test_memory_usage_unit(self):
        """Test memory usage for Qwen3 0.6B."""
        benchmark = MemoryUsageBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"\nQwen3-0.6B Memory Usage Result: {result.value} {result.unit}")
        
        # Memory usage should be positive
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_name, self.model_name)
        self.assertEqual(result.category, "performance")

    def tearDown(self):
        """Clean up after benchmarks."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()