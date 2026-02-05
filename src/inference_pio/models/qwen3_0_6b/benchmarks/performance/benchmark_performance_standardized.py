"""
Performance benchmarks for Qwen3 0.6B model.

This module contains performance-level benchmarks for the Qwen3 0.6B model,
measuring various performance metrics under different conditions.
"""

import unittest
import torch

from inference_pio.common.interfaces.benchmark_interface import (
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
    BatchProcessingBenchmark,
    ModelLoadingTimeBenchmark,
    BenchmarkRunner,
    get_performance_suite
)
from inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin


class Qwen306BPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for Qwen3 0.6B model."""

    def setUp(self):
        """Set up benchmark fixtures for Qwen3 0.6B."""
        self.plugin = create_qwen3_0_6b_plugin()
        success = self.plugin.initialize(device="cpu")  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model_name = "Qwen3-0.6B"

        # Load model for benchmarking
        model = self.plugin.load_model()
        self.assertTrue(model is not None)

    def test_performance_suite(self):
        """Run the complete performance benchmark suite for Qwen3 0.6B."""
        runner = BenchmarkRunner()

        # Get the standard performance suite
        benchmarks = get_performance_suite(self.plugin, self.model_name)

        # Run benchmarks
        results = runner.run_multiple_benchmarks(benchmarks)

        print(f"\nQwen3-0.6B Performance Suite Results:")
        for result in results:
            print(f"  {result.name}: {result.value} {result.unit}")
            self.assertEqual(result.model_name, self.model_name)
            
            # Validate performance results are reasonable
            if result.category == "performance":
                self.assertGreaterEqual(result.value, 0, f"Performance value should be non-negative: {result.name}")

    def test_detailed_inference_speeds(self):
        """Test detailed inference speeds for Qwen3 0.6B."""
        runner = BenchmarkRunner()
        
        # Test different input lengths
        benchmarks = [
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=20),
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=50),
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=100),
        ]

        results = runner.run_multiple_benchmarks(benchmarks)

        print(f"\nQwen3-0.6B Detailed Inference Speeds:")
        for result in results:
            print(f"  {result.name}: {result.value} {result.unit}")
            self.assertGreater(result.value, 0)
            self.assertEqual(result.model_name, self.model_name)
            self.assertEqual(result.category, "performance")

    def test_memory_and_loading_performance(self):
        """Test memory usage and loading time for Qwen3 0.6B."""
        runner = BenchmarkRunner()
        
        benchmarks = [
            MemoryUsageBenchmark(self.plugin, self.model_name),
            ModelLoadingTimeBenchmark(self.plugin, self.model_name),
        ]

        results = runner.run_multiple_benchmarks(benchmarks)

        print(f"\nQwen3-0.6B Memory and Loading Performance:")
        for result in results:
            print(f"  {result.name}: {result.value} {result.unit}")
            self.assertGreaterEqual(result.value, 0)
            self.assertEqual(result.model_name, self.model_name)

    def tearDown(self):
        """Clean up after benchmarks."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()