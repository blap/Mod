"""
Integration benchmarks for Qwen3 Coder Next model.

This module contains integration-level benchmarks for the Qwen3 Coder Next model,
testing how well different components work together.
"""

import unittest
import torch

from inference_pio.common.interfaces.benchmark_interface import (
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
    BatchProcessingBenchmark,
    BenchmarkRunner
)
from inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin


class Qwen3CoderNextIntegrationBenchmark(unittest.TestCase):
    """Integration benchmarks for Qwen3 Coder Next model."""

    def setUp(self):
        """Set up benchmark fixtures for Qwen3 Coder Next."""
        self.plugin = create_qwen3_coder_next_plugin()
        success = self.plugin.initialize(device="cpu")  # Using CPU for consistent benchmarks
        self.assertTrue(success)
        self.model_name = "Qwen3-Coder-Next"

        # Load model for benchmarking
        model = self.plugin.load_model()
        self.assertTrue(model is not None)

    def test_end_to_end_inference(self):
        """Test end-to-end inference pipeline for Qwen3 Coder Next."""
        runner = BenchmarkRunner()

        # Create integration benchmarks
        benchmarks = [
            InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=50),
            MemoryUsageBenchmark(self.plugin, self.model_name),
            BatchProcessingBenchmark(self.plugin, self.model_name),
        ]

        # Run benchmarks
        results = runner.run_multiple_benchmarks(benchmarks)

        print(f"\nQwen3-Coder-Next Integration Benchmark Results:")
        for result in results:
            print(f"  {result.name}: {result.value} {result.unit}")
            self.assertEqual(result.model_name, self.model_name)
            
            # Validate results are reasonable
            if result.category == "performance":
                self.assertGreaterEqual(result.value, 0, f"Performance value should be non-negative: {result.name}")

    def test_batch_processing_integration(self):
        """Test batch processing integration for Qwen3 Coder Next."""
        benchmark = BatchProcessingBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"\nQwen3-Coder-Next Batch Processing Result: {result.value} {result.unit}")
        
        # Throughput should be positive
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_name, self.model_name)
        self.assertEqual(result.category, "performance")

    def tearDown(self):
        """Clean up after benchmarks."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()