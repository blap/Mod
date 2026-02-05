"""
Standardized Memory Usage Benchmark for qwen3_4b_instruct_2507

This module benchmarks the memory usage for the qwen3_4b_instruct_2507 model using the standardized interface.
"""

import unittest

import torch

from inference_pio.common.benchmark_interface import MemoryUsageBenchmark
from inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    create_qwen3_4b_instruct_2507,
)


class Qwen34bInstruct2507MemoryUsageBenchmark(unittest.TestCase):
    """Benchmark cases for qwen3_4b_instruct_2507 memory usage using standardized interface."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_4b_instruct_2507()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model_name = "qwen3_4b_instruct_2507"

    def test_memory_usage_with_standard_interface(self):
        """Test memory usage using the standardized benchmark interface."""
        benchmark = MemoryUsageBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{self.model_name} Memory Usage: {result.value} {result.unit}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
