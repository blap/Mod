"""Memory usage benchmarks for all models."""

import torch

from src.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class MemoryUsageBenchmarks(BenchmarkBase):
    """Memory usage benchmarks for all models."""

    def test_qwen3_0_6b_memory_usage(self):
        """
        Test memory usage for Qwen3-0.6B model.

        This benchmark measures the memory consumption of the Qwen3-0.6B model
        during operation and validates that memory usage remains within reasonable bounds.
        """
        plugin = self.initialize_model("qwen3_0_6b", create_qwen3_0_6b_plugin)

        result = self.benchmark_memory_usage(plugin, "qwen3_0_6b")

        print(f"Qwen3-0.6B Memory: {result['memory_used_mb']:.2f} MB used")

        # Memory usage could be positive or negative due to GC, but should be reasonable
        self.assertLess(
            abs(result["memory_used_mb"]),
            1000,  # Less than 1GB change
            "Memory usage change should be reasonable",
        )

    def test_glm_4_7_flash_memory_usage(self):
        """
        Test memory usage for GLM-4.7-Flash model.

        This benchmark measures the memory consumption of the GLM-4.7-Flash model
        during operation and validates that memory usage remains within reasonable bounds.
        """
        plugin = self.initialize_model("glm_4_7_flash", create_glm_4_7_flash_plugin)

        result = self.benchmark_memory_usage(plugin, "glm_4_7_flash")

        print(f"GLM-4.7-Flash Memory: {result['memory_used_mb']:.2f} MB used")

        # Memory usage could be positive or negative due to GC, but should be reasonable
        self.assertLess(
            abs(result["memory_used_mb"]),
            1000,  # Less than 1GB change
            "Memory usage change should be reasonable",
        )
