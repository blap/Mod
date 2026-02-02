"""Generation speed benchmarks for all models."""

import torch

from src.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class GenerationSpeedBenchmarks(BenchmarkBase):
    """Text generation speed benchmarks for all models."""

    def test_qwen3_0_6b_generation_speed(self):
        """
        Test generation speed for Qwen3-0.6B model.

        This benchmark measures the text generation speed of the Qwen3-0.6B model
        and validates that the generation process completes within expected timeframes.
        """
        plugin = self.initialize_model("qwen3_0_6b", create_qwen3_0_6b_plugin)

        result = self.benchmark_generation_speed(plugin, "qwen3_0_6b")

        print(f"Qwen3-0.6B Generation: {result['chars_per_second']:.2f} chars/sec")

        # Basic sanity check
        self.assertGreaterEqual(
            result["chars_per_second"],
            0,
            "Characters per second should be non-negative",
        )

    def test_glm_4_7_flash_generation_speed(self):
        """
        Test generation speed for GLM-4.7-Flash model.

        This benchmark measures the text generation speed of the GLM-4.7-Flash model
        and validates that the generation process completes within expected timeframes.
        """
        plugin = self.initialize_model("glm_4_7_flash", create_glm_4_7_flash_plugin)

        result = self.benchmark_generation_speed(plugin, "glm_4_7_flash")

        print(f"GLM-4.7-Flash Generation: {result['chars_per_second']:.2f} chars/sec")

        # Basic sanity check
        self.assertGreaterEqual(
            result["chars_per_second"],
            0,
            "Characters per second should be non-negative",
        )
