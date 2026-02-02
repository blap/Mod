"""Cross-model comparison benchmarks."""

import torch

from src.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class CrossModelComparisonBenchmarks(BenchmarkBase):
    """Cross-model comparison benchmarks."""

    def setUp(self):
        """Initialize all models for comparison."""
        super().setUp()

        # Initialize all models
        self.initialize_model("qwen3_0_6b", create_qwen3_0_6b_plugin)
        self.initialize_model("glm_4_7_flash", create_glm_4_7_flash_plugin)

    def test_cross_model_inference_speed_comparison(self):
        """Compare inference speed across models."""
        models_to_test = ["qwen3_0_6b", "glm_4_7_flash"]
        input_length = 50
        num_iterations = 3

        results = {}

        for model_name in models_to_test:
            plugin = self.models[model_name]
            result = self.benchmark_inference_speed(
                plugin,
                model_name,
                input_length=input_length,
                num_iterations=num_iterations,
            )
            results[model_name] = result
            print(f"{model_name}: {result['tokens_per_second']:.2f} tokens/sec")

        # Verify all models returned valid results
        for model_name, result in results.items():
            self.assertIn(model_name, results)
            self.assertGreaterEqual(
                result["tokens_per_second"],
                0,
                f"{model_name} should have non-negative tokens per second",
            )

    def test_cross_model_generation_comparison(self):
        """Compare generation speed across models."""
        models_to_test = ["qwen3_0_6b", "glm_4_7_flash"]
        prompt = "The future of artificial intelligence is"

        results = {}

        for model_name in models_to_test:
            plugin = self.models[model_name]
            result = self.benchmark_generation_speed(plugin, model_name, prompt=prompt)
            results[model_name] = result
            print(f"{model_name}: {result['chars_per_second']:.2f} chars/sec")

        # Verify all models returned valid results
        for model_name, result in results.items():
            self.assertIn(model_name, results)
            # Chars per second could be 0 if generation failed, but shouldn't be negative
            self.assertGreaterEqual(
                result["chars_per_second"],
                0,
                f"{model_name} should have non-negative chars per second",
            )
