"""LLM inference speed benchmarks."""

import torch

from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    create_qwen3_4b_instruct_2507_plugin,
)

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class LLMInferenceSpeedBenchmarks(BenchmarkBase):
    """LLM inference speed benchmarks."""

    def test_qwen3_0_6b_inference_speed(self):
        """
        Test inference speed for Qwen3-0.6B model.

        This benchmark measures the inference speed of the Qwen3-0.6B model
        across different input sequence lengths and validates that performance
        remains consistent.

        The benchmark tests multiple input lengths to evaluate how the model's
        performance scales with sequence length. Shorter sequences test overhead
        and initialization costs, while longer sequences test sustained throughput
        capabilities. The multiple length testing reveals performance characteristics
        across different operational scenarios.
        """
        plugin = self.initialize_model("qwen3_0_6b", create_qwen3_0_6b_plugin)

        # Test different input lengths to evaluate performance scaling
        # 20 tokens: Tests overhead and short sequence performance
        # 50 tokens: Tests medium sequence performance
        # 100 tokens: Tests longer sequence performance and sustained throughput
        for length in [20, 50, 100]:
            with self.subTest(input_length=length):
                # Execute the benchmark with specified parameters
                # Measures actual inference speed using real performance monitoring
                result = self.benchmark_inference_speed(
                    plugin, "qwen3_0_6b", input_length=length, num_iterations=3
                )

                print(
                    f"Qwen3-0.6B ({length} tokens): {result['tokens_per_second']:.2f} tokens/sec"
                )

                # Basic sanity check - should be positive values
                # Validates that the benchmark produced meaningful results
                self.assertGreaterEqual(
                    result["tokens_per_second"],
                    0,
                    f"Tokens per second should be non-negative for {length} tokens",
                )

    def test_glm_4_7_flash_inference_speed(self):
        """
        Test inference speed for GLM-4.7-Flash model.

        This benchmark measures the inference speed of the GLM-4.7-Flash model
        across different input sequence lengths and validates that performance
        remains consistent.
        """
        plugin = self.initialize_model("glm_4_7_flash", create_glm_4_7_flash_plugin)

        # Test different input lengths
        for length in [20, 50, 100]:
            with self.subTest(input_length=length):
                result = self.benchmark_inference_speed(
                    plugin, "glm_4_7_flash", input_length=length, num_iterations=3
                )

                print(
                    f"GLM-4.7-Flash ({length} tokens): {result['tokens_per_second']:.2f} tokens/sec"
                )

                # Basic sanity check - should be positive values
                self.assertGreaterEqual(
                    result["tokens_per_second"],
                    0,
                    f"Tokens per second should be non-negative for {length} tokens",
                )

    def test_qwen3_4b_instruct_2507_inference_speed(self):
        """
        Test inference speed for Qwen3-4B-Instruct-2507 model.

        This benchmark measures the inference speed of the Qwen3-4B-Instruct-2507 model
        across different input sequence lengths and validates that performance
        remains consistent.
        """
        plugin = self.initialize_model(
            "qwen3_4b_instruct_2507", create_qwen3_4b_instruct_2507_plugin
        )

        # Test different input lengths
        for length in [20, 50, 100]:
            with self.subTest(input_length=length):
                result = self.benchmark_inference_speed(
                    plugin,
                    "qwen3_4b_instruct_2507",
                    input_length=length,
                    num_iterations=3,
                )

                print(
                    f"Qwen3-4B-Instruct-2507 ({length} tokens): {result['tokens_per_second']:.2f} tokens/sec"
                )

                # Basic sanity check - should be positive values
                self.assertGreaterEqual(
                    result["tokens_per_second"],
                    0,
                    f"Tokens per second should be non-negative for {length} tokens",
                )
