"""Vision-Language inference speed benchmarks."""

import torch

from src.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class VLInferenceSpeedBenchmarks(BenchmarkBase):
    """Vision-Language inference speed benchmarks."""

    def test_qwen3_vl_2b_inference_speed(self):
        """Test inference speed for Qwen3-VL-2B model."""
        plugin = self.initialize_model(
            "qwen3_vl_2b", create_qwen3_vl_2b_instruct_plugin
        )

        # Test different input lengths
        for length in [20, 50, 100]:
            with self.subTest(input_length=length):
                result = self.benchmark_inference_speed(
                    plugin, "qwen3_vl_2b", input_length=length, num_iterations=3
                )

                print(
                    f"Qwen3-VL-2B ({length} tokens): {result['tokens_per_second']:.2f} tokens/sec"
                )

                # Basic sanity check - should be positive values
                self.assertGreaterEqual(
                    result["tokens_per_second"],
                    0,
                    f"Tokens per second should be non-negative for {length} tokens",
                )
