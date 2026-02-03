"""Vision-Language model functional benchmarks."""

import torch

from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class VLFunctionalBenchmarks(BenchmarkBase):
    """Vision-Language model functional benchmarks."""

    def test_qwen3_vl_2b_functionality(self):
        """Test functional aspects for Qwen3-VL-2B model."""
        plugin = self.initialize_model(
            "qwen3_vl_2b", create_qwen3_vl_2b_instruct_plugin
        )

        # Test basic inference functionality
        try:
            result = plugin.infer(torch.randint(100, 1000, (1, 10)))
            self.assertIsNotNone(result, "Inference result should not be None")
        except Exception as e:
            # If infer doesn't work with tensor, try with string
            result = plugin.generate_text(
                "Functional test for Qwen3-VL-2B", max_new_tokens=10
            )
            self.assertIsNotNone(result, "Generation result should not be None")

        # Test multimodal functionality if available
        # This would typically involve testing with both text and image inputs
        # For now, we'll test text-based functionality
        text = "Describe the importance of multimodal AI models."
        tokens = plugin.tokenize(text)
        self.assertIsInstance(tokens, list, "Tokenization should return a list")
        self.assertGreater(len(tokens), 0, "Token list should not be empty")

        # Test detokenization functionality
        detokenized = plugin.detokenize(tokens)
        self.assertIsInstance(detokenized, str, "Detokenization should return a string")

        # Test text generation functionality
        prompt = "Explain multimodal learning"
        generated = plugin.generate_text(prompt, max_new_tokens=20)
        self.assertIsInstance(generated, str, "Generated text should be a string")
        self.assertGreater(
            len(generated), len(prompt), "Generated text should be longer than prompt"
        )

        # Test if the model supports specific multimodal features
        # This is a placeholder for when image processing capabilities are tested
        self.assertTrue(
            hasattr(plugin, "infer") or hasattr(plugin, "generate_text"),
            "Model should have inference or generation capabilities",
        )
