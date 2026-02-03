"""LLM functional benchmarks."""

import torch

from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import (
    create_qwen3_4b_instruct_2507_plugin,
)

from ....benchmark_base import BenchmarkBase  # Go up 3 more levels to reach benchmarks/


class LLMFunctionalBenchmarks(BenchmarkBase):
    """LLM functional benchmarks."""

    def test_qwen3_0_6b_functionality(self):
        """Test functional aspects for Qwen3-0.6B model."""
        plugin = self.initialize_model("qwen3_0_6b", create_qwen3_0_6b_plugin)

        # Test basic inference functionality
        try:
            result = plugin.infer(torch.randint(100, 1000, (1, 10)))
            self.assertIsNotNone(result, "Inference result should not be None")
        except Exception as e:
            # If infer doesn't work with tensor, try with string
            result = plugin.generate_text(
                "Functional test for Qwen3-0.6B", max_new_tokens=10
            )
            self.assertIsNotNone(result, "Generation result should not be None")

        # Test tokenization functionality
        text = "This is a test sentence for tokenization."
        tokens = plugin.tokenize(text)
        self.assertIsInstance(tokens, list, "Tokenization should return a list")
        self.assertGreater(len(tokens), 0, "Token list should not be empty")

        # Test detokenization functionality
        detokenized = plugin.detokenize(tokens)
        self.assertIsInstance(detokenized, str, "Detokenization should return a string")

        # Test text generation functionality
        prompt = "The future of artificial intelligence"
        generated = plugin.generate_text(prompt, max_new_tokens=20)
        self.assertIsInstance(generated, str, "Generated text should be a string")
        self.assertGreater(
            len(generated), len(prompt), "Generated text should be longer than prompt"
        )

    def test_glm_4_7_flash_functionality(self):
        """Test functional aspects for GLM-4.7-Flash model."""
        plugin = self.initialize_model("glm_4_7_flash", create_glm_4_7_flash_plugin)

        # Test basic inference functionality
        try:
            result = plugin.infer(torch.randint(100, 1000, (1, 10)))
            self.assertIsNotNone(result, "Inference result should not be None")
        except Exception as e:
            # If infer doesn't work with tensor, try with string
            result = plugin.generate_text(
                "Functional test for GLM-4.7-Flash", max_new_tokens=10
            )
            self.assertIsNotNone(result, "Generation result should not be None")

        # Test tokenization functionality
        text = "This is a test sentence for tokenization."
        tokens = plugin.tokenize(text)
        self.assertIsInstance(tokens, list, "Tokenization should return a list")
        self.assertGreater(len(tokens), 0, "Token list should not be empty")

        # Test detokenization functionality
        detokenized = plugin.detokenize(tokens)
        self.assertIsInstance(detokenized, str, "Detokenization should return a string")

        # Test text generation functionality
        prompt = "Advancements in natural language processing"
        generated = plugin.generate_text(prompt, max_new_tokens=20)
        self.assertIsInstance(generated, str, "Generated text should be a string")
        self.assertGreater(
            len(generated), len(prompt), "Generated text should be longer than prompt"
        )

    def test_qwen3_4b_instruct_2507_functionality(self):
        """Test functional aspects for Qwen3-4B-Instruct-2507 model."""
        plugin = self.initialize_model(
            "qwen3_4b_instruct_2507", create_qwen3_4b_instruct_2507_plugin
        )

        # Test basic inference functionality
        try:
            result = plugin.infer(torch.randint(100, 1000, (1, 10)))
            self.assertIsNotNone(result, "Inference result should not be None")
        except Exception as e:
            # If infer doesn't work with tensor, try with string
            result = plugin.generate_text(
                "Functional test for Qwen3-4B-Instruct-2507", max_new_tokens=10
            )
            self.assertIsNotNone(result, "Generation result should not be None")

        # Test tokenization functionality
        text = "This is a test sentence for tokenization."
        tokens = plugin.tokenize(text)
        self.assertIsInstance(tokens, list, "Tokenization should return a list")
        self.assertGreater(len(tokens), 0, "Token list should not be empty")

        # Test detokenization functionality
        detokenized = plugin.detokenize(tokens)
        self.assertIsInstance(detokenized, str, "Detokenization should return a string")

        # Test text generation functionality
        prompt = "Explain the concept of machine learning"
        generated = plugin.generate_text(prompt, max_new_tokens=20)
        self.assertIsInstance(generated, str, "Generated text should be a string")
        self.assertGreater(
            len(generated), len(prompt), "Generated text should be longer than prompt"
        )
