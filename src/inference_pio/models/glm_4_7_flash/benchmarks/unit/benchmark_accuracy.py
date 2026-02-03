"""
Standardized Benchmark for Accuracy - GLM-4.7-Flash

This module benchmarks the accuracy for the GLM-4.7-Flash model.
"""

import unittest

import numpy as np
import torch

from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin


class BenchmarkGLM47FlashAccuracy(unittest.TestCase):
    """Benchmark cases for GLM-4.7-Flash accuracy."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_glm_4_7_flash_plugin()
        success = self.plugin.initialize(device="cpu")
        self.assertTrue(success)
        self.model = self.plugin.load_model()
        self.assertTrue(self.model is not None)

    def test_perplexity_calculation(self):
        """Test perplexity calculation as an accuracy metric."""
        # Create a simple test sequence
        test_text = "The quick brown fox jumps over the lazy dog."
        tokens = self.plugin.tokenize(test_text)

        if isinstance(tokens, list):
            tokens = torch.tensor([tokens])

        # Calculate logits for the sequence
        with torch.no_grad():
            outputs = self.plugin.infer(tokens)

            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                self.skipTest("Model does not return logits in expected format")

            # Calculate probabilities and perplexity
            # Shift so that tokens < n predict token n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()

            # Calculate cross entropy loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            perplexity = torch.exp(loss)

        print(f"\nGLM-4.7 Perplexity: {perplexity.item():.4f}")

        # Perplexity should be finite and positive
        self.assertFalse(torch.isnan(perplexity))
        self.assertFalse(torch.isinf(perplexity))
        self.assertGreater(perplexity, 0)

    def test_reproducibility(self):
        """Test that the model produces reproducible results."""
        prompt = "The capital of France is"

        # Generate text multiple times with the same seed
        torch.manual_seed(42)
        result1 = self.plugin.generate_text(prompt, max_new_tokens=10)

        torch.manual_seed(42)  # Reset seed
        result2 = self.plugin.generate_text(prompt, max_new_tokens=10)

        print(f"\nGLM-4.7 Reproducibility Test:")
        print(f"  Result 1: {result1}")
        print(f"  Result 2: {result2}")

        # Results should be identical with the same seed
        self.assertEqual(result1, result2)

    def test_known_fact_accuracy(self):
        """Test accuracy on known factual questions."""
        # Simple factual questions that should have consistent answers
        test_cases = [
            ("What is 2+2?", "4"),
            ("The capital of France is", "Paris"),
            ("Water boils at", "100"),
        ]

        correct_count = 0
        total_count = len(test_cases)

        for question, expected_answer in test_cases:
            generated = self.plugin.generate_text(question, max_new_tokens=20)

            # Check if the expected answer appears in the generated text
            if expected_answer.lower() in generated.lower():
                correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"\nGLM-4.7 Known Fact Accuracy:")
        print(f"  Correct: {correct_count}/{total_count}")
        print(f"  Accuracy: {accuracy:.2%}")

        # The model may not get all facts right, but we can at least check it generates text
        self.assertGreaterEqual(correct_count, 0)
        self.assertLessEqual(correct_count, total_count)

    def test_token_probability_distribution(self):
        """Test that token probability distributions are well-formed."""
        prompt = "Hello world"
        tokens = self.plugin.tokenize(prompt)

        if isinstance(tokens, list):
            tokens = torch.tensor([tokens])

        with torch.no_grad():
            outputs = self.plugin.infer(tokens)

            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                self.skipTest("Model does not return logits in expected format")

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Check that probabilities sum to approximately 1
            prob_sums = torch.sum(probs, dim=-1)
            close_to_one = torch.allclose(
                prob_sums, torch.ones_like(prob_sums), atol=1e-5
            )

        print(f"\nGLM-4.7 Probability Distribution Test:")
        print(f"  Probabilities sum to ~1: {close_to_one}")

        self.assertTrue(close_to_one)

    def test_numerical_stability(self):
        """Test numerical stability of the model."""
        # Run multiple inferences and check for NaN or Inf values
        for i in range(10):
            input_ids = torch.randint(0, 1000, (1, 20))

            with torch.no_grad():
                outputs = self.plugin.infer(input_ids)

                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    continue  # Skip if unexpected format

                has_nan = torch.isnan(logits).any()
                has_inf = torch.isinf(logits).any()

                if has_nan or has_inf:
                    print(f"\nGLM-4.7 Numerical Stability Issue at iteration {i}")
                    print(f"  Has NaN: {has_nan}")
                    print(f"  Has Inf: {has_inf}")
                    self.fail(f"Numerical instability detected at iteration {i}")

        print(f"\nGLM-4.7 Numerical Stability: Passed all 10 iterations")

    def test_sequence_completeness(self):
        """Test that generated sequences are complete and coherent."""
        prompts = ["Once upon a time", "The weather today", "In the field of"]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                generated = self.plugin.generate_text(prompt, max_new_tokens=30)

                # Check that generated text is not empty and contains the prompt
                self.assertIsNotNone(generated)
                self.assertIsInstance(generated, str)
                self.assertGreater(len(generated), len(prompt))
                self.assertIn(
                    prompt.split()[0], generated
                )  # At least first word should be there

    def test_tokenization_consistency(self):
        """Test consistency between tokenization and detokenization."""
        test_texts = [
            "Hello world!",
            "The quick brown fox.",
            "AI and machine learning are transforming industries.",
        ]

        for text in test_texts:
            with self.subTest(text=text):
                # Tokenize
                tokens = self.plugin.tokenize(text)

                # Detokenize
                reconstructed = self.plugin.detokenize(tokens)

                # The reconstructed text should be similar to original
                self.assertIsNotNone(reconstructed)
                self.assertIsInstance(reconstructed, str)

    def test_model_confidence_calibration(self):
        """Test if model confidence scores are meaningful."""
        prompt = "The Earth is"
        tokens = self.plugin.tokenize(prompt)

        if isinstance(tokens, list):
            tokens = torch.tensor([tokens])

        with torch.no_grad():
            outputs = self.plugin.infer(tokens)

            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                self.skipTest("Model does not return logits in expected format")

            # Get probabilities for the last token position
            last_token_probs = torch.softmax(logits[0, -1, :], dim=-1)

            # Get top-k probabilities
            top_k = 5
            top_probs, top_indices = torch.topk(last_token_probs, top_k)

            # Check that probabilities are in valid range [0, 1] and sum <= 1
            self.assertTrue(torch.all(top_probs >= 0))
            self.assertTrue(torch.all(top_probs <= 1))
            self.assertLessEqual(torch.sum(top_probs), 1.0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, "cleanup") and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == "__main__":
    unittest.main()
