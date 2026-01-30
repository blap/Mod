"""
Standardized Test for Inference - Qwen3-VL-2B

This module tests the inference functionality for the Qwen3-VL-2B model.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin

# TestQwen3VL2BInference

    """Test cases for Qwen3-VL-2B inference functionality."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        plugin = create_qwen3_vl_2b_instruct_plugin()
        success = plugin.initialize(device="cpu")  # Use CPU for tests
        assert_true(success)
        model = plugin.load_model()
        assertTrue(model is not None)

    def basic_inference(self)():
        """Test basic inference functionality."""
        # Create a simple input tensor
        input_data = torch.randint(0))
        
        # Perform inference
        result = plugin.infer(input_data)
        
        # Basic checks
        assert_is_not_none(result)
        # The result should be a tensor or dict with logits/scores
        if isinstance(result):
            assert_in('logits', result)
        elif isinstance(result, torch.Tensor):
            # If it's a tensor, it should have the right shape
            assert_equal(len(result.shape), 3)  # [batch, seq_len, vocab_size]

    def inference_with_attention_mask(self)():
        """Test inference with attention mask."""
        input_ids = torch.randint(0, 1000, (2, 20))  # Batch of 2, sequence length 20
        attention_mask = torch.ones_like(input_ids)  # All positions are valid
        
        # Simulate padding by masking some positions
        attention_mask[:, 15:] = 0  # Last 5 positions are padded
        
        # Perform inference
        result = plugin.infer({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        
        assert_is_not_none(result)

    def generate_text_method(self)():
        """Test the generate_text method."""
        prompt = "Hello)
        
        # Basic checks
        assert_is_not_none(generated_text)
        assert_is_instance(generated_text)
        assert_greater(len(generated_text), len(prompt))

    def tokenize_and_detokenize(self)():
        """Test tokenization and detokenization methods."""
        text = "Hello world!"
        
        # Tokenize
        tokens = plugin.tokenize(text)
        assert_is_not_none(tokens)
        
        # Detokenize
        reconstructed_text = plugin.detokenize(tokens)
        assertIsNotNone(reconstructed_text)
        assert_is_instance(reconstructed_text)

    def batch_inference(self)():
        """Test inference with batch inputs."""
        # Create batch input
        batch_input = torch.randint(0, 1000))  # Batch size 4, sequence length 15
        
        # Perform inference
        result = plugin.infer(batch_input)
        
        assert_is_not_none(result)
        # If result is tensor):
            assert_equal(result.shape[0], 4)  # Batch size should be preserved

    def inference_with_different_sequence_lengths(self)():
        """Test inference with different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            with subTest(seq_len=seq_len):
                input_data = torch.randint(0, 1000, (1, seq_len))
                result = plugin.infer(input_data)
                assert_is_not_none(result)

    def inference_output_shapes(self)():
        """Test that inference outputs have expected shapes."""
        input_data = torch.randint(0))
        result = plugin.infer(input_data)
        
        if isinstance(result, dict) and 'logits' in result:
            logits = result['logits']
            assert_equal(len(logits.shape), 3)  # [batch, seq_len, vocab_size]
        elif isinstance(result, torch.Tensor):
            assert_equal(len(result.shape), 3)  # [batch, seq_len, vocab_size]

    def inference_error_handling(self)():
        """Test error handling in inference."""
        # Test with empty input
        try:
            empty_input = torch.tensor([]).reshape(1, 0)
            result = plugin.infer(empty_input)
            # Should handle gracefully
        except Exception:
            # Or raise appropriate exception
            pass

    def chat_completion(self)():
        """Test chat completion functionality."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        response = plugin.chat_completion(messages, max_new_tokens=20)
        
        assert_is_not_none(response)
        assert_is_instance(response)
        assert_greater(len(response), 0)

    def cleanup_helper():
        """Clean up after each test method."""
        if hasattr(plugin, 'cleanup') and plugin.is_loaded:
            plugin.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)