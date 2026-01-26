"""
Direct Test for Inference - GLM-4.7

This module tests the inference functionality for the GLM-4.7 model using direct testing.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin
from inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_is_not_none,
    assert_is_instance, assert_in, assert_greater, skip_test, run_tests
)


def setup_test():
    """Set up test fixtures."""
    plugin = create_glm_4_7_plugin()
    success = plugin.initialize(device="cpu")  # Use CPU for tests
    assert_true(success, "Plugin initialization should succeed")
    model = plugin.load_model()
    assert_is_not_none(model, "Model should not be None after loading")
    return plugin


def test_basic_inference():
    """Test basic inference functionality."""
    plugin = setup_test()

    # Create a simple input tensor
    input_data = torch.randint(0, 1000, (1, 10))

    # Perform inference
    result = plugin.infer(input_data)

    # Basic checks
    assert_is_not_none(result, "Result should not be None")
    # The result should be a tensor or dict with logits/scores
    if isinstance(result, dict):
        assert_in('logits', result, "Result dictionary should contain 'logits' key")
    elif isinstance(result, torch.Tensor):
        # If it's a tensor, it should have the right shape
        assert_equal(len(result.shape), 3, "Tensor should have 3 dimensions [batch, seq_len, vocab_size]")


def test_inference_with_attention_mask():
    """Test inference with attention mask."""
    plugin = setup_test()

    input_ids = torch.randint(0, 1000, (2, 20))  # Batch of 2, sequence length 20
    attention_mask = torch.ones_like(input_ids)  # All positions are valid

    # Simulate padding by masking some positions
    attention_mask[:, 15:] = 0  # Last 5 positions are padded

    # Perform inference
    result = plugin.infer({
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })

    assert_is_not_none(result, "Result should not be None")


def test_generate_text_method():
    """Test the generate_text method."""
    plugin = setup_test()

    prompt = "Hello, how are you?"

    # Generate text
    generated_text = plugin.generate_text(prompt, max_new_tokens=10)

    # Basic checks
    assert_is_not_none(generated_text, "Generated text should not be None")
    assert_is_instance(generated_text, str, "Generated text should be a string")
    assert_greater(len(generated_text), len(prompt), "Generated text should be longer than prompt")


def test_tokenize_and_detokenize():
    """Test tokenization and detokenization methods."""
    plugin = setup_test()

    text = "Hello world!"

    # Tokenize
    tokens = plugin.tokenize(text)
    assert_is_not_none(tokens, "Tokens should not be None")

    # Detokenize
    reconstructed_text = plugin.detokenize(tokens)
    assert_is_not_none(reconstructed_text, "Reconstructed text should not be None")
    assert_is_instance(reconstructed_text, str, "Reconstructed text should be a string")


def test_batch_inference():
    """Test inference with batch inputs."""
    plugin = setup_test()

    # Create batch input
    batch_input = torch.randint(0, 1000, (4, 15))  # Batch size 4, sequence length 15

    # Perform inference
    result = plugin.infer(batch_input)

    assert_is_not_none(result, "Result should not be None")
    # If result is tensor, check batch dimension
    if isinstance(result, torch.Tensor):
        assert_equal(result.shape[0], 4, "Batch dimension should be preserved")


def test_inference_with_different_sequence_lengths():
    """Test inference with different sequence lengths."""
    plugin = setup_test()

    for seq_len in [5, 10, 20, 50]:
        input_data = torch.randint(0, 1000, (1, seq_len))
        result = plugin.infer(input_data)
        assert_is_not_none(result, f"Result should not be None for sequence length {seq_len}")


def test_inference_output_shapes():
    """Test that inference outputs have expected shapes."""
    plugin = setup_test()

    input_data = torch.randint(0, 1000, (1, 20))
    result = plugin.infer(input_data)

    if isinstance(result, dict) and 'logits' in result:
        logits = result['logits']
        assert_equal(len(logits.shape), 3, "Logits should have 3 dimensions [batch, seq_len, vocab_size]")
    elif isinstance(result, torch.Tensor):
        assert_equal(len(result.shape), 3, "Tensor should have 3 dimensions [batch, seq_len, vocab_size]")


def test_inference_error_handling():
    """Test error handling in inference."""
    plugin = setup_test()

    # Test with empty input
    try:
        empty_input = torch.tensor([]).reshape(1, 0)
        result = plugin.infer(empty_input)
        # Should handle gracefully
    except Exception:
        # Or raise appropriate exception - this is also acceptable
        pass


def test_chat_completion():
    """Test chat completion functionality."""
    plugin = setup_test()

    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    response = plugin.chat_completion(messages, max_new_tokens=20)

    assert_is_not_none(response, "Response should not be None")
    assert_is_instance(response, str, "Response should be a string")
    assert_greater(len(response), 0, "Response should not be empty")


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_basic_inference,
        test_inference_with_attention_mask,
        test_generate_text_method,
        test_tokenize_and_detokenize,
        test_batch_inference,
        test_inference_with_different_sequence_lengths,
        test_inference_output_shapes,
        test_inference_error_handling,
        test_chat_completion
    ]

    run_tests(test_functions)