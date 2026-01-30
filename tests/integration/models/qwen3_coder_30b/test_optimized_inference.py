"""
Optimized Tests for Inference - Qwen3-Coder-30B

This module tests the inference functionality for the Qwen3-Coder-30B model with all optimizations enabled.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_is_not_none,
    assert_is_instance, assert_in, assert_greater, skip_test, run_tests
)


def setup_test():
    """Set up test fixtures with optimizations enabled."""
    plugin = create_qwen3_coder_30b_plugin()
    
    # Initialize with optimizations enabled
    success = plugin.initialize(
        device="cpu",  # Use CPU for tests
        enable_memory_management=True,
        enable_tensor_paging=True,
        enable_adaptive_batching=True,
        enable_kernel_fusion=True,
        enable_tensor_compression=True,
        enable_disk_offloading=True,
        enable_activation_offloading=True,
        enable_model_surgery=True,
        torch_compile_mode="reduce-overhead"  # Enable torch.compile for optimizations
    )
    assert_true(success, "Plugin initialization should succeed with optimizations")
    model = plugin.load_model()
    assert_is_not_none(model, "Model should not be None after loading")
    return plugin


def test_basic_inference_with_optimizations():
    """Test basic inference functionality with all optimizations enabled."""
    plugin = setup_test()

    # Create a simple input tensor
    input_data = torch.randint(0, 1000, (1, 10))

    # Perform inference
    result = plugin.infer(input_data)

    # Basic checks
    assert_is_not_none(result, "Result should not be None")
    # The result should be a string (generated text)
    assert_is_instance(result, str, "Result should be a string (generated text)")


def test_inference_with_attention_mask_optimized():
    """Test inference with attention mask and optimizations."""
    plugin = setup_test()

    prompt = "Hello, how are you?"
    
    # Tokenize with attention mask
    inputs = plugin.tokenize(prompt, return_tensors="pt", padding=True, truncation=True)
    attention_mask = torch.ones_like(inputs['input_ids'])

    # Perform inference with attention mask
    result = plugin.infer({
        'input_ids': inputs['input_ids'],
        'attention_mask': attention_mask
    })

    assert_is_not_none(result, "Result should not be None")


def test_generate_text_method_optimized():
    """Test the generate_text method with optimizations."""
    plugin = setup_test()

    prompt = "Hello, how are you?"

    # Generate text
    generated_text = plugin.generate_text(prompt, max_new_tokens=10)

    # Basic checks
    assert_is_not_none(generated_text, "Generated text should not be None")
    assert_is_instance(generated_text, str, "Generated text should be a string")
    assert_greater(len(generated_text), len(prompt), "Generated text should be longer than prompt")


def test_tokenize_and_detokenize_optimized():
    """Test tokenization and detokenization methods with optimizations."""
    plugin = setup_test()

    text = "Hello world!"

    # Tokenize
    tokens = plugin.tokenize(text)
    assert_is_not_none(tokens, "Tokens should not be None")

    # Detokenize
    reconstructed_text = plugin.detokenize(tokens['input_ids'][0] if isinstance(tokens, dict) else tokens)
    assert_is_not_none(reconstructed_text, "Reconstructed text should not be None")
    assert_is_instance(reconstructed_text, str, "Reconstructed text should be a string")


def test_batch_inference_optimized():
    """Test inference with batch inputs and optimizations."""
    plugin = setup_test()

    # Create batch input
    batch_prompts = ["Hello!", "How are you?", "What's your name?", "Nice to meet you"]
    
    # Process each prompt individually since infer expects a single string
    results = []
    for prompt in batch_prompts:
        result = plugin.infer(prompt)
        results.append(result)
        assert_is_not_none(result, f"Result should not be None for prompt: {prompt}")

    assert_equal(len(results), len(batch_prompts), "Number of results should match number of prompts")


def test_inference_with_different_sequence_lengths_optimized():
    """Test inference with different sequence lengths and optimizations."""
    plugin = setup_test()

    for seq_len in [5, 10, 20]:
        prompt = " ".join(["hello"] * seq_len)
        result = plugin.infer(prompt)
        assert_is_not_none(result, f"Result should not be None for sequence length {seq_len}")


def test_memory_management_optimized():
    """Test that memory management optimizations are active."""
    plugin = setup_test()

    # Check that memory stats can be retrieved
    memory_stats = plugin.get_memory_stats()
    assert_is_not_none(memory_stats, "Memory stats should not be None")
    assert_in('system_memory_percent', memory_stats, "Memory stats should contain system_memory_percent")


def test_compression_stats_optimized():
    """Test that compression optimizations are active."""
    plugin = setup_test()

    # Check that compression stats can be retrieved
    compression_stats = plugin.get_compression_stats()
    assert_is_not_none(compression_stats, "Compression stats should not be None")
    assert_in('compression_enabled', compression_stats, "Compression stats should contain compression_enabled")


def test_offloading_stats_optimized():
    """Test that offloading optimizations are active."""
    plugin = setup_test()

    # Check that offloading stats can be retrieved
    offloading_stats = plugin.get_offloading_stats()
    assert_is_not_none(offloading_stats, "Offloading stats should not be None")
    assert_in('offloading_enabled', offloading_stats, "Offloading stats should contain offloading_enabled")


def test_chat_completion_optimized():
    """Test chat completion functionality with optimizations."""
    plugin = setup_test()

    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    response = plugin.chat_completion(messages, max_new_tokens=20)

    assert_is_not_none(response, "Response should not be None")
    assert_is_instance(response, str, "Response should be a string")
    assert_greater(len(response), 0, "Response should not be empty")


def test_qwen3_coder_specific_optimizations():
    """Test Qwen3-Coder-30B specific optimizations."""
    plugin = setup_test()

    # Check that Qwen3 specific optimizations are applied during model loading
    info = plugin.get_model_info()
    assert_is_not_none(info, "Model info should not be None")
    assert_in('optimizations_enabled', info, "Model info should contain optimizations_enabled")
    
    optimizations = info['optimizations_enabled']
    # Check for Qwen3-specific optimizations
    assert_in('flash_attention_2', optimizations, "Flash attention 2 should be listed as optimization")


def test_model_info_with_optimizations():
    """Test that model info includes optimization information."""
    plugin = setup_test()

    info = plugin.get_model_info()
    assert_is_not_none(info, "Model info should not be None")
    assert_in('optimizations_enabled', info, "Model info should contain optimizations_enabled")
    
    optimizations = info['optimizations_enabled']
    assert_is_instance(optimizations, dict, "Optimizations should be a dictionary")


def test_model_parameters_with_optimizations():
    """Test that model parameters can be retrieved with optimizations."""
    plugin = setup_test()

    params = plugin.get_model_parameters()
    assert_is_not_none(params, "Model parameters should not be None")
    assert_in('num_parameters', params, "Parameters should contain num_parameters")


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_basic_inference_with_optimizations,
        test_inference_with_attention_mask_optimized,
        test_generate_text_method_optimized,
        test_tokenize_and_detokenize_optimized,
        test_batch_inference_optimized,
        test_inference_with_different_sequence_lengths_optimized,
        test_memory_management_optimized,
        test_compression_stats_optimized,
        test_offloading_stats_optimized,
        test_chat_completion_optimized,
        test_qwen3_coder_specific_optimizations,
        test_model_info_with_optimizations,
        test_model_parameters_with_optimizations
    ]

    run_tests(test_functions)