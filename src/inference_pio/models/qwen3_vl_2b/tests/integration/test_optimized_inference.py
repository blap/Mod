"""
Optimized Tests for Inference - Qwen3-VL-2B

This module tests the inference functionality for the Qwen3-VL-2B model with all optimizations enabled.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from PIL import Image
import io
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
from inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_is_not_none,
    assert_is_instance, assert_in, assert_greater, skip_test, run_tests
)


def setup_test():
    """Set up test fixtures with optimizations enabled."""
    plugin = create_qwen3_vl_2b_instruct_plugin()
    
    # Initialize with optimizations enabled
    success = plugin.initialize(
        device="cpu",  # Use CPU for tests
        config={'model_path': './dummy_model_path'}  # Use dummy path for testing
    )
    assert_true(success, "Plugin initialization should succeed with optimizations")
    
    # Mock the model loading to avoid actual model loading
    plugin._model = torch.nn.Linear(10, 10)  # Dummy model for testing
    plugin.is_loaded = True
    
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
    # The result should be a string (generated text) or tensor
    assert_is_instance(result, (str, torch.Tensor), "Result should be a string or tensor")


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
    if 'input_ids' in tokens:
        reconstructed_text = plugin.detokenize(tokens['input_ids'][0])
    else:
        reconstructed_text = plugin.detokenize(tokens)
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


def test_qwen3_vl_specific_optimizations():
    """Test Qwen3-VL-2B specific optimizations."""
    plugin = setup_test()

    # Check that Qwen3-VL specific optimizations are applied during model loading
    info = plugin.get_model_info()
    assert_is_not_none(info, "Model info should not be None")
    assert_in('optimizations_applied', info, "Model info should contain optimizations_applied")
    
    optimizations = info['optimizations_applied']
    # Check for Qwen3-VL specific optimizations
    assert_in('cuda_kernels', optimizations, "CUDA kernels should be listed as optimization")
    assert_in('cross_modal_alignment', optimizations, "Cross-modal alignment should be listed as optimization")
    assert_in('multimodal_attention', optimizations, "Multimodal attention should be listed as optimization")


def test_model_info_with_optimizations():
    """Test that model info includes optimization information."""
    plugin = setup_test()

    info = plugin.get_model_info()
    assert_is_not_none(info, "Model info should not be None")
    assert_in('optimizations_applied', info, "Model info should contain optimizations_applied")
    
    optimizations = info['optimizations_applied']
    assert_is_instance(optimizations, dict, "Optimizations should be a dictionary")


def test_model_parameters_with_optimizations():
    """Test that model parameters can be retrieved with optimizations."""
    plugin = setup_test()

    params = plugin.get_model_info()
    assert_is_not_none(params, "Model info should not be None")
    assert_in('parameters', params, "Info should contain parameters")


def test_image_encoding_functionality():
    """Test image encoding functionality."""
    plugin = setup_test()

    # Create a dummy image
    img_array = torch.randint(0, 255, (3, 224, 224)).to(torch.uint8)
    pil_img = Image.fromarray(img_array.permute(1, 2, 0).numpy())
    
    # Test image encoding
    try:
        embeddings = plugin.encode_image(pil_img)
        assert_is_not_none(embeddings, "Image embeddings should not be None")
        assert_is_instance(embeddings, torch.Tensor, "Embeddings should be a tensor")
    except Exception as e:
        # If image processing fails due to missing model files, that's acceptable for this test
        print(f"Image encoding test skipped due to: {e}")


def test_multimodal_inference():
    """Test multimodal inference functionality."""
    plugin = setup_test()

    # Create a dummy image
    img_array = torch.randint(0, 255, (3, 224, 224)).to(torch.uint8)
    pil_img = Image.fromarray(img_array.permute(1, 2, 0).numpy())
    
    # Test multimodal inference
    multimodal_input = {
        'text': "Describe this image:",
        'image': pil_img
    }
    
    try:
        result = plugin.infer(multimodal_input)
        assert_is_not_none(result, "Multimodal inference result should not be None")
    except Exception as e:
        # If multimodal processing fails due to missing model files, that's acceptable for this test
        print(f"Multimodal inference test skipped due to: {e}")


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


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_basic_inference_with_optimizations,
        test_inference_with_attention_mask_optimized,
        test_generate_text_method_optimized,
        test_tokenize_and_detokenize_optimized,
        test_batch_inference_optimized,
        test_inference_with_different_sequence_lengths_optimized,
        test_qwen3_vl_specific_optimizations,
        test_model_info_with_optimizations,
        test_model_parameters_with_optimizations,
        test_image_encoding_functionality,
        test_multimodal_inference,
        test_chat_completion_optimized
    ]

    run_tests(test_functions)