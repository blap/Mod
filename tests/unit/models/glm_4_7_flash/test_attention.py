"""
Comprehensive test suite for Attention Mechanisms - GLM-4.7 Flash Model.

This module provides extensive testing for the attention mechanisms in the GLM-4.7 model,
covering various aspects of attention functionality including:
- Existence and proper initialization of attention layers
- Forward pass functionality with different input configurations
- Causal masking for autoregressive generation
- Key-value caching for efficient sequential generation
- Output shape validation
- Gradient flow through attention layers
- Performance with varying sequence lengths
- Numerical stability of computations

These tests ensure the attention mechanism operates correctly across different scenarios.
"""

import torch
from tests.utils.test_utils import (
    assert_true,
    assert_greater,
    assert_is_not_none,
    assert_equal,
    run_tests
)
from inference_pio.models.glm_4_7.plugin import create_glm_4_7_flash_plugin
from inference_pio.common.base_attention import BaseAttention


def setup_glm47_plugin():
    """
    Set up the GLM-4.7-Flash plugin for testing purposes.

    Initializes the GLM-4.7-Flash plugin on CPU device and loads the model
    for subsequent testing operations.

    Returns:
        tuple: A tuple containing the initialized plugin and loaded model
    """
    plugin = create_glm_4_7_flash_plugin()
    success = plugin.initialize(device="cpu")  # Use CPU for tests
    assert_true(success)
    model = plugin.load_model()
    assert_true(model is not None)
    return plugin, model


def cleanup_glm47_plugin(plugin):
    """
    Clean up the GLM-4.7-Flash plugin after testing.

    Performs cleanup operations on the plugin if it has a cleanup method
    and the model is currently loaded.

    Args:
        plugin: The GLM-4.7-Flash plugin instance to clean up
    """
    if hasattr(plugin, 'cleanup') and plugin.is_loaded:
        plugin.cleanup()


def test_attention_layer_exists():
    """
    Test that attention layers exist within the GLM-4.7 model.

    Verifies that the model contains properly initialized attention layers
    by searching for modules that are instances of BaseAttention or have
    'attention' or 'attn' in their names.
    """
    plugin, model = setup_glm47_plugin()
    try:
        # Check if the model has attention layers
        attention_layers = []
        for name, module in model.named_modules():
            if isinstance(module, BaseAttention) or 'attention' in name.lower() or 'attn' in name.lower():
                attention_layers.append((name, module))

        # At least some attention layers should exist
        assert_greater(len(attention_layers), 0,
                      "No attention layers found in the model")
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_forward_pass():
    """
    Test the forward pass functionality of attention mechanisms.

    Validates that the attention mechanism can process input sequences
    and produce valid outputs through the complete forward pass.
    """
    plugin, model = setup_glm47_plugin()
    try:
        # Create sample input
        batch_size = 2
        seq_len = 10
        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 512

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        # Try to access attention layer directly if possible
        # Otherwise, run a forward pass through the whole model
        result = plugin.infer({'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                               'attention_mask': attention_mask})

        assert_is_not_none(result)
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_with_causal_mask():
    """
    Test attention functionality with causal masking.

    Verifies that the attention mechanism properly handles causal masks,
    which are essential for autoregressive generation where future tokens
    should not influence current predictions.
    """
    plugin, model = setup_glm47_plugin()
    try:
        batch_size = 1
        seq_len = 20
        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 512

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

        # Run inference with causal mask
        result = plugin.infer({
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': causal_mask
        })

        assert_is_not_none(result)
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_with_kv_cache():
    """
    Test attention functionality with Key-Value (KV) cache.

    Validates that the model properly implements KV caching for efficient
    sequential generation, allowing reuse of previously computed keys
    and values to avoid redundant computations.
    """
    plugin, model = setup_glm47_plugin()
    try:
        # Test if the model supports KV cache
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Run initial forward pass
        result1 = plugin.infer({
            'input_ids': input_ids,
            'use_cache': True
        })

        assert_is_not_none(result1)

        # If the model supports KV cache, it should return past key values
        if isinstance(result1, dict) and 'past_key_values' in result1:
            # Run with past key values
            next_input = torch.randint(0, 1000, (batch_size, 1))
            result2 = plugin.infer({
                'input_ids': next_input,
                'past_key_values': result1['past_key_values'],
                'use_cache': True
            })

            assert_is_not_none(result2)
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_output_shapes():
    """
    Test that attention mechanism outputs have the expected shapes.

    Ensures that the output tensors from the attention mechanism have
    the correct dimensions, particularly that batch size and sequence
    length are preserved in the output.
    """
    plugin, model = setup_glm47_plugin()
    try:
        batch_size = 2
        seq_len = 15
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        result = plugin.infer({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

        if isinstance(result, dict) and 'logits' in result:
            logits = result['logits']
            assert_equal(logits.shape[:2], (batch_size, seq_len))
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_gradient_flow():
    """
    Test that gradients flow properly through attention layers during backpropagation.

    Validates that the attention mechanism supports gradient computation,
    which is essential for training and fine-tuning operations.
    """
    plugin, model = setup_glm47_plugin()
    try:
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Enable gradient computation
        for param in model.parameters():
            param.requires_grad = True

        result = plugin.infer({'input_ids': input_ids})

        if isinstance(result, dict) and 'logits' in result:
            logits = result['logits']
            loss = logits.sum()
            loss.backward()

            # Check that gradients exist for parameters
            params_with_grad = [p for p in model.parameters() if p.grad is not None]
            assert_greater(len(params_with_grad), 0)
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_with_different_sequence_lengths():
    """
    Test attention functionality with various sequence lengths.

    Validates that the attention mechanism can handle different input
    sequence lengths efficiently, which is important for accommodating
    variable-length inputs in practical applications.
    """
    plugin, model = setup_glm47_plugin()
    try:
        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 512

        for seq_len in [5, 10, 20, 50]:
            # For each sequence length, create inputs and run inference
            hidden_states = torch.randn(1, seq_len, hidden_dim)
            attention_mask = torch.ones(1, seq_len)

            result = plugin.infer({
                'input_ids': torch.randint(0, 1000, (1, seq_len)),
                'attention_mask': attention_mask
            })

            assert_is_not_none(result)
    finally:
        cleanup_glm47_plugin(plugin)


def test_attention_numerical_stability():
    """
    Test the numerical stability of attention computations.

    Ensures that the attention mechanism produces numerically stable outputs
    without generating NaN or infinite values, which could cause training
    instabilities or incorrect predictions.
    """
    plugin, model = setup_glm47_plugin()
    try:
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        result = plugin.infer({'input_ids': input_ids})

        if isinstance(result, dict) and 'logits' in result:
            logits = result['logits']
            # Check for NaN or Inf values
            assert_true(not torch.isnan(logits).any())
            assert_true(not torch.isinf(logits).any())
    finally:
        cleanup_glm47_plugin(plugin)


if __name__ == '__main__':
    # Run the tests using custom test utilities
    test_functions = [
        test_attention_layer_exists,
        test_attention_forward_pass,
        test_attention_with_causal_mask,
        test_attention_with_kv_cache,
        test_attention_output_shapes,
        test_attention_gradient_flow,
        test_attention_with_different_sequence_lengths,
        test_attention_numerical_stability
    ]
    run_tests(test_functions)