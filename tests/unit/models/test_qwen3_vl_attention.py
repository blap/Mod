"""
Standardized Test for Attention Mechanisms - Qwen3-VL-2B

This module tests the attention mechanisms for the Qwen3-VL-2B model.
"""
import torch
from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_is_instance, assert_raises,
    run_tests
)
from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
from src.inference_pio.common.base_attention import BaseAttention


def setup_qwen3_vl_2b_plugin():
    """Set up Qwen3-VL-2B plugin for testing."""
    plugin = create_qwen3_vl_2b_instruct_plugin()
    success = plugin.initialize(device="cpu")  # Use CPU for tests
    assert_true(success)
    model = plugin.load_model()
    assert_true(model is not None)
    return plugin, model


def cleanup_qwen3_vl_2b_plugin(plugin):
    """Clean up Qwen3-VL-2B plugin after testing."""
    if hasattr(plugin, 'cleanup') and plugin.is_loaded:
        plugin.cleanup()


def test_attention_layer_exists():
    """Test that attention layers exist in the model."""
    plugin, model = setup_qwen3_vl_2b_plugin()
    try:
        # Check if the model has attention layers
        attention_layers = []
        for name, module in model.named_modules():
            if isinstance(module, BaseAttention) or 'attention' in name.lower() or 'attn' in name.lower():
                attention_layers.append((name, module))

        # At least some attention layers should exist
        assert_greater(len(attention_layers), 0)
    finally:
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_forward_pass():
    """Test the forward pass of attention mechanisms."""
    plugin, model = setup_qwen3_vl_2b_plugin()
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
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_with_causal_mask():
    """Test attention with causal masking."""
    plugin, model = setup_qwen3_vl_2b_plugin()
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
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_with_kv_cache():
    """Test attention with KV cache functionality."""
    plugin, model = setup_qwen3_vl_2b_plugin()
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
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_output_shapes():
    """Test that attention outputs have correct shapes."""
    plugin, model = setup_qwen3_vl_2b_plugin()
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
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_gradient_flow():
    """Test that gradients flow properly through attention layers."""
    plugin, model = setup_qwen3_vl_2b_plugin()
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
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_with_different_sequence_lengths():
    """Test attention with various sequence lengths."""
    plugin, model = setup_qwen3_vl_2b_plugin()
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
        cleanup_qwen3_vl_2b_plugin(plugin)


def test_attention_numerical_stability():
    """Test numerical stability of attention computations."""
    plugin, model = setup_qwen3_vl_2b_plugin()
    try:
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        result = plugin.infer({'input_ids': input_ids})

        if isinstance(result, dict) and 'logits' in result:
            logits = result['logits']
            # Check for NaN or Inf values
            assert_false(torch.isnan(logits).any())
            assert_false(torch.isinf(logits).any())
    finally:
        cleanup_qwen3_vl_2b_plugin(plugin)


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