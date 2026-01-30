"""
Tests for Spiking Neural Network (SNN) Components

This module contains comprehensive tests for SNN components in the common module.
"""
import torch
from tests.utils.test_utils import (
    assert_equal, assert_true, assert_greater, run_tests
)

from src.inference_pio.common.snn.snn_neurons import (
    LIFNeuron, IzhikevichNeuron, AdaptiveLIFNeuron
)
from src.inference_pio.common.snn.snn_layers import (
    SNNDenseLayer, SNNConvLayer, SNNTransformerBlock
)
from src.inference_pio.common.snn.snn_activations import (
    SpikingActivation, RateEncoding, TemporalEncoding
)
from src.inference_pio.common.snn.snn_utils import (
    convert_dense_to_snn, apply_snn_optimizations
)


def test_lif_neuron_basic():
    """Test basic functionality of LIF neuron."""
    batch_size = 4
    feature_size = 10
    input_tensor = torch.randn(batch_size, feature_size)
    neuron = LIFNeuron(threshold=1.0, decay=0.9)

    # Test forward pass
    spikes, membrane_potential = neuron(input_tensor)

    # Check output shapes
    assert_equal(spikes.shape, input_tensor.shape)
    assert_equal(membrane_potential.shape, input_tensor.shape)

    # Check that spikes are binary (0.0 or 1.0)
    unique_values = torch.unique(spikes)
    for val in unique_values:
        assert_true(
            torch.isclose(val, torch.tensor(0.0)) or
            torch.isclose(val, torch.tensor(1.0))
        )


def test_lif_neuron_reset_mechanism():
    """Test different reset mechanisms of LIF neuron."""
    batch_size = 4
    feature_size = 10
    # High input to trigger spikes
    input_high = torch.ones(batch_size, feature_size) * 10

    # Test zero reset mechanism
    neuron_zero = LIFNeuron(threshold=1.0, decay=0.9, reset_mechanism='zero')
    spikes_zero, _ = neuron_zero(input_high)

    # Test subtract reset mechanism
    neuron_subtract = LIFNeuron(threshold=1.0, decay=0.9, reset_mechanism='subtract')
    spikes_subtract, _ = neuron_subtract(input_high)

    # Both should produce spikes
    assert_greater(spikes_zero.sum().item(), 0)
    assert_greater(spikes_subtract.sum().item(), 0)


def test_izhikevich_neuron():
    """Test Izhikevich neuron implementation."""
    batch_size = 4
    feature_size = 10
    input_tensor = torch.randn(batch_size, feature_size)
    neuron = IzhikevichNeuron()

    # Test forward pass
    spikes, v, u = neuron(input_tensor)

    # Check output shapes
    assert_equal(spikes.shape, input_tensor.shape)
    assert_equal(v.shape, input_tensor.shape)
    assert_equal(u.shape, input_tensor.shape)


def test_adaptive_lif_neuron():
    """Test Adaptive LIF neuron implementation."""
    batch_size = 4
    feature_size = 10
    input_tensor = torch.randn(batch_size, feature_size)
    neuron = AdaptiveLIFNeuron()

    # Test forward pass
    spikes, membrane_potential, adaptation = neuron(input_tensor)

    # Check output shapes
    assert_equal(spikes.shape, input_tensor.shape)
    assert_equal(membrane_potential.shape, input_tensor.shape)
    assert_equal(adaptation.shape, input_tensor.shape)


def test_snn_dense_layer():
    """Test SNN Dense layer."""
    batch_size = 4
    in_features = 10
    out_features = 20
    input_tensor = torch.randn(batch_size, in_features)

    layer = SNNDenseLayer(
        in_features=in_features,
        out_features=out_features,
        neuron_type='LIF'
    )

    output = layer(input_tensor)

    # Check output shape
    assert_equal(output.shape, (batch_size, out_features))


def test_snn_conv_layer():
    """Test SNN Conv layer."""
    batch_size = 4
    in_channels = 3
    out_channels = 6
    height, width = 32, 32

    layer = SNNConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        neuron_type='LIF'
    )

    input_tensor = torch.randn(batch_size, in_channels, height, width)
    output = layer(input_tensor)

    # Check output shape
    assert_equal(output.shape, (batch_size, out_channels, height, width))


def test_snn_transformer_block():
    """Test SNN Transformer block."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 8

    block = SNNTransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=embed_dim * 4
    )

    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    output = block(input_tensor)

    # Check output shape
    assert_equal(output.shape, (batch_size, seq_len, embed_dim))


def test_spiking_activation():
    """Test spiking activation function."""
    batch_size = 4
    feature_size = 10
    input_tensor = torch.randn(batch_size, feature_size)
    activation = SpikingActivation(threshold=0.5)

    output = activation(input_tensor)

    # During training, output should be continuous (differentiable approximation)
    assert_equal(output.shape, input_tensor.shape)

    # Values should be between 0 and 1 (sigmoid-like)
    assert_true(torch.all(output >= 0).item() and torch.all(output <= 1).item())


def test_rate_encoding():
    """Test rate encoding."""
    batch_size = 4
    feature_size = 10
    input_tensor = torch.randn(batch_size, feature_size)
    encoder = RateEncoding(max_freq=100.0)

    positive_input = torch.abs(input_tensor)  # Use positive values
    spikes, rates = encoder(positive_input)

    # Check shapes
    assert_equal(spikes.shape, positive_input.shape)
    assert_equal(rates.shape, positive_input.shape)


def test_temporal_encoding():
    """Test temporal encoding."""
    batch_size = 4
    feature_size = 10
    input_tensor = torch.randn(batch_size, feature_size)
    encoder = TemporalEncoding(time_steps=5)

    spikes, membrane_potentials = encoder(input_tensor)

    # Check shapes
    expected_spikes_shape = (encoder.time_steps, batch_size, feature_size)
    expected_potentials_shape = (encoder.time_steps, batch_size, feature_size)

    assert_equal(spikes.shape, expected_spikes_shape)
    assert_equal(membrane_potentials.shape, expected_potentials_shape)


def test_convert_dense_to_snn():
    """Test conversion of dense model to SNN."""
    simple_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )

    snn_config = {
        'neuron_type': 'LIF',
        'threshold': 1.0,
        'decay': 0.9,
        'dropout_rate': 0.0,
        'temporal_encoding': False
    }

    converted_model = convert_dense_to_snn(simple_model, snn_config)

    # The converted model should have SNN layers instead of Linear layers
    has_snn_layers = any(
        isinstance(m, (SNNDenseLayer, SNNConvLayer))
        for m in converted_model.modules()
    )
    assert_true(has_snn_layers)


def test_apply_snn_optimizations():
    """Test SNN optimization application."""
    # Create a simple model with SNN layers
    model = torch.nn.Sequential(
        SNNDenseLayer(10, 20),
        SNNDenseLayer(20, 5)
    )

    optimization_config = {
        'pruning_ratio': 0.2,
        'quantization_bits': 8,
        'temporal_sparsity': True,
        'neural_efficiency': True
    }

    optimized_model = apply_snn_optimizations(model, optimization_config)

    # The model should still be valid after optimization
    input_tensor = torch.randn(4, 10)
    output = optimized_model(input_tensor)
    assert_equal(output.shape, (4, 5))


def test_full_snn_pipeline():
    """Test a full SNN pipeline with multiple components."""
    # Create a simple SNN model
    model = torch.nn.Sequential(
        SNNDenseLayer(10, 32, neuron_type='LIF'),
        SNNDenseLayer(32, 16, neuron_type='AdaptiveLIF'),
        SNNDenseLayer(16, 5, neuron_type='LIF')
    )

    # Test forward pass
    input_tensor = torch.randn(4, 10)
    output = model(input_tensor)

    # Check output shape
    assert_equal(output.shape, (4, 5))


if __name__ == '__main__':
    test_functions = [
        test_lif_neuron_basic,
        test_lif_neuron_reset_mechanism,
        test_izhikevich_neuron,
        test_adaptive_lif_neuron,
        test_snn_dense_layer,
        test_snn_conv_layer,
        test_snn_transformer_block,
        test_spiking_activation,
        test_rate_encoding,
        test_temporal_encoding,
        test_convert_dense_to_snn,
        test_apply_snn_optimizations,
        test_full_snn_pipeline
    ]
    run_tests(test_functions)
