"""
Tests for Spiking Neural Network (SNN) Components

This module contains comprehensive tests for the SNN components implemented in the common module.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import numpy as np
from src.inference_pio.common.snn.snn_neurons import LIFNeuron, IzhikevichNeuron, AdaptiveLIFNeuron
from src.inference_pio.common.snn.snn_layers import SNNDenseLayer, SNNConvLayer, SNNTransformerBlock
from src.inference_pio.common.snn.snn_activations import SpikingActivation, RateEncoding, TemporalEncoding
from src.inference_pio.common.snn.snn_utils import convert_dense_to_snn, apply_snn_optimizations

# TestSNNNeurons

    """Test cases for SNN neuron implementations."""

    def setup_helper():
        """Set up test fixtures."""
        batch_size = 4
        feature_size = 10
        input_tensor = torch.randn(batch_size, feature_size)

    def lif_neuron_basic(self)():
        """Test basic functionality of LIF neuron."""
        neuron = LIFNeuron(threshold=1.0, decay=0.9)
        
        # Test forward pass
        spikes, membrane_potential = neuron(input_tensor)
        
        # Check output shapes
        assert_equal(spikes.shape, input_tensor.shape)
        assert_equal(membrane_potential.shape, input_tensor.shape)
        
        # Check that spikes are binary
        unique_values = torch.unique(spikes)
        assert_true(torch.allclose(unique_values)) or 
                        torch.allclose(unique_values, torch.tensor([0.0])) or 
                        torch.allclose(unique_values, torch.tensor([1.0])))

    def lif_neuron_reset_mechanism(self)():
        """Test different reset mechanisms of LIF neuron."""
        input_high = torch.ones(batch_size, feature_size) * 10  # High input to trigger spikes
        
        # Test zero reset mechanism
        neuron_zero = LIFNeuron(threshold=1.0, decay=0.9, reset_mechanism='zero')
        spikes_zero, _ = neuron_zero(input_high)
        
        # Test subtract reset mechanism
        neuron_subtract = LIFNeuron(threshold=1.0, decay=0.9, reset_mechanism='subtract')
        spikes_subtract, _ = neuron_subtract(input_high)
        
        # Both should produce spikes
        assert_greater(spikes_zero.sum(), 0)
        assert_greater(spikes_subtract.sum(), 0)

    def izhikevich_neuron(self)():
        """Test Izhikevich neuron implementation."""
        neuron = IzhikevichNeuron()
        
        # Test forward pass
        spikes, v, u = neuron(input_tensor)
        
        # Check output shapes
        assert_equal(spikes.shape, input_tensor.shape)
        assert_equal(v.shape, input_tensor.shape)
        assert_equal(u.shape, input_tensor.shape)

    def adaptive_lif_neuron(self)():
        """Test Adaptive LIF neuron implementation."""
        neuron = AdaptiveLIFNeuron()
        
        # Test forward pass
        spikes, membrane_potential, adaptation = neuron(input_tensor)
        
        # Check output shapes
        assert_equal(spikes.shape, input_tensor.shape)
        assert_equal(membrane_potential.shape, input_tensor.shape)
        assert_equal(adaptation.shape, input_tensor.shape)

# TestSNNLayers

    """Test cases for SNN layer implementations."""

    def setup_helper():
        """Set up test fixtures."""
        batch_size = 4
        in_features = 10
        out_features = 20
        input_tensor = torch.randn(batch_size, in_features)

    def snn_dense_layer(self)():
        """Test SNN Dense layer."""
        layer = SNNDenseLayer(
            in_features=in_features,
            out_features=out_features,
            neuron_type='LIF'
        )
        
        output = layer(input_tensor)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))
        
        # Check that output is binary (spikes)
        unique_values = torch.unique(output)
        assert_true(torch.allclose(unique_values), atol=1e-6) or 
                        len(unique_values) <= 2)  # Allow for small numerical errors

    def snn_conv_layer(self)():
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
        assert_equal(output.shape, (batch_size))

    def snn_transformer_block(self)():
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
        assert_equal(output.shape, (batch_size))

# TestSNNActivations

    """Test cases for SNN activation implementations."""

    def setup_helper():
        """Set up test fixtures."""
        batch_size = 4
        feature_size = 10
        input_tensor = torch.randn(batch_size, feature_size)

    def spiking_activation(self)():
        """Test spiking activation function."""
        activation = SpikingActivation(threshold=0.5)
        
        output = activation(input_tensor)
        
        # During training, output should be continuous (differentiable approximation)
        assert_equal(output.shape, input_tensor.shape)
        
        # Values should be between 0 and 1 (sigmoid-like)
        assert_true(torch.all(output >= 0) and torch.all(output <= 1))

    def rate_encoding(self)():
        """Test rate encoding."""
        encoder = RateEncoding(max_freq=100.0)
        
        positive_input = torch.abs(input_tensor)  # Use positive values
        spikes, rates = encoder(positive_input)
        
        # Check shapes
        assert_equal(spikes.shape, positive_input.shape)
        assert_equal(rates.shape, positive_input.shape)
        
        # Check that spikes are binary
        unique_values = torch.unique(spikes)
        assert_true(torch.allclose(unique_values), atol=1e-6) or 
                        len(unique_values) <= 2)

    def temporal_encoding(self)():
        """Test temporal encoding."""
        encoder = TemporalEncoding(time_steps=5)
        
        spikes, membrane_potentials = encoder(input_tensor)
        
        # Check shapes
        expected_spikes_shape = (encoder.time_steps, batch_size, feature_size)
        expected_potentials_shape = (encoder.time_steps, batch_size, feature_size)
        
        assert_equal(spikes.shape, expected_spikes_shape)
        assert_equal(membrane_potentials.shape, expected_potentials_shape)

# TestSNNUtils

    """Test cases for SNN utility functions."""

    def setup_helper():
        """Set up test fixtures."""
        simple_model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )

    def convert_dense_to_snn(self)():
        """Test conversion of dense model to SNN."""
        snn_config = {
            'neuron_type': 'LIF',
            'threshold': 1.0,
            'decay': 0.9,
            'dropout_rate': 0.0,
            'temporal_encoding': False
        }
        
        converted_model = convert_dense_to_snn(simple_model, snn_config)
        
        # Check that linear layers were converted to SNN layers
        for name, module in converted_model.named_modules():
            if 'linear' in name.lower():  # This won't work as expected; need to check by class
                continue
        
        # Actually check by iterating through the sequential model
        # The converted model should have SNN layers instead of Linear layers
        has_snn_layers = any(isinstance(m, (SNNDenseLayer, SNNConvLayer)) for m in converted_model.modules())
        assert_true(has_snn_layers)

    def apply_snn_optimizations(self)():
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
        assert_equal(output.shape, (4))

# TestIntegration

    """Integration tests for SNN components."""

    def full_snn_pipeline(self)():
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
        assert_equal(output.shape, (4))
        
        # Check that outputs are binary (spikes)
        unique_values = torch.unique(output)
        # Allow for small numerical errors in the computation
        is_binary = all(abs(val - round(val.item())) < 1e-5 for val in unique_values)
        assert_true(is_binary)

if __name__ == '__main__':
    run_tests(test_functions)