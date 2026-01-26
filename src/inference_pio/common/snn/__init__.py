"""
Spiking Neural Network (SNN) Module for Inference-PIO

This module provides efficient SNN implementations for energy-efficient neural network inference.
"""

from .snn_neurons import LIFNeuron, IzhikevichNeuron
from .snn_layers import SNNDenseLayer, SNNConvLayer, SNNTransformerBlock
from .snn_activations import SpikingActivation, RateEncoding, TemporalEncoding
from .snn_utils import convert_dense_to_snn, apply_snn_optimizations

__all__ = [
    "LIFNeuron",
    "IzhikevichNeuron",
    "SNNDenseLayer",
    "SNNConvLayer",
    "SNNTransformerBlock",
    "SpikingActivation",
    "RateEncoding",
    "TemporalEncoding",
    "convert_dense_to_snn",
    "apply_snn_optimizations"
]