"""
Spiking Neural Network Activations Implementation

This module implements various spiking activation functions and encoding methods
for energy-efficient neural network computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpikingActivation(nn.Module):
    """
    General Spiking Activation Function
    
    Provides differentiable approximations to spiking behavior for training.
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 surrogate_width: float = 0.5,
                 activation_type: str = 'sigmoid'):
        """
        Initialize Spiking Activation
        
        Args:
            threshold: Threshold for spiking
            surrogate_width: Width of surrogate gradient
            activation_type: Type of surrogate function ('sigmoid', 'gaussian', 'triangle')
        """
        super(SpikingActivation, self).__init__()
        
        self.threshold = threshold
        self.surrogate_width = surrogate_width
        self.activation_type = activation_type
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spiking activation
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with spiking activation
        """
        if self.training:
            # During training, use differentiable approximation
            if self.activation_type == 'sigmoid':
                return torch.sigmoid(self.surrogate_width * (x - self.threshold))
            elif self.activation_type == 'gaussian':
                return torch.exp(-self.surrogate_width * (x - self.threshold)**2)
            elif self.activation_type == 'triangle':
                return torch.clamp(1 - self.surrogate_width * torch.abs(x - self.threshold), 0, 1)
            else:
                # Default to sigmoid
                return torch.sigmoid(self.surrogate_width * (x - self.threshold))
        else:
            # During inference, use hard threshold
            return (x >= self.threshold).float()


class RateEncoding(nn.Module):
    """
    Rate Encoding for Spiking Neural Networks
    
    Converts analog values to spike rates for energy-efficient computation.
    """
    
    def __init__(self, 
                 max_freq: float = 100.0,  # Hz
                 time_window: float = 0.01,  # seconds
                 threshold: float = 1.0):
        """
        Initialize Rate Encoding
        
        Args:
            max_freq: Maximum firing frequency (Hz)
            time_window: Time window for rate calculation (seconds)
            threshold: Threshold for spiking
        """
        super(RateEncoding, self).__init__()
        
        self.max_freq = max_freq
        self.time_window = time_window
        self.threshold = threshold
        self.spike_count = 0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input using rate encoding
        
        Args:
            x: Input tensor (analog values)
            
        Returns:
            spikes: Binary spike tensor
            rates: Firing rates
        """
        # Normalize input to [0, max_freq]
        normalized_x = torch.clamp(x, 0, self.threshold) / self.threshold
        firing_rates = normalized_x * self.max_freq
        
        # Generate spikes based on firing rates
        # Probability of spike in each time step
        prob_spike = firing_rates * self.time_window
        spikes = (torch.rand_like(x) < prob_spike).float()
        
        # Update spike count for statistics
        self.spike_count += spikes.sum().item()
        
        return spikes, firing_rates


class TemporalEncoding(nn.Module):
    """
    Temporal Encoding for Spiking Neural Networks
    
    Encodes information in spike timing for energy-efficient computation.
    """
    
    def __init__(self, 
                 time_steps: int = 10,
                 threshold: float = 1.0,
                 decay: float = 0.9):
        """
        Initialize Temporal Encoding
        
        Args:
            time_steps: Number of time steps to simulate
            threshold: Threshold for spiking
            decay: Decay rate for membrane potential
        """
        super(TemporalEncoding, self).__init__()
        
        self.time_steps = time_steps
        self.threshold = threshold
        self.decay = decay
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input using temporal encoding
        
        Args:
            x: Input tensor (analog values)
            
        Returns:
            spike_trains: Tensor of spike trains over time
            membrane_potentials: Tensor of membrane potentials over time
        """
        batch_size = x.size(0)
        if len(x.shape) > 1:
            output_shape = x.shape[1:]
        else:
            output_shape = []
            
        # Initialize membrane potential
        membrane_potential = torch.zeros((batch_size,) + output_shape, 
                                        dtype=x.dtype, device=x.device)
        
        # Initialize spike train
        spike_trains = torch.zeros((self.time_steps, batch_size) + output_shape, 
                                  dtype=x.dtype, device=x.device)
        membrane_potentials = torch.zeros((self.time_steps, batch_size) + output_shape, 
                                         dtype=x.dtype, device=x.device)
        
        # Simulate over time steps
        for t in range(self.time_steps):
            # Update membrane potential
            membrane_potential = self.decay * membrane_potential + x
            
            # Generate spikes
            spikes = (membrane_potential >= self.threshold).float()
            
            # Store results
            spike_trains[t] = spikes
            membrane_potentials[t] = membrane_potential.clone()
            
            # Reset membrane potential after spikes
            membrane_potential = membrane_potential - spikes * self.threshold
        
        return spike_trains, membrane_potentials


class PopulationCoding(nn.Module):
    """
    Population Coding for Spiking Neural Networks
    
    Encodes information in populations of spiking neurons for robust representation.
    """
    
    def __init__(self, 
                 n_neurons: int = 100,
                 sigma: float = 0.5,
                 threshold: float = 1.0):
        """
        Initialize Population Coding
        
        Args:
            n_neurons: Number of neurons in population
            sigma: Standard deviation of tuning curves
            threshold: Threshold for spiking
        """
        super(PopulationCoding, self).__init__()
        
        self.n_neurons = n_neurons
        self.sigma = sigma
        self.threshold = threshold
        
        # Create preferred stimuli for each neuron (evenly spaced)
        self.register_buffer('preferred_stimuli', 
                            torch.linspace(-1, 1, n_neurons))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using population coding
        
        Args:
            x: Input tensor (scalar or vector values)
            
        Returns:
            spikes: Spike tensor representing population activity
        """
        # Expand x to match population dimensions
        if x.dim() == 0 or (x.dim() == 1 and x.size(0) == 1):
            # Scalar input
            x_expanded = x.expand(self.n_neurons)
        else:
            # Vector input - expand to include population dimension
            x_expanded = x.unsqueeze(-1).expand(*x.shape, self.n_neurons)
        
        # Calculate firing rates based on distance to preferred stimuli
        distances = (x_expanded - self.preferred_stimuli) / self.sigma
        firing_rates = torch.exp(-0.5 * distances**2)
        
        # Generate spikes based on firing rates
        spikes = (torch.rand_like(firing_rates) < firing_rates).float()
        
        return spikes


class PhaseEncoding(nn.Module):
    """
    Phase Encoding for Spiking Neural Networks
    
    Encodes information in phase relationships between oscillatory activities.
    """
    
    def __init__(self, 
                 n_phases: int = 8,
                 threshold: float = 1.0):
        """
        Initialize Phase Encoding
        
        Args:
            n_phases: Number of phase bins
            threshold: Threshold for spiking
        """
        super(PhaseEncoding, self).__init__()
        
        self.n_phases = n_phases
        self.threshold = threshold
        
        # Create phase bins
        self.phase_bins = torch.linspace(0, 2 * torch.pi, n_phases + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using phase encoding
        
        Args:
            x: Input tensor (analog values normalized to [0, 2*pi])
            
        Returns:
            spikes: Spike tensor representing phase-coded activity
        """
        # Normalize input to [0, 2*pi] range
        x_normalized = ((x + 1) % (2 * torch.pi)) if x.min() < 0 else (x % (2 * torch.pi))
        
        # Create phase-coded representation
        spikes = torch.zeros((*x.shape, self.n_phases), 
                            dtype=x.dtype, device=x.device)
        
        # Assign each input to its corresponding phase bin
        for i in range(self.n_phases):
            mask = (x_normalized >= self.phase_bins[i]) & (x_normalized < self.phase_bins[i+1])
            spikes[mask, i] = 1.0
            
        return spikes


__all__ = [
    "SpikingActivation",
    "RateEncoding",
    "TemporalEncoding",
    "PopulationCoding",
    "PhaseEncoding"
]