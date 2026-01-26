"""
Spiking Neural Network Neurons Implementation

This module implements various spiking neuron models for energy-efficient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model
    
    This implementation provides a differentiable approximation of spiking behavior
    that can be trained with gradient descent.
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 reset_mechanism: str = 'subtract',
                 surrogate_grad: bool = True,
                 surrogate_width: float = 0.5):
        """
        Initialize LIF Neuron
        
        Args:
            threshold: Spike threshold value
            decay: Membrane potential decay rate
            reset_mechanism: How to reset membrane after spike ('zero' or 'subtract')
            surrogate_grad: Whether to use surrogate gradients for training
            surrogate_width: Width parameter for surrogate gradient function
        """
        super(LIFNeuron, self).__init__()
        
        self.threshold = threshold
        self.decay = decay
        self.reset_mechanism = reset_mechanism
        self.surrogate_grad = surrogate_grad
        self.surrogate_width = surrogate_width
        
        # Register buffers for membrane potential
        self.register_buffer('membrane_potential', torch.zeros(1))
        
    def forward(self, input_current):
        """
        Forward pass through LIF neuron
        
        Args:
            input_current: Input current to the neuron
            
        Returns:
            spikes: Spike tensor (binary values)
            membrane_potential: Updated membrane potential
        """
        batch_size = input_current.size(0)
        if len(input_current.shape) > 1:
            neuron_dims = input_current.shape[1:]
        else:
            neuron_dims = []
            
        # Initialize membrane potential if needed
        if self.membrane_potential.shape != (batch_size,) + neuron_dims:
            self.membrane_potential = torch.zeros((batch_size,) + neuron_dims, 
                                                 dtype=input_current.dtype, 
                                                 device=input_current.device)
        
        # Update membrane potential
        self.membrane_potential = (self.decay * self.membrane_potential + 
                                   input_current)
        
        # Generate spikes using differentiable approximation
        if self.surrogate_grad:
            # Surrogate gradient for training
            spikes = self._surrogate_spike_function(self.membrane_potential)
        else:
            # Hard threshold for inference
            spikes = (self.membrane_potential >= self.threshold).float()
        
        # Reset membrane potential after spikes
        if self.reset_mechanism == 'zero':
            self.membrane_potential = self.membrane_potential * (1 - spikes)
        elif self.reset_mechanism == 'subtract':
            self.membrane_potential = self.membrane_potential - (spikes * self.threshold)
        
        return spikes, self.membrane_potential.clone()
    
    def _surrogate_spike_function(self, membrane_potential):
        """
        Surrogate gradient function for differentiable spiking
        """
        # Use sigmoid as surrogate gradient function
        return torch.sigmoid(self.surrogate_width * 
                             (membrane_potential - self.threshold))


class IzhikevichNeuron(nn.Module):
    """
    Izhikevich Neuron Model
    
    A simplified model that captures various spiking behaviors with few parameters.
    """
    
    def __init__(self, 
                 a: float = 0.02,
                 b: float = 0.2,
                 c: float = -65.0,
                 d: float = 2.0,
                 threshold: float = 30.0,
                 surrogate_grad: bool = True,
                 surrogate_width: float = 0.5):
        """
        Initialize Izhikevich Neuron
        
        Args:
            a: Time scale of recovery variable
            b: Sensitivity of recovery variable
            c: After-spike reset value for v
            d: After-spike reset value for u
            threshold: Spike threshold
            surrogate_grad: Whether to use surrogate gradients
            surrogate_width: Width parameter for surrogate gradient
        """
        super(IzhikevichNeuron, self).__init__()
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.threshold = threshold
        self.surrogate_grad = surrogate_grad
        self.surrogate_width = surrogate_width
        
        # Register buffers for state variables
        self.register_buffer('v', torch.zeros(1))  # membrane potential
        self.register_buffer('u', torch.zeros(1))  # recovery variable
        
    def forward(self, input_current):
        """
        Forward pass through Izhikevich neuron
        
        Args:
            input_current: Input current to the neuron
            
        Returns:
            spikes: Spike tensor (binary values)
            v: Updated membrane potential
            u: Updated recovery variable
        """
        batch_size = input_current.size(0)
        if len(input_current.shape) > 1:
            neuron_dims = input_current.shape[1:]
        else:
            neuron_dims = []
            
        # Initialize state variables if needed
        if self.v.shape != (batch_size,) + neuron_dims:
            self.v = torch.full((batch_size,) + neuron_dims, self.c, 
                                dtype=input_current.dtype, 
                                device=input_current.device)
            self.u = torch.zeros((batch_size,) + neuron_dims, 
                                 dtype=input_current.dtype, 
                                 device=input_current.device)
        
        # Update state equations
        dv_dt = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + input_current
        du_dt = self.a * (self.b * self.v - self.u)
        
        self.v = self.v + dv_dt
        self.u = self.u + du_dt
        
        # Generate spikes using differentiable approximation
        if self.surrogate_grad:
            spikes = torch.sigmoid(self.surrogate_width * 
                                   (self.v - self.threshold))
        else:
            spikes = (self.v >= self.threshold).float()
        
        # Reset after spike
        self.v = torch.where(spikes.bool(), 
                             torch.full_like(self.v, self.c), 
                             self.v)
        self.u = torch.where(spikes.bool(), 
                             self.u + self.d, 
                             self.u)
        
        return spikes, self.v.clone(), self.u.clone()


class AdaptiveLIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire Neuron
    
    Includes adaptation currents to model realistic neural dynamics.
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 adaptation_decay: float = 0.9,
                 adaptation_strength: float = 0.1,
                 reset_mechanism: str = 'subtract',
                 surrogate_grad: bool = True,
                 surrogate_width: float = 0.5):
        """
        Initialize Adaptive LIF Neuron
        
        Args:
            threshold: Spike threshold value
            decay: Membrane potential decay rate
            adaptation_decay: Adaptation variable decay rate
            adaptation_strength: Strength of adaptation effect
            reset_mechanism: How to reset membrane after spike
            surrogate_grad: Whether to use surrogate gradients
            surrogate_width: Width parameter for surrogate gradient
        """
        super(AdaptiveLIFNeuron, self).__init__()
        
        self.threshold = threshold
        self.decay = decay
        self.adaptation_decay = adaptation_decay
        self.adaptation_strength = adaptation_strength
        self.reset_mechanism = reset_mechanism
        self.surrogate_grad = surrogate_grad
        self.surrogate_width = surrogate_width
        
        # Register buffers for state variables
        self.register_buffer('membrane_potential', torch.zeros(1))
        self.register_buffer('adaptation', torch.zeros(1))
        
    def forward(self, input_current):
        """
        Forward pass through adaptive LIF neuron
        
        Args:
            input_current: Input current to the neuron
            
        Returns:
            spikes: Spike tensor (binary values)
            membrane_potential: Updated membrane potential
            adaptation: Updated adaptation variable
        """
        batch_size = input_current.size(0)
        if len(input_current.shape) > 1:
            neuron_dims = input_current.shape[1:]
        else:
            neuron_dims = []
            
        # Initialize state variables if needed
        if self.membrane_potential.shape != (batch_size,) + neuron_dims:
            self.membrane_potential = torch.zeros((batch_size,) + neuron_dims, 
                                                 dtype=input_current.dtype, 
                                                 device=input_current.device)
            self.adaptation = torch.zeros((batch_size,) + neuron_dims, 
                                         dtype=input_current.dtype, 
                                         device=input_current.device)
        
        # Update membrane potential with adaptation
        self.membrane_potential = (self.decay * self.membrane_potential + 
                                   input_current - 
                                   self.adaptation_strength * self.adaptation)
        
        # Generate spikes using differentiable approximation
        if self.surrogate_grad:
            spikes = torch.sigmoid(self.surrogate_width * 
                                   (self.membrane_potential - self.threshold))
        else:
            spikes = (self.membrane_potential >= self.threshold).float()
        
        # Update adaptation variable
        self.adaptation = (self.adaptation_decay * self.adaptation + 
                           spikes)
        
        # Reset membrane potential after spikes
        if self.reset_mechanism == 'zero':
            self.membrane_potential = self.membrane_potential * (1 - spikes)
        elif self.reset_mechanism == 'subtract':
            self.membrane_potential = self.membrane_potential - (spikes * self.threshold)
        
        return spikes, self.membrane_potential.clone(), self.adaptation.clone()


__all__ = [
    "LIFNeuron",
    "IzhikevichNeuron",
    "AdaptiveLIFNeuron"
]