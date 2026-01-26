"""
Spiking Neural Network Layers Implementation

This module implements various spiking neural network layers that can replace
traditional dense, convolutional, and transformer layers for energy-efficient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .snn_neurons import LIFNeuron, AdaptiveLIFNeuron


class SNNDenseLayer(nn.Module):
    """
    Spiking Dense Layer
    
    Replaces traditional linear layers with spiking equivalents for energy efficiency.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 neuron_type: str = 'LIF',
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 dropout_rate: float = 0.0,
                 temporal_encoding: bool = False):
        """
        Initialize SNN Dense Layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            neuron_type: Type of spiking neuron ('LIF', 'AdaptiveLIF')
            threshold: Spike threshold for neurons
            decay: Decay rate for membrane potential
            dropout_rate: Dropout rate for regularization
            temporal_encoding: Whether to use temporal encoding
        """
        super(SNNDenseLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.temporal_encoding = temporal_encoding
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize spiking neurons
        if neuron_type == 'LIF':
            self.neuron = LIFNeuron(threshold=threshold, decay=decay)
        elif neuron_type == 'AdaptiveLIF':
            self.neuron = AdaptiveLIFNeuron(threshold=threshold, decay=decay)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        
        # For temporal encoding
        if temporal_encoding:
            self.spike_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN Dense Layer
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after spiking transformation
        """
        # Apply linear transformation
        pre_spike = self.linear(x)
        
        # Apply dropout if specified
        if self.dropout is not None:
            pre_spike = self.dropout(pre_spike)
        
        # Apply spiking neuron
        spikes, membrane_potential = self.neuron(pre_spike)
        
        if self.temporal_encoding:
            # Store spike history for temporal encoding
            self.spike_history.append(spikes)
            if len(self.spike_history) > 10:  # Keep last 10 timesteps
                self.spike_history.pop(0)
        
        return spikes


class SNNConvLayer(nn.Module):
    """
    Spiking Convolutional Layer
    
    Replaces traditional conv layers with spiking equivalents for energy efficiency.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 neuron_type: str = 'LIF',
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 dropout_rate: float = 0.0):
        """
        Initialize SNN Convolutional Layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: Whether to include bias term
            padding_mode: Padding mode
            neuron_type: Type of spiking neuron ('LIF', 'AdaptiveLIF')
            threshold: Spike threshold for neurons
            decay: Decay rate for membrane potential
            dropout_rate: Dropout rate for regularization
        """
        super(SNNConvLayer, self).__init__()
        
        # Convolutional transformation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Initialize spiking neurons
        if neuron_type == 'LIF':
            self.neuron = LIFNeuron(threshold=threshold, decay=decay)
        elif neuron_type == 'AdaptiveLIF':
            self.neuron = AdaptiveLIFNeuron(threshold=threshold, decay=decay)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN Convolutional Layer
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor after spiking transformation
        """
        # Apply convolution
        pre_spike = self.conv(x)
        
        # Apply dropout if specified
        if self.dropout is not None:
            pre_spike = self.dropout(pre_spike)
        
        # Apply spiking neuron
        spikes, membrane_potential = self.neuron(pre_spike)
        
        return spikes


class SNNTransformerBlock(nn.Module):
    """
    Spiking Transformer Block
    
    Replaces traditional transformer blocks with spiking equivalents for energy efficiency.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 feedforward_dim: int = None,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation: str = 'relu',
                 neuron_type: str = 'LIF',
                 threshold: float = 1.0,
                 decay: float = 0.9):
        """
        Initialize SNN Transformer Block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            activation: Activation function ('relu', 'gelu')
            neuron_type: Type of spiking neuron ('LIF', 'AdaptiveLIF')
            threshold: Spike threshold for neurons
            decay: Decay rate for membrane potential
        """
        super(SNNTransformerBlock, self).__init__()
        
        if feedforward_dim is None:
            feedforward_dim = 4 * embed_dim
        
        # Multi-head attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        
        # Initialize spiking neurons for key components
        if neuron_type == 'LIF':
            self.ffn_neuron = LIFNeuron(threshold=threshold, decay=decay)
            self.attn_neuron = LIFNeuron(threshold=threshold, decay=decay)
        elif neuron_type == 'AdaptiveLIF':
            self.ffn_neuron = AdaptiveLIFNeuron(threshold=threshold, decay=decay)
            self.attn_neuron = AdaptiveLIFNeuron(threshold=threshold, decay=decay)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
    def forward(self, 
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through SNN Transformer Block
        
        Args:
            src: Source sequence (B, T, E)
            src_mask: Attention mask (optional)
            src_key_padding_mask: Key padding mask (optional)
            
        Returns:
            Output tensor after spiking transformer block
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        
        # Apply spiking neuron to attention output
        attn_spikes, _ = self.attn_neuron(attn_output)
        
        # Residual connection and normalization
        src2 = self.norm1(src + self.dropout1(attn_spikes))
        
        # Feedforward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        
        # Apply spiking neuron to FFN output
        ffn_spikes, _ = self.ffn_neuron(ff_output)
        
        # Residual connection and normalization
        output = self.norm2(src2 + self.dropout2(ffn_spikes))
        
        return output


class SNNResidualBlock(nn.Module):
    """
    Spiking Residual Block
    
    Implements a residual connection with spiking neurons for deeper networks.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 downsample: bool = False,
                 neuron_type: str = 'LIF',
                 threshold: float = 1.0,
                 decay: float = 0.9):
        """
        Initialize SNN Residual Block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to input
            downsample: Whether to downsample
            neuron_type: Type of spiking neuron ('LIF', 'AdaptiveLIF')
            threshold: Spike threshold for neurons
            decay: Decay rate for membrane potential
        """
        super(SNNResidualBlock, self).__init__()
        
        # Main convolutional path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Spiking neurons
        if neuron_type == 'LIF':
            self.neuron1 = LIFNeuron(threshold=threshold, decay=decay)
            self.neuron2 = LIFNeuron(threshold=threshold, decay=decay)
        elif neuron_type == 'AdaptiveLIF':
            self.neuron1 = AdaptiveLIFNeuron(threshold=threshold, decay=decay)
            self.neuron2 = AdaptiveLIFNeuron(threshold=threshold, decay=decay)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SNN Residual Block
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor after spiking residual block
        """
        identity = self.shortcut(x)
        
        # First conv + BN + spiking
        out = self.conv1(x)
        out = self.bn1(out)
        out_spikes1, _ = self.neuron1(out)
        
        # Second conv + BN + spiking
        out = self.conv2(out_spikes1)
        out = self.bn2(out)
        
        # Add shortcut and apply final spiking
        out += identity
        out_spikes2, _ = self.neuron2(out)
        
        return out_spikes2


__all__ = [
    "SNNDenseLayer",
    "SNNConvLayer",
    "SNNTransformerBlock",
    "SNNResidualBlock"
]