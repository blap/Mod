"""
GLM-4.7 Fused Layer Normalization Implementation

This module implements fused layer normalization for the GLM-4.7 model.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config import GLM47FlashConfig


class FusedLayerNorm(nn.Module):
    """
    Fused Layer Normalization implementation for GLM-4.7 model.

    This implementation combines the mean/variance computation and the affine transformation
    in a single operation for improved performance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer norm."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        Forward pass for fused layer normalization.

        Args:
            input: Input tensor of shape (*, normalized_shape)

        Returns:
            Normalized tensor of the same shape as input
        """
        # Calculate mean and variance in a single pass
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        input = (input - mean) * torch.rsqrt(variance + self.eps)

        # Apply affine transformation if enabled
        if self.elementwise_affine:
            input = input * self.weight + self.bias

        return input


class FusedRMSNorm(nn.Module):
    """
    Fused Root Mean Square Normalization implementation for GLM-4.7 model.

    This implementation computes RMS normalization in a single operation for improved performance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the RMS norm."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, input):
        """
        Forward pass for fused RMS normalization.

        Args:
            input: Input tensor of shape (*, normalized_shape)

        Returns:
            Normalized tensor of the same shape as input
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(input.pow(2), dim=-1, keepdim=True) + self.eps)

        # Normalize
        input = input / rms

        # Apply affine transformation if enabled
        if self.elementwise_affine:
            input = input * self.weight

        return input


def replace_layer_norm_in_model(model: nn.Module, config: GLM47FlashConfig) -> nn.Module:
    """
    Replace standard LayerNorm modules with fused implementations in the model.

    Args:
        model: The model to modify
        config: Model configuration

    Returns:
        Modified model with fused layer norms
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Get the parameters from the original LayerNorm
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine

            # Create a new fused layer norm with the same parameters
            fused_ln = FusedLayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                device=module.weight.device if elementwise_affine else next(model.parameters()).device,
                dtype=module.weight.dtype if elementwise_affine else next(model.parameters()).dtype
            )

            # Copy the weights if they exist
            if elementwise_affine:
                with torch.no_grad():
                    fused_ln.weight.copy_(module.weight)
                    fused_ln.bias.copy_(module.bias)

            # Replace the module in the parent
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model
            for n in parent_name.split('.'):
                parent_module = getattr(parent_module, n)

            setattr(parent_module, child_name, fused_ln)

    return model


def replace_rms_norm_in_model(model: nn.Module, config: GLM47FlashConfig) -> nn.Module:
    """
    Replace standard LayerNorm modules with fused RMS normalization in the model.

    Args:
        model: The model to modify
        config: Model configuration

    Returns:
        Modified model with fused RMS norms
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Get the parameters from the original LayerNorm
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine

            # Create a new fused RMS norm with the same parameters
            fused_rms = FusedRMSNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                device=module.weight.device if elementwise_affine else next(model.parameters()).device,
                dtype=module.weight.dtype if elementwise_affine else next(model.parameters()).dtype
            )

            # Copy the weight if it exists
            if elementwise_affine:
                with torch.no_grad():
                    fused_rms.weight.copy_(module.weight)

            # Replace the module in the parent
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model
            for n in parent_name.split('.'):
                parent_module = getattr(parent_module, n)

            setattr(parent_module, child_name, fused_rms)

    return model


__all__ = [
    "FusedLayerNorm",
    "FusedRMSNorm",
    "replace_layer_norm_in_model",
    "replace_rms_norm_in_model"
]