"""
Qwen3-4B-Instruct-2507 Fused Layer Normalization Implementation

This module provides fused layer normalization for the Qwen3-4B-Instruct-2507 model.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config import Qwen34BInstruct2507Config


def replace_layer_norm_in_model(model: nn.Module, config: Qwen34BInstruct2507Config) -> nn.Module:
    """
    Replace standard LayerNorm modules with fused implementations in the model.

    Args:
        model: The model to modify
        config: Model configuration

    Returns:
        Modified model with fused layer norms
    """
    # Only apply if fused layer norm is enabled in config
    if not config.use_fused_layer_norm:
        return model
    
    # Replace LayerNorm modules with fused versions
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Try to use Apex fused layer norm if available
            try:
                from apex.normalization import FusedLayerNorm
                fused_norm = FusedLayerNorm(
                    normalized_shape=module.normalized_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine
                )
                # Copy parameters
                fused_norm.weight.data = module.weight.data
                fused_norm.bias.data = module.bias.data if module.bias is not None else None
                # Replace the module
                _set_module_by_name(model, name, fused_norm)
            except ImportError:
                # If apex is not available, use standard LayerNorm but ensure it's optimized
                # For now, we'll just keep the original module
                pass
    
    return model


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """
    Set a module by its name path.
    
    Args:
        model: The parent model
        name: Dot-separated name of the module to replace
        new_module: The new module to set
    """
    names = name.split('.')
    module = model
    for n in names[:-1]:
        module = getattr(module, n)
    setattr(module, names[-1], new_module)


__all__ = [
    "replace_layer_norm_in_model",
    "_set_module_by_name"
]