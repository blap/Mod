"""
Qwen3-4B-Instruct-2507 Bias Removal Optimization

This module provides bias removal optimization for the Qwen3-4B-Instruct-2507 model.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class BiasRemovalConfig:
    """
    Configuration for bias removal optimization.
    """
    remove_bias_after_norm: bool = True
    remove_bias_in_attention: bool = True
    remove_bias_in_mlp: bool = True
    remove_bias_in_embeddings: bool = False
    # Additional configuration parameters can be added here


def apply_bias_removal_to_model(
    model: nn.Module, 
    config: Optional[BiasRemovalConfig] = None, 
    model_type: str = "qwen3_4b"
) -> Tuple[nn.Module, Dict]:
    """
    Apply bias removal optimization to the model.

    Args:
        model: The model to optimize
        config: Bias removal configuration
        model_type: Type of model (for specific optimizations)

    Returns:
        Tuple of (optimized_model, optimization_report)
    """
    if config is None:
        config = BiasRemovalConfig()
    
    removed_components = {
        "attention_biases": 0,
        "mlp_biases": 0,
        "embedding_biases": 0,
        "other_biases": 0
    }
    
    # Remove biases based on configuration
    for name, module in model.named_modules():
        if config.remove_bias_in_attention and isinstance(module, (nn.MultiheadAttention, nn.Linear)):
            # Check if this is part of attention mechanism
            if any(part in name.lower() for part in ['attn', 'attention', 'self_attn', 'q_', 'k_', 'v_', 'proj']):
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = None
                    removed_components["attention_biases"] += 1
        elif config.remove_bias_in_mlp and isinstance(module, nn.Linear):
            # Check if this is part of MLP/feed-forward network
            if any(part in name.lower() for part in ['mlp', 'ffn', 'feed_forward', 'intermediate', 'down_proj', 'up_proj', 'gate_proj']):
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = None
                    removed_components["mlp_biases"] += 1
        elif config.remove_bias_in_embeddings and isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            # Remove bias from embeddings if configured
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = None
                removed_components["embedding_biases"] += 1
        elif config.remove_bias_after_norm and isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Remove bias after normalization if configured
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = None
                removed_components["other_biases"] += 1
    
    # Create optimization report
    total_removed = sum(removed_components.values())
    report = {
        "optimization_applied": "bias_removal",
        "model_type": model_type,
        "total_biases_removed": total_removed,
        "breakdown": removed_components,
        "config_used": {
            "remove_bias_after_norm": config.remove_bias_after_norm,
            "remove_bias_in_attention": config.remove_bias_in_attention,
            "remove_bias_in_mlp": config.remove_bias_in_mlp,
            "remove_bias_in_embeddings": config.remove_bias_in_embeddings
        }
    }
    
    return model, report


__all__ = [
    "BiasRemovalConfig",
    "apply_bias_removal_to_model"
]