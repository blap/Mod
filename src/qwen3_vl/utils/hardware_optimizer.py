"""
Hardware optimization utilities for Qwen3-VL on Intel i5-10210U + NVIDIA SM61 + NVMe SSD
"""
import torch
import torch.nn as nn
from typing import Optional


def optimize_for_hardware(model: nn.Module, 
                         use_gradient_checkpointing: bool = True,
                         use_fp16: bool = False,
                         max_batch_size: Optional[int] = None) -> nn.Module:
    """
    Optimize the model for the target hardware (Intel i5-10210U + NVIDIA SM61 + NVMe SSD).
    
    Args:
        model: The model to optimize
        use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
        use_fp16: Whether to use FP16 precision for speed/memory efficiency
        max_batch_size: Maximum batch size to optimize for
    
    Returns:
        Optimized model
    """
    # Set model to use gradient checkpointing if specified
    if use_gradient_checkpointing:
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'config'):
            model.config.use_gradient_checkpointing = True
    
    # Apply hardware-specific optimizations to hierarchical vision processor if present
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = use_gradient_checkpointing
        if hasattr(module, 'use_fp16'):
            module.use_fp16 = use_fp16
    
    # Memory-efficient initialization
    if use_fp16:
        model = model.half()
    
    return model


def configure_model_for_target_hardware(config):
    """
    Configure model configuration for optimal performance on target hardware.
    
    Args:
        config: Model configuration to modify
    
    Returns:
        Modified configuration
    """
    # Set appropriate parameters for the target hardware
    config.use_gradient_checkpointing = True  # Save memory on constrained hardware
    
    # Use appropriate attention mechanisms for the GPU
    if not hasattr(config, 'attention_implementation'):
        config.attention_implementation = "memory_efficient"
    
    # Optimize vision-specific parameters
    if hasattr(config, 'vision_num_attention_heads'):
        # Cap the number of vision attention heads for memory efficiency
        config.vision_num_attention_heads = min(config.vision_num_attention_heads, 16)
    
    # Set appropriate layer normalization epsilon for numerical stability
    if not hasattr(config, 'layer_norm_eps') or config.layer_norm_eps > 1e-5:
        config.layer_norm_eps = 1e-5
    
    return config


def get_optimal_batch_size_for_hardware() -> int:
    """
    Get the optimal batch size for the target hardware configuration.
    
    Returns:
        Optimal batch size
    """
    # For Intel i5-10210U + NVIDIA SM61, return a conservative batch size
    return 1  # Conservative default for constrained hardware


def enable_memory_efficient_attention(model: nn.Module):
    """
    Enable memory-efficient attention mechanisms in the model.
    """
    # This function would enable specific attention mechanisms optimized for the hardware
    # In practice, this would set model-specific parameters
    for module in model.modules():
        if hasattr(module, 'use_simplified_attention'):
            # Already handled in the model architecture
            pass