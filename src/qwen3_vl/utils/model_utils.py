"""
Utility functions for Qwen3-VL model operations
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


def verify_model_capacity(model: nn.Module, expected_layers: int = 32, expected_heads: int = 32):
    """
    Verify that the model maintains the expected capacity (layers and heads).
    
    Args:
        model: The model to verify
        expected_layers: Expected number of transformer layers
        expected_heads: Expected number of attention heads per layer
    
    Returns:
        Dictionary with verification results
    """
    results = {
        'layers_verified': False,
        'heads_verified': False,
        'total_parameters': 0,
        'details': {}
    }
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    results['total_parameters'] = total_params
    
    # Check for transformer layers and attention heads
    layer_count = 0
    head_count = None
    
    # Look for transformer layers in the model
    for name, module in model.named_modules():
        if 'decoder.layer' in name or 'transformer.h' in name or 'layers' in name.split('.'):
            if isinstance(module, (nn.ModuleList, nn.Sequential)) or 'layer' in name:
                # Count transformer layers
                if hasattr(module, '__len__'):
                    layer_count = len(module)
                elif hasattr(model, 'layers') and hasattr(model, 'config'):
                    # For models with config.num_hidden_layers
                    if hasattr(model.config, 'num_hidden_layers'):
                        layer_count = model.config.num_hidden_layers
                        break
    
    # If we couldn't count layers from modules, try to get from config
    if layer_count == 0 and hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        layer_count = model.config.num_hidden_layers
    
    results['layer_count'] = layer_count
    results['layers_verified'] = (layer_count == expected_layers)
    
    # Check attention heads
    if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        head_count = model.config.num_attention_heads
        results['head_count'] = head_count
        results['heads_verified'] = (head_count == expected_heads)
    else:
        # Try to find attention head information in the model
        for name, module in model.named_modules():
            if 'attn' in name or 'attention' in name:
                if hasattr(module, 'num_heads'):
                    head_count = module.num_heads
                    results['head_count'] = head_count
                    results['heads_verified'] = (head_count == expected_heads)
                    break
                elif hasattr(module, 'num_attention_heads'):
                    head_count = module.num_attention_heads
                    results['head_count'] = head_count
                    results['heads_verified'] = (head_count == expected_heads)
                    break
    
    results['details'] = {
        'expected_layers': expected_layers,
        'actual_layers': layer_count,
        'expected_heads': expected_heads,
        'actual_heads': head_count
    }
    
    return results


def get_model_memory_requirements(model: nn.Module, input_shape: Tuple[int, ...] = (1, 512)) -> Dict[str, float]:
    """
    Estimate memory requirements for the model.
    
    Args:
        model: The model to analyze
        input_shape: Shape of input tensor for estimation
    
    Returns:
        Dictionary with memory requirements in MB
    """
    # Calculate model parameters memory
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Calculate buffer memory (batch norm, etc.)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Calculate activation memory (rough estimate based on input shape and model size)
    # This is a simplified estimation - real calculation would require a forward pass
    batch_size, seq_len = input_shape
    hidden_size = getattr(model, 'config', {}).get('hidden_size', 768)  # Default to 768 if not available

    # Estimate activation size: batch_size * seq_len * hidden_size * num_layers * 2 (for both forward and gradients)
    # The factor of 2 accounts for storing activations for backpropagation
    num_layers = getattr(model.config, 'num_hidden_layers', 12)  # Default to 12 if not available
    activation_size = batch_size * seq_len * hidden_size * num_layers * 2 * 4  # 4 bytes per float32
    
    # Convert to MB
    param_mb = param_size / 1024**2
    buffer_mb = buffer_size / 1024**2
    activation_mb = activation_size / 1024**2
    
    return {
        'parameters_mb': param_mb,
        'buffers_mb': buffer_mb,
        'activations_mb': activation_mb,
        'total_mb': param_mb + buffer_mb + activation_mb
    }


def check_device_compatibility(model: nn.Module) -> Dict[str, Any]:
    """
    Check device compatibility for the model.
    
    Args:
        model: The model to check
    
    Returns:
        Dictionary with device compatibility information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(next(model.parameters()).device) if next(model.parameters(), None) is not None else 'unknown',
        'fp16_supported': False,
        'bf16_supported': False
    }
    
    if device_info['cuda_available']:
        device_info['fp16_supported'] = torch.cuda.is_bf16_supported() or True  # FP16 is widely supported
        device_info['bf16_supported'] = torch.cuda.is_bf16_supported()
        
        # Get device properties
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['device_capability'] = torch.cuda.get_device_capability(0)
        device_info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return device_info


def convert_model_to_half_precision(model: nn.Module) -> nn.Module:
    """
    Convert model to half precision (FP16) for memory efficiency.
    
    Args:
        model: The model to convert
    
    Returns:
        Converted model
    """
    return model.half()


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing to reduce memory usage during training.
    
    Args:
        model: The model to modify
    
    Returns:
        Modified model
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return model


def create_model_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Create a model checkpoint dictionary.
    
    Args:
        model: The model to checkpoint
        optimizer: Optional optimizer to include in checkpoint
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': getattr(model, 'config', None)
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    return checkpoint


def load_model_from_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load model from checkpoint.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state for
    
    Returns:
        Tuple of (model, optimizer, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the size of the model in memory.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with size information in MB and GB
    """
    param_size = 0
    buffer_size = 0

    # Calculate parameter size
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Calculate buffer size (batch norm buffers, etc.)
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    # Convert to different units
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)

    return {
        'param_bytes': param_size,
        'buffer_bytes': buffer_size,
        'total_bytes': total_size,
        'size_mb': size_mb,
        'size_gb': size_gb
    }


def get_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Estimate FLOPs for the model (simplified estimation).

    Args:
        model: The model to analyze
        input_shape: Input shape for estimation

    Returns:
        Dictionary with FLOP estimates
    """
    # This is a more accurate estimation - a full FLOP analysis would be more complex
    total_params = sum(p.numel() for p in model.parameters())

    # Estimate FLOPs based on parameters and input sequence length
    seq_len = input_shape[1] if len(input_shape) > 1 else 1
    batch_size = input_shape[0]

    # More accurate estimate: for transformer models, FLOPs are approximately:
    # 2 * total_params * seq_len * batch_size * (4/3 for attention + 2 for feedforward)
    # This is a simplified but more realistic estimate for transformer models
    attention_flops = 4 * seq_len**2 * batch_size * getattr(model.config, 'num_attention_heads', 12) * getattr(model.config, 'hidden_size', 768) // getattr(model.config, 'num_attention_heads', 12)
    feedforward_flops = 8 * seq_len * batch_size * getattr(model.config, 'hidden_size', 768) * getattr(model.config, 'intermediate_size', 3072) if hasattr(getattr(model, 'config', {}), 'intermediate_size') else 8 * seq_len * batch_size * getattr(model.config, 'hidden_size', 768) * getattr(model.config, 'hidden_size', 768) * 4

    estimated_flops = attention_flops + feedforward_flops

    return {
        'total_parameters': total_params,
        'sequence_length': seq_len,
        'batch_size': batch_size,
        'estimated_flops': estimated_flops,
        'estimated_flops_per_token': estimated_flops / (seq_len * batch_size) if seq_len * batch_size > 0 else 0
    }