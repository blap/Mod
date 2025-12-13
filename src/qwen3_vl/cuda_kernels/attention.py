"""
Python interface for attention computation CUDA kernels optimized for NVIDIA SM61 architecture
"""
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import math

# Dynamically compile the CUDA extension
curr_dir = os.path.dirname(__file__)
kernel_sources = [
    os.path.join(curr_dir, 'attention_kernel.cu'),
    os.path.join(curr_dir, 'tensor_ops.cu')
]

# Define compilation options for SM61 architecture
extra_cuda_cflags = ['-arch=sm_61', '-O3', '--use_fast_math']

try:
    # Load the compiled CUDA extension
    attention_ext = load(
        name='sm61_attention',
        sources=kernel_sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False
    )
except Exception as e:
    print(f"Warning: Could not compile CUDA extension: {e}")
    print("Falling back to PyTorch implementation")
    attention_ext = None


def attention_forward(q, k, v):
    """
    Forward pass of attention mechanism optimized for SM61 architecture
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, head_dim)
        k: Key tensor of shape (batch_size, seq_len, head_dim)  
        v: Value tensor of shape (batch_size, seq_len, head_dim)
        
    Returns:
        Output tensor of shape (batch_size, seq_len, head_dim)
    """
    if attention_ext is not None and q.is_cuda:
        # Use optimized CUDA kernel
        return attention_ext.attention_forward(q, k, v)
    else:
        # Fall back to PyTorch implementation
        return _pytorch_attention_forward(q, k, v)


def attention_backward(grad_output, q, k, v, output):
    """
    Backward pass of attention mechanism
    
    Args:
        grad_output: Gradient of the loss w.r.t. output
        q, k, v: Input tensors from forward pass
        output: Output from forward pass
        
    Returns:
        Gradients w.r.t. q, k, v
    """
    if attention_ext is not None and grad_output.is_cuda:
        # Use optimized CUDA kernel
        return attention_ext.attention_backward(grad_output, q, k, v, output)
    else:
        # Fall back to PyTorch autograd
        return _pytorch_attention_backward(grad_output, q, k, v, output)


def _pytorch_attention_forward(q, k, v):
    """PyTorch fallback implementation of attention"""
    # Compute attention scores: (batch_size, seq_len, seq_len)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, seq_len, seq_len)
    
    # Scale by square root of head dimension
    scale = math.sqrt(q.size(-1))
    attn_scores = attn_scores / scale
    
    # Apply softmax
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Compute output: (batch_size, seq_len, head_dim)
    output = torch.matmul(attn_weights, v)
    
    return output


def _pytorch_attention_backward(grad_output, q, k, v, output):
    """PyTorch fallback implementation of attention backward pass"""
    # This is a simplified version - in practice would need full autograd graph
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    
    # Recompute forward pass to enable backward
    temp_out = _pytorch_attention_forward(q, k, v)
    
    # Compute gradients
    temp_out.backward(grad_output)
    
    return q.grad, k.grad, v.grad


# For now, since we're using a fallback, define stub functions that will be implemented by the CUDA extension
def attention_kernel_launcher(q, k, v, output, config):
    """
    Launcher for attention kernel with SM61-optimized parameters
    """
    if attention_ext is not None:
        return attention_ext.attention_kernel_launcher(q, k, v, output, config)
    else:
        # Fallback implementation
        return _pytorch_attention_forward(q, k, v)