import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_dynamic_sparsity(attn_weights: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Apply dynamic sparsity by keeping only the top-k attention weights for each query position.
    This is a vectorized and more efficient implementation.
    """
    bsz, num_heads, q_len, k_len = attn_weights.size()
    
    # Calculate actual top_k to not exceed sequence length
    top_k_val = min(top_k, k_len)
    
    # Create a copy of attention weights to modify
    sparse_attn_weights = attn_weights.clone()
    
    # Vectorized approach: reshape to apply top-k across all positions at once
    # Reshape to (bsz * num_heads * q_len, k_len)
    reshaped_weights = sparse_attn_weights.view(-1, k_len)
    
    # Find top-k values for all positions at once
    top_k_values, top_k_indices = torch.topk(reshaped_weights, top_k_val, dim=-1, sorted=False)
    
    # Create a mask for top-k positions
    mask = torch.zeros_like(reshaped_weights, dtype=torch.bool)
    mask.scatter_(-1, top_k_indices, True)
    
    # Create values tensor with min values
    min_val = torch.finfo(reshaped_weights.dtype).min
    masked_weights = torch.where(mask, reshaped_weights, torch.full_like(reshaped_weights, min_val))
    
    # Reshape back to original shape
    sparse_attn_weights = masked_weights.view(bsz, num_heads, q_len, k_len)
    
    return sparse_attn_weights