"""
Basic Test for CUDA Kernels

This module contains basic tests for all CUDA kernels implemented for the various models.
"""

import torch
from src.models.specialized.glm_4_7_flash.cuda_kernels.custom_kernels import (
    GLM47FlashAttentionKernel,
    GLM47FlashMLPKernel,
    GLM47FlashRMSNormKernel,
    GLM47FlashRotaryEmbedding,
)

# Test GLM-4.7-Flash kernels
def test_glm_kernels():
    print("Testing GLM-4.7-Flash kernels...")
    
    batch_size = 2
    seq_len = 16
    d_model = 2048
    nhead = 32
    
    # Test attention kernel
    print("Testing attention kernel...")
    attn_kernel = GLM47FlashAttentionKernel(
        d_model=d_model,
        nhead=nhead,
        dropout=0.1,
        use_flash_attention=True,
        use_rotary_embeddings=True,
    )
    
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    position_ids = torch.arange(seq_len).expand(batch_size, -1)
    
    output, attn_weights = attn_kernel(query, key, value, position_ids=position_ids, need_weights=True)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test MLP kernel
    print("Testing MLP kernel...")
    mlp_kernel = GLM47FlashMLPKernel(
        d_model=d_model,
        intermediate_size=5504,
        activation_type="gelu",
        dropout=0.1,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    mlp_output = mlp_kernel(x)
    print(f"MLP output shape: {mlp_output.shape}")
    
    # Test RMSNorm kernel
    print("Testing RMSNorm kernel...")
    rmsnorm_kernel = GLM47FlashRMSNormKernel(
        normalized_shape=d_model,
        eps=1e-5,
    )
    
    norm_output = rmsnorm_kernel(x)
    print(f"RMSNorm output shape: {norm_output.shape}")
    
    # Test Rotary Embedding
    print("Testing Rotary Embedding...")
    rotary_kernel = GLM47FlashRotaryEmbedding(dim=d_model // nhead)
    
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    q_rotated, k_rotated = rotary_kernel(q, k, position_ids)
    print(f"Rotary Q output shape: {q_rotated.shape}")
    print(f"Rotary K output shape: {k_rotated.shape}")
    
    print("All GLM-4.7-Flash kernels tested successfully!")

if __name__ == "__main__":
    test_glm_kernels()