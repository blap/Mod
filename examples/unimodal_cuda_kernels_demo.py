"""
Example demonstrating the usage of unimodal CUDA kernels with language models.

This script shows how the unimodal CUDA kernels are integrated with the
GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b models.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.common.unimodal_cuda_kernels import (
    UnimodalAttentionKernel,
    UnimodalLayerNormKernel,
    UnimodalMLPKernel,
    UnimodalRMSNormKernel,
    get_unimodal_cuda_optimization_report,
)


def demonstrate_unimodal_kernels():
    """Demonstrate the usage of unimodal CUDA kernels."""
    print("=== Demonstrating Unimodal CUDA Kernels ===\n")

    # Example 1: Creating individual kernels
    print("1. Creating individual unimodal CUDA kernels:")

    # Attention kernel
    attention_kernel = UnimodalAttentionKernel(
        d_model=512, nhead=8, dropout=0.1, use_flash_attention=True, causal=True
    )
    print(f"   - Created UnimodalAttentionKernel with d_model=512, nhead=8")

    # MLP kernel
    mlp_kernel = UnimodalMLPKernel(
        d_model=512,
        intermediate_size=2048,
        activation="silu",
        use_swiglu=True,
        dropout=0.1,
    )
    print(f"   - Created UnimodalMLPKernel with d_model=512, intermediate_size=2048")

    # LayerNorm kernel
    layernorm_kernel = UnimodalLayerNormKernel(normalized_shape=512, eps=1e-5)
    print(f"   - Created UnimodalLayerNormKernel with normalized_shape=512")

    # RMSNorm kernel
    rmsnorm_kernel = UnimodalRMSNormKernel(normalized_shape=512, eps=1e-5)
    print(f"   - Created UnimodalRMSNormKernel with normalized_shape=512\n")

    # Example 2: Testing forward pass with kernels
    print("2. Testing forward pass with kernels:")

    batch_size = 2
    seq_len = 16
    d_model = 512

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # Test attention kernel
    attn_output, attn_weights = attention_kernel(query, key, value)
    print(f"   - Attention output shape: {attn_output.shape}")

    # Test MLP kernel
    mlp_output = mlp_kernel(x)
    print(f"   - MLP output shape: {mlp_output.shape}")

    # Test normalization kernels
    ln_output = layernorm_kernel(x)
    print(f"   - LayerNorm output shape: {ln_output.shape}")

    rn_output = rmsnorm_kernel(x)
    print(f"   - RMSNorm output shape: {rn_output.shape}\n")

    # Example 3: Model integration concepts
    print("3. Model integration with unimodal CUDA kernels:")
    print("   The unimodal CUDA kernels are integrated into the following models:")
    print(
        "   - GLM-4-7: Updated to use optimized attention, MLP, and normalization kernels"
    )
    print("   - Qwen3-4b-instruct-2507: Enhanced with CUDA-optimized operations")
    print("   - Qwen3-coder-30b: Improved performance with specialized kernels\n")

    # Example 4: Optimization report
    print("4. Getting optimization report:")

    # Create a simple model for demonstration
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=512, num_heads=8, batch_first=True
            )
            self.norm = torch.nn.LayerNorm(512)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(512, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 512)
            )

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out
            x = self.norm(x)
            x = x + self.mlp(x)
            return x

    simple_model = SimpleModel()

    report = get_unimodal_cuda_optimization_report(
        model=simple_model,
        d_model=512,
        nhead=8,
        intermediate_size=2048,
        model_type="general",
    )

    print(f"   - Model type: {report['model_type']}")
    print(f"   - Modules identified for optimization:")
    for module_type, count in report["modules_identified_for_optimization"].items():
        print(f"      - {module_type}: {count}")
    print(f"   - Optimization config: {report['optimization_config']}\n")

    print("=== Unimodal CUDA Kernels Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_unimodal_kernels()
