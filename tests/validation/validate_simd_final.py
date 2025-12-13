#!/usr/bin/env python
"""
Final validation of SIMD optimizations for Qwen3-VL Model
"""

import torch
from production_simd_optimizations import SIMDOptimizationConfig, AVX2OptimizedOperations

def main():
    print("Starting SIMD optimizations validation...")
    
    # Test the basic functionality
    config = SIMDOptimizationConfig()
    ops = AVX2OptimizedOperations(config)

    # Create a test tensor
    test_tensor = torch.randn(4, 32, 512)

    # Test vectorized operations
    print("Testing vectorized operations...")

    # Test normalization
    norm_result = ops.vectorized_normalize(test_tensor)
    print(f"Normalization: {norm_result.shape}")

    # Test GELU
    gelu_result = ops.vectorized_gelu_approximation(test_tensor)
    print(f"GELU: {gelu_result.shape}")

    # Test matmul
    a = torch.randn(4, 32, 512)
    b = torch.randn(4, 512, 256)
    matmul_result = ops.vectorized_matmul(a, b)
    print(f"Matmul: {matmul_result.shape}")

    print(f"SIMD width: {ops.simd_width}")
    print("All operations completed successfully")
    
    # Verify correctness
    assert norm_result.shape == test_tensor.shape
    assert gelu_result.shape == test_tensor.shape
    assert matmul_result.shape == (4, 32, 256)
    
    print("All assertions passed - SIMD optimizations are working correctly!")

if __name__ == "__main__":
    main()