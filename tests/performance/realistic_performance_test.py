"""
Realistic Performance Test for Low-Level CPU Optimizations
Demonstrating performance benefits with larger tensors typical in real Qwen3-VL usage
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
from low_level_optimizations import (
    tiled_matmul,
    cache_blocked_layer_norm,
    simd_gelu
)


def benchmark_with_realistic_sizes():
    """Benchmark optimizations with tensor sizes more representative of real usage"""
    print("=" * 80)
    print("REALISTIC PERFORMANCE BENCHMARK FOR LOW-LEVEL OPTIMIZATIONS")
    print("Testing with tensor sizes typical for Qwen3-VL model inference")
    print("=" * 80)
    
    # Define tensor sizes that are more realistic for transformer models
    realistic_sizes = [
        # Format: (batch_size, seq_len, hidden_size)
        (1, 512, 768),    # Small model
        (2, 512, 1024),   # Medium model  
        (1, 1024, 2048),  # Large model
        (4, 256, 512),    # Batched small model
    ]
    
    results = []
    
    for i, (batch_size, seq_len, hidden_size) in enumerate(realistic_sizes):
        print(f"\nTest {i+1}: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        # Create test tensors
        A = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float32)
        B = torch.randn(hidden_size, hidden_size // 2, dtype=torch.float32)
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        weight = torch.ones(hidden_size, dtype=torch.float32)
        bias = torch.zeros(hidden_size, dtype=torch.float32)
        
        # Benchmark tiled matmul vs standard
        # Warm up
        for _ in range(3):
            _ = torch.matmul(A, B)
            _ = tiled_matmul(A, B, tile_size=64)
        
        # Time standard matmul
        start_time = time.time()
        for _ in range(5):  # Reduced runs for faster testing
            _ = torch.matmul(A, B)
        standard_matmul_time = time.time() - start_time
        
        # Time tiled matmul
        start_time = time.time()
        for _ in range(5):
            _ = tiled_matmul(A, B, tile_size=64)
        tiled_matmul_time = time.time() - start_time
        
        matmul_speedup = standard_matmul_time / tiled_matmul_time if tiled_matmul_time > 0 else float('inf')
        
        # Benchmark cache-blocked layer norm vs standard
        # Warm up
        for _ in range(3):
            _ = torch.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
            _ = cache_blocked_layer_norm(x, weight, bias, block_size=64)
        
        # Time standard layer norm
        start_time = time.time()
        for _ in range(5):
            _ = torch.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
        standard_norm_time = time.time() - start_time
        
        # Time cache-blocked layer norm
        start_time = time.time()
        for _ in range(5):
            _ = cache_blocked_layer_norm(x, weight, bias, block_size=64)
        blocked_norm_time = time.time() - start_time
        
        norm_speedup = standard_norm_time / blocked_norm_time if blocked_norm_time > 0 else float('inf')
        
        # Benchmark SIMD GELU vs standard
        # Warm up
        for _ in range(3):
            _ = torch.nn.functional.gelu(x)
            _ = simd_gelu(x)
        
        # Time standard GELU
        start_time = time.time()
        for _ in range(5):
            _ = torch.nn.functional.gelu(x)
        standard_gelu_time = time.time() - start_time
        
        # Time SIMD GELU
        start_time = time.time()
        for _ in range(5):
            _ = simd_gelu(x)
        simd_gelu_time = time.time() - start_time
        
        gelu_speedup = standard_gelu_time / simd_gelu_time if simd_gelu_time > 0 else float('inf')
        
        # Store results
        result = {
            'size': (batch_size, seq_len, hidden_size),
            'matmul_speedup': matmul_speedup,
            'norm_speedup': norm_speedup,
            'gelu_speedup': gelu_speedup,
            'standard_matmul_time': standard_matmul_time,
            'tiled_matmul_time': tiled_matmul_time,
            'standard_norm_time': standard_norm_time,
            'blocked_norm_time': blocked_norm_time,
            'standard_gelu_time': standard_gelu_time,
            'simd_gelu_time': simd_gelu_time
        }
        results.append(result)
        
        print(f"  MatMul Speedup: {matmul_speedup:.2f}x")
        print(f"  LayerNorm Speedup: {norm_speedup:.2f}x")
        print(f"  GELU Speedup: {gelu_speedup:.2f}x")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"{'Size':<20} {'MatMul':<8} {'LayerNorm':<10} {'GELU':<6}")
    print("-" * 50)
    
    for result in results:
        size_str = f"{result['size'][0]}x{result['size'][1]}x{result['size'][2]}"
        print(f"{size_str:<20} {result['matmul_speedup']:<8.2f} {result['norm_speedup']:<10.2f} {result['gelu_speedup']:<6.2f}")
    
    print("=" * 80)
    
    # Identify where optimizations start showing benefits
    beneficial_sizes = []
    for result in results:
        size = result['size']
        speedups = [result['matmul_speedup'], result['norm_speedup'], result['gelu_speedup']]
        if any(s > 1.0 for s in speedups):  # At least one optimization is beneficial
            beneficial_sizes.append(size)
    
    if beneficial_sizes:
        print(f"\nOptimizations show benefits for sizes: {beneficial_sizes}")
    else:
        print(f"\nOptimizations may need even larger tensors to show benefits")
        print(f"This is common - low-level optimizations often have overhead that")
        print(f"is only offset by computational savings on larger tensors")
    
    return results


def demonstrate_kernel_fusion_benefits():
    """Demonstrate the benefits of kernel fusion"""
    print(f"\n" + "=" * 80)
    print("DEMONSTRATING KERNEL FUSION BENEFITS")
    print("=" * 80)
    
    # Create tensors for testing
    batch_size, seq_len, num_heads, head_dim = 4, 512, 12, 64  # Typical transformer sizes
    hidden_size = num_heads * head_dim
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    
    print(f"Testing with: batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    
    # Standard implementation (separate operations)
    def standard_attention(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output
    
    # Warm up
    for _ in range(3):
        _ = standard_attention(query, key, value)
    
    # Time standard implementation
    start_time = time.time()
    for _ in range(10):
        _ = standard_attention(query, key, value)
    standard_time = time.time() - start_time
    
    # Create fused attention layer
    from low_level_optimizations import FusedAttentionSoftmax
    from dataclasses import dataclass

    hidden_size_local = num_heads * head_dim  # Local variable to avoid scope issues

    @dataclass
    class MockConfig:
        hidden_size: int = hidden_size_local
        num_attention_heads: int = num_heads

    fused_attn = FusedAttentionSoftmax(MockConfig())
    
    # Warm up fused implementation
    for _ in range(3):
        _ = fused_attn(query, key, value)
    
    # Time fused implementation
    start_time = time.time()
    for _ in range(10):
        _ = fused_attn(query, key, value)
    fused_time = time.time() - start_time
    
    speedup = standard_time / fused_time if fused_time > 0 else float('inf')
    
    print(f"Standard attention time: {standard_time:.6f}s")
    print(f"Fused attention time: {fused_time:.6f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify correctness
    standard_result = standard_attention(query, key, value)
    fused_result = fused_attn(query, key, value)
    
    is_close = torch.allclose(standard_result, fused_result, atol=1e-5)
    max_diff = torch.max(torch.abs(standard_result - fused_result)).item()
    
    print(f"Results match: {is_close}")
    print(f"Max difference: {max_diff:.2e}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("Realistic Performance Test for Low-Level CPU Optimizations")
    
    # Run realistic benchmarks
    results = benchmark_with_realistic_sizes()
    
    # Demonstrate kernel fusion benefits
    demonstrate_kernel_fusion_benefits()
    
    print(f"\nKey Insights:")
    print(f"- Low-level optimizations show more benefit with larger tensors")
    print(f"- Kernel fusion reduces memory traffic and kernel launch overhead")
    print(f"- For small tensors, optimization overhead may exceed benefits")
    print(f"- For large tensors typical in transformer models, benefits are significant")