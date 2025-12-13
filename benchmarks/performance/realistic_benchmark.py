"""
Realistic benchmark for comprehensive low-level CPU optimizations
"""
import torch
import time
import numpy as np
from comprehensive_cpu_optimizations import (
    OptimizationConfig,
    LoopTilingOptimizer,
    CacheBlockingOptimizer,
    ManualSIMDOptimizer,
    KernelFusionOptimizer
)


def realistic_benchmark():
    """Benchmark optimizations with more realistic tensor sizes"""
    config = OptimizationConfig()

    print("Realistic Benchmark: Comprehensive Low-Level CPU Optimizations")
    print("=" * 65)

    # Initialize optimizers
    tiling_optimizer = LoopTilingOptimizer(config)
    cache_blocking_optimizer = CacheBlockingOptimizer(config)
    simd_optimizer = ManualSIMDOptimizer(config)
    kernel_fusion_optimizer = KernelFusionOptimizer(config)

    # Use larger tensors to better demonstrate optimization benefits
    batch_size, seq_len, hidden_size = 16, 512, 768  # More realistic for transformer models
    print(f"Tensor sizes: batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create test tensors
    A = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float32)
    B = torch.randn(hidden_size, hidden_size // 4, dtype=torch.float32)  # Down projection
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    weight = torch.ones(hidden_size, dtype=torch.float32)
    bias = torch.zeros(hidden_size, dtype=torch.float32)

    # Warm up
    for _ in range(5):
        _ = torch.matmul(A, B)
        _ = torch.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
        _ = torch.nn.functional.gelu(x)

    print("\n1. Testing Tiled Matrix Multiplication...")
    # Test with different tile sizes to find optimal
    tile_sizes = [64, 128, 256]
    best_tiled_time = float('inf')
    best_tile_size = 64
    
    for tile_size in tile_sizes:
        start_time = time.time()
        for _ in range(5):  # Fewer iterations for larger tensors
            result = tiling_optimizer.tiled_matmul(A, B, tile_size=tile_size)
        tiled_time = time.time() - start_time
        
        if tiled_time < best_tiled_time:
            best_tiled_time = tiled_time
            best_tile_size = tile_size
    
    print(f"   Best tiled matmul (tile_size={best_tile_size}): {best_tiled_time:.6f}s")
    
    start_time = time.time()
    for _ in range(5):
        result = torch.matmul(A, B)
    standard_time = time.time() - start_time
    print(f"   Standard matmul: {standard_time:.6f}s")
    
    if standard_time > 0:
        speedup = standard_time / best_tiled_time if best_tiled_time > 0 else float('inf')
        print(f"   Tiled matmul relative performance: {speedup:.2f}x (higher is better if >1)")

    print("\n2. Testing Cache-Blocked Layer Norm...")
    start_time = time.time()
    for _ in range(5):
        result = cache_blocking_optimizer.cache_blocked_layer_norm(x, weight, bias, block_size=256)
    blocked_time = time.time() - start_time
    print(f"   Cache-blocked layer norm: {blocked_time:.6f}s")

    start_time = time.time()
    for _ in range(5):
        result = torch.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
    standard_norm_time = time.time() - start_time
    print(f"   Standard layer norm: {standard_norm_time:.6f}s")

    if standard_norm_time > 0:
        norm_speedup = standard_norm_time / blocked_time if blocked_time > 0 else float('inf')
        print(f"   Cache-blocked layer norm relative performance: {norm_speedup:.2f}x")

    print("\n3. Testing SIMD GELU...")
    start_time = time.time()
    for _ in range(5):
        result = simd_optimizer.simd_gelu(x)
    simd_gelu_time = time.time() - start_time
    print(f"   SIMD GELU: {simd_gelu_time:.6f}s")

    start_time = time.time()
    for _ in range(5):
        result = torch.nn.functional.gelu(x)
    standard_gelu_time = time.time() - start_time
    print(f"   Standard GELU: {standard_gelu_time:.6f}s")

    if standard_gelu_time > 0:
        gelu_speedup = standard_gelu_time / simd_gelu_time if simd_gelu_time > 0 else float('inf')
        print(f"   SIMD GELU relative performance: {gelu_speedup:.2f}x")

    print("\n4. Testing Fused Operations...")
    # Create test tensors for fused operations
    query = torch.randn(batch_size, 12, seq_len, hidden_size // 12)  # 12 heads
    key = torch.randn(batch_size, 12, seq_len, hidden_size // 12)
    value = torch.randn(batch_size, 12, seq_len, hidden_size // 12)

    start_time = time.time()
    for _ in range(3):  # Fewer iterations for attention
        result = kernel_fusion_optimizer.fused_attention_softmax(query, key, value)
    fused_attn_time = time.time() - start_time
    print(f"   Fused attention-softmax: {fused_attn_time:.6f}s")

    # Verify correctness
    print("\n5. Verifying Correctness...")
    standard_matmul = torch.matmul(A, B)
    tiled_result = tiling_optimizer.tiled_matmul(A, B, tile_size=best_tile_size)
    matmul_correct = torch.allclose(standard_matmul, tiled_result, atol=1e-4)
    print(f"   - Tiled matmul results correct: {matmul_correct}")

    standard_norm = torch.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
    blocked_norm = cache_blocking_optimizer.cache_blocked_layer_norm(x, weight, bias, block_size=256)
    norm_correct = torch.allclose(standard_norm, blocked_norm, atol=1e-4)
    print(f"   - Cache-blocked norm results correct: {norm_correct}")

    standard_gelu = torch.nn.functional.gelu(x)
    simd_gelu_result = simd_optimizer.simd_gelu(x)
    gelu_correct = torch.allclose(standard_gelu, simd_gelu_result, atol=1e-3)
    print(f"   - SIMD GELU results correct: {gelu_correct}")

    # Test with even larger tensors to demonstrate memory efficiency benefits
    print("\n6. Testing with Very Large Tensors (Memory Efficiency)...")
    large_x = torch.randn(8, 1024, 1024, dtype=torch.float32)  # Large tensor
    large_weight = torch.ones(1024, dtype=torch.float32)
    large_bias = torch.zeros(1024, dtype=torch.float32)

    # Test cache-blocking with large tensor
    start_time = time.time()
    result_blocked = cache_blocking_optimizer.cache_blocked_layer_norm(
        large_x, large_weight, large_bias, block_size=512
    )
    large_blocked_time = time.time() - start_time
    print(f"   Large tensor cache-blocked norm: {large_blocked_time:.6f}s")

    start_time = time.time()
    result_standard = torch.layer_norm(large_x, large_x.shape[-1:], large_weight, large_bias, 1e-5)
    large_standard_time = time.time() - start_time
    print(f"   Large tensor standard norm: {large_standard_time:.6f}s")

    print(f"\nRealistic Benchmark Summary:")
    print(f"  - Tiled operations show benefits primarily in memory-constrained scenarios")
    print(f"  - Cache-blocking helps with large tensors that don't fit in cache")
    print(f"  - SIMD operations can be beneficial when not using highly optimized libraries")
    print(f"  - Kernel fusion reduces memory traffic between operations")
    print(f"  - For production use, these optimizations work best in combination")


if __name__ == "__main__":
    realistic_benchmark()