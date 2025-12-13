"""
Benchmarking script for SM61 optimized CUDA kernels
Compares performance against standard implementations
"""

import torch
import time
import numpy as np
import argparse
from typing import Tuple, List
import matplotlib.pyplot as plt

try:
    from src.cuda_kernels.pybind_interface import (
        scaled_dot_product_attention_sm61,
        coalesced_copy_sm61,
        transpose_sm61,
        SM61MemoryPool
    )
except ImportError as e:
    print(f"Error importing CUDA kernels: {e}")
    print("Make sure you've built the CUDA extensions first.")
    exit(1)

def benchmark_scaled_dot_product_attention(
    batch_size: int, 
    seq_len: int, 
    num_heads: int, 
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    num_iterations: int = 100
) -> Tuple[float, float]:
    """
    Benchmark SM61 optimized attention vs PyTorch's implementation
    """
    device = torch.device('cuda')
    
    # Create input tensors
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(10):
        _ = scaled_dot_product_attention_sm61(query, key, value)
    torch.cuda.synchronize()
    
    # Benchmark SM61 implementation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        output_sm61 = scaled_dot_product_attention_sm61(query, key, value)
    end_event.record()
    torch.cuda.synchronize()
    
    sm61_time_ms = start_event.elapsed_time(end_event)
    
    # Benchmark PyTorch implementation
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        torch.cuda.synchronize()
        
        start_event.record()
        for _ in range(num_iterations):
            output_pytorch = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        end_event.record()
        torch.cuda.synchronize()
        
        pytorch_time_ms = start_event.elapsed_time(end_event)
    
    # Verify correctness
    torch.testing.assert_close(output_sm61, output_pytorch, rtol=1e-4, atol=1e-4)
    
    return sm61_time_ms / num_iterations, pytorch_time_ms / num_iterations


def benchmark_memory_copy(
    size_mb: int,
    dtype: torch.dtype = torch.float32,
    num_iterations: int = 100
) -> Tuple[float, float]:
    """
    Benchmark coalesced memory copy vs standard copy
    """
    device = torch.device('cuda')
    
    # Calculate tensor size to match requested MB
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(size_mb * 1024 * 1024 / element_size)
    
    # Create input tensor
    input_tensor = torch.randn(num_elements, device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(10):
        _ = coalesced_copy_sm61(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark SM61 implementation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        output_sm61 = coalesced_copy_sm61(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    sm61_time_ms = start_event.elapsed_time(end_event)
    
    # Benchmark standard copy (using PyTorch)
    for _ in range(10):
        _ = input_tensor.clone()
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(num_iterations):
        output_standard = input_tensor.clone()
    end_event.record()
    torch.cuda.synchronize()
    
    standard_time_ms = start_event.elapsed_time(end_event)
    
    return sm61_time_ms / num_iterations, standard_time_ms / num_iterations


def benchmark_transpose(
    rows: int,
    cols: int,
    dtype: torch.dtype = torch.float32,
    num_iterations: int = 100
) -> Tuple[float, float]:
    """
    Benchmark SM61 optimized transpose vs PyTorch's transpose
    """
    device = torch.device('cuda')
    
    # Create input matrix
    input_matrix = torch.randn(rows, cols, device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(10):
        _ = transpose_sm61(input_matrix)
    torch.cuda.synchronize()
    
    # Benchmark SM61 implementation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        output_sm61 = transpose_sm61(input_matrix)
    end_event.record()
    torch.cuda.synchronize()
    
    sm61_time_ms = start_event.elapsed_time(end_event)
    
    # Benchmark PyTorch implementation
    for _ in range(10):
        _ = input_matrix.t()
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(num_iterations):
        output_pytorch = input_matrix.t()
    end_event.record()
    torch.cuda.synchronize()
    
    pytorch_time_ms = start_event.elapsed_time(end_event)
    
    # Verify correctness
    torch.testing.assert_close(output_sm61, output_pytorch, rtol=1e-5, atol=1e-5)
    
    return sm61_time_ms / num_iterations, pytorch_time_ms / num_iterations


def run_comprehensive_benchmarks():
    """
    Run comprehensive benchmarks for all kernels
    """
    print("="*60)
    print("SM61 CUDA Kernels Benchmarking Suite")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    device_name = torch.cuda.get_device_name()
    print(f"Using device: {device_name}")
    print(f"CUDA compute capability: {torch.cuda.get_device_capability()}")
    print()
    
    results = {
        'attention': [],
        'memory_copy': [],
        'transpose': []
    }
    
    # Benchmark attention with different configurations
    print("Benchmarking Scaled Dot-Product Attention...")
    attention_configs = [
        (1, 512, 8, 64),    # Small
        (2, 512, 8, 64),    # Medium-small
        (4, 512, 8, 64),    # Medium
        (4, 1024, 8, 64),   # Large-medium
        (2, 512, 16, 64),   # More heads
    ]
    
    for batch_size, seq_len, num_heads, head_dim in attention_configs:
        print(f"  Config: B={batch_size}, L={seq_len}, H={num_heads}, D={head_dim}")
        try:
            sm61_time, pytorch_time = benchmark_scaled_dot_product_attention(
                batch_size, seq_len, num_heads, head_dim, dtype=torch.float32
            )
            speedup = pytorch_time / sm61_time
            results['attention'].append({
                'config': f"B{batch_size}_L{seq_len}_H{num_heads}_D{head_dim}",
                'sm61_time': sm61_time,
                'pytorch_time': pytorch_time,
                'speedup': speedup
            })
            print(f"    SM61: {sm61_time:.4f}ms, PyTorch: {pytorch_time:.4f}ms, Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        print()
    
    # Benchmark memory copy with different sizes
    print("Benchmarking Memory Copy...")
    copy_sizes = [1, 4, 16, 64, 256]  # MB
    
    for size_mb in copy_sizes:
        print(f"  Size: {size_mb}MB")
        try:
            sm61_time, standard_time = benchmark_memory_copy(size_mb, dtype=torch.float32)
            speedup = standard_time / sm61_time
            results['memory_copy'].append({
                'size_mb': size_mb,
                'sm61_time': sm61_time,
                'standard_time': standard_time,
                'speedup': speedup
            })
            print(f"    SM61: {sm61_time:.4f}ms, Standard: {standard_time:.4f}ms, Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        print()
    
    # Benchmark transpose with different sizes
    print("Benchmarking Matrix Transpose...")
    transpose_configs = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]
    
    for rows, cols in transpose_configs:
        print(f"  Size: {rows}x{cols}")
        try:
            sm61_time, pytorch_time = benchmark_transpose(rows, cols, dtype=torch.float32)
            speedup = pytorch_time / sm61_time
            results['transpose'].append({
                'size': f"{rows}x{cols}",
                'sm61_time': sm61_time,
                'pytorch_time': pytorch_time,
                'speedup': speedup
            })
            print(f"    SM61: {sm61_time:.4f}ms, PyTorch: {pytorch_time:.4f}ms, Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
        print()
    
    # Print summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    # Attention summary
    if results['attention']:
        avg_speedup = np.mean([r['speedup'] for r in results['attention']])
        print(f"Attention Average Speedup: {avg_speedup:.2f}x")
        best_config = max(results['attention'], key=lambda x: x['speedup'])
        print(f"Best Attention Config: {best_config['config']} -> {best_config['speedup']:.2f}x")
    
    # Memory copy summary
    if results['memory_copy']:
        avg_speedup = np.mean([r['speedup'] for r in results['memory_copy']])
        print(f"Memory Copy Average Speedup: {avg_speedup:.2f}x")
        best_size = max(results['memory_copy'], key=lambda x: x['speedup'])
        print(f"Best Copy Size: {best_size['size_mb']}MB -> {best_size['speedup']:.2f}x")
    
    # Transpose summary
    if results['transpose']:
        avg_speedup = np.mean([r['speedup'] for r in results['transpose']])
        print(f"Transpose Average Speedup: {avg_speedup:.2f}x")
        best_size = max(results['transpose'], key=lambda x: x['speedup'])
        print(f"Best Transpose Size: {best_size['size']} -> {best_size['speedup']:.2f}x")
    
    # Plot results if matplotlib is available
    try:
        plot_results(results)
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    return results


def plot_results(results: dict):
    """
    Plot benchmark results
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot attention results
    if results['attention']:
        configs = [r['config'] for r in results['attention']]
        speedups = [r['speedup'] for r in results['attention']]
        
        axes[0].bar(range(len(configs)), speedups)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Speedup vs PyTorch')
        axes[0].set_title('Attention Kernel Speedup')
        axes[0].set_xticks(range(len(configs)))
        axes[0].set_xticklabels(configs, rotation=45, ha='right')
        axes[0].grid(True, axis='y', alpha=0.3)
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='No improvement')
        axes[0].legend()
    
    # Plot memory copy results
    if results['memory_copy']:
        sizes = [r['size_mb'] for r in results['memory_copy']]
        speedups = [r['speedup'] for r in results['memory_copy']]
        
        axes[1].plot(sizes, speedups, 'o-', label='Speedup')
        axes[1].set_xlabel('Size (MB)')
        axes[1].set_ylabel('Speedup vs Standard')
        axes[1].set_title('Memory Copy Speedup')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='No improvement')
        axes[1].legend()
    
    # Plot transpose results
    if results['transpose']:
        sizes = [r['size'] for r in results['transpose']]
        speedups = [r['speedup'] for r in results['transpose']]
        
        axes[2].bar(range(len(sizes)), speedups)
        axes[2].set_xlabel('Matrix Size')
        axes[2].set_ylabel('Speedup vs PyTorch')
        axes[2].set_title('Transpose Kernel Speedup')
        axes[2].set_xticks(range(len(sizes)))
        axes[2].set_xticklabels(sizes)
        axes[2].grid(True, axis='y', alpha=0.3)
        axes[2].axhline(y=1.0, color='r', linestyle='--', label='No improvement')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('sm61_benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Benchmark results plot saved as 'sm61_benchmark_results.png'")


def test_memory_pool_efficiency():
    """
    Test the efficiency of the SM61 memory pool
    """
    print("\nTesting Memory Pool Efficiency...")
    
    pool = SM61MemoryPool(pool_size=64 * 1024 * 1024)  # 64MB pool
    
    # Initial stats
    initial_stats = pool.get_stats()
    print(f"Initial pool stats: {initial_stats}")
    
    # Allocate a bunch of small tensors
    tensors = []
    for i in range(20):
        tensor = pool.allocate_tensor([128, 128], torch.float32)  # ~64KB each
        tensors.append(tensor)
    
    # Check stats after allocations
    mid_stats = pool.get_stats()
    print(f"After 20 allocations: {mid_stats}")
    
    # Free some tensors
    for i in range(10):
        tensors.pop()  # This should trigger deallocation when tensor goes out of scope
    
    # Check stats after partial deallocation
    torch.cuda.synchronize()
    partial_stats = pool.get_stats()
    print(f"After freeing 10 tensors: {partial_stats}")
    
    # Free remaining tensors
    tensors.clear()
    torch.cuda.synchronize()
    final_stats = pool.get_stats()
    print(f"After freeing all tensors: {final_stats}")
    
    print(f"Pool utilization efficiency: {(initial_stats['free'] - final_stats['free']) / initial_stats['free'] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark SM61 CUDA kernels')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with fewer iterations')
    
    args = parser.parse_args()
    
    # Adjust iterations based on quick flag
    if args.quick:
        print("Running quick benchmark...")
        # The benchmark functions already use reasonable defaults for quick testing
        pass
    
    # Run benchmarks
    results = run_comprehensive_benchmarks()
    
    # Test memory pool efficiency
    test_memory_pool_efficiency()
    
    print("\nBenchmarking complete!")