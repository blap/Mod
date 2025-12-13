"""
Memory Bandwidth Utilization Benchmark
Used for Phase 2.9: Memory Pooling and Pre-allocation Techniques
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import psutil
import threading
import queue


def benchmark_memory_bandwidth():
    """Benchmark memory bandwidth utilization"""
    print("Benchmarking memory bandwidth utilization...")
    
    results = {}
    
    # Test 1: Sequential read/write bandwidth
    print("Testing sequential read/write bandwidth...")
    sequential_results = test_sequential_bandwidth()
    results['sequential_bandwidth'] = sequential_results
    
    # Test 2: Random access bandwidth
    print("Testing random access bandwidth...")
    random_results = test_random_access_bandwidth()
    results['random_access_bandwidth'] = random_results
    
    # Test 3: Tensor copy bandwidth (GPU to GPU, CPU to GPU, etc.)
    if torch.cuda.is_available():
        print("Testing tensor copy bandwidth...")
        copy_results = test_tensor_copy_bandwidth()
        results['tensor_copy_bandwidth'] = copy_results
    
    # Test 4: Memory allocation bandwidth
    print("Testing allocation bandwidth...")
    alloc_results = test_allocation_bandwidth()
    results['allocation_bandwidth'] = alloc_results
    
    return results


def test_sequential_bandwidth():
    """Test sequential memory access bandwidth"""
    sizes_gb = [0.1, 0.25, 0.5, 1.0]  # GB
    results = {'sizes_gb': sizes_gb, 'bandwidth_gbps': []}
    
    for size_gb in sizes_gb:
        size_bytes = int(size_gb * 1024**3)
        num_elements = size_bytes // 4  # 4 bytes per float32
        
        # Create tensor
        tensor = torch.empty(num_elements, dtype=torch.float32)
        
        # Warm up
        tensor.fill_(1.0)
        tensor *= 2.0
        
        # Measure sequential write bandwidth
        start_time = time.perf_counter()
        tensor.fill_(1.0)
        write_time = time.perf_counter() - start_time
        
        # Measure sequential read bandwidth (with computation)
        start_time = time.perf_counter()
        result = torch.sum(tensor)
        read_time = time.perf_counter() - start_time
        
        write_bandwidth = size_gb / write_time
        read_bandwidth = size_gb / read_time
        
        results['bandwidth_gbps'].append({
            'size_gb': size_gb,
            'write_bandwidth_gbps': write_bandwidth,
            'read_bandwidth_gbps': read_bandwidth,
            'total_bandwidth_gbps': write_bandwidth + read_bandwidth
        })
    
    avg_write = np.mean([r['write_bandwidth_gbps'] for r in results['bandwidth_gbps']])
    avg_read = np.mean([r['read_bandwidth_gbps'] for r in results['bandwidth_gbps']])
    
    print(f"Sequential write bandwidth: {avg_write:.2f} GB/s")
    print(f"Sequential read bandwidth: {avg_read:.2f} GB/s")
    
    return results


def test_random_access_bandwidth():
    """Test random memory access bandwidth"""
    sizes_gb = [0.1, 0.25, 0.5]  # GB
    results = {'sizes_gb': sizes_gb, 'bandwidth_gbps': []}
    
    for size_gb in sizes_gb:
        size_bytes = int(size_gb * 1024**3)
        num_elements = size_bytes // 4  # 4 bytes per float32
        
        # Create tensor
        tensor = torch.empty(num_elements, dtype=torch.float32)
        tensor.fill_(1.0)
        
        # Create random indices for scattered access
        num_accesses = min(num_elements // 100, 100000)  # Limit for performance
        random_indices = torch.randint(0, num_elements, (num_accesses,))
        
        # Warm up
        _ = tensor[random_indices].sum()
        
        # Measure random access bandwidth
        start_time = time.perf_counter()
        result = tensor[random_indices].sum()
        access_time = time.perf_counter() - start_time
        
        # Calculate effective bandwidth (only the accessed data counts)
        accessed_bytes = num_accesses * 4  # 4 bytes per float32
        accessed_gb = accessed_bytes / 1024**3
        bandwidth = accessed_gb / access_time
        
        results['bandwidth_gbps'].append({
            'size_gb': size_gb,
            'accessed_gb': accessed_gb,
            'bandwidth_gbps': bandwidth,
            'num_accesses': num_accesses
        })
    
    avg_bandwidth = np.mean([r['bandwidth_gbps'] for r in results['bandwidth_gbps']])
    print(f"Random access bandwidth: {avg_bandwidth:.2f} GB/s")
    
    return results


def test_tensor_copy_bandwidth():
    """Test tensor copy bandwidth between different memory types"""
    if not torch.cuda.is_available():
        return {}
    
    sizes_mb = [10, 50, 100, 250, 500]  # MB
    results = {
        'sizes_mb': sizes_mb,
        'cpu_to_gpu_gbps': [],
        'gpu_to_cpu_gbps': [],
        'gpu_to_gpu_gbps': []
    }
    
    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024 * 1024)
        num_elements = size_bytes // 4  # 4 bytes per float32
        
        # Create CPU tensor
        cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
        
        # Test CPU to GPU transfer
        start_time = time.perf_counter()
        gpu_tensor = cpu_tensor.cuda()
        cpu_to_gpu_time = time.perf_counter() - start_time
        cpu_to_gpu_bw = size_mb / 1024 / cpu_to_gpu_time  # Convert to GB/s
        
        # Test GPU to GPU transfer (copy)
        gpu_tensor2 = torch.empty_like(gpu_tensor)
        start_time = time.perf_counter()
        gpu_tensor2.copy_(gpu_tensor)
        gpu_to_gpu_time = time.perf_counter() - start_time
        gpu_to_gpu_bw = size_mb / 1024 / gpu_to_gpu_time  # Convert to GB/s
        
        # Test GPU to CPU transfer
        start_time = time.perf_counter()
        cpu_tensor2 = gpu_tensor.cpu()
        gpu_to_cpu_time = time.perf_counter() - start_time
        gpu_to_cpu_bw = size_mb / 1024 / gpu_to_cpu_time  # Convert to GB/s
        
        results['cpu_to_gpu_gbps'].append(cpu_to_gpu_bw)
        results['gpu_to_cpu_gbps'].append(gpu_to_cpu_bw)
        results['gpu_to_gpu_gbps'].append(gpu_to_gpu_bw)
    
    avg_cpu_to_gpu = np.mean(results['cpu_to_gpu_gbps'])
    avg_gpu_to_cpu = np.mean(results['gpu_to_cpu_gbps'])
    avg_gpu_to_gpu = np.mean(results['gpu_to_gpu_gbps'])
    
    print(f"CPU to GPU bandwidth: {avg_cpu_to_gpu:.2f} GB/s")
    print(f"GPU to CPU bandwidth: {avg_gpu_to_cpu:.2f} GB/s")
    print(f"GPU to GPU bandwidth: {avg_gpu_to_gpu:.2f} GB/s")
    
    return results


def test_allocation_bandwidth():
    """Test memory allocation bandwidth"""
    sizes_mb = [10, 50, 100, 200, 500]  # MB
    results = {'sizes_mb': sizes_mb, 'allocation_rates': []}
    
    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024 * 1024)
        num_elements = size_bytes // 4  # 4 bytes per float32
        
        # Measure allocation rate
        start_time = time.perf_counter()
        tensor = torch.empty(num_elements, dtype=torch.float32)
        alloc_time = time.perf_counter() - start_time
        
        # Calculate allocation rate (GB/s)
        alloc_rate = size_mb / 1024 / alloc_time  # Convert to GB/s
        
        results['allocation_rates'].append(alloc_rate)
    
    avg_alloc_rate = np.mean(results['allocation_rates'])
    print(f"Allocation bandwidth: {avg_alloc_rate:.2f} GB/s")
    
    return results


def simulate_transformer_memory_access():
    """Simulate memory access patterns typical in transformer operations"""
    print("Simulating transformer memory access patterns...")
    
    # Simulate attention computation memory access
    batch_size, seq_len, hidden_dim = 1, 512, 4096
    results = {}
    
    # Q, K, V matrices
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim) 
    v = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Attention scores computation (Q @ K^T)
    start_time = time.perf_counter()
    attn_scores = torch.bmm(q, k.transpose(-2, -1))  # Batch matrix multiplication
    attn_time = time.perf_counter() - start_time
    
    # Memory accessed: Q, K, output (attn_scores)
    memory_accessed = (q.numel() + k.numel() + attn_scores.numel()) * 4  # 4 bytes per float32
    memory_gb = memory_accessed / 1024**3
    bandwidth = memory_gb / attn_time
    
    results['attention_bandwidth_gbps'] = bandwidth
    results['attention_computation_time_ms'] = attn_time * 1000
    
    print(f"Attention computation bandwidth: {bandwidth:.2f} GB/s")
    print(f"Attention computation time: {attn_time * 1000:.2f} ms")
    
    # Simulate FFN computation memory access
    ffn_input = torch.randn(batch_size, seq_len, hidden_dim)
    ffn_intermediate = torch.randn(batch_size, seq_len, 11008)  # FFN intermediate size
    
    start_time = time.perf_counter()
    # Simulate linear projection + activation
    ffn_output = torch.relu(torch.randn(batch_size, seq_len, 11008)) @ torch.randn(11008, hidden_dim)
    ffn_time = time.perf_counter() - start_time
    
    # Memory accessed: input, intermediate weights, output
    ffn_memory_accessed = (ffn_input.numel() + ffn_intermediate.numel() + ffn_output.numel()) * 4
    ffn_memory_gb = ffn_memory_accessed / 1024**3
    ffn_bandwidth = ffn_memory_gb / ffn_time
    
    results['ffn_bandwidth_gbps'] = ffn_bandwidth
    results['ffn_computation_time_ms'] = ffn_time * 1000
    
    print(f"FFN computation bandwidth: {ffn_bandwidth:.2f} GB/s")
    print(f"FFN computation time: {ffn_time * 1000:.2f} ms")
    
    return results


if __name__ == "__main__":
    print("Starting memory bandwidth utilization benchmarking...")
    
    # Run comprehensive bandwidth benchmark
    bandwidth_results = benchmark_memory_bandwidth()
    
    # Simulate transformer memory access patterns
    transformer_results = simulate_transformer_memory_access()
    
    print("\nMemory bandwidth utilization benchmarking completed!")