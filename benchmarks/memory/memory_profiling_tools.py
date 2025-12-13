"""
Memory Profiling Tools for Qwen3-VL Architecture
Used for Phase 2.9: Memory Pooling and Pre-allocation Techniques
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
from collections import defaultdict
import psutil
import os
from typing import Dict, List, Tuple, Optional


class MemoryProfiler:
    """
    Comprehensive memory profiler for analyzing allocation patterns, fragmentation,
    and overhead in the Qwen3-VL architecture.
    """
    
    def __init__(self):
        self.allocations = []
        self.deallocations = []
        self.tensor_sizes = []
        self.memory_snapshots = []
        self.fragmentation_data = []
        
    def capture_memory_snapshot(self, label: str = "") -> Dict:
        """Capture a comprehensive memory snapshot"""
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'cpu_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'gpu_memory_allocated': 0,
            'gpu_memory_reserved': 0,
            'gpu_memory_max_allocated': 0,
            'gpu_memory_max_reserved': 0
        }
        
        if torch.cuda.is_available():
            snapshot.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'gpu_memory_max_reserved': torch.cuda.max_memory_reserved() / 1024 / 1024
            })
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def profile_tensor_allocation(self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> Dict:
        """Profile allocation of a tensor with given size and dtype"""
        size_bytes = np.prod(size) * torch.tensor([], dtype=dtype).element_size()
        
        # Capture memory before allocation
        before_snapshot = self.capture_memory_snapshot(f"before_alloc_{size}")
        
        # Perform allocation
        tensor = torch.empty(size, dtype=dtype)
        
        # Capture memory after allocation
        after_snapshot = self.capture_memory_snapshot(f"after_alloc_{size}")
        
        # Calculate allocation overhead
        gpu_allocated_diff = after_snapshot['gpu_memory_allocated'] - before_snapshot['gpu_memory_allocated']
        cpu_allocated_diff = after_snapshot['cpu_memory'] - before_snapshot['cpu_memory']
        
        allocation_info = {
            'size': size,
            'dtype': dtype,
            'size_bytes': size_bytes,
            'gpu_memory_allocated_diff': gpu_allocated_diff,
            'cpu_memory_allocated_diff': cpu_allocated_diff,
            'allocation_overhead': (gpu_allocated_diff * 1024 * 1024 - size_bytes) if gpu_allocated_diff > 0 else 0,
            'tensor_ref': tensor
        }
        
        self.allocations.append(allocation_info)
        self.tensor_sizes.append(size_bytes)
        
        return allocation_info
    
    def profile_tensor_deallocation(self, tensor: torch.Tensor) -> Dict:
        """Profile deallocation of a tensor"""
        size_bytes = tensor.numel() * tensor.element_size()
        
        # Capture memory before deallocation
        before_snapshot = self.capture_memory_snapshot(f"before_dealloc_{tensor.size()}")
        
        # Delete tensor and run garbage collection
        del tensor
        gc.collect()
        
        # Capture memory after deallocation
        after_snapshot = self.capture_memory_snapshot(f"after_dealloc_{size_bytes}")
        
        deallocation_info = {
            'size_bytes': size_bytes,
            'gpu_memory_freed': before_snapshot['gpu_memory_allocated'] - after_snapshot['gpu_memory_allocated'],
            'cpu_memory_freed': before_snapshot['cpu_memory'] - after_snapshot['cpu_memory']
        }
        
        self.deallocations.append(deallocation_info)
        return deallocation_info
    
    def measure_fragmentation(self) -> Dict:
        """Measure memory fragmentation in GPU memory"""
        if not torch.cuda.is_available():
            return {'fragmentation_ratio': 0.0, 'note': 'CUDA not available'}
        
        # Calculate fragmentation ratio
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        fragmentation_ratio = 0.0
        if reserved > 0:
            fragmentation_ratio = (reserved - allocated) / reserved
        
        fragmentation_info = {
            'allocated_memory': allocated,
            'reserved_memory': reserved,
            'fragmentation_ratio': fragmentation_ratio,
            'fragmentation_bytes': reserved - allocated
        }
        
        self.fragmentation_data.append(fragmentation_info)
        return fragmentation_info
    
    def analyze_allocation_patterns(self) -> Dict:
        """Analyze tensor allocation patterns"""
        if not self.tensor_sizes:
            return {'error': 'No tensor allocation data available'}
        
        sizes_array = np.array(self.tensor_sizes)
        
        analysis = {
            'total_allocations': len(self.tensor_sizes),
            'total_size_bytes': np.sum(sizes_array),
            'mean_size_bytes': np.mean(sizes_array),
            'median_size_bytes': np.median(sizes_array),
            'std_size_bytes': np.std(sizes_array),
            'min_size_bytes': np.min(sizes_array),
            'max_size_bytes': np.max(sizes_array),
            'size_distribution': np.histogram(sizes_array, bins=20)
        }
        
        return analysis
    
    def calculate_allocation_overhead(self) -> Dict:
        """Calculate tensor allocation overhead statistics"""
        if not self.allocations:
            return {'error': 'No allocation data available'}
        
        overheads = [alloc['allocation_overhead'] for alloc in self.allocations if 'allocation_overhead' in alloc]
        
        if not overheads:
            return {'error': 'No overhead data available'}
        
        overhead_array = np.array(overheads)
        
        overhead_stats = {
            'mean_overhead_bytes': np.mean(overhead_array),
            'total_overhead_bytes': np.sum(overhead_array),
            'overhead_percentage': np.mean(overhead_array) / np.mean([alloc['size_bytes'] for alloc in self.allocations]) * 100 if self.allocations else 0,
            'max_overhead_bytes': np.max(overhead_array),
            'min_overhead_bytes': np.min(overhead_array)
        }
        
        return overhead_stats
    
    def get_memory_bandwidth_usage(self) -> Dict:
        """Estimate memory bandwidth usage based on allocation/deallocation patterns"""
        # This is a simplified estimation - in practice, you'd use more detailed profiling
        total_allocations = len(self.allocations)
        total_deallocations = len(self.deallocations)
        
        if not self.tensor_sizes:
            return {'total_data_transferred_gb': 0.0, 'estimated_bandwidth_usage': 0.0}
        
        total_size_bytes = sum(self.tensor_sizes)
        
        # Estimate bandwidth usage (simplified calculation)
        estimated_gb = total_size_bytes / (1024**3)
        
        return {
            'total_allocations': total_allocations,
            'total_deallocations': total_deallocations,
            'total_data_transferred_gb': estimated_gb,
            'estimated_bandwidth_usage': estimated_gb  # Placeholder - in practice, you'd measure actual transfer rates
        }
    
    def reset(self):
        """Reset all profiling data"""
        self.allocations = []
        self.deallocations = []
        self.tensor_sizes = []
        self.memory_snapshots = []
        self.fragmentation_data = []


def profile_memory_allocation_patterns():
    """Profile current memory allocation patterns and fragmentation"""
    profiler = MemoryProfiler()
    
    print("Starting memory allocation pattern profiling...")
    
    # Capture initial memory state
    profiler.capture_memory_snapshot("initial_state")
    
    # Profile various tensor sizes commonly used in transformer models
    common_tensor_sizes = [
        (1, 512, 4096),      # Attention output
        (1, 512, 512),       # Attention weight matrix
        (1, 8, 512, 512),    # Multi-head attention weight matrix
        (1, 512, 11008),     # FFN intermediate
        (1, 11008, 4096),    # FFN output
        (1, 512, 4096),      # KV cache (key/values)
        (1, 3, 224, 224),    # Vision input
        (1, 576, 4096),      # Patch embeddings
        (1, 4096),           # Layer norm
        (4096, 4096),        # Linear projection
        (4096, 11008),       # FFN expansion
        (11008, 4096),       # FFN compression
    ]
    
    print(f"Profiling {len(common_tensor_sizes)} common tensor sizes...")
    
    # Profile tensor allocations
    for i, size in enumerate(common_tensor_sizes):
        print(f"Profiling tensor size {i+1}/{len(common_tensor_sizes)}: {size}")
        profiler.profile_tensor_allocation(size, dtype=torch.float32)
    
    # Profile fragmentation
    fragmentation_info = profiler.measure_fragmentation()
    print(f"Fragmentation ratio: {fragmentation_info.get('fragmentation_ratio', 0):.4f}")
    
    # Analyze allocation patterns
    allocation_analysis = profiler.analyze_allocation_patterns()
    print(f"Total allocations: {allocation_analysis.get('total_allocations', 0)}")
    print(f"Mean tensor size: {allocation_analysis.get('mean_size_bytes', 0) / 1024 / 1024:.2f} MB")
    
    # Calculate allocation overhead
    overhead_stats = profiler.calculate_allocation_overhead()
    print(f"Mean allocation overhead: {overhead_stats.get('mean_overhead_bytes', 0):.2f} bytes")
    print(f"Overhead percentage: {overhead_stats.get('overhead_percentage', 0):.2f}%")
    
    # Get memory bandwidth usage
    bandwidth_usage = profiler.get_memory_bandwidth_usage()
    print(f"Estimated data transferred: {bandwidth_usage.get('total_data_transferred_gb', 0):.2f} GB")
    
    # Capture final memory state
    profiler.capture_memory_snapshot("final_state")
    
    return profiler


if __name__ == "__main__":
    profiler = profile_memory_allocation_patterns()
    print("\nMemory profiling completed successfully!")