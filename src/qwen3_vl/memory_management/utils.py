"""
Memory Management Utilities for Qwen3-VL Model
Consolidated module containing utility functions and helper classes for memory management.
"""

import torch
import torch.nn as nn
import numpy as np
import psutil
import time
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict
import logging
import threading
from dataclasses import dataclass
from enum import Enum
import math


class MemoryAccessPattern(Enum):
    """Types of memory access patterns"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    COALESCED = "coalesced"
    STRIDED = "strided"


class MemoryAccessPatternOptimizer:
    """Optimizer for different memory access patterns"""
    
    def __init__(self):
        self.pattern_stats = defaultdict(lambda: {'count': 0, 'avg_time': 0.0})
        self.lock = threading.Lock()

    def record_access_pattern(self, pattern: MemoryAccessPattern, access_time: float):
        """Record an access pattern and its performance"""
        with self.lock:
            stats = self.pattern_stats[pattern.value]
            stats['count'] += 1
            # Update average time using running average
            stats['avg_time'] = (stats['avg_time'] * (stats['count'] - 1) + access_time) / stats['count']

    def get_optimal_pattern_for_shape(self, shape: Tuple[int, ...]) -> MemoryAccessPattern:
        """Determine the optimal access pattern for a given tensor shape"""
        # For larger tensors, coalesced access is often optimal
        if len(shape) >= 2 and shape[-1] >= 32:  # Last dimension is often the one accessed in parallel
            return MemoryAccessPattern.COALESCED
        elif len(shape) == 1:
            # For 1D tensors, sequential is often best
            return MemoryAccessPattern.SEQUENTIAL
        else:
            # For other shapes, use random as default
            return MemoryAccessPattern.RANDOM

    def get_pattern_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all access patterns"""
        with self.lock:
            return dict(self.pattern_stats)


class MemoryLayoutOptimizer:
    """Optimizer for memory layouts based on tensor shapes and access patterns"""
    
    def __init__(self):
        self.layout_cache = {}
        self.cache_lock = threading.Lock()

    def optimize_layout_for_gpu(self, tensor_shape: Tuple[int, ...], 
                               device: torch.device = torch.device('cuda')) -> torch.memory_format:
        """Optimize memory layout for GPU access patterns"""
        if device.type == 'cuda':
            # For convolution operations, channels_last is often more efficient on NVIDIA GPUs
            if len(tensor_shape) == 4:  # BCHW format
                return torch.channels_last
            else:
                return torch.contiguous_format
        else:
            return torch.contiguous_format

    def align_tensor_for_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Align tensor for optimal GPU memory access"""
        # For NVIDIA GPUs, align to 64-byte boundaries for better memory throughput
        if tensor.device.type == 'cuda':
            # Check if the last dimension is aligned to 32 (warp size) or 64 (cache line)
            last_dim = tensor.shape[-1] if tensor.dim() > 0 else 1
            
            # Align to multiples of 32 for better memory access
            if last_dim % 32 != 0:
                aligned_last_dim = ((last_dim + 31) // 32) * 32
                if len(tensor.shape) == 1:
                    aligned_tensor = torch.empty((aligned_last_dim,), dtype=tensor.dtype, device=tensor.device)
                    aligned_tensor[:last_dim] = tensor
                    return aligned_tensor
                elif len(tensor.shape) == 2:
                    aligned_tensor = torch.empty((tensor.shape[0], aligned_last_dim), 
                                               dtype=tensor.dtype, device=tensor.device)
                    aligned_tensor[:, :last_dim] = tensor
                    return aligned_tensor
                else:
                    # For higher dimensions, align the last dimension
                    new_shape = list(tensor.shape)
                    new_shape[-1] = aligned_last_dim
                    aligned_tensor = torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)
                    index = [slice(None)] * len(new_shape)
                    index[-1] = slice(None, last_dim)
                    aligned_tensor[tuple(index)] = tensor
                    return aligned_tensor
        
        return tensor

    def optimize_for_memory_bandwidth(self, tensor_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Optimize tensor shape for memory bandwidth utilization"""
        # For optimal memory bandwidth, ensure the last dimension is a multiple of 32
        if len(tensor_shape) > 0:
            last_dim = tensor_shape[-1]
            if last_dim % 32 != 0:
                optimized_last_dim = ((last_dim + 31) // 32) * 32
                optimized_shape = list(tensor_shape)
                optimized_shape[-1] = optimized_last_dim
                return tuple(optimized_shape)
        
        return tensor_shape


class TensorMemoryOptimizer:
    """Optimizer for tensor memory usage"""
    
    def __init__(self):
        self.size_cache = {}
        self.cache_lock = threading.Lock()

    def optimize_tensor_allocation(self, shape: Tuple[int, ...], dtype: torch.dtype,
                                 device: torch.device) -> Tuple[Tuple[int, ...], torch.dtype]:
        """Optimize tensor allocation based on size and device"""
        # For large tensors on GPU, consider using half precision to save memory
        if device.type == 'cuda' and dtype == torch.float32:
            total_elements = np.prod(shape)
            if total_elements > 1024 * 1024:  # More than 1M elements
                return shape, torch.float16  # Use half precision for large tensors
        
        return shape, dtype

    def get_optimized_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                           device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Get an optimized tensor based on shape and device"""
        optimized_shape, optimized_dtype = self.optimize_tensor_allocation(shape, dtype, device)
        
        # Create tensor with optimized parameters
        tensor = torch.empty(optimized_shape, dtype=optimized_dtype, device=device)
        
        # If the shape was changed, we need to return the original shape view
        if optimized_shape != shape:
            # Create a view of the original size
            if len(shape) == len(optimized_shape):
                index = [slice(0, s) for s in shape]
                tensor = tensor[tuple(index)]
            else:
                # If dimensions changed, return original shape
                tensor = torch.empty(shape, dtype=dtype, device=device)
        
        return tensor


class MemoryUsageMonitor:
    """Monitor memory usage and provide optimization suggestions"""
    
    def __init__(self):
        self.usage_history = []
        self.history_size = 100
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start monitoring memory usage"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring memory usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.monitoring:
            usage_info = self.get_memory_usage()
            with self.lock:
                self.usage_history.append((time.time(), usage_info))
                if len(self.usage_history) > self.history_size:
                    self.usage_history.pop(0)
            time.sleep(1)  # Monitor every second

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information"""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory = 0
        gpu_memory_used = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_used = torch.cuda.memory_allocated(0)
        
        return {
            'system_memory_percent': system_memory.percent,
            'system_memory_available_gb': system_memory.available / (1024**3),
            'system_memory_used_gb': system_memory.used / (1024**3),
            'gpu_memory_percent': gpu_memory_used / gpu_memory if gpu_memory > 0 else 0,
            'gpu_memory_used_gb': gpu_memory_used / (1024**3),
            'gpu_memory_total_gb': gpu_memory / (1024**3)
        }

    def get_usage_trend(self) -> Dict[str, float]:
        """Get memory usage trend analysis"""
        with self.lock:
            if len(self.usage_history) < 2:
                return {'system_trend': 0.0, 'gpu_trend': 0.0}
            
            # Calculate trends
            first_time, first_usage = self.usage_history[0]
            last_time, last_usage = self.usage_history[-1]
            
            time_diff = last_time - first_time
            if time_diff <= 0:
                return {'system_trend': 0.0, 'gpu_trend': 0.0}
            
            system_trend = (last_usage['system_memory_percent'] - first_usage['system_memory_percent']) / time_diff
            gpu_trend = (last_usage['gpu_memory_percent'] - first_usage['gpu_memory_percent']) / time_diff
            
            return {
                'system_trend': system_trend,
                'gpu_trend': gpu_trend
            }

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on memory usage"""
        usage = self.get_memory_usage()
        suggestions = []
        
        if usage['system_memory_percent'] > 80:
            suggestions.append("System memory usage is high. Consider using memory-efficient data loading or increasing swap space.")
        
        if usage['gpu_memory_percent'] > 80:
            suggestions.append("GPU memory usage is high. Consider using gradient checkpointing, mixed precision, or tensor compression.")
        
        trend = self.get_usage_trend()
        if trend['gpu_trend'] > 0.1:  # Memory usage increasing rapidly
            suggestions.append("GPU memory usage is increasing rapidly. Consider implementing memory pooling or early defragmentation.")
        
        return suggestions


class HardwareSpecificOptimizer:
    """Optimizer tailored for specific hardware configurations"""
    
    def __init__(self, compute_capability: Tuple[int, int] = (6, 1),  # SM61
                 memory_size_gb: float = 8.0):
        self.compute_capability = compute_capability
        self.memory_size_gb = memory_size_gb
        
        # Based on compute capability, set hardware-specific optimizations
        self.shared_memory_per_block = 48 * 1024 if compute_capability >= (6, 0) else 49152  # 48KB for SM61
        self.max_threads_per_block = 1024
        self.warp_size = 32
        
        # Memory access optimization based on hardware
        self.memory_access_optimizer = MemoryAccessPatternOptimizer()
        self.layout_optimizer = MemoryLayoutOptimizer()

    def get_optimal_block_size(self, tensor_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        """Get optimal block size for CUDA kernels based on tensor shape and hardware"""
        # For SM61 architecture, optimize for warp size and shared memory constraints
        if len(tensor_shape) >= 2:
            # Use 2D block for 2D+ tensors
            max_block_size = int(math.sqrt(self.max_threads_per_block))
            height = min(tensor_shape[-2], max_block_size) if len(tensor_shape) >= 2 else 1
            width = min(tensor_shape[-1], max_block_size) if len(tensor_shape) >= 1 else 1
            depth = 1
            
            # Ensure block size is a multiple of warp size for optimal performance
            height = ((height + self.warp_size - 1) // self.warp_size) * self.warp_size
            width = ((width + self.warp_size - 1) // self.warp_size) * self.warp_size
            
            # Clamp to max thread count
            while height * width > self.max_threads_per_block:
                if height > width:
                    height //= 2
                else:
                    width //= 2
            
            return (height, width, depth)
        else:
            # For 1D tensors, use 1D block
            block_size = min(tensor_shape[0] if tensor_shape else 1, self.max_threads_per_block)
            block_size = ((block_size + self.warp_size - 1) // self.warp_size) * self.warp_size
            return (block_size, 1, 1)

    def optimize_for_hardware(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply hardware-specific optimizations to a tensor"""
        # Align tensor for optimal memory access on this hardware
        aligned_tensor = self.layout_optimizer.align_tensor_for_gpu(tensor)
        
        # Choose appropriate memory format based on hardware and tensor shape
        memory_format = self.layout_optimizer.optimize_layout_for_gpu(
            aligned_tensor.shape, aligned_tensor.device
        )
        
        # Apply memory format if appropriate
        if memory_format == torch.channels_last and aligned_tensor.dim() == 4:
            aligned_tensor = aligned_tensor.to(memory_format=memory_format)
        
        return aligned_tensor


class MemoryEfficiencyAnalyzer:
    """Analyzer for memory efficiency metrics"""
    
    def __init__(self):
        self.allocation_stats = defaultdict(int)
        self.tensor_sizes = []
        self.compression_stats = {
            'total_compressed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0
        }

    def record_tensor_allocation(self, shape: Tuple[int, ...], dtype: torch.dtype):
        """Record tensor allocation for efficiency analysis"""
        size_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        self.tensor_sizes.append(size_bytes)
        self.allocation_stats[dtype] += 1

    def record_compression(self, original_size: int, compressed_size: int):
        """Record compression statistics"""
        self.compression_stats['total_compressed'] += 1
        self.compression_stats['total_original_size'] += original_size
        self.compression_stats['total_compressed_size'] += compressed_size

    def get_efficiency_report(self) -> Dict[str, Any]:
        """Get memory efficiency report"""
        if not self.tensor_sizes:
            return {'message': 'No tensor allocation data available'}
        
        avg_tensor_size = sum(self.tensor_sizes) / len(self.tensor_sizes)
        largest_tensor = max(self.tensor_sizes) if self.tensor_sizes else 0
        smallest_tensor = min(self.tensor_sizes) if self.tensor_sizes else 0
        
        compression_ratio = (
            self.compression_stats['total_compressed_size'] / self.compression_stats['total_original_size']
            if self.compression_stats['total_original_size'] > 0 else 1.0
        )
        
        return {
            'average_tensor_size_bytes': avg_tensor_size,
            'largest_tensor_bytes': largest_tensor,
            'smallest_tensor_bytes': smallest_tensor,
            'allocation_count_by_dtype': dict(self.allocation_stats),
            'compression_ratio': compression_ratio,
            'memory_saved_by_compression_bytes': 
                self.compression_stats['total_original_size'] - self.compression_stats['total_compressed_size'],
            'total_compression_operations': self.compression_stats['total_compressed']
        }


# Global instances for common use
_global_access_pattern_optimizer = MemoryAccessPatternOptimizer()
_global_layout_optimizer = MemoryLayoutOptimizer()
_global_tensor_optimizer = TensorMemoryOptimizer()
_global_memory_monitor = MemoryUsageMonitor()
_global_hardware_optimizer = HardwareSpecificOptimizer()
_global_efficiency_analyzer = MemoryEfficiencyAnalyzer()


def get_access_pattern_optimizer() -> MemoryAccessPatternOptimizer:
    """Get the global access pattern optimizer"""
    return _global_access_pattern_optimizer


def get_layout_optimizer() -> MemoryLayoutOptimizer:
    """Get the global layout optimizer"""
    return _global_layout_optimizer


def get_tensor_optimizer() -> TensorMemoryOptimizer:
    """Get the global tensor optimizer"""
    return _global_tensor_optimizer


def get_memory_monitor() -> MemoryUsageMonitor:
    """Get the global memory monitor"""
    return _global_memory_monitor


def get_hardware_optimizer() -> HardwareSpecificOptimizer:
    """Get the global hardware optimizer"""
    return _global_hardware_optimizer


def get_efficiency_analyzer() -> MemoryEfficiencyAnalyzer:
    """Get the global efficiency analyzer"""
    return _global_efficiency_analyzer


def optimize_tensor_for_hardware(tensor: torch.Tensor) -> torch.Tensor:
    """Apply hardware-specific optimizations to a tensor"""
    hw_optimizer = get_hardware_optimizer()
    return hw_optimizer.optimize_for_hardware(tensor)


def record_tensor_allocation(shape: Tuple[int, ...], dtype: torch.dtype):
    """Record tensor allocation for efficiency analysis"""
    analyzer = get_efficiency_analyzer()
    analyzer.record_tensor_allocation(shape, dtype)


def get_memory_efficiency_report() -> Dict[str, Any]:
    """Get the global memory efficiency report"""
    analyzer = get_efficiency_analyzer()
    return analyzer.get_efficiency_report()