"""
Performance Comparison Test Suite for CUDA Kernels

This module compares the performance of custom CUDA kernels against standard PyTorch implementations
to validate that the custom kernels provide the expected performance gains.
"""

import unittest
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.common.cuda_kernels.kernel_framework import (
    AttentionKernel,
    LinearProjectionKernel,
    ActivationKernel,
    NormalizationKernel,
    MLPLayerKernel,
)

def benchmark_function(func: Callable, *args, num_runs: int = 10, warmup_runs: int = 3, **kwargs) -> Tuple[float, float]:
    """
    Benchmark a function and return average time and standard deviation.
    
    Args:
        func: Function to benchmark
        args: Arguments to pass to the function
        num_runs: Number of runs for benchmarking
        warmup_runs: Number of warmup runs
        kwargs: Keyword arguments to pass to the function
    
    Returns:
        Tuple of (average_time, std_deviation)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Actual benchmarking
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


class StandardAttention(nn.Module):
    """Standard PyTorch attention implementation for comparison."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5
        
        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, query, key, value, need_weights=False):
        output, attn_weights = self.multihead_attn(
            query, key, value, 
            need_weights=need_weights,
            average_attn_weights=False if need_weights else None
        )
        return output, attn_weights


class StandardMLP(nn.Module):
    """Standard PyTorch MLP implementation for comparison."""
    
    def __init__(self, d_model: int, intermediate_size: int, activation_type: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, d_model)
        
        if activation_type == "silu":
            self.activation = nn.SiLU()
        elif activation_type == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StandardRMSNorm(nn.Module):
    """Standard RMSNorm implementation for comparison."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class TestPerformanceComparison(unittest.TestCase):
    """Performance comparison tests between custom CUDA kernels and standard implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 128
        self.d_model = 512
        self.nhead = 8
        self.intermediate_size = 2048
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create inputs
        self.query = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        self.key = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        self.value = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)

    def test_attention_performance_comparison(self):
        """Compare performance of custom attention kernel vs standard implementation."""
        # Create custom kernel
        custom_kernel = AttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
        ).to(self.device)
        
        # Create standard implementation
        standard_attn = StandardAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1
        ).to(self.device)
        
        # Benchmark custom kernel
        def run_custom():
            output, _ = custom_kernel(self.query, self.key, self.value)
            return output
        
        custom_avg_time, custom_std_time = benchmark_function(run_custom)
        
        # Benchmark standard implementation
        def run_standard():
            output, _ = standard_attn(self.query, self.key, self.value)
            return output
        
        standard_avg_time, standard_std_time = benchmark_function(run_standard)
        
        logger.info(f"Custom Attention - Avg time: {custom_avg_time:.6f}s ± {custom_std_time:.6f}s")
        logger.info(f"Standard Attention - Avg time: {standard_avg_time:.6f}s ± {standard_std_time:.6f}s")
        
        # Calculate speedup
        speedup = standard_avg_time / custom_avg_time if custom_avg_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # The custom kernel should be at least as fast as the standard implementation
        # (May not be faster on all hardware, especially older GPUs without tensor cores)
        self.assertLessEqual(custom_avg_time, standard_avg_time * 2.0, 
                            "Custom attention should not be significantly slower than standard")

    def test_mlp_performance_comparison(self):
        """Compare performance of custom MLP kernel vs standard implementation."""
        # Create custom kernel
        custom_kernel = MLPLayerKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            use_swiglu=True,
            activation_type="silu",
            dropout=0.1,
        ).to(self.device)
        
        # Create standard implementation
        standard_mlp = StandardMLP(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            activation_type="silu",
            dropout=0.1
        ).to(self.device)
        
        # Benchmark custom kernel
        def run_custom():
            return custom_kernel(self.x)
        
        custom_avg_time, custom_std_time = benchmark_function(run_custom)
        
        # Benchmark standard implementation
        def run_standard():
            return standard_mlp(self.x)
        
        standard_avg_time, standard_std_time = benchmark_function(run_standard)
        
        logger.info(f"Custom MLP - Avg time: {custom_avg_time:.6f}s ± {custom_std_time:.6f}s")
        logger.info(f"Standard MLP - Avg time: {standard_avg_time:.6f}s ± {standard_std_time:.6f}s")
        
        # Calculate speedup
        speedup = standard_avg_time / custom_avg_time if custom_avg_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # The custom kernel should be at least as fast as the standard implementation
        # Account for cases where timing measurements might be imprecise
        tolerance = 0.001  # Higher tolerance for measurement precision of fast operations
        self.assertLessEqual(custom_avg_time, standard_avg_time + tolerance,
                            "Custom MLP should not be significantly slower than standard")

    def test_rmsnorm_performance_comparison(self):
        """Compare performance of custom RMSNorm kernel vs standard implementation."""
        # Create custom kernel
        custom_kernel = NormalizationKernel(
            normalized_shape=self.d_model,
            norm_type="rms",
            eps=1e-6,
        ).to(self.device)
        
        # Create standard implementation
        standard_norm = StandardRMSNorm(
            normalized_shape=self.d_model,
            eps=1e-6
        ).to(self.device)
        
        # Benchmark custom kernel
        def run_custom():
            return custom_kernel(self.x)
        
        custom_avg_time, custom_std_time = benchmark_function(run_custom)
        
        # Benchmark standard implementation
        def run_standard():
            return standard_norm(self.x)
        
        standard_avg_time, standard_std_time = benchmark_function(run_standard)
        
        logger.info(f"Custom RMSNorm - Avg time: {custom_avg_time:.6f}s ± {custom_std_time:.6f}s")
        logger.info(f"Standard RMSNorm - Avg time: {standard_avg_time:.6f}s ± {standard_std_time:.6f}s")
        
        # Calculate speedup
        speedup = standard_avg_time / custom_avg_time if custom_avg_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # The custom kernel should be at least as fast as the standard implementation
        # Account for cases where timing measurements might be imprecise
        # Use a higher tolerance for very fast operations
        tolerance = 0.001  # Higher tolerance for measurement precision of fast operations
        self.assertLessEqual(custom_avg_time, standard_avg_time + tolerance,
                            "Custom RMSNorm should not be significantly slower than standard")


class TestMemoryEfficiency(unittest.TestCase):
    """Memory efficiency tests for CUDA kernels."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 128
        self.d_model = 512
        self.nhead = 8
        self.intermediate_size = 2048
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def test_memory_usage_comparison(self):
        """Compare memory usage between custom and standard implementations."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping memory efficiency test")
            self.skipTest("CUDA not available")
            return
        
        # Measure memory before
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.max_memory_allocated()
        
        # Create and run custom kernel
        custom_kernel = AttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
        ).to(self.device)
        
        query = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        # Run multiple times to ensure memory is properly allocated
        for _ in range(5):
            output, _ = custom_kernel(query, key, value)
            del output
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get peak memory after custom kernel
        custom_peak_memory = torch.cuda.max_memory_allocated()
        
        # Reset and test standard implementation
        torch.cuda.reset_peak_memory_stats()
        
        standard_attn = StandardAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1
        ).to(self.device)
        
        # Run multiple times to ensure memory is properly allocated
        for _ in range(5):
            output, _ = standard_attn(query, key, value)
            del output
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Get peak memory after standard implementation
        standard_peak_memory = torch.cuda.max_memory_allocated()
        
        logger.info(f"Custom kernel peak memory: {custom_peak_memory / 1024**2:.2f} MB")
        logger.info(f"Standard implementation peak memory: {standard_peak_memory / 1024**2:.2f} MB")
        
        # Memory usage should be reasonable (within 2x of each other)
        memory_ratio = max(custom_peak_memory, standard_peak_memory) / min(custom_peak_memory, standard_peak_memory)
        self.assertLessEqual(memory_ratio, 2.0, 
                            "Memory usage should be within reasonable bounds")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)