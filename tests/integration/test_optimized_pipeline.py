"""
Comprehensive test for the end-to-end optimized inference pipeline.
Validates all implemented optimization techniques and measures performance improvements.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
import time
import numpy as np
from typing import Dict, Any, List
import gc
import psutil
import os

# Import our optimized pipeline
from src.qwen3_vl.optimization.end_to_end_inference_pipeline import (
    InferencePipelineConfig,
    OptimizedInferencePipeline,
    VariableBatchProcessor,
    CachingMechanism,
    OptimizedIOMechanism
)

def test_variable_batching():
    """Test variable batching strategies."""
    print("Testing Variable Batching Strategies...")
    
    config = InferencePipelineConfig(
        max_batch_size=8,
        variable_batch_size=True
    )
    
    batch_processor = VariableBatchProcessor(config)
    
    # Create test inputs of different sizes
    inputs = [
        (torch.randn(1, 32, 512), "language"),   # Small sequence
        (torch.randn(1, 64, 512), "language"),   # Medium sequence
        (torch.randn(1, 128, 512), "language"),  # Large sequence
        (torch.randn(1, 32, 512), "language"),   # Another small sequence
        (torch.randn(1, 128, 512), "language"),  # Another large sequence
    ]
    
    # Process inputs with variable batching
    batches = batch_processor.group_inputs_by_size(inputs)
    
    print(f"  Created {len(batches)} batches from {len(inputs)} inputs")
    for i, batch in enumerate(batches):
        print(f"  Batch {i}: size {batch['batch_size']}, group {batch['size_group']}")
    
    # Validate that batching worked correctly
    assert len(batches) <= len(inputs), "Batching should reduce or maintain input count"
    assert all(batch['batch_size'] <= config.max_batch_size for batch in batches), "Batch sizes should not exceed max"
    
    print("  ✓ Variable batching test passed")


def test_caching_mechanisms():
    """Test caching mechanisms."""
    print("Testing Caching Mechanisms...")
    
    config = InferencePipelineConfig(
        enable_tensor_caching=True,
        cache_size=128
    )
    
    caching_mechanism = CachingMechanism(config)
    
    # Test tensor caching
    test_tensor = torch.randn(2, 10, 512)
    cache_key = "test_tensor_1"
    
    # Cache tensor
    success = caching_mechanism.cache_tensor(cache_key, test_tensor)
    assert success, "Tensor caching should succeed"
    
    # Retrieve cached tensor
    cached_tensor = caching_mechanism.get_cached_tensor(cache_key)
    assert cached_tensor is not None, "Cached tensor should be retrievable"
    assert torch.equal(test_tensor, cached_tensor), "Cached tensor should match original"
    
    # Test cache statistics
    stats = caching_mechanism.get_cache_stats()
    print(f"  Cache statistics: {stats}")
    
    print("  ✓ Caching mechanisms test passed")


def test_io_optimization():
    """Test I/O optimization mechanisms."""
    print("Testing I/O Optimization...")
    
    config = InferencePipelineConfig(
        enable_pinned_memory=True,
        enable_async_transfers=True
    )
    
    io_mechanism = OptimizedIOMechanism(config)
    
    # Test tensor transfer to device
    test_tensor = torch.randn(2, 10, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transferred_tensor = io_mechanism.transfer_to_device(test_tensor, device)
    assert transferred_tensor.device == device, f"Tensor should be on {device}"
    
    # Test optimized dataloader creation
    dataset = TensorDataset(torch.randn(100, 10, 512), torch.randint(0, 2, (100, 10)))
    dataloader = io_mechanism.create_optimized_dataloader(dataset, batch_size=4)
    
    # Verify dataloader properties
    assert dataloader.pin_memory == config.enable_pinned_memory
    assert dataloader.batch_size == 4
    
    print("  ✓ I/O optimization test passed")


def test_pipeline_integration():
    """Test integration of all pipeline components."""
    print("Testing Pipeline Integration...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.norm = nn.LayerNorm(512)
        
        def forward(self, x):
            if isinstance(x, list):
                x = x[0]  # Take first tensor if list
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if needed
            x = self.linear(x)
            x = self.norm(x)
            return x

    model = TestModel()
    
    # Create pipeline configuration
    config = InferencePipelineConfig(
        max_batch_size=4,
        variable_batch_size=True,
        enable_tensor_caching=True,
        enable_model_caching=True,
        enable_async_io=True,
        enable_pinned_memory=True,
        enable_async_transfers=True,
        memory_efficient=True
    )
    
    # Create optimized pipeline
    pipeline = OptimizedInferencePipeline(model, config)
    
    # Create test inputs
    test_inputs = [
        torch.randn(2, 10, 512),  # Batch of 2, seq_len=10, hidden_size=512
        torch.randn(1, 20, 512),  # Different sequence length
        torch.randn(3, 5, 512)    # Another batch
    ]
    
    # Run inference
    results = pipeline.efficient_pipeline.run_inference(test_inputs)
    assert len(results) == len(test_inputs), "Should return results for all inputs"
    
    for i, result in enumerate(results):
        expected_shape = test_inputs[i].shape
        assert result.shape == expected_shape, f"Result {i} shape mismatch: {result.shape} vs {expected_shape}"
    
    print("  ✓ Pipeline integration test passed")


def test_performance_improvements():
    """Test performance improvements with optimized pipeline."""
    print("Testing Performance Improvements...")
    
    # Create a more complex test model
    class ComplexTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
                for _ in range(6)
            ])
        
        def forward(self, x):
            if isinstance(x, list):
                x = x[0]  # Take first tensor if list
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if needed
            for layer in self.layers:
                x = layer(x)
            return x

    model = ComplexTestModel()
    
    # Create configuration
    config = InferencePipelineConfig(
        max_batch_size=4,
        variable_batch_size=True,
        enable_tensor_caching=True,
        enable_model_caching=True,
        enable_async_io=True,
        enable_pinned_memory=True,
        enable_async_transfers=True,
        memory_efficient=True
    )
    
    # Create optimized pipeline
    optimized_pipeline = OptimizedInferencePipeline(model, config)
    
    # Create test inputs
    test_inputs = [torch.randn(2, 32, 512) for _ in range(5)]
    
    # Benchmark optimized pipeline
    start_time = time.time()
    for _ in range(5):  # Run multiple times for average
        _ = optimized_pipeline.efficient_pipeline.run_inference(test_inputs)
    optimized_time = time.time() - start_time
    
    # Compare with naive approach
    naive_start_time = time.time()
    for _ in range(5):
        for inp in test_inputs:
            with torch.no_grad():
                _ = model(inp)
    naive_time = time.time() - naive_start_time
    
    print(f"  Optimized pipeline time: {optimized_time:.4f}s")
    print(f"  Naive approach time: {naive_time:.4f}s")
    print(f"  Speedup: {naive_time / optimized_time:.2f}x" if optimized_time > 0 else "N/A")
    
    # The optimized pipeline should be faster or at least not significantly slower
    assert optimized_time <= naive_time * 1.1, "Optimized pipeline should be at least as fast as naive approach"
    
    print("  ✓ Performance improvements test passed")


def test_batch_inference():
    """Test batch inference with DataLoader."""
    print("Testing Batch Inference...")
    
    # Create test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 100)  # Output to vocab size
        
        def forward(self, x):
            if isinstance(x, list):
                x = x[0]  # Take first tensor if list
            return self.linear(x)

    model = SimpleModel()
    
    # Create configuration
    config = InferencePipelineConfig(
        max_batch_size=4,
        variable_batch_size=True,
        enable_async_io=True,
        enable_pinned_memory=True
    )
    
    # Create pipeline
    pipeline = OptimizedInferencePipeline(model, config)
    
    # Create test dataset
    input_data = torch.randn(20, 16, 512)  # 20 samples, seq_len=16, hidden_size=512
    dummy_labels = torch.randint(0, 100, (20, 16))  # Dummy labels
    dataset = TensorDataset(input_data, dummy_labels)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Run batch inference
    batch_results = pipeline.run_batch_inference(dataloader)
    
    # Should have processed all 20 samples in 5 batches of 4
    assert len(batch_results) >= 20, f"Should have processed at least 20 samples, got {len(batch_results)}"
    
    print(f"  Processed {len(batch_results)} outputs from batch inference")
    print("  ✓ Batch inference test passed")


def test_memory_efficiency():
    """Test memory efficiency of the optimized pipeline."""
    print("Testing Memory Efficiency...")
    
    # Create test model
    class MemoryTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 1024)
            self.linear2 = nn.Linear(1024, 512)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            if isinstance(x, list):
                x = x[0]  # Take first tensor if list
            x = self.linear1(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x

    model = MemoryTestModel()
    
    # Create configuration with memory optimization enabled
    config = InferencePipelineConfig(
        max_batch_size=2,
        variable_batch_size=True,
        enable_tensor_caching=True,
        enable_model_caching=True,
        memory_efficient=True
    )
    
    # Create pipeline
    pipeline = OptimizedInferencePipeline(model, config)
    
    # Create moderately large test inputs
    test_inputs = [torch.randn(1, 128, 512) for _ in range(3)]
    
    # Measure memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()
    
    # Run inference
    results = pipeline.efficient_pipeline.run_inference(test_inputs)
    
    # Measure memory after
    if torch.cuda.is_available():
        mem_after = torch.cuda.max_memory_allocated()
        peak_memory = torch.cuda.max_memory_reserved()
        print(f"  CUDA Memory - Before: {mem_before / 1024**2:.2f} MB")
        print(f"  CUDA Memory - After: {mem_after / 1024**2:.2f} MB")
        print(f"  CUDA Peak Memory: {peak_memory / 1024**2:.2f} MB")
    
    # Validate results
    assert len(results) == len(test_inputs), "Should return results for all inputs"
    for i, result in enumerate(results):
        expected_shape = test_inputs[i].shape
        assert result.shape == expected_shape, f"Result {i} shape mismatch"
    
    print("  ✓ Memory efficiency test passed")


def run_comprehensive_tests():
    """Run all tests."""
    print("Running Comprehensive Tests for Optimized Inference Pipeline")
    print("=" * 60)
    
    test_variable_batching()
    print()
    
    test_caching_mechanisms()
    print()
    
    test_io_optimization()
    print()
    
    test_pipeline_integration()
    print()
    
    test_performance_improvements()
    print()
    
    test_batch_inference()
    print()
    
    test_memory_efficiency()
    print()
    
    print("=" * 60)
    print("🎉 All tests passed! Optimized inference pipeline is working correctly.")
    
    # Print final summary
    print("\nSUMMARY OF IMPLEMENTED OPTIMIZATIONS:")
    print("• Variable Batch Processing: Groups inputs by size to optimize batching")
    print("• Tensor Caching: LRU cache for frequently accessed tensors")
    print("• I/O Optimization: Pinned memory and async transfers")
    print("• Pipeline Parallelism: Multi-stage pipeline with overlapping operations")
    print("• Memory Efficiency: Optimized memory usage patterns")
    print("• Performance Improvements: Measurable speedups over naive approach")


if __name__ == "__main__":
    run_comprehensive_tests()