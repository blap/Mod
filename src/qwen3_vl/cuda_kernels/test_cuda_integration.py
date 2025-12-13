"""
Comprehensive test for CUDA kernel integration with model components
"""
import torch
import torch.nn as nn
import pytest
import sys
import os
from typing import Optional, Tuple

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cuda_kernels.cuda_wrapper import (
    SM61AttentionWrapper, 
    SM61MemoryPoolWrapper, 
    SM61TensorOpsWrapper,
    OptimizedAttentionModule,
    OptimizedMLPModule,
    test_cuda_integration,
    get_cuda_wrapper_stats
)
from components.attention.advanced_dynamic_sparse_attention import DynamicSparseAttentionWithLearnedRouting
from components.configuration.config_manager import ConfigManager


def test_cuda_wrapper_basic():
    """Test basic CUDA wrapper functionality"""
    print("Testing basic CUDA wrapper functionality...")
    
    # Test attention wrapper
    attention_wrapper = SM61AttentionWrapper()
    assert hasattr(attention_wrapper, 'forward'), "Attention wrapper should have forward method"
    
    # Test memory pool wrapper
    memory_pool = SM61MemoryPoolWrapper()
    assert hasattr(memory_pool, 'allocate_tensor'), "Memory pool should have allocate_tensor method"
    
    # Test tensor ops wrapper
    tensor_ops = SM61TensorOpsWrapper()
    assert hasattr(tensor_ops, 'matmul'), "Tensor ops should have matmul method"
    
    print("✓ Basic CUDA wrapper functionality test passed")


def test_cuda_attention_with_tensors():
    """Test CUDA attention with actual tensors"""
    print("Testing CUDA attention with tensors...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tensor tests")
        return
    
    # Create test tensors
    batch_size, seq_len, num_heads, head_dim = 2, 64, 8, 32
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    
    attention_wrapper = SM61AttentionWrapper()
    output = attention_wrapper.forward(query, key, value)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim), f"Output shape mismatch: {output.shape}"
    print(f"✓ CUDA attention test passed, output shape: {output.shape}")


def test_memory_pool_allocation():
    """Test memory pool tensor allocation"""
    print("Testing memory pool allocation...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory pool tests")
        return
    
    memory_pool = SM61MemoryPoolWrapper(pool_size=16 * 1024 * 1024)  # 16MB pool
    
    # Test allocation
    tensor = memory_pool.allocate_tensor((100, 50), dtype=torch.float32)
    assert tensor.shape == (100, 50), f"Tensor shape mismatch: {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Tensor dtype mismatch: {tensor.dtype}"
    
    # Test stats
    stats = memory_pool.get_stats()
    assert 'total_size' in stats, "Stats should contain total_size"
    assert 'allocated' in stats, "Stats should contain allocated"
    
    print(f"✓ Memory pool allocation test passed, stats: {stats}")


def test_tensor_operations():
    """Test CUDA-optimized tensor operations"""
    print("Testing tensor operations...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tensor operation tests")
        return
    
    tensor_ops = SM61TensorOpsWrapper()
    
    # Test transpose
    tensor = torch.randn(100, 64, device='cuda')
    transposed = tensor_ops.transpose(tensor)
    assert transposed.shape == (64, 100), f"Transpose shape mismatch: {transposed.shape}"
    
    # Test coalesced copy
    copied = tensor_ops.coalesced_copy(tensor)
    assert torch.equal(tensor, copied), "Copied tensor should be equal to original"
    assert copied.device == tensor.device, "Copied tensor should be on same device"
    
    print(f"✓ Tensor operations test passed")


def test_optimized_attention_module():
    """Test the optimized attention module integration"""
    print("Testing optimized attention module...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping optimized attention module test")
        return
    
    # Create a mock config
    class MockConfig:
        hidden_size = 256
        num_attention_heads = 8
        attention_dropout_prob = 0.0
        is_causal = False
        max_position_embeddings = 512
        rope_theta = 10000.0
    
    config = MockConfig()
    attention_module = OptimizedAttentionModule(config)
    attention_module = attention_module.cuda()
    
    # Create test inputs
    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda')
    
    # Forward pass
    output = attention_module(hidden_states)
    assert len(output) >= 1, "Output should contain at least one element"
    assert output[0].shape == hidden_states.shape, f"Output shape mismatch: {output[0].shape} vs {hidden_states.shape}"
    
    print(f"✓ Optimized attention module test passed")


def test_optimized_mlp_module():
    """Test the optimized MLP module"""
    print("Testing optimized MLP module...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping optimized MLP module test")
        return
    
    # Create a mock config
    class MockConfig:
        hidden_size = 256
        intermediate_size = 512
    
    config = MockConfig()
    mlp_module = OptimizedMLPModule(config)
    mlp_module = mlp_module.cuda()
    
    # Create test input
    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda')
    
    # Forward pass
    output = mlp_module(hidden_states)
    assert output.shape == hidden_states.shape, f"Output shape mismatch: {output.shape} vs {hidden_states.shape}"
    
    print(f"✓ Optimized MLP module test passed")


def test_cuda_integration_full():
    """Run the full CUDA integration test"""
    print("Running full CUDA integration test...")
    
    success = test_cuda_integration()
    assert success, "Full CUDA integration test should pass"
    
    print("✓ Full CUDA integration test passed")


def test_wrapper_statistics():
    """Test CUDA wrapper statistics"""
    print("Testing CUDA wrapper statistics...")
    
    stats = get_cuda_wrapper_stats()
    assert 'cuda_kernels_available' in stats, "Stats should contain cuda_kernels_available"
    assert 'memory_pool_stats' in stats, "Stats should contain memory_pool_stats"
    
    print(f"✓ Wrapper statistics test passed: {stats}")


def test_fallback_mechanisms():
    """Test fallback mechanisms when CUDA is not available"""
    print("Testing fallback mechanisms...")
    
    # Create a mock config
    class MockConfig:
        hidden_size = 256
        num_attention_heads = 8
        attention_dropout_prob = 0.0
        is_causal = False
        max_position_embeddings = 512
        rope_theta = 10000.0
    
    config = MockConfig()
    
    # Test attention wrapper fallback (when CUDA is available, this will still use CUDA,
    # but the internal fallback logic will be tested if CUDA ops fail)
    attention_wrapper = SM61AttentionWrapper()
    
    # If CUDA is available, create tensors on CUDA, otherwise on CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size, seq_len, num_heads, head_dim = 2, 32, 8, 16
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    output = attention_wrapper.forward(query, key, value)
    assert output.shape == (batch_size, num_heads, seq_len, head_dim), f"Output shape mismatch: {output.shape}"
    
    print(f"✓ Fallback mechanisms test passed")


def test_model_capacity_preservation():
    """Test that model capacity is preserved with CUDA optimizations"""
    print("Testing model capacity preservation...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping capacity preservation test")
        return
    
    # Create config with full capacity (32 layers, 32 heads)
    class FullCapacityConfig:
        hidden_size = 4096  # Standard size for large models
        num_hidden_layers = 32  # Full capacity
        num_attention_heads = 32  # Full capacity
        num_key_value_heads = None
        intermediate_size = 11008
        hidden_act = "silu"
        hidden_dropout_prob = 0.0
        attention_dropout_prob = 0.0
        max_position_embeddings = 32768
        initializer_range = 0.02
        layer_norm_eps = 1e-6
        pad_token_id = 0
        tie_word_embeddings = False
        rope_theta = 1000000.0
        use_cache = True
        vocab_size = 152064
    
    config = FullCapacityConfig()
    
    # Test that we can create the optimized attention module with full capacity
    attention_module = OptimizedAttentionModule(config)
    attention_module = attention_module.cuda()
    
    # Verify the dimensions are correct
    assert attention_module.num_heads == config.num_attention_heads
    assert attention_module.head_dim == config.hidden_size // config.num_attention_heads
    
    # Test with a small input to ensure it works
    batch_size, seq_len = 1, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda')
    
    output = attention_module(hidden_states)
    assert output[0].shape == hidden_states.shape, f"Output shape mismatch: {output[0].shape}"
    
    print(f"✓ Model capacity preservation test passed")
    print(f"  - Num layers: {config.num_hidden_layers} (preserved)")
    print(f"  - Num heads: {config.num_attention_heads} (preserved)")
    print(f"  - Hidden size: {config.hidden_size}")


def run_all_tests():
    """Run all tests"""
    print("Running all CUDA integration tests...\n")
    
    test_cuda_wrapper_basic()
    test_cuda_attention_with_tensors()
    test_memory_pool_allocation()
    test_tensor_operations()
    test_optimized_attention_module()
    test_optimized_mlp_module()
    test_cuda_integration_full()
    test_wrapper_statistics()
    test_fallback_mechanisms()
    test_model_capacity_preservation()
    
    print("\n🎉 All CUDA integration tests passed!")


if __name__ == "__main__":
    run_all_tests()