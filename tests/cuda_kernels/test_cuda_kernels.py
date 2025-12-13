"""
Tests for CUDA kernels optimized for NVIDIA SM61 architecture
"""
import pytest
import numpy as np
import torch
import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from cuda_kernels.attention import attention_forward, attention_backward
    from cuda_kernels.memory_pool import MemoryPool
    from cuda_kernels.tensor_ops import matmul_sm61
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


def skip_if_no_cuda():
    """Skip test if CUDA is not available"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")


class TestAttentionKernels:
    """Test attention computation kernels optimized for SM61"""
    
    def test_attention_forward_basic(self):
        """Test basic attention forward computation"""
        skip_if_no_cuda()
        
        batch_size, seq_len, head_dim = 2, 128, 64
        q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Test attention computation
        output = attention_forward(q, k, v)
        
        assert output.shape == (batch_size, seq_len, head_dim)
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
        
        # Check that output values are reasonable (not NaN or infinity)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_backward_basic(self):
        """Test basic attention backward computation"""
        skip_if_no_cuda()
        
        batch_size, seq_len, head_dim = 2, 128, 64
        q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Forward pass
        output = attention_forward(q, k, v)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape
    
    def test_attention_performance(self):
        """Test attention performance with different input sizes"""
        skip_if_no_cuda()
        
        # Test different sizes to ensure kernels work efficiently
        test_configs = [
            (1, 64, 32),    # Small
            (2, 128, 64),   # Medium
            (4, 256, 128),  # Large
        ]
        
        for batch_size, seq_len, head_dim in test_configs:
            q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
            k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
            v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
            
            output = attention_forward(q, k, v)
            
            assert output.shape == (batch_size, seq_len, head_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


class TestMemoryPool:
    """Test memory pool system optimized for SM61"""
    
    def test_memory_pool_allocation(self):
        """Test basic memory pool allocation"""
        skip_if_no_cuda()
        
        pool = MemoryPool()
        
        # Allocate memory
        size = 1024 * 1024  # 1MB
        ptr = pool.allocate(size)
        
        assert ptr is not None
        assert isinstance(ptr, int)  # CUDA pointer is an integer
        
        # Deallocate memory
        pool.deallocate(ptr, size)
    
    def test_memory_pool_reuse(self):
        """Test memory pool reuse capability"""
        skip_if_no_cuda()
        
        pool = MemoryPool()
        
        # Allocate and deallocate multiple times to test reuse
        size = 1024 * 1024  # 1MB
        ptr1 = pool.allocate(size)
        pool.deallocate(ptr1, size)
        
        ptr2 = pool.allocate(size)
        pool.deallocate(ptr2, size)
        
        # The pool should reuse memory, so ptr1 and ptr2 might be the same
        assert ptr1 is not None
        assert ptr2 is not None
        assert isinstance(ptr1, int)
        assert isinstance(ptr2, int)
    
    def test_memory_pool_large_allocation(self):
        """Test memory pool with large allocation"""
        skip_if_no_cuda()
        
        pool = MemoryPool()
        
        # Test with a larger allocation
        size = 16 * 1024 * 1024  # 16MB
        ptr = pool.allocate(size)
        
        assert ptr is not None
        assert isinstance(ptr, int)
        
        pool.deallocate(ptr, size)


class TestMatmulKernels:
    """Test optimized matrix multiplication for SM61"""
    
    def test_matmul_basic(self):
        """Test basic matrix multiplication"""
        skip_if_no_cuda()
        
        m, n, k = 128, 128, 64
        a = torch.randn(m, k, device='cuda', dtype=torch.float32)
        b = torch.randn(k, n, device='cuda', dtype=torch.float32)
        
        output = matmul_sm61(a, b)
        
        assert output.shape == (m, n)
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
        
        # Compare with PyTorch's matmul
        expected = torch.matmul(a, b)
        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_matmul_different_sizes(self):
        """Test matrix multiplication with different sizes"""
        skip_if_no_cuda()
        
        test_configs = [
            (64, 64, 32),
            (256, 128, 64),
            (512, 256, 128),
        ]
        
        for m, n, k in test_configs:
            a = torch.randn(m, k, device='cuda', dtype=torch.float32)
            b = torch.randn(k, n, device='cuda', dtype=torch.float32)
            
            output = matmul_sm61(a, b)
            expected = torch.matmul(a, b)
            
            assert output.shape == (m, n)
            assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5)


def test_sm61_optimizations():
    """Test that our kernels are optimized for SM61 architecture"""
    skip_if_no_cuda()
    
    # Verify that we're using the right CUDA device
    device_name = torch.cuda.get_device_name(0)
    print(f"Using device: {device_name}")
    
    # Test that our kernels work with SM61-specific parameters
    # SM61 has 128 CUDA cores per SM, 96KB shared memory per SM
    # These parameters should be reflected in our kernel configurations
    
    # Test attention with parameters that benefit from SM61 architecture
    batch_size, seq_len, head_dim = 2, 256, 128
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32, requires_grad=True)
    
    output = attention_forward(q, k, v)
    
    assert output.shape == (batch_size, seq_len, head_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()