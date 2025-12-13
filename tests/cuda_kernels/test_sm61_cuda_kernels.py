import torch
import pytest
import numpy as np
import unittest
from src.cuda_kernels.pybind_interface import (
    scaled_dot_product_attention_sm61,
    coalesced_copy_sm61,
    transpose_sm61,
    SM61MemoryPool
)

class TestSM61CudaKernels(unittest.TestCase):
    """Test suite for SM61 optimized CUDA kernels"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - ensure CUDA is available"""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        
        cls.device = torch.device('cuda')
    
    def test_scaled_dot_product_attention_basic(self):
        """Test basic functionality of SM61 optimized attention kernel"""
        batch_size, seq_len, num_heads, head_dim = 2, 16, 8, 64
        
        # Create random tensors
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        
        # Run SM61 optimized attention
        output = scaled_dot_product_attention_sm61(query, key, value)
        
        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, num_heads, head_dim))
        
        # Verify output is on the correct device
        self.assertTrue(output.device == self.device)
        
        # Verify output dtype matches input
        self.assertEqual(output.dtype, torch.float32)
    
    def test_scaled_dot_product_attention_half_precision(self):
        """Test attention kernel with half precision (float16)"""
        batch_size, seq_len, num_heads, head_dim = 2, 16, 8, 64
        
        # Create random tensors in float16
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float16)
        
        # Run SM61 optimized attention
        output = scaled_dot_product_attention_sm61(query, key, value)
        
        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, num_heads, head_dim))
        
        # Verify output is on the correct device
        self.assertTrue(output.device == self.device)
        
        # Verify output dtype matches input
        self.assertEqual(output.dtype, torch.float16)
    
    def test_scaled_dot_product_attention_vs_pytorch(self):
        """Compare SM61 optimized attention with PyTorch's implementation"""
        batch_size, seq_len, num_heads, head_dim = 2, 32, 4, 16
        
        # Create random tensors
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        
        # SM61 optimized implementation
        output_sm61 = scaled_dot_product_attention_sm61(query, key, value)
        
        # PyTorch implementation for comparison
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            output_pytorch = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        
        # Check if results are close (considering numerical differences due to implementation differences)
        torch.testing.assert_close(output_sm61, output_pytorch, rtol=1e-5, atol=1e-5)
    
    def test_coalesced_copy_functionality(self):
        """Test coalesced memory copy kernel"""
        # Create a tensor with known values
        input_tensor = torch.arange(1024, dtype=torch.float32, device=self.device).reshape(32, 32)
        
        # Copy using SM61 optimized kernel
        output_tensor = coalesced_copy_sm61(input_tensor)
        
        # Verify output shape
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        
        # Verify values are preserved
        torch.testing.assert_close(output_tensor, input_tensor)
        
        # Verify output is on the correct device
        self.assertTrue(output_tensor.device == self.device)
    
    def test_coalesced_copy_half_precision(self):
        """Test coalesced memory copy with half precision"""
        # Create a tensor with known values in float16
        input_tensor = torch.arange(512, dtype=torch.float16, device=self.device).reshape(16, 32)
        
        # Copy using SM61 optimized kernel
        output_tensor = coalesced_copy_sm61(input_tensor)
        
        # Verify output shape
        self.assertEqual(output_tensor.shape, input_tensor.shape)
        
        # Verify values are preserved
        torch.testing.assert_close(output_tensor, input_tensor)
    
    def test_transpose_functionality(self):
        """Test matrix transpose kernel with bank conflict avoidance"""
        rows, cols = 64, 32
        
        # Create a test matrix
        input_matrix = torch.randn(rows, cols, dtype=torch.float32, device=self.device)
        
        # Transpose using SM61 optimized kernel
        transposed = transpose_sm61(input_matrix)
        
        # Verify output shape (should be transposed)
        self.assertEqual(transposed.shape, (cols, rows))
        
        # Verify values are correctly transposed
        expected = input_matrix.t()  # PyTorch transpose
        torch.testing.assert_close(transposed, expected, rtol=1e-5, atol=1e-5)
    
    def test_transpose_half_precision(self):
        """Test matrix transpose with half precision"""
        rows, cols = 32, 64
        
        # Create a test matrix in float16
        input_matrix = torch.randn(rows, cols, dtype=torch.float16, device=self.device)
        
        # Transpose using SM61 optimized kernel
        transposed = transpose_sm61(input_matrix)
        
        # Verify output shape (should be transposed)
        self.assertEqual(transposed.shape, (cols, rows))
        
        # Verify values are correctly transposed
        expected = input_matrix.t()  # PyTorch transpose
        torch.testing.assert_close(transposed, expected, rtol=1e-3, atol=1e-3)  # Slightly relaxed for float16
    
    def test_memory_pool_basic_allocation(self):
        """Test basic memory pool functionality"""
        pool = SM61MemoryPool(pool_size=16 * 1024 * 1024)  # 16MB pool
        
        # Allocate a tensor
        tensor = pool.allocate_tensor([1024, 1024], torch.float32)
        
        # Verify tensor properties
        self.assertEqual(tensor.shape, (1024, 1024))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(tensor.device.type == 'cuda')
        
        # Verify memory pool stats
        stats = pool.get_stats()
        self.assertGreater(stats['allocated'], 0)
        self.assertLessEqual(stats['allocated'] + stats['free'], 16 * 1024 * 1024)
    
    def test_memory_pool_multiple_allocations(self):
        """Test multiple allocations and deallocations"""
        pool = SM61MemoryPool(pool_size=32 * 1024 * 1024)  # 32MB pool
        
        # Initial stats
        initial_stats = pool.get_stats()
        initial_free = initial_stats['free']
        
        # Allocate multiple tensors
        tensors = []
        for i in range(5):
            tensor = pool.allocate_tensor([512, 512], torch.float32)  # ~1MB each
            tensors.append(tensor)
        
        # Check that memory was allocated
        mid_stats = pool.get_stats()
        self.assertLess(mid_stats['free'], initial_free)
        
        # Delete tensors (should trigger deallocation)
        del tensors
        torch.cuda.synchronize()  # Ensure deallocation happens
        
        # Check that memory was returned to pool
        final_stats = pool.get_stats()
        # Note: Memory may not be fully returned immediately due to pooling strategy
        self.assertLessEqual(final_stats['allocated'], mid_stats['allocated'])
    
    def test_memory_pool_different_sizes(self):
        """Test memory pool with different allocation sizes"""
        pool = SM61MemoryPool(pool_size=16 * 1024 * 1024)  # 16MB pool
        
        # Allocate tensors of different sizes
        small_tensor = pool.allocate_tensor([64, 64], torch.float32)      # Small allocation
        medium_tensor = pool.allocate_tensor([256, 256], torch.float32)   # Medium allocation
        large_tensor = pool.allocate_tensor([512, 512], torch.float32)    # Large allocation
        
        # Verify all tensors were created successfully
        self.assertEqual(small_tensor.shape, (64, 64))
        self.assertEqual(medium_tensor.shape, (256, 256))
        self.assertEqual(large_tensor.shape, (512, 512))
        
        # Check memory pool stats
        stats = pool.get_stats()
        total_used = stats['allocated']
        self.assertGreater(total_used, 0)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with empty tensors
        empty_query = torch.empty(0, 0, 0, 0, device=self.device, dtype=torch.float32)
        empty_key = torch.empty(0, 0, 0, 0, device=self.device, dtype=torch.float32)
        empty_value = torch.empty(0, 0, 0, 0, device=self.device, dtype=torch.float32)
        
        # This should handle gracefully (implementation-dependent)
        try:
            output = scaled_dot_product_attention_sm61(empty_query, empty_key, empty_value)
            # If it doesn't throw, verify the output is also empty with correct shape
            self.assertEqual(output.numel(), 0)
        except Exception:
            # Some implementations might throw for empty tensors, which is also valid
            pass
        
        # Test with invalid device (CPU tensors)
        cpu_tensor = torch.randn(2, 4, 2, 8)  # CPU tensor
        with self.assertRaises(RuntimeError):
            scaled_dot_product_attention_sm61(cpu_tensor, cpu_tensor, cpu_tensor)


class TestPerformanceComparisons(unittest.TestCase):
    """Performance comparison tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - ensure CUDA is available"""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        
        cls.device = torch.device('cuda')
    
    def test_attention_performance(self):
        """Basic performance test for attention kernel"""
        import time
        
        batch_size, seq_len, num_heads, head_dim = 4, 128, 8, 64
        
        # Create test tensors
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device, dtype=torch.float32)
        
        # Warm up
        for _ in range(5):
            _ = scaled_dot_product_attention_sm61(query, key, value)
        torch.cuda.synchronize()
        
        # Time SM61 implementation
        start_time = time.time()
        for _ in range(10):
            output_sm61 = scaled_dot_product_attention_sm61(query, key, value)
        torch.cuda.synchronize()
        sm61_time = time.time() - start_time
        
        # Time PyTorch implementation for comparison
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            start_time = time.time()
            for _ in range(10):
                output_pytorch = torch.nn.functional.scaled_dot_product_attention(query, key, value)
            torch.cuda.synchronize()
            pytorch_time = time.time() - start_time
        
        print(f"\nAttention Performance Comparison:")
        print(f"SM61 Optimized: {sm61_time:.4f}s")
        print(f"PyTorch Default: {pytorch_time:.4f}s")
        print(f"Speedup: {pytorch_time/sm61_time:.2f}x")
        
        # Verify results are still correct
        torch.testing.assert_close(output_sm61, output_pytorch, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()