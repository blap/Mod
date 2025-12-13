"""
Python interface for tensor operations optimized for NVIDIA SM61 architecture
"""
import torch
from torch.utils.cpp_extension import load
import os

# Dynamically compile the CUDA extension
curr_dir = os.path.dirname(__file__)
kernel_sources = [
    os.path.join(curr_dir, 'tensor_ops.cu')
]

# Define compilation options for SM61 architecture
extra_cuda_cflags = ['-arch=sm_61', '-O3', '--use_fast_math']

try:
    # Load the compiled CUDA extension
    tensor_ops_ext = load(
        name='sm61_tensor_ops',
        sources=kernel_sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False
    )
except Exception as e:
    print(f"Warning: Could not compile CUDA extension: {e}")
    print("Falling back to PyTorch implementation")
    tensor_ops_ext = None


def matmul_sm61(a, b):
    """
    Matrix multiplication optimized for SM61 architecture
    
    Args:
        a: Tensor of shape (m, k)
        b: Tensor of shape (k, n)
        
    Returns:
        Result tensor of shape (m, n)
    """
    if tensor_ops_ext is not None and a.is_cuda and b.is_cuda:
        # Use optimized CUDA kernel
        return tensor_ops_ext.matmul_sm61(a, b)
    else:
        # Fall back to PyTorch implementation
        return torch.matmul(a, b)


def softmax_sm61(input, dim=-1):
    """
    Softmax operation optimized for SM61 architecture
    
    Args:
        input: Input tensor
        dim: Dimension along which to apply softmax
        
    Returns:
        Softmax output tensor
    """
    if tensor_ops_ext is not None and input.is_cuda:
        # Use optimized CUDA kernel
        return tensor_ops_ext.softmax_sm61(input, dim)
    else:
        # Fall back to PyTorch implementation
        return torch.softmax(input, dim=dim)


# Memory pool class for Python
class MemoryPool:
    """
    Memory pool optimized for SM61 architecture with proper synchronization
    """
    def __init__(self, pool_size: int = 64 * 1024 * 1024):  # 64MB default
        self.pool_size = pool_size
        if tensor_ops_ext is not None:
            try:
                self._pool = tensor_ops_ext.SM61MemoryPool(pool_size)
            except Exception:
                # Fallback to original method if SM61MemoryPool is not available
                self._pool = tensor_ops_ext.MemoryPool() if hasattr(tensor_ops_ext, 'MemoryPool') else None
        else:
            self._pool = None

    def allocate(self, size):
        """
        Allocate memory from the pool

        Args:
            size: Size in bytes to allocate

        Returns:
            Memory pointer or tensor
        """
        if self._pool is not None:
            try:
                return self._pool.allocate(size)
            except AttributeError:
                # Fallback for older implementation
                return self._pool.allocate(size)
        else:
            # Fall back to standard CUDA allocation
            return torch.cuda.FloatTensor(size // 4).data_ptr()  # Assuming float32

    def deallocate(self, ptr, size):
        """
        Deallocate memory back to the pool

        Args:
            ptr: Memory pointer to deallocate
            size: Size of the memory block
        """
        if self._pool is not None:
            try:
                self._pool.deallocate(ptr, size)
            except AttributeError:
                # Fallback for older implementation
                self._pool.deallocate(ptr, size)
        else:
            # Memory was allocated with standard CUDA allocation, no pooling
            pass

    def get_stats(self):
        """
        Get memory pool statistics
        """
        if self._pool is not None:
            try:
                return self._pool.get_stats()
            except AttributeError:
                # Return default stats if method doesn't exist
                return {
                    "total_size": self.pool_size,
                    "allocated": 0,
                    "free": self.pool_size,
                    "fragmentation": 0.0,
                    "num_free_blocks": 1
                }
        else:
            return {
                "total_size": 0,
                "allocated": 0,
                "free": 0,
                "fragmentation": 0.0,
                "num_free_blocks": 0
            }

    def synchronize(self):
        """
        Synchronize the memory pool operations with the GPU
        """
        if self._pool is not None:
            try:
                # Try to call synchronization methods if available
                if hasattr(self._pool, 'synchronize'):
                    self._pool.synchronize()
                elif hasattr(self._pool, 'stream_synchronize'):
                    self._pool.stream_synchronize()
            except Exception:
                # If synchronization methods don't exist or fail, use PyTorch sync
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        else:
            # Use PyTorch synchronization as fallback
            if torch.cuda.is_available():
                torch.cuda.synchronize()