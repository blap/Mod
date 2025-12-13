"""
Minimal setup for SM61-optimized CUDA kernels
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='sm61_cuda_kernels_minimal',
    ext_modules=[
        CUDAExtension(
            name='sm61_cuda_kernels_minimal',
            sources=[
                'pybind_interface.cpp',
                'attention_sm61.cu',
                'attention_kernel_impl.cu',
                'tensor_ops.cu',
                'memory_pool.cu',
                'block_sparse_attention.cu',
                'cuda_launchers.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--maxrregcount=64',  # Optimize register usage for SM61
                    '-gencode', 'arch=compute_60,code=sm_60',  # Include compute capability 6.0 as well
                    '-gencode', 'arch=compute_61,code=sm_61',  # SM61 specific
                    '-gencode', 'arch=compute_61,code=compute_61',  # For forward compatibility
                    '-lineinfo',  # Include line info for profiling
                    '-Xptxas', '-v',  # Verbose PTX compilation
                    '-Xcompiler', '-Wall,-Wextra',  # Additional compiler warnings
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)