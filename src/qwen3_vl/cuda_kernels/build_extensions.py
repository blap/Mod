#!/usr/bin/env python
"""
Build script for SM61 CUDA kernels
This script compiles the CUDA extensions for the SM61 architecture optimizations
"""
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Define the CUDA extension
sm61_extension = CUDAExtension(
    name='sm61_cuda_kernels',
    sources=[
        os.path.join(curr_dir, 'pybind_interface.cpp'),
        os.path.join(curr_dir, 'attention_sm61.cu'),
        os.path.join(curr_dir, 'attention_kernel_impl.cu'),
        os.path.join(curr_dir, 'tensor_ops.cu'),
        os.path.join(curr_dir, 'memory_pool.cu'),
        os.path.join(curr_dir, 'block_sparse_attention.cu'),
        os.path.join(curr_dir, 'cuda_launchers.cu'),
    ],
    extra_compile_args={
        'cxx': ['-O3', '/std:c++14'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '--maxrregcount=64',  # Optimize register usage for SM61
            '-gencode', 'arch=compute_60,code=sm_60',  # Include compute capability 6.0 as well
            '-gencode', 'arch=compute_61,code=sm_61',  # SM61 specific
            '-gencode', 'arch=compute_61,code=compute_61',  # For forward compatibility
            '-lineinfo',  # Include line info for profiling
            '-Xptxas', '-v',  # Verbose PTX compilation
        ]
    }
)

setup(
    name='sm61_cuda_kernels',
    ext_modules=[sm61_extension],
    cmdclass={
        'build_ext': BuildExtension
    }
)