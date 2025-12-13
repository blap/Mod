"""
Setup script for SM61-optimized CUDA kernels
Builds the CUDA extensions for the Qwen3-VL model with optimizations for NVIDIA SM61 architecture
"""
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import sys
from pathlib import Path

def get_cuda_arch_flags():
    """Get CUDA architecture flags optimized for SM61"""
    # SM61 (Pascal) architecture flags
    arch_flags = ['-gencode', 'arch=compute_60,code=sm_60',
                  '-gencode', 'arch=compute_61,code=sm_61',
                  '-gencode', 'arch=compute_61,code=compute_61']
    return arch_flags

def get_cuda_root():
    """Get the CUDA root directory"""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if cuda_home is None:
        # Common CUDA installation paths on Windows
        possible_paths = [
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7',
            'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cuda_home = path
                break
    
    return cuda_home

def get_include_dirs():
    """Get include directories for CUDA"""
    include_dirs = []
    cuda_home = get_cuda_root()
    
    if cuda_home:
        include_dirs.append(os.path.join(cuda_home, 'include'))
    
    # Add PyTorch include directories
    include_dirs.extend(torch.utils.cpp_extension.include_paths())
    
    return include_dirs

def get_library_dirs():
    """Get library directories for CUDA"""
    library_dirs = []
    cuda_home = get_cuda_root()
    
    if cuda_home:
        if sys.platform.startswith('win'):
            library_dirs.append(os.path.join(cuda_home, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_home, 'lib64'))
    
    return library_dirs

def get_libraries():
    """Get required CUDA libraries"""
    if sys.platform.startswith('win'):
        return ['cudart', 'cublas', 'curand', 'cusparse']
    else:
        return ['cudart', 'cublas', 'curand', 'cusparse', 'cufft']

# Define the CUDA extension
sm61_extension = CUDAExtension(
    name='sm61_cuda_kernels',
    sources=[
        'pybind_interface.cpp',
        'sm61_optimized_kernels.cu',
        'attention_sm61.cu',
        'matmul_sm61.cu',
        'memory_ops_sm61.cu',
        'transpose_sm61.cu',
        'memory_pool.cu',
    ],
    include_dirs=get_include_dirs(),
    library_dirs=get_library_dirs(),
    libraries=get_libraries(),
    extra_compile_args={
        'cxx': [
            '-O3',
            '-std=c++14',
            '/std:c++14' if sys.platform.startswith('win') else '',  # Windows-specific flag
            '-DTORCH_API_INCLUDE_EXTENSION_H',
            '-D_GLIBCXX_USE_CXX11_ABI=0'
        ],
        'nvcc': [
            '-O3',
            '--use_fast_math',              # Optimize for speed
            '-lineinfo',                     # Include line info for debugging
            '-maxrregcount=64',              # Optimize register usage for SM61
            '-Xptxas', '-v',                # Verbose PTX compilation
            '-gencode', 'arch=compute_60,code=sm_60',    # SM60 support
            '-gencode', 'arch=compute_61,code=sm_61',    # SM61 support (primary target)
            '-gencode', 'arch=compute_61,code=compute_61', # Forward compatibility
            '--ptxas-options=-v',           # Verbose assembly info
            '--compiler-options', "'-fPIC'", # Position independent code
            '--expt-relaxed-constexpr',      # Experimental relaxed constexpr
        ] + get_cuda_arch_flags()
    }
)

setup(
    name='sm61_cuda_kernels',
    version='1.0.0',
    description='SM61-optimized CUDA kernels for Qwen3-VL model',
    ext_modules=[sm61_extension],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'pybind11>=2.6.0'
    ]
)