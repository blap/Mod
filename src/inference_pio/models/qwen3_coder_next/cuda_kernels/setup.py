from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure we are in the right directory or handle paths correctly
# This setup file is meant to be run to compile the kernels

setup(
    name='qwen3_coder_next_kernels',
    ext_modules=[
        CUDAExtension(
            name='qwen3_coder_next_cuda_kernels',
            sources=[
                'bindings.cpp',
                'deltanet_kernel.cu',
                'moe_kernel.cu',
                'attention_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
