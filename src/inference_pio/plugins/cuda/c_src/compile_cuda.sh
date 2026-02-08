#!/bin/bash
# CUDA Compilation Script (Linux)

echo "Building CUDA Plugin (libtensor_ops_cuda.so)..."

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

nvcc -shared -Xcompiler -fPIC -O3 -o libtensor_ops_cuda.so tensor_ops_cuda.cu -I../../common

if [ $? -eq 0 ]; then
    echo "Build Successful: libtensor_ops_cuda.so"
else
    echo "Build Failed"
    exit 1
fi
