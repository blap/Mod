#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <cuda_runtime.h>

// Configuration for SM61 optimized tensor operations
#define MATMUL_TILE_SIZE 16
#define MATMUL_BLOCK_SIZE 256  // 16x16 block of threads
#define SM61_MAX_SHARED_MEMORY_PER_BLOCK 48 * 1024  // 48KB in bytes

// Structure for kernel launch parameters
struct MatmulConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_size;
};

// Optimized matrix multiplication kernel for SM61
__global__ void matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k
);

// Function to compute optimal kernel configuration for SM61
MatmulConfig get_matmul_config(int m, int n, int k);

// Kernel for optimized attention computation with proper memory coalescing
__global__ void coalesced_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int head_dim
);

#endif // TENSOR_OPS_H