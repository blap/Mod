#ifndef ATTENTION_KERNEL_H
#define ATTENTION_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Configuration constants for SM61 architecture
#define SM61_MAX_THREADS_PER_BLOCK 1024
#define SM61_MAX_SHARED_MEMORY_PER_BLOCK 48 * 1024  // 48KB in bytes
#define WARP_SIZE 32
#define MAX_REGISTERS_PER_THREAD 255

// Attention kernel configuration
#define ATTENTION_TILE_SIZE 32
#define ATTENTION_WARPS_PER_BLOCK 4
#define ATTENTION_THREADS_PER_BLOCK (ATTENTION_WARPS_PER_BLOCK * WARP_SIZE)

// Structure to hold kernel launch parameters
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_size;
};

// Forward declaration of attention kernel
__global__ void attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads
);

// Function to compute optimal kernel configuration for SM61
KernelConfig get_attention_config(int batch_size, int seq_len, int head_dim);

#endif // ATTENTION_KERNEL_H