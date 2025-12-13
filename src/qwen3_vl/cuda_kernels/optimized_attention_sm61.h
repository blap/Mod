#ifndef OPTIMIZED_ATTENTION_SM61_H
#define OPTIMIZED_ATTENTION_SM61_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Configuration constants for SM61 architecture with optimizations
#define SHARED_MEM_SIZE 48000  // Max shared memory per block (48KB)
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_WARPS_PER_SM 64

// Optimized tile size for SM61 to balance occupancy and arithmetic intensity
#define OPTIMIZED_TILE_SIZE 16  // Reduced from 32 to improve occupancy
#define OPTIMIZED_WARPS_PER_BLOCK 8  // 8 warps = 256 threads for better occupancy

// Define the maximum sequence length that fits in shared memory
#define MAX_SEQ_LEN_SHARED 512

// Warp-level reduction functions for better performance
__device__ float warp_reduce_sum(float val);
__device__ float warp_reduce_max(float val);

/**
 * @brief Optimized scaled dot-product attention kernel for SM61 with better occupancy
 * Implements: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
 * Optimizations:
 * - Reduced tile size for better occupancy
 * - Warp-level reductions for softmax
 * - Improved memory access patterns
 */
template<typename T>
__global__ void optimized_scaled_dot_product_attention_kernel(
    const T* __restrict__ query,     // [batch_size, seq_len, num_heads, head_dim]
    const T* __restrict__ key,       // [batch_size, seq_len, num_heads, head_dim]
    const T* __restrict__ value,     // [batch_size, seq_len, num_heads, head_dim]
    T* __restrict__ output,          // [batch_size, seq_len, num_heads, head_dim]
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

/**
 * @brief Optimized attention kernel using shared memory with padding to avoid bank conflicts
 * Uses shared memory to cache frequently accessed data with bank conflict avoidance
 */
template<typename T>
__global__ void optimized_scaled_dot_product_attention_shared_mem_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

/**
 * @brief Optimized matrix multiplication kernel with better arithmetic intensity
 */
template<typename T>
__global__ void optimized_matmul_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int n, int k
);

/**
 * @brief Launch configuration helper for SM61 optimized attention kernel
 * Includes register usage optimization
 */
template<typename T>
cudaError_t launch_optimized_scaled_dot_product_attention(
    const T* query,
    const T* key,
    const T* value,
    T* output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0
);

/**
 * @brief Optimized half-precision attention kernel for mixed precision computing
 */
__global__ void optimized_scaled_dot_product_attention_half_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

/**
 * @brief Launch function for half-precision optimized attention
 */
cudaError_t launch_optimized_scaled_dot_product_attention_half(
    const __half* query,
    const __half* key, 
    const __half* value,
    __half* output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0
);

#endif // OPTIMIZED_ATTENTION_SM61_H