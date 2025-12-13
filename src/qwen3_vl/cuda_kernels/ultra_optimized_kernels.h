/*
 * Ultra-Optimized CUDA Kernels Header for SM61 Architecture
 * Provides interfaces to state-of-the-art optimization techniques
 */

#ifndef ULTRA_OPTIMIZED_KERNELS_H
#define ULTRA_OPTIMIZED_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Structure for kernel configuration
struct UltraKernelConfig {
    dim3 block_dim;
    dim3 grid_dim;
    size_t shared_mem_size;
    int occupancy;
};

// Custom memory pool class declaration
class UltraMemoryPool {
private:
    char* memory_pool;
    size_t pool_size;
    size_t* block_offsets;
    size_t* block_sizes;
    bool* block_allocated;
    size_t num_blocks;
    size_t min_block_size;
    cudaStream_t* streams;
    size_t stream_count;

public:
    UltraMemoryPool(size_t size = 128 * 1024 * 1024, size_t min_block = 1024, size_t stream_count = 4);
    ~UltraMemoryPool();

    void* allocate(size_t size, cudaStream_t stream = 0);
    void deallocate(void* ptr, cudaStream_t stream = 0);
};

// Custom 16-bit floating point format
struct CustomFloat16 {
    uint16_t data;
    
    __device__ __forceinline__ CustomFloat16(float f);
    __device__ __forceinline__ operator float() const;
};

// Warp-level optimization functions
__device__ __forceinline__ float warp_reduce_sum_ptx(float val);
__device__ __forceinline__ float warp_reduce_max_ptx(float val);

// Ultra-optimized kernel launch functions
template<int HEAD_DIM>
cudaError_t launch_ultra_optimized_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int seq_len,
    int num_heads,
    float scale_factor,
    cudaStream_t stream = 0
);

cudaError_t launch_ultra_optimized_matmul(
    const float* a,
    const float* b,
    float* c,
    int m, int n, int k,
    cudaStream_t stream = 0
);

cudaError_t launch_ultra_quantized_matmul(
    const __half* a,
    const __half* b,
    __half* c,
    int m, int n, int k,
    float a_scale, float b_scale, float c_scale,
    cudaStream_t stream = 0
);

cudaError_t launch_ultra_low_latency_softmax(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    cudaStream_t stream = 0
);

cudaError_t launch_custom_precision_matmul(
    const CustomFloat16* a,
    const CustomFloat16* b,
    CustomFloat16* c,
    int m, int n, int k,
    cudaStream_t stream = 0
);

cudaError_t launch_ultra_optimized_layer_norm(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int batch_size,
    int hidden_dim,
    float eps = 1e-5f,
    cudaStream_t stream = 0
);

// Configuration helper functions
template<int HEAD_DIM>
UltraKernelConfig get_ultra_attention_config(int batch_size, int seq_len, int num_heads);

#endif // ULTRA_OPTIMIZED_KERNELS_H