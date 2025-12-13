#ifndef BLOCK_SPARSE_ATTENTION_H
#define BLOCK_SPARSE_ATTENTION_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Configuration constants for SM61 architecture
#define SM61_MAX_THREADS_PER_BLOCK 1024
#define SM61_MAX_SHARED_MEMORY_PER_BLOCK 48 * 1024  // 48KB in bytes
#define WARP_SIZE 32
#define MAX_REGISTERS_PER_THREAD 255

// Block-sparse attention configuration
#define BLOCK_SPARSE_TILE_SIZE 32
#define BLOCK_SPARSE_WARPS_PER_BLOCK 4
#define BLOCK_SPARSE_THREADS_PER_BLOCK (BLOCK_SPARSE_WARPS_PER_BLOCK * WARP_SIZE)
#define BLOCK_SPARSE_BLOCK_DIM_X 32
#define BLOCK_SPARSE_BLOCK_DIM_Y 8

// Memory-efficient operations configuration
#define MEMORY_EFFICIENT_BLOCK_DIM_X 32
#define MEMORY_EFFICIENT_BLOCK_DIM_Y 8

// High-performance matmul configuration for SM61
#define SM61_MATMUL_TILE_SIZE 16

// Enum for operation types
typedef enum {
    OP_MATMUL = 0,
    OP_ADD = 1,
    OP_MUL = 2,
    OP_ACTIVATION = 3
} OpType;

typedef enum {
    MEM_OP_COPY = 0,
    MEM_OP_SET = 1
} MemOpType;

// Structure to hold kernel launch parameters for block-sparse attention
struct BlockSparseConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_size;
};

// Structure to hold kernel launch parameters for memory-efficient operations
struct MemoryEfficientConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_size;
};

// Structure to hold kernel launch parameters for matmul
struct MatmulConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_size;
};

// Block-sparse attention kernel for SM61 architecture
__global__ void block_sparse_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    const int* __restrict__ block_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    int block_size
);

// Function to compute optimal kernel configuration for block-sparse attention on SM61
BlockSparseConfig get_block_sparse_attention_config(int batch_size, int seq_len, int head_dim, int num_heads, int block_size);

// Optimized memory-efficient operations kernel for SM61
__global__ void memory_efficient_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    OpType op_type
);

// Function to compute optimal kernel configuration for memory-efficient operations on SM61
MemoryEfficientConfig get_memory_efficient_config(int batch_size, int seq_len, int hidden_dim);

// Optimized memory management kernel for SM61
__global__ void memory_management_kernel(
    void** __restrict__ ptrs,
    size_t* __restrict__ sizes,
    int num_ops,
    MemOpType op_type
);

// High-performance matrix operations kernel for SM61
__global__ void high_performance_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k,
    bool use_tensor_cores  // This will be ignored for SM61 as it doesn't have tensor cores
);

// Function to compute optimal kernel configuration for high-performance matmul on SM61
MatmulConfig get_high_performance_matmul_config(int m, int n, int k);

#endif // BLOCK_SPARSE_ATTENTION_H