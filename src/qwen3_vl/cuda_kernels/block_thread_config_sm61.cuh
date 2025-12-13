#ifndef BLOCK_THREAD_CONFIG_SM61_CUH
#define BLOCK_THREAD_CONFIG_SM61_CUH

#include <cuda_runtime.h>

// Block and thread configuration constants for SM61 architecture
// SM61 (Pascal) specifications:
// - 128 CUDA cores per SM
// - 64K 32-bit registers per SM
// - 48KB or 96KB shared memory per SM (configurable)
// - Max 32 active warps per SM
// - Max 1024 threads per block

#define SM61_MAX_THREADS_PER_BLOCK 1024
#define SM61_MAX_WARPS_PER_BLOCK 32  // 1024 threads / 32 threads per warp
#define SM61_WARP_SIZE 32
#define SM61_MAX_SHARED_MEMORY_PER_BLOCK 48 * 1024  // 48KB in bytes (default config)

// Configuration for attention operations
#define ATTENTION_HEADS_PER_BLOCK 8
#define ATTENTION_THREADS_PER_HEAD 128
#define ATTENTION_MAX_THREADS_PER_BLOCK (ATTENTION_HEADS_PER_BLOCK * ATTENTION_THREADS_PER_HEAD)

// Configuration for matrix operations
#define MATMUL_TILE_SIZE 16
#define MATMUL_BLOCK_SIZE (MATMUL_TILE_SIZE * MATMUL_TILE_SIZE)  // 256 threads per block

// Function to calculate optimal block size for SM61
inline dim3 get_optimal_block_size(int operation_type) {
    switch (operation_type) {
        case 0:  // Attention operation
            return dim3(ATTENTION_THREADS_PER_HEAD, ATTENTION_HEADS_PER_BLOCK, 1);
        case 1:  // Matrix multiplication
            return dim3(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
        default:
            return dim3(256, 1, 1);  // Default block size
    }
}

// Function to calculate optimal grid size for SM61
inline dim3 get_optimal_grid_size(int n_elements, dim3 block_size) {
    int total_threads_per_block = block_size.x * block_size.y * block_size.z;
    int num_blocks = (n_elements + total_threads_per_block - 1) / total_threads_per_block;
    
    // For 2D operations, we might want a 2D grid
    int grid_x = (int)sqrtf((float)num_blocks);
    int grid_y = (num_blocks + grid_x - 1) / grid_x;
    
    return dim3(grid_x, grid_y, 1);
}

#endif // BLOCK_THREAD_CONFIG_SM61_CUH