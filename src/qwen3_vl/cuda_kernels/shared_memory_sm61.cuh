#ifndef SHARED_MEMORY_SM61_CUH
#define SHARED_MEMORY_SM61_CUH

#include <cuda_runtime.h>

// Shared memory optimization constants for SM61 architecture
// SM61 has 48KB or 96KB of shared memory per SM (configurable)

#define SM61_MAX_SHARED_MEMORY_PER_BLOCK 48 * 1024  // 48KB default
#define SM61_MAX_BANKS 32  // Number of shared memory banks
#define SM61_BANK_SIZE 4   // Size of each bank in bytes (for 32-bit words)

// Bank conflict avoidance strategies
#define SM61_BANK_CONFLICT_THRESHOLD 4  // More than this many concurrent accesses causes conflicts

// Structure for managing shared memory allocation
struct SharedMemoryConfig {
    size_t attention_buffer_size;    // Size for attention computation buffers
    size_t matrix_buffer_size;       // Size for matrix operation buffers
    size_t intermediate_results_size; // Size for temporary results
    size_t total_required_size;      // Total size required
};

// Function to calculate shared memory requirements for attention operations
inline SharedMemoryConfig calculate_attention_shared_memory(int seq_len, int head_dim, int batch_size) {
    SharedMemoryConfig config;
    
    // For attention: need space for Q, K, V matrices and attention scores
    // Each thread might need to store a portion of these
    config.attention_buffer_size = seq_len * head_dim * sizeof(float) * 3;  // Q, K, V
    config.intermediate_results_size = seq_len * seq_len * sizeof(float);   // Attention scores
    
    config.total_required_size = config.attention_buffer_size + config.intermediate_results_size;
    
    // Ensure we don't exceed the maximum shared memory per block
    if (config.total_required_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        // Fallback: use smaller buffers and process in chunks
        config.total_required_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK * 0.8; // Use 80% to be safe
    }
    
    return config;
}

// Function to calculate shared memory requirements for matrix operations
inline SharedMemoryConfig calculate_matmul_shared_memory(int m, int n, int k) {
    SharedMemoryConfig config;
    
    // For tiled matrix multiplication, we need space for two tiles
    int tile_size = 16;  // Common tile size for SM61
    config.matrix_buffer_size = 2 * tile_size * tile_size * sizeof(float);  // Two tiles: A and B
    
    // Also need space for the result tile
    config.intermediate_results_size = tile_size * tile_size * sizeof(float);
    
    config.total_required_size = config.matrix_buffer_size + config.intermediate_results_size;
    
    return config;
}

// Macro to add padding to avoid bank conflicts in shared memory access
// This adds padding to the second dimension to avoid conflicts
#define SM61_SHARED_MEM_PADDING 8  // Add 8 floats of padding

#endif // SHARED_MEMORY_SM61_CUH