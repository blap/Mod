/*
 * SM61-Optimized CUDA Kernels Header
 * Defines the interface between Python and CUDA kernels optimized for NVIDIA SM61 architecture
 * Compute Capability 6.1 - Pascal Architecture
 */

#ifndef SM61_OPTIMIZED_KERNELS_H
#define SM61_OPTIMIZED_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Configuration constants for SM61 architecture
#define SM61_MAX_THREADS_PER_BLOCK 1024
#define SM61_WARP_SIZE 32
#define SM61_MAX_SHARED_MEMORY 48 * 1024  // 48KB per block
#define SM61_MAX_REGISTERS_PER_THREAD 255

// Define register bank size for avoiding conflicts
#define REGISTER_BANK_SIZE 8

/**
 * @brief Launch configuration for SM61 optimized attention kernel
 */
struct SM61AttentionConfig {
    int block_size_x;
    int block_size_y;
    int grid_size_x;
    int grid_size_y;
    size_t shared_memory_size;
    cudaStream_t stream;
    
    // Constructor with default values optimized for SM61
    SM61AttentionConfig(int seq_len, int head_dim) {
        // Optimize for SM61 architecture
        block_size_x = min(256, SM61_MAX_THREADS_PER_BLOCK);
        block_size_y = 1;
        
        // Grid size based on sequence length
        grid_size_x = (seq_len + block_size_x - 1) / block_size_x;
        grid_size_y = 1;
        
        // Shared memory size depends on head dimension and sequence length
        shared_memory_size = min(
            (size_t)(seq_len * head_dim * sizeof(float)), 
            (size_t)SM61_MAX_SHARED_MEMORY
        );
        
        stream = 0;  // Default stream
    }
};

/**
 * @brief Launch configuration for SM61 optimized matmul kernel
 */
struct SM61MatmulConfig {
    dim3 block_size;
    dim3 grid_size;
    size_t shared_memory_size;
    cudaStream_t stream;
    bool use_tensor_cores;  // Will be false for SM61 (no tensor cores)
    
    // Constructor with default values optimized for SM61
    SM61MatmulConfig(int m, int n, int k) {
        // For SM61, use 16x16 or 32x32 thread blocks for optimal performance
        int tile_size = (m > 1024 || n > 1024) ? 32 : 16;
        
        block_size = dim3(tile_size, tile_size);
        grid_size = dim3((n + tile_size - 1) / tile_size, (m + tile_size - 1) / tile_size);
        
        // Shared memory for tile-based matmul
        shared_memory_size = 2 * tile_size * tile_size * sizeof(float);
        
        stream = 0;  // Default stream
        use_tensor_cores = false;  // SM61 doesn't have tensor cores
    }
};

/**
 * @brief Launch configuration for SM61 optimized memory copy kernel
 */
struct SM61MemoryCopyConfig {
    int block_size;
    int grid_size;
    cudaStream_t stream;
    int elements_per_thread;  // Number of elements each thread processes
    
    // Constructor with default values optimized for SM61
    SM61MemoryCopyConfig(size_t total_elements) {
        block_size = 256;  // Good for coalesced access
        grid_size = (total_elements + block_size - 1) / block_size;
        stream = 0;
        
        // Process multiple elements per thread to reduce launch overhead
        elements_per_thread = min(8, (int)(total_elements / (grid_size * block_size)));
        if (elements_per_thread < 1) elements_per_thread = 1;
    }
};

/**
 * @brief Launch configuration for SM61 optimized transpose kernel
 */
struct SM61TransposeConfig {
    dim3 block_size;
    dim3 grid_size;
    size_t shared_memory_size;
    cudaStream_t stream;
    bool use_padding;  // Whether to use padding to avoid bank conflicts
    
    // Constructor with default values optimized for SM61
    SM61TransposeConfig(int rows, int cols) {
        // Use 32x32 blocks to optimize for 32-thread warps
        int tile_size = 32;
        block_size = dim3(tile_size, tile_size);
        grid_size = dim3((cols + tile_size - 1) / tile_size, (rows + tile_size - 1) / tile_size);
        
        // Shared memory for tile + padding to avoid bank conflicts
        shared_memory_size = (tile_size + 1) * tile_size * sizeof(float);
        stream = 0;
        use_padding = true;  // Enable padding to avoid bank conflicts
    }
};

/**
 * @brief Optimized scaled dot-product attention kernel for SM61
 * 
 * Implements: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
 * Optimized for:
 * - Memory coalescing on GP104 architecture
 * - Shared memory usage optimized for 48KB per block
 * - Warp-efficient computation patterns
 * 
 * @param query Query tensor [batch_size, seq_len, num_heads, head_dim]
 * @param key Key tensor [batch_size, seq_len, num_heads, head_dim]
 * @param value Value tensor [batch_size, seq_len, num_heads, head_dim]
 * @param output Output tensor [batch_size, seq_len, num_heads, head_dim]
 * @param scale_factor Scaling factor for attention scores
 * @param batch_size Number of batches
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 * @param config SM61-specific configuration
 * @return cudaError_t CUDA error code
 */
template<typename T>
cudaError_t launch_scaled_dot_product_attention_sm61(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    const SM61AttentionConfig& config
);

/**
 * @brief Optimized block-sparse attention kernel for SM61
 * 
 * Implements attention with sparse block patterns to reduce computation
 * and memory access for SM61 architecture.
 * 
 * @param query Query tensor [batch_size, seq_len, num_heads, head_dim]
 * @param key Key tensor [batch_size, seq_len, num_heads, head_dim]
 * @param value Value tensor [batch_size, seq_len, num_heads, head_dim]
 * @param output Output tensor [batch_size, seq_len, num_heads, head_dim]
 * @param block_mask Sparse block mask indicating which blocks to compute
 * @param scale_factor Scaling factor for attention scores
 * @param batch_size Number of batches
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 * @param block_size Size of each sparse block
 * @param config SM61-specific configuration
 * @return cudaError_t CUDA error code
 */
template<typename T>
cudaError_t launch_block_sparse_attention_sm61(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    const int* __restrict__ block_mask,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int block_size,
    const SM61AttentionConfig& config
);

/**
 * @brief High-performance matrix multiplication kernel for SM61
 * 
 * Implements optimized GEMM using tile-based algorithm optimized for SM61
 * memory hierarchy and compute capabilities.
 * 
 * @param a Input matrix A [m, k]
 * @param b Input matrix B [k, n]
 * @param c Output matrix C [m, n]
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for C
 * @param config SM61-specific configuration
 * @return cudaError_t CUDA error code
 */
template<typename T>
cudaError_t launch_high_performance_matmul_sm61(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int n, int k,
    float alpha = 1.0f, float beta = 0.0f,
    const SM61MatmulConfig& config = SM61MatmulConfig(0, 0, 0)
);

/**
 * @brief Memory-efficient operations kernel for SM61
 * 
 * Performs various memory-efficient operations like element-wise add/mul/activation
 * with optimized memory access patterns for SM61 architecture.
 * 
 * @param input Input tensor
 * @param output Output tensor
 * @param weight Weight tensor (for operations that need it)
 * @param op_type Type of operation (0=matmul, 1=add, 2=mul, 3=activation)
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 * @param config SM61-specific configuration
 * @return cudaError_t CUDA error code
 */
template<typename T>
cudaError_t launch_memory_efficient_ops_sm61(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    int op_type,
    int batch_size,
    int seq_len,
    int hidden_dim,
    const SM61MemoryCopyConfig& config
);

/**
 * @brief Coalesced memory copy kernel for SM61
 * 
 * Optimized memory copy with coalesced access patterns for SM61 architecture.
 * 
 * @param dst Destination tensor
 * @param src Source tensor
 * @param n_elements Number of elements to copy
 * @param config SM61-specific configuration
 * @return cudaError_t CUDA error code
 */
template<typename T>
cudaError_t launch_coalesced_memory_copy_sm61(
    T* __restrict__ dst,
    const T* __restrict__ src,
    size_t n_elements,
    const SM61MemoryCopyConfig& config
);

/**
 * @brief Optimized matrix transpose kernel for SM61
 * 
 * Transpose matrix with bank-conflict avoidance for SM61 architecture.
 * 
 * @param output Output tensor (transposed)
 * @param input Input tensor
 * @param rows Number of rows in input
 * @param cols Number of columns in input
 * @param config SM61-specific configuration
 * @return cudaError_t CUDA error code
 */
template<typename T>
cudaError_t launch_transpose_sm61(
    T* __restrict__ output,
    const T* __restrict__ input,
    int rows,
    int cols,
    const SM61TransposeConfig& config
);

/**
 * @brief SM61-optimized memory pool class
 * 
 * Implements a memory pool optimized for SM61 architecture with
 * consideration for its memory hierarchy and compute characteristics.
 */
class SM61MemoryPool {
private:
    void* pool_memory;
    size_t pool_size;
    std::vector<bool> block_free;  // Tracks free blocks
    std::vector<size_t> block_sizes;  // Sizes of each block
    mutable std::mutex pool_mutex;  // Mutex for thread safety
    
public:
    explicit SM61MemoryPool(size_t size = 64 * 1024 * 1024);  // Default 64MB
    ~SM61MemoryPool();
    
    /**
     * @brief Allocate memory from the pool
     * @param size Size of memory to allocate
     * @return Pointer to allocated memory, or nullptr if allocation fails
     */
    void* allocate(size_t size);
    
    /**
     * @brief Deallocate memory back to the pool
     * @param ptr Pointer to memory to deallocate
     * @param size Size of memory to deallocate
     */
    void deallocate(void* ptr, size_t size);
    
    /**
     * @brief Clear all memory from the pool
     */
    void clear();
    
    /**
     * @brief Get memory pool statistics
     */
    struct Stats {
        size_t total_size;
        size_t allocated;
        size_t free;
        double fragmentation;
        size_t num_free_blocks;
    };
    
    Stats get_stats() const;
    
    /**
     * @brief Defragment the memory pool
     */
    void defragment();
};

// Explicit template instantiations for commonly used types
extern template cudaError_t launch_scaled_dot_product_attention_sm61<float>(
    const float*, const float*, const float*, float*, float, int, int, int, int, const SM61AttentionConfig&);

extern template cudaError_t launch_scaled_dot_product_attention_sm61<half>(
    const half*, const half*, const half*, half*, float, int, int, int, int, const SM61AttentionConfig&);

extern template cudaError_t launch_block_sparse_attention_sm61<float>(
    const float*, const float*, const float*, float*, const int*, float, int, int, int, int, int, const SM61AttentionConfig&);

extern template cudaError_t launch_block_sparse_attention_sm61<half>(
    const half*, const half*, const half*, half*, const int*, float, int, int, int, int, int, const SM61AttentionConfig&);

extern template cudaError_t launch_high_performance_matmul_sm61<float>(
    const float*, const float*, float*, int, int, int, float, float, const SM61MatmulConfig&);

extern template cudaError_t launch_high_performance_matmul_sm61<half>(
    const half*, const half*, half*, int, int, int, float, float, const SM61MatmulConfig&);

extern template cudaError_t launch_memory_efficient_ops_sm61<float>(
    const float*, float*, const float*, int, int, int, int, const SM61MemoryCopyConfig&);

extern template cudaError_t launch_memory_efficient_ops_sm61<half>(
    const half*, half*, const half*, int, int, int, int, const SM61MemoryCopyConfig&);

extern template cudaError_t launch_coalesced_memory_copy_sm61<float>(
    float*, const float*, size_t, const SM61MemoryCopyConfig&);

extern template cudaError_t launch_coalesced_memory_copy_sm61<half>(
    half*, const half*, size_t, const SM61MemoryCopyConfig&);

extern template cudaError_t launch_transpose_sm61<float>(
    float*, const float*, int, int, const SM61TransposeConfig&);

extern template cudaError_t launch_transpose_sm61<half>(
    half*, const half*, int, int, const SM61TransposeConfig&);

#endif // SM61_OPTIMIZED_KERNELS_H