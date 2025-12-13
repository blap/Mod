/*
 * SM61-Optimized CUDA Kernels Implementation
 * Optimized for NVIDIA SM61 Architecture (Compute Capability 6.1)
 * Features specific optimizations for memory access, register usage, and compute patterns
 */

#include "sm61_optimized_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// SM61-specific configurations
#define SM61_MAX_THREADS_PER_BLOCK 1024
#define SM61_WARP_SIZE 32
#define SM61_MAX_SHARED_MEMORY 48 * 1024  // 48KB per block
#define SM61_MAX_REGISTERS_PER_THREAD 255

/**
 * @brief Optimized scaled dot-product attention kernel for SM61
 */
template<typename T>
__global__ void scaled_dot_product_attention_sm61_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    // Calculate thread indices
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;

    if (token_id >= seq_len) return;

    // Use shared memory to store intermediate results
    extern __shared__ float sdata[];
    float* s_attention_scores = &sdata[0];
    float* s_softmax_denominator = &sdata[seq_len];

    // Calculate query * key_t for all keys
    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;

        // Compute dot product: query[token_id] * key[k_idx]
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            int q_idx = ((batch_id * seq_len + token_id) * num_heads + head_id) * head_dim + d;
            int k_idx_full = ((batch_id * seq_len + k_idx) * num_heads + head_id) * head_dim + d;

            score += static_cast<float>(query[q_idx]) * static_cast<float>(key[k_idx_full]);
        }

        score *= scale_factor;
        s_attention_scores[k_idx] = score;

        // Track maximum for numerical stability in softmax
        max_score = fmaxf(max_score, score);
    }

    // Apply softmax with numerical stability
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        s_attention_scores[k_idx] = expf(s_attention_scores[k_idx] - max_score);
        sum_exp += s_attention_scores[k_idx];
    }

    // Normalize attention weights
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        s_attention_scores[k_idx] /= sum_exp;
    }

    // Compute weighted sum: attention_weights * values
    for (int d = 0; d < head_dim; d++) {
        float result = 0.0f;

        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            int v_idx = ((batch_id * seq_len + k_idx) * num_heads + head_id) * head_dim + d;
            result += s_attention_scores[k_idx] * static_cast<float>(value[v_idx]);
        }

        int out_idx = ((batch_id * seq_len + token_id) * num_heads + head_id) * head_dim + d;
        output[out_idx] = static_cast<T>(result);
    }
}

/**
 * @brief Optimized block-sparse attention kernel for SM61
 */
template<typename T>
__global__ void block_sparse_attention_sm61_kernel(
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
    int block_size
) {
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int block_row = blockIdx.z;
    int block_col = blockIdx.w;  // Note: using 4D grid indexing
    
    // Check if this block should be computed based on mask
    int mask_idx = block_row * gridDim.w + block_col;
    if (block_mask[mask_idx] == 0) return;  // Skip if block is masked out
    
    // Calculate thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Shared memory for caching blocks
    extern __shared__ float sdata[];
    float* s_q_block = &sdata[0];
    float* s_k_block = &sdata[block_size * head_dim];
    float* s_v_block = &sdata[block_size * head_dim + block_size * head_dim];
    
    // Load query block into shared memory
    int q_base = ((batch_id * seq_len + block_row * block_size + ty) * num_heads + head_id) * head_dim + tx;
    if (block_row * block_size + ty < seq_len && tx < head_dim) {
        s_q_block[ty * head_dim + tx] = static_cast<float>(query[q_base]);
    } else {
        s_q_block[ty * head_dim + tx] = 0.0f;
    }
    
    // Load key block into shared memory
    int k_base = ((batch_id * seq_len + block_col * block_size + ty) * num_heads + head_id) * head_dim + tx;
    if (block_col * block_size + ty < seq_len && tx < head_dim) {
        s_k_block[ty * head_dim + tx] = static_cast<float>(key[k_base]);
    } else {
        s_k_block[ty * head_dim + tx] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute attention scores for this block
    float scores[8];  // Store scores for multiple tokens (unrolled)
    
    #pragma unroll 8
    for (int i = 0; i < 8 && (tx * 8 + i) < block_size; i++) {
        int token_idx = tx * 8 + i;
        if (block_row * block_size + token_idx < seq_len) {
            scores[i] = 0.0f;
            
            #pragma unroll 16
            for (int d = 0; d < head_dim; d++) {
                scores[i] += s_q_block[token_idx * head_dim + d] * s_k_block[ty * head_dim + d];
            }
            scores[i] *= scale_factor;
        }
    }
    
    __syncthreads();
    
    // Load value block into shared memory
    int v_base = ((batch_id * seq_len + block_col * block_size + ty) * num_heads + head_id) * head_dim + tx;
    if (block_col * block_size + ty < seq_len && tx < head_dim) {
        s_v_block[ty * head_dim + tx] = static_cast<float>(value[v_base]);
    } else {
        s_v_block[ty * head_dim + tx] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute output for this block
    #pragma unroll 8
    for (int i = 0; i < 8 && (tx * 8 + i) < block_size; i++) {
        int token_idx = tx * 8 + i;
        if (block_row * block_size + token_idx < seq_len) {
            float result = 0.0f;
            
            #pragma unroll 16
            for (int d = 0; d < block_size && block_col * block_size + d < seq_len; d++) {
                result += scores[i] * s_v_block[d * head_dim + (token_idx % head_dim)];
            }
            
            int out_idx = ((batch_id * seq_len + block_row * block_size + token_idx) * num_heads + head_id) * head_dim + (token_idx % head_dim);
            output[out_idx] = static_cast<T>(result);
        }
    }
}

/**
 * @brief High-performance matrix multiplication kernel for SM61
 */
template<typename T>
__global__ void high_performance_matmul_sm61_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m, int n, int k,
    float alpha, float beta
) {
    // Use tile size optimized for SM61 (warp size multiple)
    const int TILE_SIZE = 16;
    
    // Shared memory for tile caching
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];
    
    // Calculate thread and block indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles of a and b
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < m && a_col < k) {
            shared_a[threadIdx.y][threadIdx.x] = static_cast<float>(a[row * k + a_col]);
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < n && b_row < k) {
            shared_b[threadIdx.y][threadIdx.x] = static_cast<float>(b[b_row * n + col]);
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result to output
    if (row < m && col < n) {
        int c_idx = row * n + col;
        c[c_idx] = static_cast<T>(alpha * sum + beta * static_cast<float>(c[c_idx]));
    }
}

/**
 * @brief Memory-efficient operations kernel for SM61
 */
template<typename T>
__global__ void memory_efficient_ops_sm61_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    int op_type,  // 0=matmul, 1=add, 2=mul, 3=activation
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_dim;
    
    if (idx >= total_elements) return;
    
    float val = static_cast<float>(input[idx]);
    
    switch (op_type) {
        case 0:  // Matmul (not applicable here, this is element-wise)
            output[idx] = static_cast<T>(val);
            break;
        case 1:  // Add
            output[idx] = static_cast<T>(val + static_cast<float>(weight[idx % hidden_dim]));
            break;
        case 2:  // Mul
            output[idx] = static_cast<T>(val * static_cast<float>(weight[idx % hidden_dim]));
            break;
        case 3:  // Activation (SiLU)
            output[idx] = static_cast<T>(val / (1.0f + expf(-val)));  // SiLU activation
            break;
        default:
            output[idx] = static_cast<T>(val);
            break;
    }
}

/**
 * @brief Coalesced memory copy kernel for SM61
 */
template<typename T>
__global__ void coalesced_memory_copy_sm61_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    size_t n_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread copies multiple elements to reduce launch overhead
    size_t elements_per_thread = (n_elements + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    
    for (size_t i = 0; i < elements_per_thread; i++) {
        size_t actual_idx = idx * elements_per_thread + i;
        if (actual_idx < n_elements) {
            dst[actual_idx] = src[actual_idx];
        }
    }
}

/**
 * @brief Optimized matrix transpose kernel for SM61 with bank conflict avoidance
 */
template<typename T>
__global__ void transpose_sm61_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int rows,
    int cols
) {
    // Use 32x32 thread blocks for optimal memory access
    __shared__ T tile[32][33]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load data into shared memory with coalesced access
    for (int j = 0; j < 32; j += 8) {
        if (y + j < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Write transposed data back with coalesced access
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    for (int j = 0; j < 32; j += 8) {
        if (y + j < cols && x < rows) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Template implementations
template<typename T>
cudaError_t launch_scaled_dot_product_attention_sm61(
    const T* query,
    const T* key,
    const T* value,
    T* output,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    const SM61AttentionConfig& config
) {
    // Calculate grid and block dimensions
    dim3 block_size(config.block_size_x, config.block_size_y);
    dim3 grid_size(config.grid_size_x, num_heads, batch_size);
    
    // Calculate shared memory size
    size_t shared_mem_size = config.shared_memory_size;
    
    // Launch kernel
    scaled_dot_product_attention_sm61_kernel<T><<<grid_size, block_size, shared_mem_size, config.stream>>>(
        query, key, value, output, scale_factor, batch_size, seq_len, num_heads, head_dim
    );
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_block_sparse_attention_sm61(
    const T* query,
    const T* key,
    const T* value,
    T* output,
    const int* block_mask,
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int block_size,
    const SM61AttentionConfig& config
) {
    // Calculate grid dimensions for sparse blocks
    int num_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid_size(batch_size, num_heads, num_blocks, num_blocks);
    dim3 block_size_2d(32, 32);  // Using 2D block for spatial locality
    
    // Calculate shared memory requirements
    size_t shared_mem_size = 3 * block_size * head_dim * sizeof(float);
    
    // Launch kernel
    block_sparse_attention_sm61_kernel<T><<<grid_size, block_size_2d, shared_mem_size, config.stream>>>(
        query, key, value, output, block_mask, scale_factor, 
        batch_size, seq_len, num_heads, head_dim, block_size
    );
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_high_performance_matmul_sm61(
    const T* a,
    const T* b,
    T* c,
    int m, int n, int k,
    float alpha, float beta,
    const SM61MatmulConfig& config
) {
    // Use tile size optimized for SM61
    const int TILE_SIZE = 16;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    
    // Shared memory for tiles
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    // Launch kernel
    high_performance_matmul_sm61_kernel<T><<<grid_size, block_size, shared_mem_size, config.stream>>>(
        a, b, c, m, n, k, alpha, beta
    );
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_memory_efficient_ops_sm61(
    const T* input,
    T* output,
    const T* weight,
    int op_type,
    int batch_size,
    int seq_len,
    int hidden_dim,
    const SM61MemoryCopyConfig& config
) {
    int total_elements = batch_size * seq_len * hidden_dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    memory_efficient_ops_sm61_kernel<T><<<grid_size, block_size, 0, config.stream>>>(
        input, output, weight, op_type, batch_size, seq_len, hidden_dim
    );
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_coalesced_memory_copy_sm61(
    T* dst,
    const T* src,
    size_t n_elements,
    const SM61MemoryCopyConfig& config
) {
    int block_size = config.block_size;
    int grid_size = config.grid_size;
    
    coalesced_memory_copy_sm61_kernel<T><<<grid_size, block_size, 0, config.stream>>>(
        dst, src, n_elements
    );
    
    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_transpose_sm61(
    T* output,
    const T* input,
    int rows,
    int cols,
    const SM61TransposeConfig& config
) {
    dim3 block_size = config.block_size;
    dim3 grid_size = config.grid_size;
    
    // Shared memory for tile + padding to avoid bank conflicts
    size_t shared_mem_size = config.shared_memory_size;
    
    transpose_sm61_kernel<T><<<grid_size, block_size, shared_mem_size, config.stream>>>(
        output, input, rows, cols
    );
    
    return cudaGetLastError();
}

// Implementation of SM61MemoryPool
SM61MemoryPool::SM61MemoryPool(size_t size) : pool_size(size) {
    cudaError_t err = cudaMalloc(&pool_memory, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate SM61 memory pool: " + std::string(cudaGetErrorString(err)));
    }
    
    // Initialize block tracking structures
    size_t min_block_size = 256; // Minimum block size in bytes
    size_t num_blocks = size / min_block_size;
    
    block_free.resize(num_blocks, true);
    block_sizes.resize(num_blocks, min_block_size);
    
    // If there's a remainder, create one more block with remaining size
    if (size % min_block_size != 0) {
        block_sizes.push_back(size % min_block_size);
        block_free.push_back(true);
    }
}

SM61MemoryPool::~SM61MemoryPool() {
    if (pool_memory) {
        cudaFree(pool_memory);
    }
}

void* SM61MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Find a free block that's large enough
    for (size_t i = 0; i < block_free.size(); i++) {
        if (block_free[i] && block_sizes[i] >= size) {
            block_free[i] = false;
            char* ptr = static_cast<char*>(pool_memory) + (i * block_sizes[0]); // Assuming uniform block size for simplicity
            return static_cast<void*>(ptr);
        }
    }
    
    // No suitable block found, return nullptr
    return nullptr;
}

void SM61MemoryPool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Calculate which block this pointer corresponds to
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(pool_memory);
    size_t block_idx = offset / block_sizes[0]; // Assuming uniform block size
    
    if (block_idx < block_free.size()) {
        block_free[block_idx] = true;
    }
}

void SM61MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // Mark all blocks as free
    std::fill(block_free.begin(), block_free.end(), true);
}

SM61MemoryPool::Stats SM61MemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    Stats stats;
    stats.total_size = pool_size;
    stats.allocated = 0;
    stats.free = 0;
    stats.num_free_blocks = 0;
    
    for (size_t i = 0; i < block_free.size(); i++) {
        if (block_free[i]) {
            stats.free += block_sizes[i];
            stats.num_free_blocks++;
        } else {
            stats.allocated += block_sizes[i];
        }
    }
    
    // Calculate fragmentation (simplified calculation)
    stats.fragmentation = (stats.num_free_blocks > 0) ? 
        static_cast<double>(stats.num_free_blocks) / static_cast<double>(block_free.size()) : 0.0;
    
    return stats;
}

void SM61MemoryPool::defragment() {
    // For SM61, defragmentation involves consolidating free blocks
    // This is a simplified implementation - a full implementation would
    // require more complex memory management
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // In a real implementation, we would consolidate adjacent free blocks
    // For now, we'll just ensure memory is properly tracked
    cudaDeviceSynchronize();  // Synchronize to ensure all operations are complete
}

// Explicit template instantiations
template cudaError_t launch_scaled_dot_product_attention_sm61<float>(
    const float*, const float*, const float*, float*, float, int, int, int, int, const SM61AttentionConfig&);

template cudaError_t launch_scaled_dot_product_attention_sm61<half>(
    const half*, const half*, const half*, half*, float, int, int, int, int, const SM61AttentionConfig&);

template cudaError_t launch_block_sparse_attention_sm61<float>(
    const float*, const float*, const float*, float*, const int*, float, int, int, int, int, int, const SM61AttentionConfig&);

template cudaError_t launch_block_sparse_attention_sm61<half>(
    const half*, const half*, const half*, half*, const int*, float, int, int, int, int, int, const SM61AttentionConfig&);

template cudaError_t launch_high_performance_matmul_sm61<float>(
    const float*, const float*, float*, int, int, int, float, float, const SM61MatmulConfig&);

template cudaError_t launch_high_performance_matmul_sm61<half>(
    const half*, const half*, half*, int, int, int, float, float, const SM61MatmulConfig&);

template cudaError_t launch_memory_efficient_ops_sm61<float>(
    const float*, float*, const float*, int, int, int, int, const SM61MemoryCopyConfig&);

template cudaError_t launch_memory_efficient_ops_sm61<half>(
    const half*, half*, const half*, int, int, int, int, const SM61MemoryCopyConfig&);

template cudaError_t launch_coalesced_memory_copy_sm61<float>(
    float*, const float*, size_t, const SM61MemoryCopyConfig&);

template cudaError_t launch_coalesced_memory_copy_sm61<half>(
    half*, const half*, size_t, const SM61MemoryCopyConfig&);

template cudaError_t launch_transpose_sm61<float>(
    float*, const float*, int, int, const SM61TransposeConfig&);

template cudaError_t launch_transpose_sm61<half>(
    half*, const half*, int, int, const SM61TransposeConfig&);