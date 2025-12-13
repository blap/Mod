/*
 * Optimized Attention Computation Kernel for NVIDIA SM61 Architecture
 * Compute Capability 6.1 - Pascal Architecture
 * Features:
 * - Memory coalescing optimized for GP104 architecture
 * - Shared memory usage optimized for 96KB per SM
 * - Warp-efficient computation patterns
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Configuration constants for SM61 architecture
#define SHARED_MEM_SIZE 48000  // Max shared memory per block (48KB)
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_WARPS_PER_SM 64

// Define the maximum sequence length that fits in shared memory
#define MAX_SEQ_LEN_SHARED 512

/**
 * @brief Optimized scaled dot-product attention kernel for SM61
 * Implements: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
 */
template<typename T>
__global__ void scaled_dot_product_attention_kernel(
    const T* __restrict__ query,     // [batch_size, seq_len, num_heads, head_dim]
    const T* __restrict__ key,       // [batch_size, seq_len, num_heads, head_dim]  
    const T* __restrict__ value,     // [batch_size, seq_len, num_heads, head_dim]
    T* __restrict__ output,          // [batch_size, seq_len, num_heads, head_dim]
    float scale_factor,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    // Calculate indices
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (token_id >= seq_len) return;
    
    // Shared memory for temporary computations
    extern __shared__ float sdata[];
    
    // Compute attention scores for this token against all keys
    float max_score = -INFINITY;
    float sum_exp_scores = 0.0f;
    
    // Temporary storage for attention weights
    float att_weights[MAX_SEQ_LEN_SHARED];
    
    // Compute query * key_t for all keys
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
        att_weights[k_idx] = score;
        
        // Track maximum for numerical stability in softmax
        max_score = fmaxf(max_score, score);
    }
    
    // Apply softmax with numerical stability
    float sum = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        att_weights[k_idx] = expf(att_weights[k_idx] - max_score);
        sum += att_weights[k_idx];
    }
    
    // Normalize attention weights
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        att_weights[k_idx] /= sum;
    }
    
    // Compute weighted sum: attention_weights * values
    for (int d = 0; d < head_dim; d++) {
        float result = 0.0f;
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            int v_idx = ((batch_id * seq_len + k_idx) * num_heads + head_id) * head_dim + d;
            result += att_weights[k_idx] * static_cast<float>(value[v_idx]);
        }
        
        int out_idx = ((batch_id * seq_len + token_id) * num_heads + head_id) * head_dim + d;
        output[out_idx] = static_cast<T>(result);
    }
}

/**
 * @brief Optimized attention kernel using shared memory for better performance
 * Uses shared memory to cache frequently accessed data
 */
template<typename T>
__global__ void scaled_dot_product_attention_shared_mem_kernel(
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
    // Calculate indices
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (token_id >= seq_len) return;
    
    // Shared memory allocation - dynamic based on head_dim
    extern __shared__ char s_mem_char[];
    float* s_mem_float = reinterpret_cast<float*>(s_mem_char);
    
    // Partition shared memory
    // First part: for caching query vector
    float* s_query = &s_mem_float[0];
    // Second part: for intermediate attention scores
    float* s_att_scores = &s_mem_float[head_dim];
    
    // Load query vector into shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        if (d < head_dim) {
            int q_idx = ((batch_id * seq_len + token_id) * num_heads + head_id) * head_dim + d;
            s_query[d] = static_cast<float>(query[q_idx]);
        }
    }
    
    __syncthreads();
    
    // Pre-calculate key-value pairs for this token
    float max_score = -INFINITY;
    
    // Process keys in chunks to fit in shared memory
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += blockDim.x) {
        int k_idx = chunk_start + threadIdx.x;
        
        float score = 0.0f;
        if (k_idx < seq_len) {
            // Compute dot product with cached query
            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                int k_full_idx = ((batch_id * seq_len + k_idx) * num_heads + head_id) * head_dim + d;
                score += s_query[d] * static_cast<float>(key[k_full_idx]);
            }
            score *= scale_factor;
            s_att_scores[threadIdx.x] = score;
            max_score = fmaxf(max_score, score);
        } else {
            s_att_scores[threadIdx.x] = -INFINITY;
        }
        
        __syncthreads();
    }
    
    // Apply softmax with numerical stability
    // First, subtract max for numerical stability
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (i < seq_len) {
            s_att_scores[i] = expf(s_att_scores[i] - max_score);
        }
    }
    
    __syncthreads();
    
    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (i < seq_len) {
            sum_exp += s_att_scores[i];
        }
    }
    
    // Perform reduction to get total sum
    __shared__ float s_sum_exp[WARP_SIZE];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Each warp performs partial reduction
    float warp_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (i < seq_len) {
            warp_sum += s_att_scores[i];
        }
    }
    
    // Reduce within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }
    
    if (lane_id == 0) {
        s_sum_exp[warp_id] = warp_sum;
    }
    
    __syncthreads();
    
    // Reduce across warps
    if (warp_id == 0) {
        float block_sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? s_sum_exp[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
        
        if (lane_id == 0) {
            s_sum_exp[0] = block_sum;
        }
    }
    
    __syncthreads();
    
    // Normalize attention scores
    float total_sum = s_sum_exp[0];
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (i < seq_len) {
            s_att_scores[i] /= total_sum;
        }
    }
    
    __syncthreads();
    
    // Compute final output: weighted sum of values
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float result = 0.0f;
        
        #pragma unroll 8
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            int v_idx = ((batch_id * seq_len + k_idx) * num_heads + head_id) * head_dim + d;
            result += s_att_scores[k_idx] * static_cast<float>(value[v_idx]);
        }
        
        int out_idx = ((batch_id * seq_len + token_id) * num_heads + head_id) * head_dim + d;
        output[out_idx] = static_cast<T>(result);
    }
}

/**
 * @brief Launch configuration helper for SM61 optimized attention kernel
 */
template<typename T>
cudaError_t launch_scaled_dot_product_attention(
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
) {
    // Determine optimal block and grid dimensions for SM61
    dim3 block_dim(min(seq_len, MAX_THREADS_PER_BLOCK));
    dim3 grid_dim(batch_size, num_heads, (seq_len + block_dim.x - 1) / block_dim.x);
    
    // Calculate required shared memory
    size_t shared_mem_size = (head_dim + seq_len) * sizeof(float);
    
    // Use the shared memory optimized version if it fits
    if (shared_mem_size <= SHARED_MEM_SIZE && seq_len <= MAX_SEQ_LEN_SHARED) {
        scaled_dot_product_attention_shared_mem_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
            query, key, value, output, scale_factor, batch_size, seq_len, num_heads, head_dim
        );
    } else {
        // Fallback to basic kernel if shared memory requirements too high
        size_t basic_shared_mem = seq_len * sizeof(float); // Only for attention weights
        scaled_dot_product_attention_kernel<<<grid_dim, block_dim, basic_shared_mem, stream>>>(
            query, key, value, output, scale_factor, batch_size, seq_len, num_heads, head_dim
        );
    }
    
    return cudaGetLastError();
}

// Implementation of coalesced memory copy kernel
template<typename T>
__global__ void coalesced_memory_copy(
    T* __restrict__ dst,
    const T* __restrict__ src,
    size_t n_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        dst[idx] = src[idx];
    }
}

// Implementation of coalesced matrix transpose kernel
template<typename T>
__global__ void coalesced_matrix_transpose(
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

// Explicit template instantiations
template cudaError_t launch_scaled_dot_product_attention<float>(
    const float*, const float*, const float*, float*, float, int, int, int, int, cudaStream_t);

template cudaError_t launch_scaled_dot_product_attention<half>(
    const half*, const half*, const half*, half*, float, int, int, int, int, cudaStream_t);

// Explicit template instantiations for the coalesced operations
template __global__ void coalesced_memory_copy<float>(
    float* __restrict__ dst,
    const float* __restrict__ src,
    size_t n_elements
);

template __global__ void coalesced_memory_copy<half>(
    half* __restrict__ dst,
    const half* __restrict__ src,
    size_t n_elements
);

template __global__ void coalesced_matrix_transpose<float>(
    float* __restrict__ output,
    const float* __restrict__ input,
    int rows,
    int cols
);

template __global__ void coalesced_matrix_transpose<half>(
    half* __restrict__ output,
    const half* __restrict__ input,
    int rows,
    int cols
);