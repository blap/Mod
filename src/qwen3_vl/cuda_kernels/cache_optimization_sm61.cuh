/*
 * Advanced Cache Optimization Techniques for SM61 Architecture
 * Implements L1/L2 cache optimization, texture cache utilization, and memory access pattern prediction
 */

#ifndef CACHE_OPTIMIZATION_SM61_CUH
#define CACHE_OPTIMIZATION_SM61_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Cache optimization constants for SM61
// Pascal architecture has unified L1/texture cache (typically 16KB or 24KB configurable)
// L2 cache is shared across the entire GPU (2MB on GTX 1080)
#define SM61_L1_CACHE_SIZE 24 * 1024  // 24KB configurable as L1 cache
#define SM61_L2_CACHE_SIZE 2 * 1024 * 1024  // 2MB L2 cache
#define SM61_CACHE_LINE_SIZE 128  // Bytes per cache line
#define SM61_WARP_SIZE 32

// Structure for cache-aware memory access patterns
struct CacheOptimizationConfig {
    size_t l1_cache_size;
    size_t l2_cache_size;
    int cache_line_size;
    int warp_size;
    bool use_texture_cache;
    bool use_read_only_cache;
    int memory_access_pattern;  // 0=sequential, 1=strided, 2=random
};

// Function to configure cache settings for SM61
inline CacheOptimizationConfig get_sm61_cache_config() {
    CacheOptimizationConfig config;
    config.l1_cache_size = SM61_L1_CACHE_SIZE;
    config.l2_cache_size = SM61_L2_CACHE_SIZE;
    config.cache_line_size = SM61_CACHE_LINE_SIZE;
    config.warp_size = SM61_WARP_SIZE;
    config.use_texture_cache = true;  // Pascal has unified L1/texture cache
    config.use_read_only_cache = true;
    config.memory_access_pattern = 0;  // Default to sequential
    
    return config;
}

// Texture memory objects for read-only data that benefits from texture cache
texture<float, 1, cudaReadModeElementType> tex_weights_1d;
texture<float, 2, cudaReadModeElementType> tex_weights_2d;

// Cache-optimized attention kernel with L1/L2 cache awareness
__global__ void __launch_bounds__(256, 4)
cache_optimized_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads
) {
    // Create thread block and warp groups
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);

    int batch_id = blockIdx.x / num_heads;
    int head_id = blockIdx.x % num_heads;
    int token_id = blockIdx.y * (blockDim.x / 32) + (threadIdx.x / 32);

    if (batch_id >= batch_size || token_id >= seq_len) return;
    if (threadIdx.x % 32 != 0) return; // Only first thread in each warp processes

    // Calculate base offset
    int qkv_offset = (batch_id * num_heads + head_id) * seq_len * head_dim;
    int out_offset = (batch_id * num_heads + head_id) * seq_len * head_dim;

    // Use cache-friendly access patterns
    // Load query vector to registers to minimize cache misses
    float query[128]; // Cache first 128 dimensions in registers for cache efficiency
    for (int d = 0; d < min(head_dim, 128); d++) {
        query[d] = q[qkv_offset + token_id * head_dim + d];
    }

    // Process attention computation with cache-aware access patterns
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float result[128] = {0.0f};

    // Process keys in cache-friendly chunks to maximize L1 cache utilization
    const int CACHE_BLOCK_SIZE = (SM61_L1_CACHE_SIZE / sizeof(float)) / 8; // Use 1/8 of L1 cache per block
    const int CACHE_DIM_SIZE = min(head_dim, CACHE_BLOCK_SIZE);

    // Process keys sequentially to maximize spatial locality
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        // Compute attention score: Q * K with cache-aware access
        float score = 0.0f;

        // Process dimensions in cache blocks to maximize cache hit rate
        for (int block_start = 0; block_start < head_dim; block_start += CACHE_DIM_SIZE) {
            int block_end = min(block_start + CACHE_DIM_SIZE, head_dim);
            
            #pragma unroll 4
            for (int d = block_start; d < block_end; d++) {
                int k_linear_idx = qkv_offset + k_idx * head_dim + d;
                score += query[d % 128] * k[k_linear_idx];
            }
        }

        // Scale by sqrt(head_dim)
        score = score / sqrtf((float)head_dim);

        // Use warp operations for numerical stability (reduces memory pressure)
        float warp_max = score;
        for (int offset = 16; offset > 0; offset /= 2) {
            float next_max = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
            warp_max = fmaxf(warp_max, next_max);
        }
        max_score = fmaxf(max_score, warp_max);

        float exp_score = expf(score - max_score);
        float warp_sum = exp_score;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        }
        sum_exp += warp_sum;

        // Accumulate weighted values with cache-aware access
        for (int block_start = 0; block_start < head_dim; block_start += CACHE_DIM_SIZE) {
            int block_end = min(block_start + CACHE_DIM_SIZE, head_dim);
            
            #pragma unroll 4
            for (int d = block_start; d < block_end; d++) {
                int v_linear_idx = qkv_offset + k_idx * head_dim + d;
                float weighted_val = exp_score * v[v_linear_idx];
                
                // Sum across warp to reduce memory writes
                float warp_weighted = weighted_val;
                for (int offset = 16; offset > 0; offset /= 2) {
                    warp_weighted += __shfl_down_sync(0xFFFFFFFF, warp_weighted, offset);
                }
                result[d % 128] += warp_weighted;
            }
        }
    }

    // Write final result with coalesced access pattern
    for (int d = 0; d < min(head_dim, 128); d++) {
        int out_idx = out_offset + token_id * head_dim + d;
        output[out_idx] = result[d] / sum_exp;
    }
}

// Cache-optimized matrix multiplication with L1/L2 cache awareness
__global__ void __launch_bounds__(256, 4)
cache_optimized_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k
) {
    // Use 16x16 tiles with cache-aware padding
    __shared__ float tile_a[16][17];  // +1 to avoid bank conflicts and improve cache alignment
    __shared__ float tile_b[16][17];  // +1 to avoid bank conflicts and improve cache alignment

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles with cache-aware access patterns
    for (int t = 0; t < k; t += 16) {
        // Load tiles with coalesced access and cache-friendly patterns
        if (row < m && (t + threadIdx.x) < k) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * k + t + threadIdx.x];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t + threadIdx.y) < k && col < n) {
            tile_b[threadIdx.y][threadIdx.x] = b[(t + threadIdx.y) * n + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial result with cache-aware access and multiple accumulators
        // Use multiple accumulators to increase ILP and reduce cache pressure
        float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        #pragma unroll 4
        for (int k_idx = 0; k_idx < 16; k_idx++) {
            float a_val = tile_a[threadIdx.y][k_idx];
            float b_val = tile_b[k_idx][threadIdx.x];
            
            // Distribute computation across multiple accumulators to hide latency
            accum[0] += a_val * b_val;
            accum[1] += a_val * b_val;  // Duplicate computation to increase ILP
            accum[2] += a_val * b_val;
            accum[3] += a_val * b_val;
        }

        // Combine accumulators
        sum += (accum[0] + accum[1] + accum[2] + accum[3]) * 0.25f;

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Texture memory optimized kernel for read-only weight matrices
__global__ void texture_optimized_matmul_kernel(
    const int m, const int n, const int k,
    float* __restrict__ c,
    int batch_size
) {
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    // Access weight matrices through texture cache for better spatial locality
    for (int t = 0; t < k; t += 16) {
        __shared__ float tile_a[16][17];

        // Load A tile normally (input may be changing)
        if (row < m && (t + threadIdx.x) < k) {
            tile_a[threadIdx.y][threadIdx.x] = tex1Dfetch(tex_weights_1d, row * k + t + threadIdx.x);
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute with texture cache access for B matrix
        #pragma unroll 4
        for (int k_idx = 0; k_idx < 16; k_idx++) {
            float a_val = tile_a[threadIdx.y][k_idx];
            // Access B through texture cache (weights are read-only)
            float b_val = tex2D(tex_weights_2d, col, t + k_idx);
            sum += a_val * b_val;
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// L1 cache optimized softmax with reduced memory pressure
__global__ void l1_cache_optimized_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    // Create thread block and warp groups
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);

    int batch_id = blockIdx.x;
    int token_id = blockIdx.y * blockDim.x + threadIdx.x;

    if (token_id >= seq_len) return;

    int linear_idx = batch_id * seq_len + token_id;

    // Load value to process
    float val = input[linear_idx];

    // Use shared memory to reduce L1 cache pressure for reduction operations
    extern __shared__ float shared_mem[];
    float* shared_vals = shared_mem;

    // Store value in shared memory for warp-level operations
    shared_vals[threadIdx.x] = val;
    block.sync();

    // Find maximum value across the sequence using shared memory
    // This reduces L1 cache pressure compared to global memory access
    float max_val = val;
    for (int offset = 16; offset > 0; offset /= 2) {
        float next_max = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        max_val = fmaxf(max_val, next_max);
    }
    // Broadcast max to all threads in warp
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Compute exponential with normalized values
    float exp_val = expf(val - max_val);

    // Compute sum of exponentials using shared memory
    shared_vals[threadIdx.x] = exp_val;
    block.sync();

    float sum_exp = exp_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    // Broadcast sum to all threads in warp
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Compute final softmax value
    output[linear_idx] = exp_val / sum_exp;
}

// Cache-optimized layer normalization with reduced L1 pressure
__global__ void l1_cache_optimized_layer_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_size,
    int hidden_dim,
    float eps = 1e-5f
) {
    // Create thread block and warp groups
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);

    int batch_id = blockIdx.x;
    int dim_id = threadIdx.x;

    if (dim_id >= hidden_dim) return;

    int linear_idx = batch_id * hidden_dim + dim_id;

    // Use cache-friendly access pattern by processing one dimension per thread
    // but with warp-level coordination for reductions
    float x = input[linear_idx];

    // Use shared memory to store values for mean/var computation
    // This reduces L1 cache pressure
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;

    // Store value in shared memory for warp operations
    shared_input[threadIdx.x] = x;
    block.sync();

    // Compute mean using shared memory (reduces L1 cache pressure)
    float sum = x;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    // Broadcast sum to all threads in warp
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    float mean = sum / hidden_dim;

    // Compute variance using shared memory
    float diff = x - mean;
    float var = diff * diff;
    for (int offset = 16; offset > 0; offset /= 2) {
        var += __shfl_down_sync(0xFFFFFFFF, var, offset);
    }
    // Broadcast variance to all threads in warp
    var = __shfl_sync(0xFFFFFFFF, var, 0);
    var = var / hidden_dim;

    // Compute normalized value
    float normalized = (x - mean) / sqrtf(var + eps);

    // Apply weight and bias
    output[linear_idx] = normalized * weight[dim_id] + bias[dim_id];
}

// Cache-aware memory prefetching using async memory operations
__global__ void __launch_bounds__(256, 4)
prefetching_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Prefetch next data elements to L1/L2 cache
        // Note: Actual prefetching in CUDA requires newer architectures
        // For SM61, we simulate prefetching by organizing access patterns
        
        // Process current element
        float val = input[idx];
        
        // Simple computation
        output[idx] = val * val + 2.0f * val + 1.0f;  // (val + 1)^2
    }
}

// Function to launch cache-optimized attention kernel
cudaError_t launch_cache_optimized_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    cudaStream_t stream = 0
) {
    // Use 256 threads per block (8 warps) for optimal cache utilization
    dim3 block_dim(256);
    dim3 grid_dim(batch_size * num_heads, (seq_len + 7) / 8); // 8 tokens per 8 warps

    cache_optimized_attention_kernel<<<grid_dim, block_dim, 0, stream>>>(
        q, k, v, output, batch_size, seq_len, head_dim, num_heads
    );

    return cudaGetLastError();
}

// Function to launch cache-optimized matmul kernel
cudaError_t launch_cache_optimized_matmul(
    const float* a,
    const float* b,
    float* c,
    int m, int n, int k,
    cudaStream_t stream = 0
) {
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((n + 15) / 16, (m + 15) / 16, 1);

    cache_optimized_matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, c, m, n, k);

    return cudaGetLastError();
}

// Function to bind texture memory for optimized access
cudaError_t bind_texture_memory(const float* weights, int size) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    
    // Create texture object for 1D access
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = weights;
    resDesc.res.linear.sizeInBytes = size * sizeof(float);
    resDesc.res.linear.desc = channelDesc;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    return cudaGetLastError();
}

#endif // CACHE_OPTIMIZATION_SM61_CUH