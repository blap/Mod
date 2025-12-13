#include "attention_kernel.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Optimized attention kernel for SM61 architecture
__global__ void attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads
) {
    // Calculate indices
    int batch_id = blockIdx.x / num_heads;
    int head_id = blockIdx.x % num_heads;
    int seq_id = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_id >= batch_size || seq_id >= seq_len) return;
    
    // Shared memory for caching Q, K, V values
    extern __shared__ float shared_mem[];
    float* shared_q = shared_mem;
    float* shared_k = shared_mem + head_dim * ATTENTION_TILE_SIZE;
    float* shared_v = shared_mem + 2 * head_dim * ATTENTION_TILE_SIZE;
    float* shared_scores = shared_mem + 3 * head_dim * ATTENTION_TILE_SIZE;
    
    // Calculate base pointers for current batch and head
    int qkv_offset = (batch_id * num_heads + head_id) * seq_len * head_dim;
    int out_offset = (batch_id * num_heads + head_id) * seq_len * head_dim;
    
    // Initialize attention scores - use shared memory for variable-length arrays
    float* scores = &shared_mem[3 * head_dim * ATTENTION_TILE_SIZE];
    float* values = &shared_mem[3 * head_dim * ATTENTION_TILE_SIZE + ATTENTION_TILE_SIZE];

    // Initialize output values
    for (int d = 0; d < head_dim; d++) {
        values[d] = 0.0f;
    }
    
    // Process attention computation
    for (int k_seq = 0; k_seq < seq_len; k_seq += ATTENTION_TILE_SIZE) {
        // Load Q values to shared memory
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            int q_idx = qkv_offset + seq_id * head_dim + d;
            if (d < head_dim && seq_id < seq_len) {
                shared_q[threadIdx.y * head_dim + d] = q[q_idx];
            }
        }
        
        // Load K values to shared memory
        for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                int k_linear_idx = qkv_offset + (k_seq + k_idx) * head_dim + d;
                if (d < head_dim) {
                    shared_k[k_idx * head_dim + d] = k[k_linear_idx];
                }
            }
        }
        
        __syncthreads();
        
        // Compute attention scores for current tile
        for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
            float score = 0.0f;
            
            // Compute dot product between Q and K
            for (int d = 0; d < head_dim; d++) {
                score += shared_q[threadIdx.y * head_dim + d] * shared_k[k_idx * head_dim + d];
            }
            
            // Scale by sqrt(head_dim)
            score = score / sqrtf((float)head_dim);
            
            // Store the score
            shared_scores[k_idx] = score;
        }
        
        __syncthreads();
        
        // Apply softmax to scores in shared memory
        // For simplicity, we'll implement a basic softmax across the tile
        float max_score = -INFINITY;
        for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
            max_score = fmaxf(max_score, shared_scores[k_idx]);
        }
        
        // Compute sum of exponentials for normalization
        float exp_sum = 0.0f;
        for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
            float exp_score = expf(shared_scores[k_idx] - max_score);
            shared_scores[k_idx] = exp_score;  // Store exp values
            exp_sum += exp_score;
        }
        
        // Normalize the scores
        for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
            shared_scores[k_idx] = shared_scores[k_idx] / exp_sum;
        }
        
        __syncthreads();
        
        // Load V values to shared memory
        for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                int v_linear_idx = qkv_offset + (k_seq + k_idx) * head_dim + d;
                if (d < head_dim) {
                    shared_v[k_idx * head_dim + d] = v[v_linear_idx];
                }
            }
        }
        
        __syncthreads();
        
        // Compute weighted sum of V values
        for (int d = 0; d < head_dim; d++) {
            float weighted_val = 0.0f;
            for (int k_idx = 0; k_idx < ATTENTION_TILE_SIZE && (k_seq + k_idx) < seq_len; k_idx++) {
                weighted_val += shared_scores[k_idx] * shared_v[k_idx * head_dim + d];
            }
            values[d] += weighted_val;
        }
        
        __syncthreads();
    }
    
    // Write results to output
    for (int d = 0; d < head_dim; d++) {
        int out_idx = out_offset + seq_id * head_dim + d;
        if (seq_id < seq_len) {
            output[out_idx] = values[d];
        }
    }
}

// Function to compute optimal kernel configuration for SM61
KernelConfig get_attention_config(int batch_size, int seq_len, int head_dim) {
    KernelConfig config;
    
    // For SM61: 128 cores per SM, max 64 warps per SM (2048 threads per SM)
    // Optimize for 4 warps per block (128 threads per block) for better occupancy
    int threads_per_block = ATTENTION_THREADS_PER_BLOCK;
    int threads_y = ATTENTION_WARPS_PER_BLOCK;  // 4 warps
    int threads_x = WARP_SIZE;                  // 32 threads per warp
    
    config.block = dim3(threads_x, threads_y, 1);
    
    // Calculate grid dimensions
    int num_heads = 1;  // Simplified for single head - in practice this would be configurable
    config.grid = dim3(batch_size * num_heads, (seq_len + config.block.y - 1) / config.block.y, 1);
    
    // Calculate shared memory size needed
    size_t shared_mem_per_block = 3 * head_dim * ATTENTION_TILE_SIZE * sizeof(float) +  // For Q, K, V
                                  ATTENTION_TILE_SIZE * sizeof(float) +  // For scores
                                  head_dim * sizeof(float);              // For values
    config.shared_mem_size = shared_mem_per_block;
    
    // Ensure we don't exceed SM61's shared memory limit (48KB per block)
    if (config.shared_mem_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        config.shared_mem_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK;
    }
    
    return config;
}