/*
 * Optimized Memory Copy Routines for SM61 Architecture
 * Implements coalesced, vectorized, and warp-optimized memory copy operations
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

using namespace cooperative_groups;

// Optimized memory copy with coalesced access pattern
__global__ void coalesced_memory_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory utilization
    const int ELEMENTS_PER_THREAD = 4;
    size_t elements_to_process = (count + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    size_t start_idx = idx * ELEMENTS_PER_THREAD;
    
    if (start_idx < count) {
        for (int i = 0; i < ELEMENTS_PER_THREAD && start_idx + i < count; i++) {
            dst[start_idx + i] = src[start_idx + i];
        }
    }
}

// Vectorized memory copy using float4 for better bandwidth utilization
__global__ void vectorized_memory_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements at a time using float4
    size_t vec_idx = idx * 4;
    size_t vec_count = count / 4;
    
    if (vec_idx < vec_count * 4) {
        float4 src_vec = reinterpret_cast<const float4*>(src)[idx];
        reinterpret_cast<float4*>(dst)[idx] = src_vec;
    }
    
    // Handle remaining elements that don't fit in vector
    size_t remaining_start = vec_count * 4;
    size_t remaining_idx = remaining_start + threadIdx.x;
    
    if (remaining_idx < count) {
        dst[remaining_idx] = src[remaining_idx];
    }
}

// Warp-optimized memory copy with shared memory staging
__global__ void warp_optimized_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    extern __shared__ float shared_mem[];
    
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each warp processes a chunk of data
    size_t warp_size = 32;
    size_t warp_id = threadIdx.x / warp_size;
    size_t lane_id = threadIdx.x % warp_size;
    
    // Calculate the chunk this warp will process
    size_t elements_per_warp = (count + gridDim.x * (blockDim.x / warp_size) - 1) / 
                              (gridDim.x * (blockDim.x / warp_size));
    size_t warp_start = (blockIdx.x * (blockDim.x / warp_size) + warp_id) * elements_per_warp;
    
    // Process elements in the warp's chunk
    for (size_t i = lane_id; i < elements_per_warp; i += warp_size) {
        size_t idx = warp_start + i;
        if (idx < count) {
            dst[idx] = src[idx];
        }
    }
}

// Optimized memory copy with memory access pattern optimization
__global__ void pattern_optimized_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count,
    size_t stride  // For strided access patterns
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate global index based on stride
    size_t global_idx = idx * stride;
    
    if (global_idx < count) {
        // Ensure we don't exceed bounds
        size_t elements_to_copy = min((size_t)stride, count - global_idx);
        
        for (size_t i = 0; i < elements_to_copy; i++) {
            if (global_idx + i < count) {
                dst[global_idx + i] = src[global_idx + i];
            }
        }
    }
}

// Asynchronous memory copy with CUDA streams
__global__ void async_memory_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

// Optimized transpose with shared memory for better memory access
__global__ void optimized_transpose_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int width,
    int height
) {
    // Tile size for shared memory
    const int TILE_SIZE = 32;
    
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Calculate indices
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile from source matrix to shared memory
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        int current_y = y + i;
        if (x < width && current_y < height) {
            tile[threadIdx.y + i][threadIdx.x] = src[current_y * width + x];
        }
    }
    
    __syncthreads();
    
    // Calculate transposed indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write transposed tile to destination matrix
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        int current_y = y + i;
        if (x < height && current_y < width) {
            dst[current_y * height + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// Optimized memory copy for half precision (FP16)
__global__ void fp16_optimized_copy_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 2 half values at a time using half2
    size_t half2_idx = idx * 2;
    size_t half2_count = count / 2;
    
    if (half2_idx < half2_count * 2) {
        __half2 src_vec = reinterpret_cast<const __half2*>(src)[idx];
        reinterpret_cast<__half2*>(dst)[idx] = src_vec;
    }
    
    // Handle remaining element if count is odd
    if (count % 2 == 1) {
        size_t last_idx = count - 1;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            dst[last_idx] = src[last_idx];
        }
    }
}

// Optimized memory copy for attention-specific patterns
__global__ void attention_pattern_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads
) {
    // Handle reshaping for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    
    if (batch_id >= batch_size || head_id >= num_heads) return;
    
    // Each block handles one (batch, head) pair
    for (int token_idx = threadIdx.x; token_idx < seq_len; token_idx += blockDim.x) {
        for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
            // Source: [batch, seq, heads, head_dim]
            int src_idx = (batch_id * seq_len * num_heads * head_dim) + 
                         (token_idx * num_heads * head_dim) + 
                         (head_id * head_dim) + d;
            
            // Destination: [batch, heads, seq, head_dim] 
            int dst_idx = (batch_id * num_heads * seq_len * head_dim) + 
                         (head_id * seq_len * head_dim) + 
                         (token_idx * head_dim) + d;
            
            if (src_idx < batch_size * seq_len * num_heads * head_dim &&
                dst_idx < batch_size * num_heads * seq_len * head_dim) {
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

// Memory copy with padding to avoid bank conflicts
__global__ void padded_memory_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int cols,
    int padded_cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        // Source is unpadded, destination is padded
        int src_idx = row * cols + col;
        int dst_idx = row * padded_cols + col;
        dst[dst_idx] = src[src_idx];
    }
}

// Optimized copy for block-sparse patterns
__global__ void block_sparse_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ block_mask,  // 1 where copy should happen, 0 otherwise
    int rows,
    int cols,
    int block_size
) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Calculate global indices
    int global_row = block_row * block_size + thread_row;
    int global_col = block_col * block_size + thread_col;
    
    // Check if this block is active in the sparse pattern
    int block_mask_idx = block_row * ((cols + block_size - 1) / block_size) + block_col;
    if (block_mask[block_mask_idx] == 0) {
        // This block is masked out, fill with zeros
        if (global_row < rows && global_col < cols) {
            int dst_idx = global_row * cols + global_col;
            dst[dst_idx] = 0.0f;
        }
        return;
    }
    
    // Copy data if in bounds
    if (global_row < rows && global_col < cols) {
        int src_idx = global_row * cols + global_col;
        int dst_idx = global_row * cols + global_col;
        dst[dst_idx] = src[src_idx];
    }
}

// Memory copy with warp-level optimizations for attention QKV splitting
__global__ void qkv_split_copy_kernel(
    const float* __restrict__ src,  // Combined QKV tensor
    float* __restrict__ q,          // Q tensor
    float* __restrict__ k,          // K tensor
    float* __restrict__ v,          // V tensor
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_dim;
    
    if (idx >= total_elements) return;
    
    // Split the combined QKV tensor into separate Q, K, V tensors
    // Input format: [batch, seq, 3*hidden_dim] 
    // Output format: [batch, seq, hidden_dim] for each of Q, K, V
    int src_base = idx * 3;  // Each position has 3 values: Q, K, V
    
    // Calculate which batch, seq, and hidden dim we're processing
    int batch = idx / (seq_len * hidden_dim);
    int remaining = idx % (seq_len * hidden_dim);
    int seq = remaining / hidden_dim;
    int hidden = remaining % hidden_dim;
    
    // Calculate destination indices
    int q_idx = batch * seq_len * hidden_dim + seq * hidden_dim + hidden;
    int k_idx = q_idx;
    int v_idx = q_idx;
    
    // Copy values to separate tensors
    q[q_idx] = src[src_base];
    k[k_idx] = src[src_base + 1];
    v[v_idx] = src[src_base + 2];
}

// Function to launch coalesced memory copy
cudaError_t launch_coalesced_memory_copy(
    const float* src,
    float* dst,
    size_t count,
    cudaStream_t stream = 0
) {
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    
    coalesced_memory_copy_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, count);
    return cudaGetLastError();
}

// Function to launch vectorized memory copy
cudaError_t launch_vectorized_memory_copy(
    const float* src,
    float* dst,
    size_t count,
    cudaStream_t stream = 0
) {
    int block_size = 256;
    int grid_size = (count + 3) / 4;  // For float4 processing
    
    vectorized_memory_copy_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, count);
    return cudaGetLastError();
}

// Function to launch warp-optimized copy
cudaError_t launch_warp_optimized_copy(
    const float* src,
    float* dst,
    size_t count,
    cudaStream_t stream = 0
) {
    int block_size = 256;  // 8 warps per block
    int grid_size = (count + block_size - 1) / block_size;
    
    size_t shared_mem_size = 4 * block_size * sizeof(float);  // For staging data
    
    warp_optimized_copy_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(src, dst, count);
    return cudaGetLastError();
}

// Function to launch optimized transpose
cudaError_t launch_optimized_transpose(
    const float* src,
    float* dst,
    int width,
    int height,
    cudaStream_t stream = 0
) {
    dim3 block_size(32, 32);
    dim3 grid_size((width + 31) / 32, (height + 31) / 32);
    
    optimized_transpose_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, width, height);
    return cudaGetLastError();
}

// Function to launch attention pattern copy
cudaError_t launch_attention_pattern_copy(
    const float* src,
    float* dst,
    int batch_size,
    int seq_len,
    int head_dim,
    int num_heads,
    cudaStream_t stream = 0
) {
    dim3 grid_size(batch_size, num_heads);
    dim3 block_size(min(32, seq_len), min(32, head_dim));
    
    attention_pattern_copy_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, batch_size, seq_len, head_dim, num_heads
    );
    return cudaGetLastError();
}

// Function to launch QKV split copy
cudaError_t launch_qkv_split_copy(
    const float* src,
    float* q,
    float* k,
    float* v,
    int batch_size,
    int seq_len,
    int hidden_dim,
    cudaStream_t stream = 0
) {
    int total_elements = batch_size * seq_len * hidden_dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    qkv_split_copy_kernel<<<grid_size, block_size, 0, stream>>>(
        src, q, k, v, batch_size, seq_len, hidden_dim
    );
    return cudaGetLastError();
}

// Memory copy with error checking
template<typename T>
cudaError_t safe_memory_copy(T* dst, const T* src, size_t count, cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), kind);
    if (err != cudaSuccess) {
        return err;
    }
    
    // Synchronize to catch any asynchronous errors
    return cudaDeviceSynchronize();
}

// Asynchronous memory copy with events for synchronization
cudaError_t async_memory_copy_with_events(
    float* dst,
    const float* src,
    size_t count,
    cudaStream_t stream,
    cudaEvent_t start_event,
    cudaEvent_t stop_event
) {
    // Record start event
    cudaEventRecord(start_event, stream);
    
    // Launch async copy
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    
    async_memory_copy_kernel<<<grid_size, block_size, 0, stream>>>(src, dst, count);
    
    // Record stop event
    cudaEventRecord(stop_event, stream);
    
    return cudaGetLastError();
}