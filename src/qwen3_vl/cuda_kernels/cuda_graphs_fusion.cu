/*
 * Custom CUDA Graphs for Kernel Fusion on SM61 Architecture
 * Implements graph-based execution for fusing multiple operations to reduce kernel launch overhead
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Graph node for attention computation
struct AttentionGraphNode {
    float* q;
    float* k;
    float* v;
    float* output;
    int batch_size;
    int seq_len;
    int head_dim;
    int num_heads;
    
    cudaGraphNode_t node;
};

// Graph node for matrix multiplication
struct MatmulGraphNode {
    float* a;
    float* b;
    float* c;
    int m, n, k;
    
    cudaGraphNode_t node;
};

// Graph node for activation function
struct ActivationGraphNode {
    float* input;
    float* output;
    int size;
    
    cudaGraphNode_t node;
};

// CUDA Graph Manager for kernel fusion
class CUDAGraphManager {
private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    std::vector<cudaGraphNode_t> nodes_;
    cudaStream_t stream_;
    
public:
    CUDAGraphManager() {
        cudaGraphCreate(&graph_, 0);
        cudaStreamCreate(&stream_);
    }
    
    ~CUDAGraphManager() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    
    // Add attention node to graph
    cudaGraphNode_t add_attention_node(
        float* q, float* k, float* v, float* output,
        int batch_size, int seq_len, int head_dim, int num_heads
    ) {
        cudaKernelNodeParams attention_params = {0};
        
        void* attention_args[] = {
            &q, &k, &v, &output, 
            &batch_size, &seq_len, &head_dim, &num_heads
        };
        
        attention_params.func = (void*)fused_attention_kernel;
        attention_params.gridDim = dim3((batch_size * num_heads + 3) / 4, (seq_len + 63) / 64);
        attention_params.blockDim = dim3(256);
        attention_params.sharedMemBytes = (head_dim * 32 + seq_len) * sizeof(float);
        attention_params.kernelParams = (void**)attention_args;
        attention_params.extra = nullptr;
        
        cudaGraphNode_t attention_node;
        cudaGraphAddKernelNode(&attention_node, graph_, nullptr, 0, &attention_params);
        nodes_.push_back(attention_node);
        
        return attention_node;
    }
    
    // Add matmul node to graph
    cudaGraphNode_t add_matmul_node(float* a, float* b, float* c, int m, int n, int k) {
        cudaKernelNodeParams matmul_params = {0};
        
        void* matmul_args[] = {&a, &b, &c, &m, &n, &k};
        
        matmul_params.func = (void*)fused_matmul_kernel;
        matmul_params.gridDim = dim3((n + 15) / 16, (m + 15) / 16);
        matmul_params.blockDim = dim3(256);
        matmul_params.sharedMemBytes = 2 * 16 * 16 * sizeof(float);  // For tiles
        matmul_params.kernelParams = (void**)matmul_args;
        matmul_params.extra = nullptr;
        
        cudaGraphNode_t matmul_node;
        cudaGraphAddKernelNode(&matmul_node, graph_, nullptr, 0, &matmul_params);
        nodes_.push_back(matmul_node);
        
        return matmul_node;
    }
    
    // Add activation node to graph
    cudaGraphNode_t add_activation_node(float* input, float* output, int size) {
        cudaKernelNodeParams activation_params = {0};
        
        void* activation_args[] = {&input, &output, &size};
        
        activation_params.func = (void*)fused_activation_kernel;
        activation_params.gridDim = dim3((size + 255) / 256);
        activation_params.blockDim = dim3(256);
        activation_params.sharedMemBytes = 0;
        activation_params.kernelParams = (void**)activation_args;
        activation_params.extra = nullptr;
        
        cudaGraphNode_t activation_node;
        cudaGraphAddKernelNode(&activation_node, graph_, nullptr, 0, &activation_params);
        nodes_.push_back(activation_node);
        
        return activation_node;
    }
    
    // Create dependencies between nodes
    void add_dependency(cudaGraphNode_t from, cudaGraphNode_t to) {
        cudaGraphAddDependencies(graph_, &from, &to, 1);
    }
    
    // Instantiate and launch the graph
    cudaError_t instantiate_and_launch() {
        // Instantiate the graph
        cudaError_t err = cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
        if (err != cudaSuccess) return err;
        
        // Launch the graph
        return cudaGraphLaunch(graph_exec_, stream_);
    }
    
    // Synchronize the stream
    cudaError_t synchronize() {
        return cudaStreamSynchronize(stream_);
    }
    
    // Create a fused attention-matmul graph for transformer layer
    cudaError_t create_transformer_fusion_graph(
        float* q, float* k, float* v, 
        float* attn_output,
        float* attn_weights,  // After attention projection
        float* ff_input,      // After residual connection
        float* ff_intermediate, // After first FF layer
        float* ff_output,     // After second FF layer
        int batch_size, int seq_len, int head_dim, int num_heads,
        int hidden_dim, int intermediate_dim
    ) {
        // Create nodes for the transformer layer
        auto attn_node = add_attention_node(q, k, v, attn_output, 
                                          batch_size, seq_len, head_dim, num_heads);
        
        auto proj_node = add_matmul_node(attn_output, attn_weights, ff_input,
                                       batch_size * seq_len, hidden_dim, hidden_dim);
        
        auto ff1_node = add_matmul_node(ff_input, attn_weights, ff_intermediate,  // Reusing attn_weights for example
                                      batch_size * seq_len, intermediate_dim, hidden_dim);
        
        auto act_node = add_activation_node(ff_intermediate, ff_intermediate, 
                                         batch_size * seq_len * intermediate_dim);
        
        auto ff2_node = add_matmul_node(ff_intermediate, attn_weights, ff_output,  // Reusing for example
                                      batch_size * seq_len, hidden_dim, intermediate_dim);
        
        // Add dependencies to create the execution order
        add_dependency(attn_node, proj_node);
        add_dependency(proj_node, ff1_node);
        add_dependency(ff1_node, act_node);
        add_dependency(act_node, ff2_node);
        
        return cudaSuccess;
    }
};

// Fused kernel implementations

// Fused attention kernel with optional layer norm
__global__ void fused_attention_kernel(
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

    // Load query vector to registers
    float query[128]; // Cache first 128 dimensions
    for (int d = 0; d < min(head_dim, 128); d++) {
        query[d] = q[qkv_offset + token_id * head_dim + d];
    }

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float result[128] = {0.0f};

    // Process keys with warp-level coordination
    for (int k_idx = warp.thread_rank(); k_idx < seq_len; k_idx += warp.size()) {
        // Compute attention score: Q * K
        float score = 0.0f;
        
        // Use warp operations for partial dot product
        #pragma unroll 8
        for (int d = 0; d < min(head_dim, 128); d++) {
            int k_linear_idx = qkv_offset + k_idx * head_dim + d;
            float partial = query[d] * k[k_linear_idx];
            
            // Sum across warp
            for (int offset = 16; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
            }
            score += partial;
        }
        
        // Compute remaining dimensions
        for (int d = 128; d < head_dim; d += warp.size()) {
            int adjusted_d = d + warp.thread_rank();
            if (adjusted_d < head_dim) {
                int k_linear_idx = qkv_offset + k_idx * head_dim + adjusted_d;
                score += q[qkv_offset + token_id * head_dim + adjusted_d] * k[k_linear_idx];
            }
        }
        
        // Scale by sqrt(head_dim)
        score = score / sqrtf((float)head_dim);
        
        // Use warp operations for numerical stability
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
        
        // Accumulate weighted values
        #pragma unroll 8
        for (int d = 0; d < min(head_dim, 128); d++) {
            int v_linear_idx = qkv_offset + k_idx * head_dim + d;
            float weighted_val = exp_score * v[v_linear_idx];
            
            // Sum across warp
            float warp_weighted = weighted_val;
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_weighted += __shfl_down_sync(0xFFFFFFFF, warp_weighted, offset);
            }
            result[d] += warp_weighted;
        }
    }

    // Write final result
    for (int d = 0; d < min(head_dim, 128); d++) {
        int out_idx = out_offset + token_id * head_dim + d;
        output[out_idx] = result[d] / sum_exp;
    }
}

// Fused matmul kernel
__global__ void fused_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k
) {
    // Use 16x16 tiles with proper padding
    __shared__ float tile_a[16][17];  // 17 to avoid bank conflicts
    __shared__ float tile_b[16][17];  // 17 to avoid bank conflicts

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < k; t += 16) {
        // Load tiles with coalesced access
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

        // Compute partial result with unrolling for ILP
        #pragma unroll 4
        for (int k_idx = 0; k_idx < 16; k_idx++) {
            sum += tile_a[threadIdx.y][k_idx] * tile_b[k_idx][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Fused activation kernel
__global__ void fused_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        
        // SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
        float result = x / (1.0f + expf(-x));
        
        output[idx] = result;
    }
}

// Fused layer norm + attention + residual
__global__ void fused_layer_norm_attention_residual_kernel(
    const float* __restrict__ input,
    const float* __restrict__ norm_weights,
    const float* __restrict__ norm_bias,
    const float* __restrict__ q_weights,
    const float* __restrict__ k_weights, 
    const float* __restrict__ v_weights,
    const float* __restrict__ attn_weights,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads,
    float eps = 1e-5f
) {
    // Create thread block and warp groups
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int batch_id = blockIdx.x;
    int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_id >= batch_size || seq_id >= seq_len) return;
    
    int linear_idx = batch_id * seq_len * hidden_dim + seq_id * hidden_dim;
    
    // Step 1: Layer normalization
    extern __shared__ float shared_mem[];
    float* input_cache = shared_mem;
    float* norm_cache = shared_mem + hidden_dim;
    
    // Load input to shared memory
    for (int d = 0; d < hidden_dim; d++) {
        input_cache[d] = input[linear_idx + d];
    }
    
    // Compute mean and variance for layer norm
    float sum = 0.0f, sq_sum = 0.0f;
    for (int d = 0; d < hidden_dim; d++) {
        sum += input_cache[d];
        sq_sum += input_cache[d] * input_cache[d];
    }
    
    float mean = sum / hidden_dim;
    float var = sq_sum / hidden_dim - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Normalize and apply weights/bias
    for (int d = 0; d < hidden_dim; d++) {
        norm_cache[d] = (input_cache[d] - mean) * inv_std * norm_weights[d] + norm_bias[d];
    }
    
    // Step 2: Apply Q, K, V projections
    // For simplicity, assume dense projections are applied separately
    // In practice, these would be fused as well
    
    // Step 3: Perform attention (simplified)
    // This would be a full attention computation with Q*K*V
    // For this example, we'll just pass through the normalized values
    for (int d = 0; d < hidden_dim; d++) {
        output[linear_idx + d] = norm_cache[d] + input[linear_idx + d]; // Add residual
    }
}

// Fused MLP block: FC1 -> Activation -> FC2
__global__ void fused_mlp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ fc1_weights,
    const float* __restrict__ fc2_weights,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int intermediate_dim
) {
    int batch_seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_dim;
    
    if (batch_seq_id >= batch_size * seq_len) return;
    
    // Each thread processes one sequence element
    int input_base = batch_seq_id * hidden_dim;
    int output_base = batch_seq_id * hidden_dim;
    
    // FC1: hidden_dim -> intermediate_dim
    float intermediate[4096]; // Assuming max intermediate size
    for (int i = 0; i < intermediate_dim; i++) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
            sum += input[input_base + h] * fc1_weights[i * hidden_dim + h];
        }
        intermediate[i] = sum;
    }
    
    // Apply activation (SiLU)
    for (int i = 0; i < intermediate_dim; i++) {
        intermediate[i] = intermediate[i] / (1.0f + expf(-intermediate[i]));
    }
    
    // FC2: intermediate_dim -> hidden_dim
    for (int h = 0; h < hidden_dim; h++) {
        float sum = 0.0f;
        for (int i = 0; i < intermediate_dim; i++) {
            sum += intermediate[i] * fc2_weights[h * intermediate_dim + i];
        }
        output[output_base + h] = sum;
    }
}

// Function to create and execute a fused transformer layer graph
cudaError_t execute_fused_transformer_layer(
    float* input,
    float* q_weights, float* k_weights, float* v_weights, float* attn_weights,
    float* fc1_weights, float* fc2_weights,
    float* output,
    int batch_size, int seq_len, int hidden_dim, int num_heads, int intermediate_dim,
    cudaStream_t stream = 0
) {
    CUDAGraphManager graph_manager;
    
    // Calculate derived dimensions
    int head_dim = hidden_dim / num_heads;
    
    // Create device memory for intermediate results
    float *attn_output, *norm_output, *intermediate_output;
    cudaMalloc(&attn_output, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&norm_output, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&intermediate_output, batch_size * seq_len * intermediate_dim * sizeof(float));
    
    // Create the fused graph
    cudaError_t err = graph_manager.create_transformer_fusion_graph(
        input, input, input,  // Using input as Q, K, V for simplicity
        attn_output, attn_weights,
        norm_output,
        intermediate_output,
        output,
        batch_size, seq_len, head_dim, num_heads,
        hidden_dim, intermediate_dim
    );
    
    if (err != cudaSuccess) {
        cudaFree(attn_output);
        cudaFree(norm_output);
        cudaFree(intermediate_output);
        return err;
    }
    
    // Instantiate and launch the graph
    err = graph_manager.instantiate_and_launch();
    if (err != cudaSuccess) {
        cudaFree(attn_output);
        cudaFree(norm_output);
        cudaFree(intermediate_output);
        return err;
    }
    
    // Synchronize
    err = graph_manager.synchronize();
    
    // Cleanup
    cudaFree(attn_output);
    cudaFree(norm_output);
    cudaFree(intermediate_output);
    
    return err;
}

// Function to create a custom CUDA graph for attention + MLP fusion
cudaGraphExec_t create_attention_mlp_fusion_graph(
    float* input,
    float* attn_weights,
    float* mlp_weights1,
    float* mlp_weights2,
    float* output,
    int batch_size, int seq_len, int hidden_dim, int intermediate_dim,
    cudaStream_t stream
) {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    cudaGraphCreate(&graph, 0);
    
    // Create memory for intermediate results
    float *attn_result, *mlp_input, *mlp_intermediate;
    cudaMalloc(&attn_result, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&mlp_input, batch_size * seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&mlp_intermediate, batch_size * seq_len * intermediate_dim * sizeof(float));
    
    // Add nodes to the graph
    cudaGraphNode_t attn_node, add_node, mlp1_node, act_node, mlp2_node;
    
    // Attention node
    {
        cudaKernelNodeParams attn_params = {0};
        void* attn_args[] = {&input, &input, &input, &attn_result, 
                            &batch_size, &seq_len, &hidden_dim, &batch_size};
        attn_params.func = (void*)fused_attention_kernel;
        attn_params.gridDim = dim3((batch_size + 3) / 4, (seq_len + 63) / 64);
        attn_params.blockDim = dim3(256);
        attn_params.sharedMemBytes = (hidden_dim * 32 + seq_len) * sizeof(float);
        attn_params.kernelParams = (void**)attn_args;
        cudaGraphAddKernelNode(&attn_node, graph, nullptr, 0, &attn_params);
    }
    
    // Add residual connection
    {
        cudaKernelNodeParams add_params = {0};
        void* add_args[] = {&input, &attn_result, &mlp_input, 
                           &batch_size, &seq_len, &hidden_dim};
        add_params.func = (void*)fused_add_kernel;
        add_params.gridDim = dim3((batch_size * seq_len * hidden_dim + 255) / 256);
        add_params.blockDim = dim3(256);
        add_params.sharedMemBytes = 0;
        add_params.kernelParams = (void**)add_args;
        cudaGraphAddKernelNode(&add_node, graph, &attn_node, 1, &add_params);
    }
    
    // MLP first layer
    {
        cudaKernelNodeParams mlp1_params = {0};
        void* mlp1_args[] = {&mlp_input, &mlp_weights1, &mlp_intermediate, 
                            &batch_size, &intermediate_dim, &hidden_dim};
        mlp1_params.func = (void*)fused_matmul_kernel;
        mlp1_params.gridDim = dim3((intermediate_dim + 15) / 16, (batch_size * seq_len + 15) / 16);
        mlp1_params.blockDim = dim3(256);
        mlp1_params.sharedMemBytes = 2 * 16 * 16 * sizeof(float);
        mlp1_params.kernelParams = (void**)mlp1_args;
        cudaGraphAddKernelNode(&mlp1_node, graph, &add_node, 1, &mlp1_params);
    }
    
    // Activation
    {
        cudaKernelNodeParams act_params = {0};
        int total_intermediate = batch_size * seq_len * intermediate_dim;
        void* act_args[] = {&mlp_intermediate, &mlp_intermediate, &total_intermediate};
        act_params.func = (void*)fused_activation_kernel;
        act_params.gridDim = dim3((total_intermediate + 255) / 256);
        act_params.blockDim = dim3(256);
        act_params.sharedMemBytes = 0;
        act_params.kernelParams = (void**)act_args;
        cudaGraphAddKernelNode(&act_node, graph, &mlp1_node, 1, &act_params);
    }
    
    // MLP second layer
    {
        cudaKernelNodeParams mlp2_params = {0};
        void* mlp2_args[] = {&mlp_intermediate, &mlp_weights2, &output, 
                            &batch_size, &hidden_dim, &intermediate_dim};
        mlp2_params.func = (void*)fused_matmul_kernel;
        mlp2_params.gridDim = dim3((hidden_dim + 15) / 16, (batch_size * seq_len + 15) / 16);
        mlp2_params.blockDim = dim3(256);
        mlp2_params.sharedMemBytes = 2 * 16 * 16 * sizeof(float);
        mlp2_params.kernelParams = (void**)mlp2_args;
        cudaGraphAddKernelNode(&mlp2_node, graph, &act_node, 1, &mlp2_params);
    }
    
    // Instantiate the graph
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    // Cleanup graph (not the executable)
    cudaGraphDestroy(graph);
    
    return graphExec;
}

// Simple fused add kernel for residual connections
__global__ void fused_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int batch_size, int seq_len, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * hidden_dim;
    
    if (idx < total_size) {
        c[idx] = a[idx] + b[idx];
    }
}