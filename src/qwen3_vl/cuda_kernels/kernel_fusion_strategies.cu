/*
 * Kernel Fusion Strategies for SM61 Architecture
 * Implements various fusion patterns to reduce kernel launch overhead and memory traffic
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>

using namespace cooperative_groups;

// Structure to represent a fused kernel operation
struct FusedOperation {
    enum OpType {
        LINEAR,
        LAYER_NORM,
        ACTIVATION,
        ATTENTION,
        MATMUL,
        ADD,
        RESHAPE
    };
    
    OpType type;
    void* params;
    size_t param_size;
    
    FusedOperation(OpType t, void* p, size_t s) : type(t), params(p), param_size(s) {}
};

// Fused LayerNorm + Linear + Activation
__global__ void fused_layer_norm_linear_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ ln_weight,
    const float* __restrict__ ln_bias,
    const float* __restrict__ linear_weight,
    const float* __restrict__ linear_bias,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps = 1e-5f
) {
    // Create thread block and warp groups
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int batch_seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;
    
    if (batch_seq_id >= total_elements) return;
    
    // Calculate base indices
    int batch_id = batch_seq_id / seq_len;
    int seq_id = batch_seq_id % seq_len;
    int input_base = batch_seq_id * hidden_dim;
    
    // Step 1: Compute LayerNorm statistics
    float sum = 0.0f, sq_sum = 0.0f;
    for (int d = 0; d < hidden_dim; d++) {
        float val = input[input_base + d];
        sum += val;
        sq_sum += val * val;
    }
    
    // Use warp operations to reduce across hidden dimension
    float thread_sum = sum;
    float thread_sq_sum = sq_sum;
    
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        thread_sq_sum += __shfl_down_sync(0xFFFFFFFF, thread_sq_sum, offset);
    }
    
    // Broadcast final sum to all threads
    float final_sum = __shfl_sync(0xFFFFFFFF, thread_sum, 0);
    float final_sq_sum = __shfl_sync(0xFFFFFFFF, thread_sq_sum, 0);
    
    float mean = final_sum / hidden_dim;
    float var = final_sq_sum / hidden_dim - mean * mean;
    float inv_std = rsqrtf(var + eps);
    
    // Step 2: Apply LayerNorm and Linear transformation in one pass
    extern __shared__ float shared_mem[];
    float* norm_vals = shared_mem;
    
    for (int d = 0; d < hidden_dim; d++) {
        float norm_val = (input[input_base + d] - mean) * inv_std * ln_weight[d] + ln_bias[d];
        norm_vals[d] = norm_val;
    }
    
    __syncthreads();
    
    // Step 3: Linear transformation + Activation
    // For simplicity, assume output dimension equals input dimension
    for (int d = 0; d < hidden_dim; d++) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            sum += norm_vals[k] * linear_weight[d * hidden_dim + k];
        }
        sum += linear_bias[d];
        
        // Apply activation (SiLU: x * sigmoid(x))
        float activated = sum / (1.0f + expf(-sum));
        
        int output_idx = input_base + d;
        output[output_idx] = activated;
    }
}

// Fused Attention + Add + LayerNorm
__global__ void fused_attention_add_layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ qkv_weights,
    const float* __restrict__ attn_output_weights,
    const float* __restrict__ norm_weights,
    const float* __restrict__ norm_bias,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads,
    float eps = 1e-5f
) {
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
    int input_offset = (batch_id * seq_len + token_id) * hidden_dim;

    // Step 1: Apply QKV projection
    float q[128], k[128], v[128]; // Cache first 128 dimensions
    for (int d = 0; d < min(head_dim, 128); d++) {
        int qkv_idx = input_offset + head_id * head_dim + d;
        q[d] = input[input_offset + d] * qkv_weights[0 * hidden_dim * head_dim + d * head_dim + d]; // Simplified
        k[d] = input[input_offset + d] * qkv_weights[1 * hidden_dim * head_dim + d * head_dim + d];
        v[d] = input[input_offset + d] * qkv_weights[2 * hidden_dim * head_dim + d * head_dim + d];
    }

    // Step 2: Compute attention
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float attn_result[128] = {0.0f};

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        // Compute attention score: Q * K
        float score = 0.0f;
        for (int d = 0; d < min(head_dim, 128); d++) {
            int k_linear_idx = qkv_offset + k_idx * head_dim + d;
            score += q[d] * k[d]; // Simplified
        }
        
        // Scale by sqrt(head_dim)
        score = score / sqrtf((float)head_dim);
        
        // Update max for numerical stability
        max_score = fmaxf(max_score, score);
        
        float exp_score = expf(score - max_score);
        
        // Accumulate weighted values
        for (int d = 0; d < min(head_dim, 128); d++) {
            attn_result[d] += exp_score * v[d]; // Simplified
        }
        sum_exp += exp_score;
    }

    // Normalize attention result
    for (int d = 0; d < min(head_dim, 128); d++) {
        attn_result[d] = attn_result[d] / sum_exp;
    }

    // Step 3: Apply attention output projection
    float proj_result[128] = {0.0f};
    for (int d = 0; d < min(head_dim, 128); d++) {
        for (int k = 0; k < head_dim; k++) {
            proj_result[d] += attn_result[k] * attn_output_weights[d * head_dim + k];
        }
    }

    // Step 4: Add residual connection
    float residual[128];
    for (int d = 0; d < min(hidden_dim, 128); d++) {
        residual[d] = input[input_offset + d] + proj_result[d % head_dim];
    }

    // Step 5: Apply layer normalization to residual
    float sum = 0.0f, sq_sum = 0.0f;
    for (int d = 0; d < min(hidden_dim, 128); d++) {
        float val = residual[d];
        sum += val;
        sq_sum += val * val;
    }

    float mean = sum / min(hidden_dim, 128);
    float var = sq_sum / min(hidden_dim, 128) - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Write final output
    for (int d = 0; d < min(hidden_dim, 128); d++) {
        int out_idx = input_offset + d;
        output[out_idx] = (residual[d] - mean) * inv_std * norm_weights[d] + norm_bias[d];
    }
}

// Fused MLP block: Linear1 + Activation + Linear2 + Add residual
__global__ void fused_mlp_block_kernel(
    const float* __restrict__ input,
    const float* __restrict__ fc1_weights,
    const float* __restrict__ fc1_bias,
    const float* __restrict__ fc2_weights,
    const float* __restrict__ fc2_bias,
    const float* __restrict__ residual_input,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int intermediate_dim
) {
    int batch_seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_dim;
    
    if (batch_seq_id >= batch_size * seq_len) return;
    
    int input_base = batch_seq_id * hidden_dim;
    int output_base = input_base;
    int residual_base = input_base;
    
    // Process one sequence element per thread
    // FC1: hidden_dim -> intermediate_dim
    float intermediate[4096]; // Assuming max intermediate size
    for (int i = 0; i < intermediate_dim; i++) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
            sum += input[input_base + h] * fc1_weights[i * hidden_dim + h];
        }
        sum += fc1_bias[i];
        
        // Apply activation (GeLU approximation)
        float x = sum;
        float tanh_arg = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        intermediate[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
    
    // FC2: intermediate_dim -> hidden_dim
    for (int h = 0; h < hidden_dim; h++) {
        float sum = 0.0f;
        for (int i = 0; i < intermediate_dim; i++) {
            sum += intermediate[i] * fc2_weights[h * intermediate_dim + i];
        }
        sum += fc2_bias[h];
        
        // Add residual connection
        output[output_base + h] = sum + residual_input[residual_base + h];
    }
}

// Fused QKV projection + attention computation
__global__ void fused_qkv_attention_kernel(
    const float* __restrict__ input,
    const float* __restrict__ q_weights,
    const float* __restrict__ k_weights,
    const float* __restrict__ v_weights,
    const float* __restrict__ attn_weights,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads
) {
    int batch_id = blockIdx.x / num_heads;
    int head_id = blockIdx.x % num_heads;
    int token_id = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_id >= batch_size || token_id >= seq_len) return;

    // Calculate base offset
    int input_offset = (batch_id * seq_len + token_id) * hidden_dim;
    int head_offset = head_id * head_dim;
    int out_offset = (batch_id * num_heads + head_id) * seq_len * head_dim + token_id * head_dim;

    // Compute Q, K, V projections in shared memory
    extern __shared__ float shared_mem[];
    float* q_vals = shared_mem;
    float* k_vals = shared_mem + head_dim;
    float* v_vals = shared_mem + 2 * head_dim;

    // Project input to Q, K, V
    for (int d = 0; d < head_dim; d++) {
        float input_val = input[input_offset + head_offset + d];
        
        // Q projection
        float q_sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            q_sum += input[input_offset + k] * q_weights[head_offset * hidden_dim + d * hidden_dim + k];
        }
        q_vals[d] = q_sum;
        
        // K projection
        float k_sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            k_sum += input[input_offset + k] * k_weights[head_offset * hidden_dim + d * hidden_dim + k];
        }
        k_vals[d] = k_sum;
        
        // V projection
        float v_sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            v_sum += input[input_offset + k] * v_weights[head_offset * hidden_dim + d * hidden_dim + k];
        }
        v_vals[d] = v_sum;
    }

    __syncthreads();

    // Compute attention scores
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float result[1024] = {0.0f}; // Assuming max head_dim

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        // Load K values for this key position
        float k_vals_k[1024];
        for (int d = 0; d < head_dim; d++) {
            int k_offset = (batch_id * num_heads + head_id) * seq_len * head_dim + k_idx * head_dim + d;
            k_vals_k[d] = k_vals[d]; // Simplified - in practice, we'd load the k_idx-th K
        }
        
        // Compute attention score: Q * K
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vals[d] * k_vals_k[d];
        }
        
        // Scale by sqrt(head_dim)
        score = score / sqrtf((float)head_dim);
        
        // Update max for numerical stability
        max_score = fmaxf(max_score, score);
        
        float exp_score = expf(score - max_score);
        
        // Load V values for this key position and accumulate
        for (int d = 0; d < head_dim; d++) {
            int v_offset = (batch_id * num_heads + head_id) * seq_len * head_dim + k_idx * head_dim + d;
            result[d] += exp_score * v_vals[d]; // Simplified
        }
        sum_exp += exp_score;
    }

    // Normalize and apply output projection
    for (int d = 0; d < head_dim; d++) {
        result[d] = result[d] / sum_exp;
        
        // Apply output projection
        float out_sum = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            out_sum += result[k] * attn_weights[d * head_dim + k];
        }
        
        int final_out_idx = (batch_id * seq_len + token_id) * hidden_dim + head_offset + d;
        output[final_out_idx] = out_sum;
    }
}

// Fused embedding + position encoding
__global__ void fused_embedding_pos_encoding_kernel(
    const int* __restrict__ input_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ pos_embeddings,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size
) {
    int batch_seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;
    
    if (batch_seq_id >= total_elements) return;
    
    int batch_id = batch_seq_id / seq_len;
    int seq_id = batch_seq_id % seq_len;
    int token_id = input_ids[batch_seq_id];
    
    if (token_id >= vocab_size) return; // Out of bounds check
    
    int output_base = batch_seq_id * hidden_dim;
    int token_emb_base = token_id * hidden_dim;
    int pos_emb_base = seq_id * hidden_dim;
    
    // Add token embedding and position embedding
    for (int d = 0; d < hidden_dim; d++) {
        output[output_base + d] = token_embeddings[token_emb_base + d] + 
                                 pos_embeddings[pos_emb_base + d];
    }
}

// Fused softmax + sampling for generation
__global__ void fused_softmax_sampling_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int* __restrict__ samples,
    float temperature,
    int batch_size,
    int vocab_size
) {
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_id >= batch_size) return;
    
    int logit_base = batch_id * vocab_size;
    
    // Find max for numerical stability
    float max_logit = -INFINITY;
    for (int v = 0; v < vocab_size; v++) {
        max_logit = fmaxf(max_logit, logits[logit_base + v]);
    }
    
    // Compute exp with temperature scaling and sum for normalization
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        float exp_val = expf((logits[logit_base + v] - max_logit) / temperature);
        probs[logit_base + v] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize probabilities
    for (int v = 0; v < vocab_size; v++) {
        probs[logit_base + v] = probs[logit_base + v] / sum_exp;
    }
    
    // Simple sampling (in practice, you'd use a proper random number generator)
    float random_val = 0.5f; // Placeholder - would use actual RNG
    float cum_sum = 0.0f;
    int sampled_token = 0;
    
    for (int v = 0; v < vocab_size; v++) {
        cum_sum += probs[logit_base + v];
        if (random_val <= cum_sum) {
            sampled_token = v;
            break;
        }
    }
    
    samples[batch_id] = sampled_token;
}

// Multi-head attention fusion with query-key-value processing
__global__ void fused_multi_head_attention_kernel(
    const float* __restrict__ input,
    const float* __restrict__ qkv_weights,
    const float* __restrict__ attn_output_weights,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads
) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int token_id = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_id >= batch_size || head_id >= num_heads || token_id >= seq_len) return;
    
    int input_base = (batch_id * seq_len + token_id) * hidden_dim;
    int head_offset = head_id * head_dim;
    int qkv_base = (batch_id * num_heads + head_id) * seq_len * head_dim;
    
    // Shared memory for caching Q, K, V values
    extern __shared__ float shared_mem[];
    float* shared_q = shared_mem;
    float* shared_k = shared_mem + head_dim;
    float* shared_v = shared_mem + 2 * head_dim;
    float* shared_scores = shared_mem + 3 * head_dim;
    
    // Compute Q, K, V projections for this head and token
    for (int d = 0; d < head_dim; d++) {
        float input_val = input[input_base + head_offset + d];
        
        // Q projection
        float q_sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            q_sum += input[input_base + k] * qkv_weights[0 * hidden_dim * hidden_dim + head_offset * hidden_dim + d * hidden_dim + k];
        }
        shared_q[d] = q_sum;
        
        // K projection (for all sequence positions)
        float k_sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            k_sum += input[input_base + k] * qkv_weights[1 * hidden_dim * hidden_dim + head_offset * hidden_dim + d * hidden_dim + k];
        }
        shared_k[d] = k_sum;  // This is simplified - in reality, we'd compute K for all positions
        
        // V projection (for all sequence positions)
        float v_sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            v_sum += input[input_base + k] * qkv_weights[2 * hidden_dim * hidden_dim + head_offset * hidden_dim + d * hidden_dim + k];
        }
        shared_v[d] = v_sum;  // This is simplified - in reality, we'd compute V for all positions
    }
    
    __syncthreads();
    
    // Compute attention for this token across all key positions
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float result[1024] = {0.0f};  // Assuming max head_dim
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        // Compute attention score: Q * K (for key at k_idx)
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int k_offset = (batch_id * num_heads + head_id) * seq_len * head_dim + k_idx * head_dim + d;
            score += shared_q[d] * shared_k[d];  // Simplified
        }
        
        // Scale by sqrt(head_dim)
        score = score / sqrtf((float)head_dim);
        
        // Update max for numerical stability
        max_score = fmaxf(max_score, score);
        
        float exp_score = expf(score - max_score);
        shared_scores[k_idx] = exp_score;
        sum_exp += exp_score;
    }
    
    __syncthreads();
    
    // Normalize scores and compute weighted sum of V values
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        shared_scores[k_idx] = shared_scores[k_idx] / sum_exp;
        
        for (int d = 0; d < head_dim; d++) {
            int v_offset = (batch_id * num_heads + head_id) * seq_len * head_dim + k_idx * head_dim + d;
            result[d] += shared_scores[k_idx] * shared_v[d];  // Simplified
        }
    }
    
    __syncthreads();
    
    // Apply output projection for this head
    for (int d = 0; d < head_dim; d++) {
        float out_sum = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            out_sum += result[k] * attn_output_weights[head_offset * head_dim + d * head_dim + k];
        }
        
        int out_idx = (batch_id * seq_len + token_id) * hidden_dim + head_offset + d;
        output[out_idx] = out_sum;
    }
}

// Function to launch fused layer norm + linear + activation
cudaError_t launch_fused_layer_norm_linear_activation(
    const float* input,
    const float* ln_weight,
    const float* ln_bias,
    const float* linear_weight,
    const float* linear_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    cudaStream_t stream = 0
) {
    int total_elements = batch_size * seq_len;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    size_t shared_mem_size = hidden_dim * sizeof(float);
    
    fused_layer_norm_linear_activation_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, ln_weight, ln_bias, linear_weight, linear_bias, output,
        batch_size, seq_len, hidden_dim
    );
    
    return cudaGetLastError();
}

// Function to launch fused attention + add + layer norm
cudaError_t launch_fused_attention_add_layernorm(
    const float* input,
    const float* qkv_weights,
    const float* attn_output_weights,
    const float* norm_weights,
    const float* norm_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads,
    cudaStream_t stream = 0
) {
    dim3 block_size(256);
    dim3 grid_size(num_heads * batch_size, (seq_len + 7) / 8);  // 8 tokens per 8 warps
    
    size_t shared_mem_size = min(hidden_dim, 128) * sizeof(float);
    
    fused_attention_add_layernorm_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, qkv_weights, attn_output_weights, norm_weights, norm_bias, output,
        batch_size, seq_len, hidden_dim, head_dim, num_heads
    );
    
    return cudaGetLastError();
}

// Function to launch fused MLP block
cudaError_t launch_fused_mlp_block(
    const float* input,
    const float* fc1_weights,
    const float* fc1_bias,
    const float* fc2_weights,
    const float* fc2_bias,
    const float* residual_input,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream = 0
) {
    int total_elements = batch_size * seq_len;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_mlp_block_kernel<<<grid_size, block_size, 0, stream>>>(
        input, fc1_weights, fc1_bias, fc2_weights, fc2_bias, residual_input, output,
        batch_size, seq_len, hidden_dim, intermediate_dim
    );
    
    return cudaGetLastError();
}

// Function to launch fused QKV + attention
cudaError_t launch_fused_qkv_attention(
    const float* input,
    const float* q_weights,
    const float* k_weights,
    const float* v_weights,
    const float* attn_weights,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads,
    cudaStream_t stream = 0
) {
    dim3 block_size(256);
    dim3 grid_size(num_heads * batch_size, (seq_len + block_size.x - 1) / block_size.x);
    
    size_t shared_mem_size = 3 * head_dim * sizeof(float);
    
    fused_qkv_attention_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, q_weights, k_weights, v_weights, attn_weights, output,
        batch_size, seq_len, hidden_dim, head_dim, num_heads
    );
    
    return cudaGetLastError();
}

// Function to launch fused multi-head attention
cudaError_t launch_fused_multi_head_attention(
    const float* input,
    const float* qkv_weights,
    const float* attn_output_weights,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int head_dim,
    int num_heads,
    cudaStream_t stream = 0
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size, num_heads, (seq_len + block_size.x - 1) / block_size.x);
    
    size_t shared_mem_size = (3 * head_dim + seq_len) * sizeof(float);
    
    fused_multi_head_attention_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, qkv_weights, attn_output_weights, output,
        batch_size, seq_len, hidden_dim, head_dim, num_heads
    );
    
    return cudaGetLastError();
}