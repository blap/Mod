#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../common/tensor.h"

// Helper for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

// --- 1. Int8 Support (DP4A) ---
#if __CUDA_ARCH__ >= 610
#define HAS_DP4A
#endif

__global__ void matmul_int8_kernel(const int8_t* A, const int8_t* B, int* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0;
#ifdef HAS_DP4A
        // Vectorized loop using __dp4a
        // Requires K to be multiple of 4 and data packed in int
        // Assuming tight packing for this demo optimization kernel
        int k_vec = K / 4;
        const int* A_ptr = (const int*)A;
        const int* B_ptr = (const int*)B;

        for (int k = 0; k < k_vec; k++) {
            int a_val = A_ptr[row * k_vec + k];
            // Accessing B is complex if not transposed or packed correctly for coalescing.
            // Assuming B is [K/4, N] packed int for simplicity of this specific kernel
            // Real impl needs specialized packing.
            // Fallback to scalar read for robustness in "General" kernel
            // Implementation Stub for DP4A logic:
            int b_val = B_ptr[k * N + col]; // Strided access, bad.
            sum = __dp4a(a_val, b_val, sum);
        }
#else
        // Fallback Scalar
        for(int k=0; k<K; k++) {
            sum += A[row*K + k] * B[k*N + col];
        }
#endif
        C[row * N + col] = sum;
    }
}

// --- 2. Memory Optimizations ---

EXPORT void tensor_advise_memory(Tensor* t, int advice_enum, int device_id) {
    if (t->device_id >= 0) {
        // advice_enum maps to cudaMemAdvise...
        // 1: SetReadMostly, 2: SetPreferredLocation, 3: SetAccessedBy
        cudaMemoryAdvise advice = cudaMemAdviseSetReadMostly;
        if (advice_enum == 2) advice = cudaMemAdviseSetPreferredLocation;
        if (advice_enum == 3) advice = cudaMemAdviseSetAccessedBy;

        cudaMemAdvise(t->data, t->size * sizeof(float), advice, device_id);
    }
}

// --- 3. Stream Priorities ---
EXPORT void* create_priority_stream(int priority) {
    // priority: -1 (High), 0 (Default)
    cudaStream_t stream;
    int least, greatest;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    int cuda_prio = (priority < 0) ? greatest : least;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, cuda_prio);
    return (void*)stream;
}

// --- Standard Kernels (Optimized) ---

__global__ void fill_kernel(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

__global__ void add_kernel_vec4(float4* a, float4* b, float4* out, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vout;
        vout.x = va.x + vb.x;
        vout.y = va.y + vb.y;
        vout.z = va.z + vb.z;
        vout.w = va.w + vb.w;
        out[idx] = vout;
    }
}

__global__ void add_kernel_scalar(float* a, float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}

__global__ void mul_kernel_vec4(float4* a, float4* b, float4* out, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vout;
        vout.x = va.x * vb.x;
        vout.y = va.y * vb.y;
        vout.z = va.z * vb.z;
        vout.w = va.w * vb.w;
        out[idx] = vout;
    }
}

__global__ void mul_kernel_scalar(float* a, float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * b[idx];
}

__device__ float silu_op(float x) { return x * (1.0f / (1.0f + expf(-x))); }

__global__ void silu_kernel_vec4(float4* input, float4* out, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        float4 v = input[idx];
        v.x = silu_op(v.x);
        v.y = silu_op(v.y);
        v.z = silu_op(v.z);
        v.w = silu_op(v.w);
        out[idx] = v;
    }
}

__global__ void silu_kernel_scalar(float* input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = silu_op(input[idx]);
}

__global__ void fused_add_mul_kernel_vec4(float4* a, float4* b, float4* c, float4* out, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc = c[idx];
        float4 vout;
        vout.x = va.x + vb.x * vc.x;
        vout.y = va.y + vb.y * vc.y;
        vout.z = va.z + vb.z * vc.z;
        vout.w = va.w + vb.w * vc.w;
        out[idx] = vout;
    }
}
__global__ void fused_add_mul_kernel_scalar(float* a, float* b, float* c, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx] * c[idx];
}

__global__ void gelu_kernel(float* input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = x * cdf;
    }
}

__global__ void swiglu_kernel(float* gate, float* up, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float u = up[idx];
        float sg = g / (1.0f + expf(-g));
        out[idx] = sg * u;
    }
}

__global__ void fused_gate_up_swiglu_kernel(float* gate_up, float* out, int hidden, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int row = idx / hidden;
        int col = idx % hidden;
        int in_idx = row * (hidden * 2) + col;
        float g = gate_up[in_idx];
        float u = gate_up[in_idx + hidden];
        float sg = g / (1.0f + expf(-g));
        out[idx] = sg * u;
    }
}

__global__ void rms_norm_kernel(float* input, float* weight, float* out, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row < rows) {
        float sum_sq = 0.0f;
        for (int i = 0; i < cols; i++) {
            float val = input[row * cols + i];
            sum_sq += val * val;
        }
        float scale = 1.0f / sqrtf(sum_sq / cols + eps);
        for (int i = 0; i < cols; i++) {
            out[row * cols + i] = input[row * cols + i] * scale * weight[i];
        }
    }
}

__global__ void fused_add_rms_norm_row_kernel(float* x, const float* residual, const float* weight, float* out, int hidden_size, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[row * hidden_size + i] + residual[row * hidden_size + i];
        x[row * hidden_size + i] = val; // Store back
        sum_sq += val * val;
    }
    // Shared Memory Padding to avoid bank conflicts
    // 32 threads + 1 float padding? Standard reduction doesn't strictly need it if stride is power of 2
    // But declaring volatile helps.
    __shared__ float s_mean;
    if (tid == 0) s_mean = 0.0f;
    __syncthreads();
    atomicAdd(&s_mean, sum_sq);
    __syncthreads();
    float inv_rms = rsqrtf(s_mean / hidden_size + eps);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out[row * hidden_size + i] = x[row * hidden_size + i] * inv_rms * weight[i];
    }
}

__global__ void softmax_kernel(float* input, float* out, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    float max_val = -1e20f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = input[row * cols + i];
        max_val = fmaxf(max_val, val);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    static __shared__ float s_max[32];
    int lane = tid % warpSize;
    int warp = tid / warpSize;
    if (lane == 0) s_max[warp] = max_val;
    __syncthreads();
    if (tid == 0) {
        float block_max = -1e20f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) block_max = fmaxf(block_max, s_max[i]);
        s_max[0] = block_max;
    }
    __syncthreads();
    float global_max = s_max[0];
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += expf(input[row * cols + i] - global_max);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    static __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) block_sum += s_sum[i];
        s_sum[0] = block_sum;
    }
    __syncthreads();
    float global_sum = s_sum[0];
    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = expf(input[row * cols + i] - global_max) * inv_sum;
    }
}

__global__ void matmul_kernel_naive(float* A, float* B, float* C, int TotalM, int K, int N, int batch_count, int M, int broadcast_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < TotalM && col < N) {
        float sum = 0.0f;
        int batch_idx = row / M;
        float* B_base = broadcast_B ? B : (B + batch_idx * K * N);
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B_base[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transposed_kernel(float* A, float* B, float* C, int TotalM, int K, int N, int batch_count, int M, int broadcast_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < TotalM && col < N) {
        float sum = 0.0f;
        int batch_idx = row / M;
        float* B_base = broadcast_B ? B : (B + batch_idx * N * K);
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B_base[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

__global__ void linear_kernel_naive(float* input, float* weight, float* bias, float* out, int TotalRows, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < TotalRows && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[row * K + k] * weight[col * K + k];
        }
        if (bias) sum += bias[col];
        out[row * N + col] = sum;
    }
}

__global__ void rope_kernel(float* q, float* k, float* cos, float* sin, float* out_q, float* out_k, int total_tokens, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (idx < total_tokens * half_dim) {
        int token_idx = idx / half_dim;
        int dim_idx = idx % half_dim;
        float c = cos[idx];
        float s = sin[idx];
        int qk_idx = token_idx * dim + dim_idx;
        float qr = q[qk_idx];
        float qi = q[qk_idx + half_dim];
        float kr = k[qk_idx];
        float ki = k[qk_idx + half_dim];
        out_q[qk_idx] = qr * c - qi * s;
        out_q[qk_idx + half_dim] = qr * s + qi * c;
        out_k[qk_idx] = kr * c - ki * s;
        out_k[qk_idx + half_dim] = kr * s + ki * c;
    }
}

__global__ void conv2d_kernel_naive(float* input, float* weight, float* bias, float* out, int N, int C_in, int H_in, int W_in, int C_out, int KH, int KW, int H_out, int W_out, int stride, int padding, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx < total) {
        int w_out_idx = idx % W_out;
        int h_out_idx = (idx / W_out) % H_out;
        int c_out_idx = (idx / (W_out * H_out)) % C_out;
        int n_idx = idx / (W_out * H_out * C_out);
        int C_in_group = C_in / groups;
        int C_out_group = C_out / groups;
        int g = c_out_idx / C_out_group;
        float sum = 0.0f;
        int h_in_base = h_out_idx * stride - padding;
        int w_in_base = w_out_idx * stride - padding;
        for (int c = 0; c < C_in_group; c++) {
            int c_in = g * C_in_group + c;
            for (int i = 0; i < KH; i++) {
                for (int j = 0; j < KW; j++) {
                    int h_in = h_in_base + i;
                    int w_in = w_in_base + j;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int in_idx = ((n_idx * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int w_idx = ((c_out_idx * C_in_group + c) * KH + i) * KW + j;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        if (bias) sum += bias[c_out_idx];
        out[idx] = sum;
    }
}

__global__ void count_value_kernel(float* data, int size, float value, int* out_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int match = 0;
    if (idx < size) { if (fabsf(data[idx] - value) < 1e-6) match = 1; }
    if (match) atomicAdd(out_count, 1);
}

__device__ int g_gather_idx = 0;
__global__ void reset_gather_idx() { g_gather_idx = 0; }
__global__ void gather_by_value_kernel(float* input, float* indices, int size, float value, float* out_data, float* out_indices, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (fabsf(indices[idx] - value) < 1e-6) {
            int pos = atomicAdd(&g_gather_idx, 1);
            out_indices[pos] = (float)idx;
            int src_base = idx * hidden_size;
            int dst_base = pos * hidden_size;
            for (int j = 0; j < hidden_size; j++) { out_data[dst_base + j] = input[src_base + j]; }
        }
    }
}

__global__ void scatter_add_kernel(float* out, float* src, float* indices, int count, int hidden_size, int total_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = count * hidden_size;
    if (idx < total_elements) {
        int row_idx = idx / hidden_size;
        int col_idx = idx % hidden_size;
        int target_row = (int)indices[row_idx];
        if (target_row >= 0 && target_row < total_rows) {
            float val = src[idx];
            atomicAdd(&out[target_row * hidden_size + col_idx], val);
        }
    }
}

__global__ void deltanet_recurrence_kernel(float* q, float* k, float* v, float* beta, float* state, float* out, int B, int S, int H, int D) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    extern __shared__ float s_mem[];
    float* global_state = state + (b*H + h) * D * D;
    for (int i = tid; i < D*D; i += blockDim.x) { s_mem[i] = global_state[i]; }
    __syncthreads();
    for (int t = 0; t < S; t++) {
        int offset = ((b*S + t)*H + h);
        float* q_vec = q + offset * D;
        float* k_vec = k + offset * D;
        float* v_vec = v + offset * D;
        float* out_vec = out + offset * D;
        float b_val = beta[offset];
        float k_val = k_vec[tid];
        float v_val = v_vec[tid];
        for (int i = tid; i < D*D; i += blockDim.x) {
            int r = i / D; int c = i % D;
            float old = s_mem[i];
            float kv = k_vec[r] * v_vec[c];
            s_mem[i] = b_val * old + kv;
        }
        __syncthreads();
        if (tid < D) {
            float sum = 0.0f;
            for (int i = 0; i < D; i++) { sum += q_vec[i] * s_mem[i*D + tid]; }
            out_vec[tid] = sum;
        }
        __syncthreads();
    }
    for (int i = tid; i < D*D; i += blockDim.x) { global_state[i] = s_mem[i]; }
}

__global__ void topk_kernel_naive(float* input, int cols, int k, float* out_val, float* out_idx, int rows) {
    int row = blockIdx.x;
    if (row < rows) {
        if (threadIdx.x == 0) {
            float* in_row = input + row * cols;
            float* val_row = out_val + row * k;
            float* idx_row = out_idx + row * k;
            for (int i = 0; i < k; i++) { val_row[i] = -1e20f; idx_row[i] = -1.0f; }
            for (int i = 0; i < cols; i++) {
                float val = in_row[i];
                int pos = -1;
                for (int j = 0; j < k; j++) { if (val > val_row[j]) { pos = j; break; } }
                if (pos != -1) {
                    for (int j = k - 1; j > pos; j--) { val_row[j] = val_row[j - 1]; idx_row[j] = idx_row[j - 1]; }
                    val_row[pos] = val; idx_row[pos] = (float)i;
                }
            }
        }
    }
}

__global__ void dequantize_kernel(const int8_t* input, const float* scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { output[idx] = (float)input[idx] * (*scale); }
}

__global__ void fp32_to_bf16_kernel(float* in, unsigned short* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int f = __float_as_uint(in[idx]);
        out[idx] = (unsigned short)(f >> 16);
    }
}

__global__ void bf16_to_fp32_kernel(unsigned short* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int f = ((unsigned int)in[idx]) << 16;
        out[idx] = __uint_as_float(f);
    }
}

__global__ void verify_tokens_kernel(int* draft, int* target, int* out_count, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int matches = 0;
        for (int i = 0; i < len; i++) {
            if (draft[i] == target[i]) matches++; else break;
        }
        *out_count = matches;
    }
}

__global__ void embed_kernel(float* weight, float* indices, float* out, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / hidden;
    int dim_idx = idx % hidden;
    int emb_idx = (int)indices[token_idx];
    out[idx] = weight[emb_idx * hidden + dim_idx];
}

__global__ void permute_kernel(float* input, float* out, int size, int* strides_in, int* strides_out, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int temp = idx;
        int in_offset = 0;
        for (int i = 0; i < ndim; i++) {
            int c = temp / strides_out[i];
            temp %= strides_out[i];
            in_offset += c * strides_in[i];
        }
        out[idx] = input[in_offset];
    }
}

__global__ void fused_attention_kernel(float* Q, float* K, float* V, float* O, float scale, int seq_len, int head_dim) {
    // Basic Scaled Dot Product Attention implementation
    int bh = blockIdx.x; // Batch * Heads
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    float q_val = Q[bh * head_dim + tid];
    float max_score = -1e20f;
    float sum_score = 0.0f;
    float acc_o = 0.0f;

    // First pass: Compute scores and max for softmax
    // Naive serial loop per thread over seq_len (inefficient but correct/functional)
    // To do efficiently we need shared memory reduction.
    // Simplifying to parallel reduction requires multi-pass or persistent thread block.
    // Implementing Online Softmax logic per thread? No, need reduction across HeadDim for DotProduct.
    //
    // Strategy: Each thread computes part of dot product? No, standard is each thread handles one dimension of Q/K?
    // Correct approach for simple kernel: One thread block per query head.
    // Threads cooperatively load Q, iterate K/V.

    extern __shared__ float s_mem[]; // Size head_dim (Q)
    float* s_q = s_mem;
    s_q[tid] = q_val;
    __syncthreads();

    // Online Softmax loop
    for(int t = 0; t < seq_len; t++) {
        float k_val = K[(bh * seq_len + t) * head_dim + tid];
        float dot = s_q[tid] * k_val;

        // Block Reduce Dot
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, offset);

        float score = dot * scale; // Only thread 0 has full sum? No, need broadcast or sync.
        score = __shfl_sync(0xffffffff, score, 0); // Broadcast score to all threads

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_val = expf(score - max_score);
        float alpha = expf(old_max - max_score);

        sum_score = sum_score * alpha + exp_val;

        // Accumulate Output
        float v_val = V[(bh * seq_len + t) * head_dim + tid];
        acc_o = acc_o * alpha + v_val * exp_val;
    }

    O[bh * head_dim + tid] = acc_o / sum_score;
}

__global__ void paged_attention_kernel(float* q, float* k_cache, float* v_cache, int* block_tables, int* context_lens, float* out, float scale, int batch_size, int heads, int head_dim, int page_size, int max_blocks_per_seq) {
    int b = blockIdx.x; int h = blockIdx.y; int tid = threadIdx.x;
    if (b >= batch_size || h >= heads) return;
    float q_val = 0.0f;
    if (tid < head_dim) q_val = q[(b*heads + h)*head_dim + tid];
    extern __shared__ float s_mem[];
    float* s_q = s_mem;
    if (tid < head_dim) s_q[tid] = q_val;
    __syncthreads();
    int ctx_len = context_lens[b];
    int num_pages = (ctx_len + page_size - 1) / page_size;
    float sum_score = 0.0f;
    float max_score = -1e20f;
    float acc_o = 0.0f;
    for (int p = 0; p < num_pages; p++) {
        int block_idx = block_tables[b * max_blocks_per_seq + p];
        int num_tokens_in_page = (p == num_pages - 1) ? (ctx_len - p*page_size) : page_size;
        if (num_tokens_in_page <= 0) break;
        for (int t = 0; t < num_tokens_in_page; t++) {
            float* k_ptr = k_cache + ((block_idx * page_size + t) * heads + h) * head_dim;
            float dot = 0.0f;
            if (tid < head_dim) dot = s_q[tid] * k_ptr[tid];
            for (int offset = blockDim.x / 2; offset > 0; offset /= 2) dot += __shfl_down_sync(0xffffffff, dot, offset);
            float score = dot * scale;
            score = __shfl_sync(0xffffffff, score, 0);
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_val = expf(score - max_score);
            float alpha = expf(old_max - max_score);
            sum_score = sum_score * alpha + exp_val;
            float* v_ptr = v_cache + ((block_idx * page_size + t) * heads + h) * head_dim;
            float v_val = (tid < head_dim) ? v_ptr[tid] : 0.0f;
            acc_o = acc_o * alpha + v_val * exp_val;
        }
    }
    if (tid < head_dim) out[(b*heads + h)*head_dim + tid] = acc_o / sum_score;
}

__global__ void fused_split_rope_kernel(float* qkv, float* cos, float* sin, float* q_out, float* k_out, float* v_out, int total_tokens, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden = heads * head_dim;
    int total_elements = total_tokens * hidden;
    if (idx < total_elements) {
        int token_idx = idx / hidden;
        int dim_idx = idx % hidden;
        int d = dim_idx % head_dim;
        int half_dim = head_dim / 2;
        float q_val = qkv[token_idx * 3 * hidden + dim_idx];
        float k_val = qkv[token_idx * 3 * hidden + hidden + dim_idx];
        float v_val = qkv[token_idx * 3 * hidden + 2 * hidden + dim_idx];
        float c = 1.0f, s = 0.0f;
        if (d < head_dim) {
             int rot_idx = token_idx * (head_dim/2) + (d % half_dim);
             c = cos[rot_idx];
             s = sin[rot_idx];
             float val_r, val_i;
             if (d < half_dim) {
                 val_r = q_val; float q_val_i = qkv[token_idx * 3 * hidden + dim_idx + half_dim];
                 q_val = val_r * c - q_val_i * s;
                 val_r = k_val; float k_val_i = qkv[token_idx * 3 * hidden + hidden + dim_idx + half_dim];
                 k_val = val_r * c - k_val_i * s;
             } else {
                 val_i = q_val; float q_val_r = qkv[token_idx * 3 * hidden + dim_idx - half_dim];
                 q_val = q_val_r * s + val_i * c;
                 val_i = k_val; float k_val_r = qkv[token_idx * 3 * hidden + hidden + dim_idx - half_dim];
                 k_val = k_val_r * s + val_i * c;
             }
        }
        q_out[idx] = q_val; k_out[idx] = k_val; v_out[idx] = v_val;
    }
}

__global__ void precompute_freqs_kernel(float* cos, float* sin, int end, int half, float theta) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < end*half) {
        int i = idx / half; int j = idx % half;
        float freq = 1.0f / powf(theta, (float)(2*j) / (half*2));
        float val = i * freq;
        cos[idx] = cosf(val); sin[idx] = sinf(val);
    }
}

__global__ void slice_kernel(float* in, float* out, int size, int* start, int* h_in, int* h_out, int ndim) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        int temp=idx; int offset=0;
        for(int i=0; i<ndim; i++) {
            int c=temp/h_out[i]; temp%=h_out[i];
            offset+=(start[i]+c)*h_in[i];
        }
        out[idx]=in[offset];
    }
}

__global__ void slice_kernel_device(float* in, float* out, int size, float* start, int* h_in, int* h_out, int ndim) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        int temp=idx; int offset=0;
        for(int i=0; i<ndim; i++) {
            int c=temp/h_out[i]; temp%=h_out[i];
            offset+=((int)start[i]+c)*h_in[i];
        }
        out[idx]=in[offset];
    }
}

__global__ void set_slice_kernel(float* dst, float* src, int size, int* start, int* h_dst, int* h_src, int ndim) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        int temp=idx; int offset=0;
        for(int i=0; i<ndim; i++) {
            int c=temp/h_src[i]; temp%=h_src[i];
            offset+=(start[i]+c)*h_dst[i];
        }
        dst[offset]=src[idx];
    }
}

__global__ void set_slice_kernel_device(float* dst, float* src, int size, float* start, int* h_dst, int* h_src, int ndim) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        int temp=idx; int offset=0;
        for(int i=0; i<ndim; i++) {
            int c=temp/h_src[i]; temp%=h_src[i];
            offset+=((int)start[i]+c)*h_dst[i];
        }
        dst[offset]=src[idx];
    }
}

// Exports
EXPORT Tensor* create_tensor(int* shape, int ndim, int device_id) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim; t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->size = 1; for (int i = 0; i < ndim; i++) t->size *= shape[i];
    t->device_id = device_id;
    if (device_id >= 0) { CUDA_CHECK(cudaSetDevice(device_id)); CUDA_CHECK(cudaMalloc((void**)&t->data, t->size * sizeof(float))); }
    else { t->data = (float*)malloc(t->size * sizeof(float)); }
    return t;
}
EXPORT void free_tensor(Tensor* t) { if (t) { if (t->device_id >= 0) CUDA_CHECK(cudaFree(t->data)); else free(t->data); free(t->shape); free(t); } }
EXPORT void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    if (out->device_id >= 0) {
        if (out->size % 4 == 0) {
            int vec_size = out->size / 4;
            int threads = 256; int blocks = (vec_size + threads - 1) / threads;
            add_kernel_vec4<<<blocks, threads>>>((float4*)a->data, (float4*)b->data, (float4*)out->data, vec_size);
        } else {
            int threads = 256; int blocks = (out->size + threads - 1) / threads;
            add_kernel_scalar<<<blocks, threads>>>(a->data, b->data, out->data, out->size);
        }
    }
}
EXPORT void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    if (out->device_id >= 0) {
        if (out->size % 4 == 0) {
            int vec_size = out->size / 4;
            int threads = 256; int blocks = (vec_size + threads - 1) / threads;
            mul_kernel_vec4<<<blocks, threads>>>((float4*)a->data, (float4*)b->data, (float4*)out->data, vec_size);
        } else {
            int threads = 256; int blocks = (out->size + threads - 1) / threads;
            mul_kernel_scalar<<<blocks, threads>>>(a->data, b->data, out->data, out->size);
        }
    }
}
EXPORT void tensor_silu(Tensor* input, Tensor* out) {
    if (out->device_id >= 0) {
        if (out->size % 4 == 0) {
             int vec_size = out->size / 4;
             int threads = 256; int blocks = (vec_size + threads - 1) / threads;
             silu_kernel_vec4<<<blocks, threads>>>((float4*)input->data, (float4*)out->data, vec_size);
        } else {
             int threads = 256; int blocks = (out->size + threads - 1) / threads;
             silu_kernel_scalar<<<blocks, threads>>>(input->data, out->data, out->size);
        }
    }
}
EXPORT void tensor_fused_add_mul(Tensor* a, Tensor* b, Tensor* c, Tensor* out) {
    if (out->device_id >= 0) {
        if (out->size % 4 == 0) {
            int vec_size = out->size / 4;
            int threads = 256; int blocks = (vec_size + threads - 1) / threads;
            fused_add_mul_kernel_vec4<<<blocks, threads>>>((float4*)a->data, (float4*)b->data, (float4*)c->data, (float4*)out->data, vec_size);
        } else {
            int threads = 256; int blocks = (out->size + threads - 1) / threads;
            fused_add_mul_kernel_scalar<<<blocks, threads>>>(a->data, b->data, c->data, out->data, out->size);
        }
    }
}
EXPORT void tensor_gelu(Tensor* input, Tensor* out) { if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; gelu_kernel<<<blocks, threads>>>(input->data, out->data, out->size); } }
EXPORT void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) { if (out->device_id >= 0) { int rows = input->size / input->shape[input->ndim - 1]; int cols = input->shape[input->ndim - 1]; int threads = 1; int blocks = rows; rms_norm_kernel<<<blocks, threads>>>(input->data, weight->data, out->data, rows, cols, eps); } }
EXPORT void tensor_softmax(Tensor* input, Tensor* out) { if (out->device_id >= 0) { int rows = input->size / input->shape[input->ndim - 1]; int cols = input->shape[input->ndim - 1]; int threads = 1; int blocks = rows; softmax_kernel<<<blocks, threads>>>(input->data, out->data, rows, cols); } }
EXPORT void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) { if (out->device_id >= 0) { int M = a->shape[a->ndim-2]; int K = a->shape[a->ndim-1]; int N = b->shape[b->ndim-1]; int TotalM = a->size / K; int broadcast_B = (b->ndim == 2); dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16); matmul_kernel_naive<<<blocks, threads>>>(a->data, b->data, out->data, TotalM, K, N, TotalM/M, M, broadcast_B); } }
EXPORT void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out) { if (out->device_id >= 0) { int M = a->shape[a->ndim-2]; int K = a->shape[a->ndim-1]; int N = b->shape[b->ndim-2]; int TotalM = a->size / K; int broadcast_B = (b->ndim == 2); dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16); matmul_transposed_kernel<<<blocks, threads>>>(a->data, b->data, out->data, TotalM, K, N, TotalM/M, M, broadcast_B); } }
EXPORT void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) { if (out->device_id >= 0) { int K = input->shape[input->ndim - 1]; int TotalRows = input->size / K; int N = weight->shape[0]; dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalRows + 15) / 16); linear_kernel_naive<<<blocks, threads>>>(input->data, weight->data, bias ? bias->data : NULL, out->data, TotalRows, K, N); } }
EXPORT void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) { if (out_q->device_id >= 0) { int dim = q->shape[q->ndim - 1]; int total_tokens = q->size / dim; int threads = 256; int blocks = (total_tokens * (dim/2) + threads - 1) / threads; rope_kernel<<<blocks, threads>>>(q->data, k->data, cos->data, sin->data, out_q->data, out_k->data, total_tokens, dim); } }
EXPORT void tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding, int groups) { if (out->device_id >= 0) { int N = input->shape[0]; int C_in = input->shape[1]; int H_in = input->shape[2]; int W_in = input->shape[3]; int C_out = weight->shape[0]; int KH = weight->shape[2]; int KW = weight->shape[3]; int H_out = out->shape[2]; int W_out = out->shape[3]; int total = out->size; int threads = 256; int blocks = (total + threads - 1) / threads; conv2d_kernel_naive<<<blocks, threads>>>(input->data, weight->data, bias ? bias->data : NULL, out->data, N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out, stride, padding, groups); } }
EXPORT void tensor_swiglu(Tensor* gate, Tensor* up, Tensor* out) { if (out->device_id >= 0) { int size = out->size; int threads = 256; int blocks = (size + threads - 1) / threads; swiglu_kernel<<<blocks, threads>>>(gate->data, up->data, out->data, size); } }
EXPORT void tensor_fused_gate_up_swiglu(Tensor* gate_up, Tensor* out) { if (out->device_id >= 0) { int hidden = out->shape[out->ndim - 1]; int size = out->size; int threads = 256; int blocks = (size + threads - 1) / threads; fused_gate_up_swiglu_kernel<<<blocks, threads>>>(gate_up->data, out->data, hidden, size); } }
EXPORT void tensor_argmax(Tensor* input, Tensor* out) {
    if (input->device_id >= 0) {
        int cols = input->shape[input->ndim - 1];
        int rows = input->size / cols;
        argmax_kernel<<<rows, 1>>>(input->data, out->data, cols);
    }
}
EXPORT void tensor_embed(Tensor* weight, Tensor* indices, Tensor* out) {
    if (out->device_id >= 0) {
        int hidden = weight->shape[1];
        int size = out->size;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        embed_kernel<<<blocks, threads>>>(weight->data, indices->data, out->data, hidden);
    }
}
EXPORT void tensor_permute(Tensor* input, Tensor* out, int* dims) {
    if (out->device_id >= 0) {
        int ndim = input->ndim;
        int h_in[8], h_out[8];
        int h_in_strides[8]; // Strides of INPUT tensor

        h_in_strides[ndim-1] = 1;
        for(int i=ndim-2; i>=0; i--) h_in_strides[i] = h_in_strides[i+1] * input->shape[i+1];
        h_out[ndim-1] = 1;
        for(int i=ndim-2; i>=0; i--) h_out[i] = h_out[i+1] * out->shape[i+1];
        for(int i=0; i<ndim; i++) { h_in[i] = h_in_strides[dims[i]]; }

        int* d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, 2*ndim*4));
        CUDA_CHECK(cudaMemcpy(d_params, h_in, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params+ndim, h_out, ndim*4, cudaMemcpyHostToDevice));

        int size = out->size;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        permute_kernel<<<blocks, threads>>>(input->data, out->data, size, d_params, d_params+ndim, ndim);
        CUDA_CHECK(cudaFree(d_params));
    }
}
EXPORT void tensor_reshape(Tensor* input, Tensor* out) {
    if (out->device_id >= 0) {
        CUDA_CHECK(cudaMemcpy(out->data, input->data, input->size * 4, cudaMemcpyDeviceToDevice));
    }
}
EXPORT void tensor_cat(Tensor** tensors, int num_tensors, int axis, Tensor* out) {
    if (out->device_id < 0) return;
    int outer_dim = 1;
    for(int i=0; i<axis; i++) outer_dim *= out->shape[i];
    int inner_dim = 1;
    for(int i=axis+1; i<out->ndim; i++) inner_dim *= out->shape[i];
    int offset_accum = 0;
    for(int i=0; i<num_tensors; i++) {
        Tensor* t = tensors[i];
        int dim = t->shape[axis];
        for(int o=0; o<outer_dim; o++) {
            size_t src_offset = o * dim * inner_dim;
            size_t dst_offset = (o * out->shape[axis] + offset_accum) * inner_dim;
            size_t size = dim * inner_dim;
            CUDA_CHECK(cudaMemcpy(out->data + dst_offset, t->data + src_offset, size * 4, cudaMemcpyDeviceToDevice));
        }
        offset_accum += dim;
    }
}
EXPORT void tensor_scaled_dot_product_attention(Tensor* q, Tensor* k, Tensor* v, Tensor* out, float scale) {
    // Stub or real simple impl? Required for standard.
    // Use paged attention with dummy table?
    // Implementing basic launch of fused kernel placeholder
    int seq_len = k->shape[1];
    int head_dim = q->shape[2];
    fused_attention_kernel<<<q->shape[0]*q->shape[1], 128>>>(q->data, k->data, v->data, out->data, scale, seq_len, head_dim);
}
EXPORT void tensor_count_value(Tensor* t, float value, int* count) { if (t->device_id >= 0) { int* d_count; CUDA_CHECK(cudaMalloc((void**)&d_count, sizeof(int))); CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int))); int threads = 256; int blocks = (t->size + threads - 1) / threads; count_value_kernel<<<blocks, threads>>>(t->data, t->size, value, d_count); CUDA_CHECK(cudaMemcpy(count, d_count, sizeof(int), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaFree(d_count)); } }
EXPORT void tensor_gather_by_value(Tensor* input, Tensor* indices, float value, Tensor* out_data, Tensor* out_indices) { if (input->device_id >= 0) { reset_gather_idx<<<1, 1>>>(); int hidden_size = input->shape[input->ndim - 1]; int size = indices->size; int threads = 256; int blocks = (size + threads - 1) / threads; gather_by_value_kernel<<<blocks, threads>>>(input->data, indices->data, size, value, out_data->data, out_indices->data, hidden_size); } }
EXPORT void tensor_scatter_add_by_index(Tensor* out, Tensor* src, Tensor* indices) { if (out->device_id >= 0) { int count = indices->size; int hidden_size = src->shape[src->ndim - 1]; int total_rows = out->size / hidden_size; int total_elements = count * hidden_size; int threads = 256; int blocks = (total_elements + threads - 1) / threads; scatter_add_kernel<<<blocks, threads>>>(out->data, src->data, indices->data, count, hidden_size, total_rows); } }
EXPORT void tensor_load_data(Tensor* t, float* buffer, int size) { if (t->device_id >= 0) CUDA_CHECK(cudaMemcpy(t->data, buffer, size*4, cudaMemcpyHostToDevice)); else memcpy(t->data, buffer, size*4); }
EXPORT void tensor_get_data(Tensor* t, float* buffer, int size) { if (t->device_id >= 0) CUDA_CHECK(cudaMemcpy(buffer, t->data, size*4, cudaMemcpyDeviceToHost)); else memcpy(buffer, t->data, size*4); }
EXPORT void tensor_topk(Tensor* input, int k, Tensor* out_values, Tensor* out_indices) {
    if (input->device_id >= 0) {
        int cols = input->shape[input->ndim-1];
        int rows = input->size / cols;
        int threads = 1; int blocks = rows;
        topk_kernel_naive<<<blocks, threads>>>(input->data, cols, k, out_values->data, out_indices->data, rows);
    }
}
EXPORT void tensor_deltanet_recurrence(Tensor* q, Tensor* k, Tensor* v, Tensor* beta, Tensor* state, Tensor* out) {
    if (q->device_id >= 0) {
        int B = q->shape[0]; int S = q->shape[1]; int H = q->shape[2]; int D = q->shape[3];
        int blocks_x = B; int blocks_y = H; dim3 blocks(blocks_x, blocks_y);
        int threads = 256; int shared_size = D * D * sizeof(float);
        deltanet_recurrence_kernel<<<blocks, threads, shared_size>>>(q->data, k->data, v->data, beta->data, state->data, out->data, B, S, H, D);
    }
}
EXPORT void tensor_slice_device(Tensor* input, Tensor* out, Tensor* start_indices) {
    if (out->device_id >= 0) {
        int ndim = input->ndim;
        int* h_in = (int*)malloc(ndim*4); int* h_out = h_in + ndim;
        h_in[ndim-1]=1; h_out[ndim-1]=1;
        for(int i=ndim-2; i>=0; i--) { h_in[i] = h_in[i+1]*input->shape[i+1]; h_out[i] = h_out[i+1]*out->shape[i+1]; }
        int* d_strides;
        CUDA_CHECK(cudaMalloc((void**)&d_strides, 2*ndim*4));
        CUDA_CHECK(cudaMemcpy(d_strides, h_in, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strides+ndim, h_out, ndim*4, cudaMemcpyHostToDevice));
        int size = out->size; int threads=256; int blocks=(size+255)/256;
        slice_kernel_device<<<blocks, threads>>>(input->data, out->data, size, start_indices->data, d_strides, d_strides+ndim, ndim);
        CUDA_CHECK(cudaFree(d_strides)); free(h_in);
    }
}
EXPORT void tensor_set_slice_device(Tensor* dst, Tensor* src, Tensor* start_indices) {
    if (dst->device_id >= 0) {
        int ndim = dst->ndim;
        int* strides = (int*)malloc(ndim * 2 * sizeof(int));
        int* s_dst = strides; int* s_src = strides + ndim;
        s_dst[ndim-1] = 1; s_src[ndim-1] = 1;
        for(int i=ndim-2; i>=0; i--) { s_dst[i] = s_dst[i+1]*dst->shape[i+1]; s_src[i] = s_src[i+1]*src->shape[i+1]; }
        int* d_strides;
        CUDA_CHECK(cudaMalloc((void**)&d_strides, 2*ndim*4));
        CUDA_CHECK(cudaMemcpy(d_strides, s_dst, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strides+ndim, s_src, ndim*4, cudaMemcpyHostToDevice));
        int size = src->size; int threads=256; int blocks=(size+255)/256;
        set_slice_kernel_device<<<blocks, threads>>>(dst->data, src->data, size, start_indices->data, d_strides, d_strides+ndim, ndim);
        CUDA_CHECK(cudaFree(d_strides)); free(strides);
    }
}
EXPORT void tensor_fused_add_rms_norm(Tensor* x, Tensor* residual, Tensor* weight, Tensor* out, float eps) {
    if (x->device_id >= 0) {
        int hidden_size = x->shape[x->ndim - 1];
        int rows = x->size / hidden_size;
        int threads = 256;
        if (hidden_size < 256) threads = hidden_size;
        fused_add_rms_norm_row_kernel<<<rows, threads>>>(x->data, residual->data, weight->data, out->data, hidden_size, eps);
    }
}
EXPORT void tensor_dequantize(Tensor* input, Tensor* scale, Tensor* out) {
    if (out->device_id >= 0) {
        int8_t* in_ptr = (int8_t*)input->data;
        int size = out->size; int threads = 256; int blocks = (size + threads - 1) / threads;
        dequantize_kernel<<<blocks, threads>>>(in_ptr, scale->data, out->data, size);
    }
}
EXPORT void tensor_convert_fp32_bf16(Tensor* fp32, Tensor* bf16) {
    if (fp32->device_id >= 0) {
        int size = fp32->size; int threads = 256; int blocks = (size + threads - 1) / threads;
        fp32_to_bf16_kernel<<<blocks, threads>>>(fp32->data, (unsigned short*)bf16->data, size);
    }
}
EXPORT void tensor_convert_bf16_fp32(Tensor* bf16, Tensor* fp32) {
    if (bf16->device_id >= 0) {
        int size = fp32->size; int threads = 256; int blocks = (size + threads - 1) / threads;
        bf16_to_fp32_kernel<<<blocks, threads>>>((unsigned short*)bf16->data, fp32->data, size);
    }
}
EXPORT void tensor_verify_tokens(Tensor* draft, Tensor* target, Tensor* out_count) {
    if (draft->device_id >= 0) {
        int len = draft->size;
        verify_tokens_kernel<<<1, 1>>>((int*)draft->data, (int*)target->data, (int*)out_count->data, len);
    }
}
EXPORT void tensor_copy_p2p(Tensor* src, Tensor* dst) {
    if (src->device_id >= 0 && dst->device_id >= 0) {
        size_t size = src->size; if (dst->size < size) size = dst->size;
        CUDA_CHECK(cudaMemcpyPeer(dst->data, dst->device_id, src->data, src->device_id, size * sizeof(float)));
    }
}
EXPORT void tensor_copy_offset(Tensor* src, int src_offset, Tensor* dst, int dst_offset, int count) {
    if (src->device_id >= 0 && dst->device_id >= 0) {
        CUDA_CHECK(cudaMemcpy(dst->data + dst_offset, src->data + src_offset, count * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}
// Benchmark
EXPORT void tensor_benchmark_matmul(int M, int N, int K, int trials, double* out_time_ms) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*4); cudaMalloc(&d_B, K*N*4); cudaMalloc(&d_C, M*N*4);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<trials; i++) {
        int TotalM = M;
        dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16);
        matmul_kernel_naive<<<blocks, threads>>>(d_A, d_B, d_C, TotalM, K, N, 1, M, 0);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    *out_time_ms = ms / trials;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaEventDestroy(start); cudaEventDestroy(stop);
}
// Slice Host
__global__ void slice_kernel_host(float* in, float* out, int size, int* start, int* h_in, int* h_out, int ndim) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        int temp=idx; int offset=0;
        for(int i=0; i<ndim; i++) {
            int c=temp/h_out[i]; temp%=h_out[i];
            offset+=(start[i]+c)*h_in[i];
        }
        out[idx]=in[offset];
    }
}
EXPORT void tensor_slice(Tensor* input, Tensor* out, int* start_indices, int* slice_shapes) {
    if (out->device_id >= 0) {
        int ndim = input->ndim;
        int* h_in = (int*)malloc(ndim*4); int* h_out = h_in + ndim;
        h_in[ndim-1]=1; h_out[ndim-1]=1;
        for(int i=ndim-2; i>=0; i--) { h_in[i] = h_in[i+1]*input->shape[i+1]; h_out[i] = h_out[i+1]*out->shape[i+1]; }
        int* d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, 3*ndim*4));
        CUDA_CHECK(cudaMemcpy(d_params, start_indices, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params+ndim, h_in, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params+2*ndim, h_out, ndim*4, cudaMemcpyHostToDevice));
        int size = out->size; int threads=256; int blocks=(size+255)/256;
        slice_kernel_host<<<blocks, threads>>>(input->data, out->data, size, d_params, d_params+ndim, d_params+2*ndim, ndim);
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync required for d_params free
        CUDA_CHECK(cudaFree(d_params)); free(h_in);
    }
}
EXPORT void tensor_set_slice(Tensor* dst, Tensor* src, int* start_indices) {
    if (dst->device_id >= 0) {
        int ndim = dst->ndim;
        int* strides = (int*)malloc(ndim * 2 * sizeof(int));
        int* s_dst = strides; int* s_src = strides + ndim;
        s_dst[ndim-1] = 1; s_src[ndim-1] = 1;
        for(int i=ndim-2; i>=0; i--) { s_dst[i] = s_dst[i+1]*dst->shape[i+1]; s_src[i] = s_src[i+1]*src->shape[i+1]; }
        int* d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, 3*ndim*4));
        CUDA_CHECK(cudaMemcpy(d_params, start_indices, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params+ndim, s_dst, ndim*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params+2*ndim, s_src, ndim*4, cudaMemcpyHostToDevice));
        int size = src->size; int threads=256; int blocks=(size+255)/256;
        set_slice_kernel<<<blocks, threads>>>(dst->data, src->data, size, d_params, d_params+ndim, d_params+2*ndim, ndim);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_params)); free(strides);
    }
}
EXPORT void tensor_set_tuning_param(int param_id, int value) { if (param_id == 0) g_matmul_tuning_config = value; }
EXPORT void tensor_precompute_freqs_cis(int dim, int end, float theta, Tensor* out_cos, Tensor* out_sin) { if (out_cos->device_id >= 0) { int half = dim/2; int total = end*half; int th=256; int bl=(total+255)/256; precompute_freqs_kernel<<<bl, th>>>(out_cos->data, out_sin->data, end, half, theta); } }
