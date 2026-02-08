// ... (Previous implementation same as before) ...
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// ... (Existing Kernels fill, add, mul, silu, gelu, rms, matmul, linear, softmax, rope) ...
// Keeping them intact. Assuming previous file content is preserved. I will append new kernels.

__global__ void fill_kernel(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}
__global__ void add_kernel(float* a, float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] + b[idx];
}
__global__ void mul_kernel(float* a, float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = a[idx] * b[idx];
}
__global__ void silu_kernel(float* input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { float x = input[idx]; out[idx] = x / (1.0f + expf(-x)); }
}
__global__ void gelu_kernel(float* input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { float x = input[idx]; float c = 0.044715f; float sqrt_2_pi = 0.7978845608f; out[idx] = 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + c * x * x * x))); }
}
__global__ void rms_norm_kernel(float* input, float* weight, float* out, int rows, int cols, float eps) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= rows) return;
    float sum_sq = 0.0f; int base = row_idx * cols;
    for (int j = 0; j < cols; j++) { float val = input[base + j]; sum_sq += val * val; }
    float rms = rsqrtf(sum_sq / cols + eps);
    for (int j = 0; j < cols; j++) { out[base + j] = (input[base + j] * rms) * weight[j]; }
}
__global__ void matmul_kernel_naive(float* A, float* B, float* C, int TotalM, int K, int N, int Batch, int M_per_Batch, int broadcast_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < TotalM && col < N) {
        float sum = 0.0f; int batch_idx = row / M_per_Batch;
        int a_base = row * K; int b_base_start = broadcast_B ? 0 : (batch_idx * K * N);
        for (int k = 0; k < K; k++) { sum += A[a_base + k] * B[b_base_start + k * N + col]; }
        C[row * N + col] = sum;
    }
}
__global__ void matmul_transposed_kernel(float* A, float* B, float* C, int TotalM, int K, int N, int Batch, int M_per_Batch, int broadcast_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < TotalM && col < N) {
        float sum = 0.0f; int batch_idx = row / M_per_Batch;
        int a_base = row * K; int b_base_start = broadcast_B ? 0 : (batch_idx * N * K);
        for (int k = 0; k < K; k++) { sum += A[a_base + k] * B[b_base_start + col * K + k]; }
        C[row * N + col] = sum;
    }
}
__global__ void linear_kernel_naive(float* input, float* weight, float* bias, float* out, int TotalRows, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < TotalRows && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) { sum += input[row * K + k] * weight[col * K + k]; }
        if (bias) sum += bias[col];
        out[row * N + col] = sum;
    }
}
__global__ void softmax_kernel(float* input, float* out, int rows, int cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= rows) return;
    int base = row_idx * cols; float max_val = -1e9f;
    for (int j = 0; j < cols; j++) { float val = input[base + j]; if (val > max_val) max_val = val; }
    float sum_exp = 0.0f;
    for (int j = 0; j < cols; j++) { float val = expf(input[base + j] - max_val); out[base + j] = val; sum_exp += val; }
    float inv_sum = 1.0f / sum_exp;
    for (int j = 0; j < cols; j++) { out[base + j] *= inv_sum; }
}
__global__ void rope_kernel(float* q, float* k, float* cos, float* sin, float* out_q, float* out_k, int total_tokens, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; int half_dim = dim / 2;
    if (idx >= total_tokens * half_dim) return;
    int token_idx = idx / half_dim; int feat_idx = idx % half_dim;
    int base = token_idx * dim;
    // RoPE assumes cos/sin provided for this token.
    // If we use the new precompute, we should map appropriately.
    // Assuming cos input is [MaxSeq, HalfDim]
    // We need to look up based on position?
    // The previous kernel assumed cos matched q shape.
    // If we change the calling logic, we need to update this kernel or the caller.
    // Let's assume the caller passes the sliced cos/sin corresponding to the tokens in q.
    // E.g. if q is [Batch, 1, Dim] (next token), cos is [1, HalfDim] (for that pos).
    // The previous implementation accessed `cos[base + feat_idx]` which implies
    // cos has same flattened structure as q (so it repeats for heads etc).
    // If we use `precompute_freqs_cis`, we get [Seq, HalfDim].
    // We need to broadcast.

    // Updated Logic:
    // Pass stride for cos/sin? Or assume they are [Batch*Seq, HalfDim]?
    // Let's assume standard [Batch*Seq, HalfDim] or [TotalTokens, HalfDim] broadcasted effectively.
    // Since we can't change signature easily without breaking things, let's assume the input `cos` pointer
    // points to a buffer that ALIGNS with `q`.
    // i.e., Caller must expand/broadcast `cos` before calling `rope`?
    // OR we change the kernel to use `token_idx` into `cos` directly if `cos` is smaller.
    // Let's assume `cos` is [TotalTokens, HalfDim].

    // Current code: `float c = cos[base + feat_idx];` where base = token_idx * dim.
    // If `cos` is [TotalTokens, HalfDim], index should be `token_idx * half_dim + feat_idx`.
    // NOTE: This is a change. The previous code assumed [TotalTokens, Dim] or similar.
    // Let's fix this for correctness with new precompute.

    int cos_idx = token_idx * half_dim + feat_idx;

    float q_r = q[base + feat_idx];
    float q_i = q[base + feat_idx + half_dim];
    float k_r = k[base + feat_idx];
    float k_i = k[base + feat_idx + half_dim];

    float c = cos[cos_idx];
    float s = sin[cos_idx];

    out_q[base + feat_idx] = q_r * c - q_i * s;
    out_q[base + feat_idx + half_dim] = q_r * s + q_i * c;
    out_k[base + feat_idx] = k_r * c - k_i * s;
    out_k[base + feat_idx + half_dim] = k_r * s + k_i * c;
}

// ... (Exported functions) ...
// (Re-declaring them implies overwriting the file, so I include all)

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
EXPORT void free_tensor(Tensor* t) {
    if (t) { if (t->device_id >= 0) CUDA_CHECK(cudaFree(t->data)); else free(t->data); free(t->shape); free(t); }
}
EXPORT void tensor_fill(Tensor* t, float value) {
    if (t->device_id >= 0) { int threads = 256; int blocks = (t->size + threads - 1) / threads; fill_kernel<<<blocks, threads>>>(t->data, value, t->size); CUDA_CHECK(cudaDeviceSynchronize()); }
    else { for (int i = 0; i < t->size; i++) t->data[i] = value; }
}
EXPORT void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; add_kernel<<<blocks, threads>>>(a->data, b->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); }
}
EXPORT void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; mul_kernel<<<blocks, threads>>>(a->data, b->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); }
}
EXPORT void tensor_gelu(Tensor* input, Tensor* out) {
    if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; gelu_kernel<<<blocks, threads>>>(input->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); }
}
EXPORT void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) {
    if (out->device_id >= 0) { int rows = input->size / input->shape[input->ndim - 1]; int cols = input->shape[input->ndim - 1]; int threads = 1; int blocks = rows; rms_norm_kernel<<<blocks, threads>>>(input->data, weight->data, out->data, rows, cols, eps); CUDA_CHECK(cudaDeviceSynchronize()); }
}
EXPORT void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    if (out->device_id >= 0) {
        int adim = a->ndim; int bdim = b->ndim;
        int M = a->shape[adim-2]; int K = a->shape[adim-1]; int N = b->shape[bdim-1];
        int TotalM = a->size / K; int broadcast_B = (bdim == 2);
        dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16);
        matmul_kernel_naive<<<blocks, threads>>>(a->data, b->data, out->data, TotalM, K, N, TotalM / M, M, broadcast_B);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
EXPORT void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out) {
    if (out->device_id >= 0) {
        int adim = a->ndim; int bdim = b->ndim;
        int M = a->shape[adim-2]; int K = a->shape[adim-1]; int N = b->shape[bdim-2];
        int TotalM = a->size / K; int broadcast_B = (bdim == 2);
        dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16);
        matmul_transposed_kernel<<<blocks, threads>>>(a->data, b->data, out->data, TotalM, K, N, TotalM / M, M, broadcast_B);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
EXPORT void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) {
    if (out->device_id >= 0) {
        int K = input->shape[input->ndim - 1]; int TotalRows = input->size / K; int N = weight->shape[0];
        dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalRows + 15) / 16);
        linear_kernel_naive<<<blocks, threads>>>(input->data, weight->data, bias ? bias->data : NULL, out->data, TotalRows, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
EXPORT void tensor_softmax(Tensor* input, Tensor* out) {
    if (out->device_id >= 0) { int rows = input->size / input->shape[input->ndim - 1]; int cols = input->shape[input->ndim - 1]; int threads = 1; int blocks = rows; softmax_kernel<<<blocks, threads>>>(input->data, out->data, rows, cols); CUDA_CHECK(cudaDeviceSynchronize()); }
}
EXPORT void tensor_silu(Tensor* input, Tensor* out) {
    if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; silu_kernel<<<blocks, threads>>>(input->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); }
}
EXPORT void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) {
    if (out_q->device_id >= 0) {
        int dim = q->shape[q->ndim - 1]; int total_tokens = q->size / dim;
        int threads = 256; int blocks = (total_tokens * (dim/2) + threads - 1) / threads;
        rope_kernel<<<blocks, threads>>>(q->data, k->data, cos->data, sin->data, out_q->data, out_k->data, total_tokens, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
EXPORT void tensor_load_data(Tensor* t, float* buffer, int size) {
    if (t->device_id >= 0) { CUDA_CHECK(cudaMemcpy(t->data, buffer, size * sizeof(float), cudaMemcpyHostToDevice)); }
    else { memcpy(t->data, buffer, size * sizeof(float)); }
}
EXPORT void tensor_get_data(Tensor* t, float* buffer, int size) {
    if (t->device_id >= 0) { CUDA_CHECK(cudaMemcpy(buffer, t->data, size * sizeof(float), cudaMemcpyDeviceToHost)); }
    else { memcpy(buffer, t->data, size * sizeof(float)); }
}

// --- NEW OPS (Slice / Precompute) ---

__global__ void slice_kernel(float* input, float* out, int size, int* start_indices, int* in_strides, int* out_strides, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp = idx;
    int in_idx = 0;
    for(int d = 0; d < ndim; d++) {
        int coord = temp / out_strides[d];
        temp %= out_strides[d];
        in_idx += (start_indices[d] + coord) * in_strides[d];
    }
    out[idx] = input[in_idx];
}

EXPORT void tensor_slice(Tensor* input, Tensor* out, int* start_indices, int* slice_shapes) {
    // We need to pass array params to kernel.
    // Simplification: ndim <= 4 usually.
    if (out->device_id >= 0) {
        int ndim = input->ndim;
        int* h_in_strides = (int*)malloc(ndim * sizeof(int));
        int* h_out_strides = (int*)malloc(ndim * sizeof(int));

        h_in_strides[ndim-1] = 1; h_out_strides[ndim-1] = 1;
        for(int i = ndim - 2; i >= 0; i--) {
            h_in_strides[i] = h_in_strides[i+1] * input->shape[i+1];
            h_out_strides[i] = h_out_strides[i+1] * out->shape[i+1];
        }

        // Alloc device memory for strides/indices
        int* d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, 3 * ndim * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_params, start_indices, ndim * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params + ndim, h_in_strides, ndim * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params + 2*ndim, h_out_strides, ndim * sizeof(int), cudaMemcpyHostToDevice));

        int size = out->size;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        slice_kernel<<<blocks, threads>>>(input->data, out->data, size, d_params, d_params + ndim, d_params + 2*ndim, ndim);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_params));
        free(h_in_strides);
        free(h_out_strides);
    }
}

__global__ void precompute_freqs_kernel(float* cos_out, float* sin_out, int end, int half_dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end * half_dim) return;

    int t = idx / half_dim;
    int i = idx % half_dim;

    float freq = 1.0f / powf(theta, (float)(i * 2) / (half_dim * 2));
    float val = t * freq;

    cos_out[idx] = cosf(val);
    sin_out[idx] = sinf(val);
}

EXPORT void tensor_precompute_freqs_cis(int dim, int end, float theta, Tensor* out_cos, Tensor* out_sin) {
    if (out_cos->device_id >= 0) {
        int half_dim = dim / 2;
        int total = end * half_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        precompute_freqs_kernel<<<blocks, threads>>>(out_cos->data, out_sin->data, end, half_dim, theta);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
