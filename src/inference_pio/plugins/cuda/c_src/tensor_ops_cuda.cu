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

#define EXPORT

// Kernels
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
    if (idx < size) {
        float x = input[idx];
        out[idx] = x * (1.0f / (1.0f + expf(-x)));
    }
}

__global__ void gelu_kernel(float* input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Tanh approximation
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

__global__ void softmax_kernel(float* input, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float max_val = -1e9f;
        for(int i=0; i<cols; i++) if(input[row*cols+i] > max_val) max_val = input[row*cols+i];

        float sum = 0.0f;
        for(int i=0; i<cols; i++) {
            float val = expf(input[row*cols+i] - max_val);
            out[row*cols+i] = val;
            sum += val;
        }
        float inv_sum = 1.0f / sum;
        for(int i=0; i<cols; i++) out[row*cols+i] *= inv_sum;
    }
}

// Naive MatMul [M, K] x [K, N] -> [M, N]
__global__ void matmul_kernel_naive(float* A, float* B, float* C, int TotalM, int K, int N, int batch_count, int M, int broadcast_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < TotalM && col < N) {
        float sum = 0.0f;
        int batch_idx = row / M;
        // If broadcast_B is true, B is shared across batches (2D)
        // If false, B is batched [Batch, K, N]
        float* B_base = broadcast_B ? B : (B + batch_idx * K * N);

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B_base[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Naive MatMul Transposed B: [M, K] x [N, K] -> [M, N]
__global__ void matmul_transposed_kernel(float* A, float* B, float* C, int TotalM, int K, int N, int batch_count, int M, int broadcast_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < TotalM && col < N) {
        float sum = 0.0f;
        int batch_idx = row / M;
        float* B_base = broadcast_B ? B : (B + batch_idx * N * K); // B is [Batch, N, K] or [N, K]

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B_base[col * K + k]; // B is [N, K] -> B[col, k]
        }
        C[row * N + col] = sum;
    }
}

__global__ void linear_kernel_naive(float* input, float* weight, float* bias, float* out, int TotalRows, int K, int N) {
    // Input [TotalRows, K], Weight [N, K], Bias [N] -> Out [TotalRows, N]
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
    // idx iterates over [total_tokens * half_dim]

    if (idx < total_tokens * half_dim) {
        int token_idx = idx / half_dim;
        int dim_idx = idx % half_dim;

        // We need seq position to index cos/sin correctly?
        // Wait, input cos/sin is assumed pre-sliced or aligned to [total_tokens, half_dim].
        // If not, we assume simple broadcasting or 1:1 mapping if shape matches.
        // Assuming 1:1 for simplicity as handled by Python slicing.

        float c = cos[dim_idx]; // Simplified: assume cos is [HalfDim] broadcasted? No, usually [Seq, HalfDim]
        // But here we rely on the Python caller passing correct buffers.
        // If total_tokens > cos_len, this is wrong.
        // Assuming cos/sin are [Total_Tokens, Half_Dim] flattened.
        c = cos[idx];
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

__global__ void conv2d_kernel_naive(float* input, float* weight, float* bias, float* out,
                                    int N, int C_in, int H_in, int W_in,
                                    int C_out, int KH, int KW,
                                    int H_out, int W_out,
                                    int stride, int padding, int groups) {
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

// MoE Kernels
__global__ void count_value_kernel(float* data, int size, float value, int* out_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int match = 0;
    if (idx < size) {
        if (fabsf(data[idx] - value) < 1e-6) match = 1;
    }
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
            for (int j = 0; j < hidden_size; j++) {
                out_data[dst_base + j] = input[src_base + j];
            }
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
EXPORT void tensor_fill(Tensor* t, float value) { if (t->device_id >= 0) { int threads = 256; int blocks = (t->size + threads - 1) / threads; fill_kernel<<<blocks, threads>>>(t->data, value, t->size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_add(Tensor* a, Tensor* b, Tensor* out) { if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; add_kernel<<<blocks, threads>>>(a->data, b->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_mul(Tensor* a, Tensor* b, Tensor* out) { if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; mul_kernel<<<blocks, threads>>>(a->data, b->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_gelu(Tensor* input, Tensor* out) { if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; gelu_kernel<<<blocks, threads>>>(input->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_silu(Tensor* input, Tensor* out) { if (out->device_id >= 0) { int threads = 256; int blocks = (out->size + threads - 1) / threads; silu_kernel<<<blocks, threads>>>(input->data, out->data, out->size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) { if (out->device_id >= 0) { int rows = input->size / input->shape[input->ndim - 1]; int cols = input->shape[input->ndim - 1]; int threads = 1; int blocks = rows; rms_norm_kernel<<<blocks, threads>>>(input->data, weight->data, out->data, rows, cols, eps); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_softmax(Tensor* input, Tensor* out) { if (out->device_id >= 0) { int rows = input->size / input->shape[input->ndim - 1]; int cols = input->shape[input->ndim - 1]; int threads = 1; int blocks = rows; softmax_kernel<<<blocks, threads>>>(input->data, out->data, rows, cols); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) { if (out->device_id >= 0) { int M = a->shape[a->ndim-2]; int K = a->shape[a->ndim-1]; int N = b->shape[b->ndim-1]; int TotalM = a->size / K; int broadcast_B = (b->ndim == 2); dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16); matmul_kernel_naive<<<blocks, threads>>>(a->data, b->data, out->data, TotalM, K, N, TotalM/M, M, broadcast_B); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out) { if (out->device_id >= 0) { int M = a->shape[a->ndim-2]; int K = a->shape[a->ndim-1]; int N = b->shape[b->ndim-2]; int TotalM = a->size / K; int broadcast_B = (b->ndim == 2); dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalM + 15) / 16); matmul_transposed_kernel<<<blocks, threads>>>(a->data, b->data, out->data, TotalM, K, N, TotalM/M, M, broadcast_B); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) { if (out->device_id >= 0) { int K = input->shape[input->ndim - 1]; int TotalRows = input->size / K; int N = weight->shape[0]; dim3 threads(16, 16); dim3 blocks((N + 15) / 16, (TotalRows + 15) / 16); linear_kernel_naive<<<blocks, threads>>>(input->data, weight->data, bias ? bias->data : NULL, out->data, TotalRows, K, N); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) { if (out_q->device_id >= 0) { int dim = q->shape[q->ndim - 1]; int total_tokens = q->size / dim; int threads = 256; int blocks = (total_tokens * (dim/2) + threads - 1) / threads; rope_kernel<<<blocks, threads>>>(q->data, k->data, cos->data, sin->data, out_q->data, out_k->data, total_tokens, dim); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding, int groups) { if (out->device_id >= 0) { int N = input->shape[0]; int C_in = input->shape[1]; int H_in = input->shape[2]; int W_in = input->shape[3]; int C_out = weight->shape[0]; int KH = weight->shape[2]; int KW = weight->shape[3]; int H_out = out->shape[2]; int W_out = out->shape[3]; int total = out->size; int threads = 256; int blocks = (total + threads - 1) / threads; conv2d_kernel_naive<<<blocks, threads>>>(input->data, weight->data, bias ? bias->data : NULL, out->data, N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out, stride, padding, groups); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_swiglu(Tensor* gate, Tensor* up, Tensor* out) { if (out->device_id >= 0) { int size = out->size; int threads = 256; int blocks = (size + threads - 1) / threads; swiglu_kernel<<<blocks, threads>>>(gate->data, up->data, out->data, size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_count_value(Tensor* t, float value, int* count) { if (t->device_id >= 0) { int* d_count; CUDA_CHECK(cudaMalloc((void**)&d_count, sizeof(int))); CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int))); int threads = 256; int blocks = (t->size + threads - 1) / threads; count_value_kernel<<<blocks, threads>>>(t->data, t->size, value, d_count); CUDA_CHECK(cudaMemcpy(count, d_count, sizeof(int), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaFree(d_count)); } }
EXPORT void tensor_gather_by_value(Tensor* input, Tensor* indices, float value, Tensor* out_data, Tensor* out_indices) { if (input->device_id >= 0) { reset_gather_idx<<<1, 1>>>(); int hidden_size = input->shape[input->ndim - 1]; int size = indices->size; int threads = 256; int blocks = (size + threads - 1) / threads; gather_by_value_kernel<<<blocks, threads>>>(input->data, indices->data, size, value, out_data->data, out_indices->data, hidden_size); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_scatter_add_by_index(Tensor* out, Tensor* src, Tensor* indices) { if (out->device_id >= 0) { int count = indices->size; int hidden_size = src->shape[src->ndim - 1]; int total_rows = out->size / hidden_size; int total_elements = count * hidden_size; int threads = 256; int blocks = (total_elements + threads - 1) / threads; scatter_add_kernel<<<blocks, threads>>>(out->data, src->data, indices->data, count, hidden_size, total_rows); CUDA_CHECK(cudaDeviceSynchronize()); } }
EXPORT void tensor_load_data(Tensor* t, float* buffer, int size) { if (t->device_id >= 0) CUDA_CHECK(cudaMemcpy(t->data, buffer, size*4, cudaMemcpyHostToDevice)); else memcpy(t->data, buffer, size*4); }
EXPORT void tensor_get_data(Tensor* t, float* buffer, int size) { if (t->device_id >= 0) CUDA_CHECK(cudaMemcpy(buffer, t->data, size*4, cudaMemcpyDeviceToHost)); else memcpy(buffer, t->data, size*4); }
// Missing: slice, precompute (add empty or copy from previous if not modified, assuming typical impl)
// Adding minimal versions to prevent link error
__global__ void slice_kernel(float* in, float* out, int size, int* start, int* h_in, int* h_out, int ndim) { int idx = blockIdx.x*blockDim.x + threadIdx.x; if(idx<size) { int temp=idx; int offset=0; for(int i=0; i<ndim; i++) { int c=temp/h_out[i]; temp%=h_out[i]; offset+=(start[i]+c)*h_in[i]; } out[idx]=in[offset]; } }
EXPORT void tensor_slice(Tensor* input, Tensor* out, int* start_indices, int* slice_shapes) { if (out->device_id >= 0) { int ndim = input->ndim; int* h_in = (int*)malloc(ndim*4); int* h_out = h_in + ndim; h_in[ndim-1]=1; h_out[ndim-1]=1; for(int i=ndim-2; i>=0; i--) { h_in[i] = h_in[i+1]*input->shape[i+1]; h_out[i] = h_out[i+1]*out->shape[i+1]; } int* d_params; CUDA_CHECK(cudaMalloc((void**)&d_params, 3*ndim*4)); CUDA_CHECK(cudaMemcpy(d_params, start_indices, ndim*4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_params+ndim, h_in, ndim*4, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_params+2*ndim, h_out, ndim*4, cudaMemcpyHostToDevice)); int size = out->size; int threads=256; int blocks=(size+255)/256; slice_kernel<<<blocks, threads>>>(input->data, out->data, size, d_params, d_params+ndim, d_params+2*ndim, ndim); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaFree(d_params)); free(h_in); } }
