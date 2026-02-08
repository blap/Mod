// ... (Previous content of tensor_ops.c) ...
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

// ... (Keep existing memory management) ...
// Re-implementing create/free to ensure context (simplified for append)
EXPORT Tensor* create_tensor(int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->size = 1;
    for (int i = 0; i < ndim; i++) t->size *= shape[i];
    t->data = (float*)malloc(t->size * sizeof(float));
    return t;
}
EXPORT void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) free(t->data);
        if (t->shape) free(t->shape);
        free(t);
    }
}

// ... (Keep existing ops: fill, add, mul, matmul, linear, softmax, silu, rms_norm, rope, quantize) ...
EXPORT void tensor_fill(Tensor* t, float value) {
    #pragma omp parallel for
    for (int i = 0; i < t->size; i++) t->data[i] = value;
}
EXPORT void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] + b->data[i];
}
EXPORT void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    #pragma omp parallel for
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] * b->data[i];
}
// Naive 3D Matmul re-included for completeness in this file rewrite
EXPORT void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    int adim = a->ndim;
    int bdim = b->ndim;
    int Batch = (adim > 2) ? a->shape[0] : 1;
    int M = a->shape[adim-2];
    int K = a->shape[adim-1];
    int N = b->shape[bdim-1];

    #pragma omp parallel for collapse(2)
    for (int b_idx = 0; b_idx < Batch; b_idx++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    int a_idx = (adim > 2 ? b_idx * M * K : 0) + i * K + k;
                    int b_idx_offset = (bdim > 2 ? b_idx * K * N : 0) + k * N + j;
                    sum += a->data[a_idx] * b->data[b_idx_offset];
                }
                out->data[(adim > 2 ? b_idx * M * N : 0) + i * N + j] = sum;
            }
        }
    }
}
EXPORT void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) {
    int M = input->shape[0];
    int K = input->shape[1];
    int N = weight->shape[0];
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += input->data[i * K + k] * weight->data[j * K + k];
            if (bias) sum += bias->data[j];
            out->data[i * N + j] = sum;
        }
    }
}
EXPORT void tensor_softmax(Tensor* input, Tensor* out) {
    int rows = input->size / input->shape[input->ndim - 1];
    int cols = input->shape[input->ndim - 1];
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float max_val = -1e9;
        for (int j = 0; j < cols; j++) if (input->data[i * cols + j] > max_val) max_val = input->data[i * cols + j];
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = expf(input->data[i * cols + j] - max_val);
            out->data[i * cols + j] = val;
            sum_exp += val;
        }
        for (int j = 0; j < cols; j++) out->data[i * cols + j] /= sum_exp;
    }
}
EXPORT void tensor_silu(Tensor* input, Tensor* out) {
    #pragma omp parallel for
    for (int i = 0; i < input->size; i++) {
        float x = input->data[i];
        out->data[i] = x / (1.0f + expf(-x));
    }
}
EXPORT void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) {
    int rows = input->size / input->shape[input->ndim - 1];
    int cols = input->shape[input->ndim - 1];
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < cols; j++) sum_sq += input->data[i * cols + j] * input->data[i * cols + j];
        float rms = sqrtf(sum_sq / cols + eps);
        for (int j = 0; j < cols; j++) out->data[i * cols + j] = (input->data[i * cols + j] / rms) * weight->data[j];
    }
}
EXPORT void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) {
    int size = q->size;
    int dim = q->shape[q->ndim - 1];
    int half_dim = dim / 2;
    int num_tokens = size / dim;
    #pragma omp parallel for
    for (int i = 0; i < num_tokens; i++) {
        int base_idx = i * dim;
        for (int j = 0; j < half_dim; j++) {
            float q_r = q->data[base_idx + j];
            float q_i = q->data[base_idx + j + half_dim];
            float k_r = k->data[base_idx + j];
            float k_i = k->data[base_idx + j + half_dim];
            float c = cos->data[base_idx + j];
            float s = sin->data[base_idx + j];
            out_q->data[base_idx + j] = q_r * c - q_i * s;
            out_q->data[base_idx + j + half_dim] = q_r * s + q_i * c;
            out_k->data[base_idx + j] = k_r * c - k_i * s;
            out_k->data[base_idx + j + half_dim] = k_r * s + k_i * c;
        }
    }
}
EXPORT void tensor_load_data(Tensor* t, float* buffer, int size) { memcpy(t->data, buffer, size * sizeof(float)); }
EXPORT void tensor_get_data(Tensor* t, float* buffer, int size) { memcpy(buffer, t->data, size * sizeof(float)); }
EXPORT void tensor_quantize_int8(Tensor* input, int8_t* out_data, float* scale) {
    float max_val = 0.0f;
    for (int i = 0; i < input->size; i++) if (fabsf(input->data[i]) > max_val) max_val = fabsf(input->data[i]);
    *scale = max_val / 127.0f;
    float inv_scale = 1.0f / (*scale + 1e-9f);
    #pragma omp parallel for
    for (int i = 0; i < input->size; i++) {
        float val = input->data[i] * inv_scale;
        if (val > 127.0f) val = 127.0f; else if (val < -127.0f) val = -127.0f;
        out_data[i] = (int8_t)roundf(val);
    }
}

// --- NEW OPS ---

EXPORT void tensor_argmax(Tensor* input, Tensor* out) {
    // Argmax along last dimension
    // Output shape should have last dim removed (flattened in C logic, Python handles shape)
    int rows = input->size / input->shape[input->ndim - 1];
    int cols = input->shape[input->ndim - 1];

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float max_val = -1e9;
        int max_idx = 0;
        for (int j = 0; j < cols; j++) {
            if (input->data[i * cols + j] > max_val) {
                max_val = input->data[i * cols + j];
                max_idx = j;
            }
        }
        out->data[i] = (float)max_idx; // Storing index as float to reuse float tensor type
    }
}

EXPORT void tensor_embed(Tensor* weight, Tensor* indices, Tensor* out) {
    // Gather rows from weight based on indices
    // Weight: [NumEmbed, Dim]
    // Indices: [Batch, Seq]
    // Out: [Batch, Seq, Dim]

    int num_indices = indices->size;
    int embed_dim = weight->shape[1];

    #pragma omp parallel for
    for (int i = 0; i < num_indices; i++) {
        int idx = (int)indices->data[i];
        if (idx < 0 || idx >= weight->shape[0]) idx = 0; // Boundary check (pad or error)

        // Copy row
        memcpy(out->data + i * embed_dim, weight->data + idx * embed_dim, embed_dim * sizeof(float));
    }
}

EXPORT void tensor_cat(Tensor** inputs, int count, int axis, Tensor* out) {
    // Concatenation. Simplified for:
    // Axis 0 (Batch): Stack [A, B] -> [A+B, ...]
    // Axis 1 (Seq): [B, S1, D], [B, S2, D] -> [B, S1+S2, D] (Assuming 3D)

    // Naive implementation for 2 tensors on axis 1 (common for KV cache [B, H, S, D])
    // or axis 2 if [B, S, D]
    // For simplicity, we just implement memcpy loops based on strides.
    // Assuming inputs[0] and inputs[1] have compatible shapes.

    if (count != 2) return; // Only 2 supported for now
    Tensor* A = inputs[0];
    Tensor* B = inputs[1];

    // Example: Axis 2 (Seq for [B, H, S, D])
    // Or Axis 1 (Seq for [B, S, D])

    // We calculate block size to copy
    int outer_loops = 1;
    for(int i=0; i<axis; ++i) outer_loops *= A->shape[i];

    int stride_A = 1;
    for(int i=axis+1; i<A->ndim; ++i) stride_A *= A->shape[i];
    int stride_B = stride_A; // B must match

    int dim_A = A->shape[axis];
    int dim_B = B->shape[axis];

    int block_size_A = dim_A * stride_A;
    int block_size_B = dim_B * stride_B;

    float* out_ptr = out->data;
    float* a_ptr = A->data;
    float* b_ptr = B->data;

    for(int i=0; i<outer_loops; ++i) {
        memcpy(out_ptr, a_ptr, block_size_A * sizeof(float));
        out_ptr += block_size_A;
        a_ptr += block_size_A;

        memcpy(out_ptr, b_ptr, block_size_B * sizeof(float));
        out_ptr += block_size_B;
        b_ptr += block_size_B;
    }
}

// Forward declarations for Loader and Image Ops
EXPORT int open_safetensors(const char* filepath);
EXPORT int load_tensor_data(const char* name, float* buffer, int size);
EXPORT void close_safetensors();
EXPORT void image_resize_bilinear(float* input, int channels, int h, int w, float* output, int target_h, int target_w);
EXPORT void image_normalize(float* image, int channels, int h, int w, float* mean, float* std);
EXPORT void image_rescale(float* image, int size, float scale);
