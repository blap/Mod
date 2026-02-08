#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Basic float tensor structure
typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

// --- Memory Management ---

Tensor* create_tensor(int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));

    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->size *= shape[i];
    }

    t->data = (float*)malloc(t->size * sizeof(float));
    return t;
}

void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) free(t->data);
        if (t->shape) free(t->shape);
        free(t);
    }
}

// --- Operations ---

void tensor_fill(Tensor* t, float value) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
}

void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    // Naive element-wise add (assuming shapes match or broadcasting handled in python)
    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    // 2D Matmul: A[M, K] @ B[K, N] -> Out[M, N]
    // A: [..., M, K], B: [..., K, N]
    // Simplified for 2D.

    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    // Naive O(n^3) implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }
}

void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) {
    // input: [M, K], weight: [N, K] (transposed standard), bias: [N]
    // out: [M, N]

    int M = input->shape[0]; // Simplified for 2D input
    int K = input->shape[1];
    int N = weight->shape[0];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += input->data[i * K + k] * weight->data[j * K + k]; // weight is [N, K]
            }
            if (bias) {
                sum += bias->data[j];
            }
            out->data[i * N + j] = sum;
        }
    }
}

void tensor_softmax(Tensor* input, Tensor* out) {
    // Softmax along last dim
    int rows = input->size / input->shape[input->ndim - 1];
    int cols = input->shape[input->ndim - 1];

    for (int i = 0; i < rows; i++) {
        float max_val = -1e9;
        for (int j = 0; j < cols; j++) {
            if (input->data[i * cols + j] > max_val) {
                max_val = input->data[i * cols + j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = expf(input->data[i * cols + j] - max_val);
            out->data[i * cols + j] = val;
            sum_exp += val;
        }

        for (int j = 0; j < cols; j++) {
            out->data[i * cols + j] /= sum_exp;
        }
    }
}

void tensor_silu(Tensor* input, Tensor* out) {
    for (int i = 0; i < input->size; i++) {
        float x = input->data[i];
        out->data[i] = x / (1.0f + expf(-x));
    }
}

void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) {
    // Last dim norm
    int rows = input->size / input->shape[input->ndim - 1];
    int cols = input->shape[input->ndim - 1];

    for (int i = 0; i < rows; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_sq += input->data[i * cols + j] * input->data[i * cols + j];
        }
        float rms = sqrtf(sum_sq / cols + eps);

        for (int j = 0; j < cols; j++) {
            out->data[i * cols + j] = (input->data[i * cols + j] / rms) * weight->data[j];
        }
    }
}

void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) {
    // Simplified 1D RoPE
    // q, k: [..., dim]
    // cos, sin: [..., dim]
    // Assuming contiguous memory

    int size = q->size;
    int dim = q->shape[q->ndim - 1];
    int half_dim = dim / 2;
    int num_tokens = size / dim;

    for (int i = 0; i < num_tokens; i++) {
        int base_idx = i * dim;
        for (int j = 0; j < half_dim; j++) {
            // Pair (j, j + half_dim)
            float q_r = q->data[base_idx + j];
            float q_i = q->data[base_idx + j + half_dim];
            float k_r = k->data[base_idx + j];
            float k_i = k->data[base_idx + j + half_dim];

            float c = cos->data[base_idx + j]; // Assuming broadcast or matched shape
            float s = sin->data[base_idx + j];

            // Rotate Q
            out_q->data[base_idx + j] = q_r * c - q_i * s;
            out_q->data[base_idx + j + half_dim] = q_r * s + q_i * c;

            // Rotate K
            out_k->data[base_idx + j] = k_r * c - k_i * s;
            out_k->data[base_idx + j + half_dim] = k_r * s + k_i * c;
        }
    }
}

// Helper to copy data from python buffer
void tensor_load_data(Tensor* t, float* buffer, int size) {
    memcpy(t->data, buffer, size * sizeof(float));
}

// Helper to retrieve data
void tensor_get_data(Tensor* t, float* buffer, int size) {
    memcpy(buffer, t->data, size * sizeof(float));
}
