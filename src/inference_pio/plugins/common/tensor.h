#ifndef INFERENCE_PIO_TENSOR_H
#define INFERENCE_PIO_TENSOR_H

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared Tensor Structure
typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
    int device_id; // -1 for CPU, >=0 for CUDA GPU ID
} Tensor;

// Core Memory Management
EXPORT Tensor* create_tensor(int* shape, int ndim, int device_id);
EXPORT void free_tensor(Tensor* t);

// Utility
EXPORT void tensor_fill(Tensor* t, float value);
EXPORT void tensor_load_data(Tensor* t, float* buffer, int size);
EXPORT void tensor_get_data(Tensor* t, float* buffer, int size);

// Math
EXPORT void tensor_add(Tensor* a, Tensor* b, Tensor* out);
EXPORT void tensor_mul(Tensor* a, Tensor* b, Tensor* out);
EXPORT void tensor_matmul(Tensor* a, Tensor* b, Tensor* out);
EXPORT void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out);
EXPORT void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out);
EXPORT void tensor_softmax(Tensor* input, Tensor* out);
EXPORT void tensor_silu(Tensor* input, Tensor* out);
EXPORT void tensor_gelu(Tensor* input, Tensor* out);
EXPORT void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps);
EXPORT void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k);

// New Ops
EXPORT void tensor_slice(Tensor* input, Tensor* out, int* start_indices, int* slice_shapes);
EXPORT void tensor_precompute_freqs_cis(int dim, int end, float theta, Tensor* out_cos, Tensor* out_sin);
EXPORT void tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding, int groups);

// Ops that were implicitly exported before but good to decl
EXPORT void tensor_argmax(Tensor* input, Tensor* out);
EXPORT void tensor_embed(Tensor* weight, Tensor* indices, Tensor* out);
EXPORT void tensor_cat(Tensor** inputs, int count, int axis, Tensor* out);

// Loader / Image
typedef struct SafetensorsContext SafetensorsContext;
EXPORT SafetensorsContext* open_safetensors(const char* filepath);
EXPORT int load_tensor_data(SafetensorsContext* ctx, const char* name, float* buffer, int size);
EXPORT void close_safetensors(SafetensorsContext* ctx);
EXPORT void image_resize_bilinear(float* input, int channels, int h, int w, float* output, int target_h, int target_w);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_PIO_TENSOR_H
