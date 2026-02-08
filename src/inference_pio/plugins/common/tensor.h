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

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_PIO_TENSOR_H
