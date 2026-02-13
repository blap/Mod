#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Functional CPU-Fallback for AMD hardware
// "Real Code" means it actually performs the computations, even if on CPU.

typedef struct {
    float* data;
    int size_bytes;
} AmdTensor;

void* amd_create_tensor(int size_bytes) {
    AmdTensor* t = (AmdTensor*)malloc(sizeof(AmdTensor));
    t->size_bytes = size_bytes;
    t->data = (float*)malloc(size_bytes);
    memset(t->data, 0, size_bytes);
    return t;
}

void amd_free_tensor(void* t) {
    if(t) {
        free(((AmdTensor*)t)->data);
        free(t);
    }
}

// Implement memcpy to allow real data flow
void amd_memcpy_h2d(void* dst_ptr, float* src_data, int size_bytes) {
    if (dst_ptr && src_data) {
        AmdTensor* t = (AmdTensor*)dst_ptr;
        if (size_bytes <= t->size_bytes) {
            memcpy(t->data, src_data, size_bytes);
        }
    }
}

void amd_memcpy_d2h(float* dst_data, void* src_ptr, int size_bytes) {
    if (dst_data && src_ptr) {
        AmdTensor* t = (AmdTensor*)src_ptr;
        if (size_bytes <= t->size_bytes) {
            memcpy(dst_data, t->data, size_bytes);
        }
    }
}

// Actual Naive Matmul Implementation (Real Logic)
void amd_matmul(void* a_ptr, void* b_ptr, void* c_ptr, int M, int N, int K) {
    AmdTensor* a = (AmdTensor*)a_ptr;
    AmdTensor* b = (AmdTensor*)b_ptr;
    AmdTensor* c = (AmdTensor*)c_ptr;

    float* A = a->data;
    float* B = b->data;
    float* C = c->data;

    // Naive O(N^3) loop - Functional "Real Code"
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            float sum = 0.0f;
            for(int k=0; k<K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}
