#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#define LOAD_LIB(name) LoadLibraryA(name)
#define GET_PROC(lib, name) GetProcAddress((HMODULE)lib, name)
#define CLOSE_LIB(lib) FreeLibrary((HMODULE)lib)
#else
#include <dlfcn.h>
#define LOAD_LIB(name) dlopen(name, RTLD_LAZY)
#define GET_PROC(lib, name) dlsym(lib, name)
#define CLOSE_LIB(lib) dlclose(lib)
#endif

#include "cl_minimal.h"

#ifndef VENDOR_FILTER
#define VENDOR_FILTER "AMD"
#endif

// --- 1. Global State & Function Pointers ---

static void* g_cl_lib = NULL;
static cl_context g_ctx = NULL;
static cl_command_queue g_queue = NULL;
static cl_program g_program = NULL;

// Kernels
static cl_kernel k_fill = NULL;
static cl_kernel k_add = NULL;
static cl_kernel k_sub = NULL;
static cl_kernel k_mul = NULL;
static cl_kernel k_div = NULL;
static cl_kernel k_matmul = NULL;
static cl_kernel k_rms_norm = NULL;
static cl_kernel k_silu = NULL;
static cl_kernel k_gelu = NULL;
static cl_kernel k_rope = NULL;
static cl_kernel k_softmax = NULL;
static cl_kernel k_topk = NULL;

// Function Pointers
static PTR_clGetPlatformIDs p_clGetPlatformIDs = NULL;
static PTR_clGetPlatformInfo p_clGetPlatformInfo = NULL;
static PTR_clGetDeviceIDs p_clGetDeviceIDs = NULL;
static PTR_clCreateContext p_clCreateContext = NULL;
static PTR_clCreateCommandQueue p_clCreateCommandQueue = NULL;
static PTR_clCreateBuffer p_clCreateBuffer = NULL;
static PTR_clReleaseMemObject p_clReleaseMemObject = NULL;
static PTR_clEnqueueWriteBuffer p_clEnqueueWriteBuffer = NULL;
static PTR_clEnqueueReadBuffer p_clEnqueueReadBuffer = NULL;
static PTR_clCreateProgramWithSource p_clCreateProgramWithSource = NULL;
static PTR_clBuildProgram p_clBuildProgram = NULL;
static PTR_clCreateKernel p_clCreateKernel = NULL;
static PTR_clSetKernelArg p_clSetKernelArg = NULL;
static PTR_clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel = NULL;
static PTR_clFinish p_clFinish = NULL;
static PTR_clGetProgramBuildInfo p_clGetProgramBuildInfo = NULL;

// --- 2. Kernel Source Code ---

const char* KERNEL_SOURCE =
"__kernel void fill(__global float* data, float val) {\n"
"    int idx = get_global_id(0);\n"
"    data[idx] = val;\n"
"}\n"
"__kernel void add(__global const float* a, __global const float* b, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    out[idx] = a[idx] + b[idx];\n"
"}\n"
"__kernel void sub(__global const float* a, __global const float* b, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    out[idx] = a[idx] - b[idx];\n"
"}\n"
"__kernel void mul(__global const float* a, __global const float* b, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    out[idx] = a[idx] * b[idx];\n"
"}\n"
"__kernel void div_op(__global const float* a, __global const float* b, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    out[idx] = a[idx] / b[idx];\n"
"}\n"
"__kernel void matmul(const int M, const int N, const int K,\n"
"                     __global const float* A,\n"
"                     __global const float* B,\n"
"                     __global float* C) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k=0; k<K; k++) {\n"
"            sum += A[row*K + k] * B[k*N + col];\n"
"        }\n"
"        C[row*N + col] = sum;\n"
"    }\n"
"}\n"
"__kernel void rms_norm(__global const float* x, __global const float* w, __global float* out, float eps, int size) {\n"
"    int row = get_global_id(0);\n"
"    // One thread per row (simplified). For real perf use reduction.\n"
"    // Assuming 1D launch over rows\n"
"    // But wait, standard rms_norm is per token (row).\n"
"    // Input is flattened. We need stride.\n"
"    // Kernel arg 'size' is hidden_dim.\n"
"    int offset = row * size;\n"
"    float sum_sq = 0.0f;\n"
"    for(int i=0; i<size; i++) {\n"
"        float v = x[offset + i];\n"
"        sum_sq += v * v;\n"
"    }\n"
"    float scale = rsqrt(sum_sq / size + eps);\n"
"    for(int i=0; i<size; i++) {\n"
"        out[offset + i] = x[offset + i] * scale * w[i];\n"
"    }\n"
"}\n"
"__kernel void silu(__global const float* x, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    float val = x[idx];\n"
"    out[idx] = val / (1.0f + exp(-val));\n"
"}\n"
"__kernel void gelu(__global const float* x, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    float val = x[idx];\n"
"    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (val + 0.044715f * val * val * val)));\n"
"    out[idx] = val * cdf;\n"
"}\n"
"__kernel void rope(__global const float* q, __global const float* k,\n"
"                   __global const float* cos_t, __global const float* sin_t,\n"
"                   __global float* out_q, __global float* out_k,\n"
"                   int dim, int total_tokens) {\n"
"    int idx = get_global_id(0);\n"
"    int half_dim = dim / 2;\n"
"    if (idx < total_tokens * half_dim) {\n"
"        int token_idx = idx / half_dim;\n"
"        int dim_idx = idx % half_dim;\n"
"        // Assuming cos/sin are [Total_Tokens, Half_Dim] flattened\n"
"        float c = cos_t[idx];\n"
"        float s = sin_t[idx];\n"
"        int qk_idx = token_idx * dim + dim_idx;\n"
"        float qr = q[qk_idx];\n"
"        float qi = q[qk_idx + half_dim];\n"
"        float kr = k[qk_idx];\n"
"        float ki = k[qk_idx + half_dim];\n"
"        out_q[qk_idx] = qr * c - qi * s;\n"
"        out_q[qk_idx + half_dim] = qr * s + qi * c;\n"
"        out_k[qk_idx] = kr * c - ki * s;\n"
"        out_k[qk_idx + half_dim] = kr * s + ki * c;\n"
"    }\n"
"}\n"
"__kernel void softmax(__global const float* x, __global float* out, int cols) {\n"
"    int row = get_global_id(0);\n"
"    int offset = row * cols;\n"
"    float max_val = -1e20f;\n"
"    for(int i=0; i<cols; i++) {\n"
"        float v = x[offset + i];\n"
"        if(v > max_val) max_val = v;\n"
"    }\n"
"    float sum = 0.0f;\n"
"    for(int i=0; i<cols; i++) {\n"
"        float v = exp(x[offset + i] - max_val);\n"
"        out[offset + i] = v;\n"
"        sum += v;\n"
"    }\n"
"    for(int i=0; i<cols; i++) {\n"
"        out[offset + i] /= sum;\n"
"    }\n"
"}\n"
"__kernel void topk(__global const float* input, int cols, int k,\n"
"                   __global float* out_val, __global float* out_idx) {\n"
"    int row = get_global_id(0);\n"
"    int offset = row * cols;\n"
"    int out_offset = row * k;\n"
"    // Naive selection sort for small K\n"
"    for (int i=0; i<k; i++) {\n"
"        out_val[out_offset + i] = -1e20f;\n"
"        out_idx[out_offset + i] = -1.0f;\n"
"    }\n"
"    for (int i=0; i<cols; i++) {\n"
"        float val = input[offset + i];\n"
"        // Insert\n"
"        int pos = -1;\n"
"        for (int j=0; j<k; j++) {\n"
"             if (val > out_val[out_offset + j]) {\n"
"                 pos = j;\n"
"                 break;\n"
"             }\n"
"        }\n"
"        if (pos != -1) {\n"
"            for (int j=k-1; j>pos; j--) {\n"
"                out_val[out_offset + j] = out_val[out_offset + j-1];\n"
"                out_idx[out_offset + j] = out_idx[out_offset + j-1];\n"
"            }\n"
"            out_val[out_offset + pos] = val;\n"
"            out_idx[out_offset + pos] = (float)i;\n"
"        }\n"
"    }\n"
"}\n";

// --- 3. Helper Functions ---

#define CHECK_CL(err, msg) if(err != CL_SUCCESS) { printf("[OpenCL Error] %s: %d\n", msg, err); return; }
#define LOAD_SYM(name) p_##name = (PTR_##name)GET_PROC(g_cl_lib, #name); if(!p_##name) { printf("Failed to load symbol %s\n", #name); return 0; }

int init_opencl() {
    if (g_ctx) return 1; // Already initialized

    // 1. Load Library
#ifdef _WIN32
    g_cl_lib = LOAD_LIB("OpenCL.dll");
#else
    g_cl_lib = LOAD_LIB("libOpenCL.so");
    if(!g_cl_lib) g_cl_lib = LOAD_LIB("libOpenCL.so.1");
#endif

    if (!g_cl_lib) {
        printf("Failed to load OpenCL library\n");
        return 0;
    }

    // 2. Load Symbols
    LOAD_SYM(clGetPlatformIDs);
    LOAD_SYM(clGetPlatformInfo);
    LOAD_SYM(clGetDeviceIDs);
    LOAD_SYM(clCreateContext);
    LOAD_SYM(clCreateCommandQueue);
    LOAD_SYM(clCreateBuffer);
    LOAD_SYM(clReleaseMemObject);
    LOAD_SYM(clEnqueueWriteBuffer);
    LOAD_SYM(clEnqueueReadBuffer);
    LOAD_SYM(clCreateProgramWithSource);
    LOAD_SYM(clBuildProgram);
    LOAD_SYM(clCreateKernel);
    LOAD_SYM(clSetKernelArg);
    LOAD_SYM(clEnqueueNDRangeKernel);
    LOAD_SYM(clFinish);
    LOAD_SYM(clGetProgramBuildInfo);

    // 3. Select Platform
    cl_uint num_platforms = 0;
    p_clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) return 0;

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    p_clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_platform_id selected_plat = platforms[0];
    char buffer[128];
    int found = 0;
    for (cl_uint i = 0; i < num_platforms; i++) {
        p_clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 128, buffer, NULL);
        // Case-insensitive strstr would be better but simple check is enough for now
        if (strstr(buffer, VENDOR_FILTER) || strstr(buffer, "Advanced Micro Devices")) {
             selected_plat = platforms[i];
             found = 1;
             break;
        }
    }
    free(platforms);

    if (!found) {
        printf("[OpenCL] Warning: Vendor '%s' not found. Using default platform.\n", VENDOR_FILTER);
    }

    // 4. Device
    cl_device_id device;
    cl_uint num_devices;
    if (p_clGetDeviceIDs(selected_plat, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) != CL_SUCCESS) {
        printf("[OpenCL] No GPU found.\n");
        return 0;
    }

    // 5. Context & Queue
    cl_int err;
    g_ctx = p_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;

    // Check OpenCL version for queue creation (2.0 vs 1.2)
    // For simplicity assuming 1.2+ is fine with basic CreateCommandQueue (deprecated in 2.0 but still works often, or use WithProperties)
    // Actually clCreateCommandQueue is deprecated in 2.0, should use clCreateCommandQueueWithProperties.
    // But dlsym loading usually maps to the available one. Let's try basic.
    g_queue = p_clCreateCommandQueue(g_ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
         // Try finding WithProperties if this failed? For now assume 1.2 compat.
         printf("[OpenCL] Failed to create queue: %d\n", err);
         return 0;
    }

    // 6. Build Program
    g_program = p_clCreateProgramWithSource(g_ctx, 1, &KERNEL_SOURCE, NULL, &err);
    err = p_clBuildProgram(g_program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        p_clGetProgramBuildInfo(g_program, device, 0x1183, 0, NULL, &len); // CL_PROGRAM_BUILD_LOG
        char* log = (char*)malloc(len);
        p_clGetProgramBuildInfo(g_program, device, 0x1183, len, log, NULL);
        printf("[OpenCL] Build Error:\n%s\n", log);
        free(log);
        return 0;
    }

    // 7. Create Kernels
    k_fill = p_clCreateKernel(g_program, "fill", &err);
    k_add = p_clCreateKernel(g_program, "add", &err);
    k_sub = p_clCreateKernel(g_program, "sub", &err);
    k_mul = p_clCreateKernel(g_program, "mul", &err);
    k_div = p_clCreateKernel(g_program, "div_op", &err);
    k_matmul = p_clCreateKernel(g_program, "matmul", &err);
    k_rms_norm = p_clCreateKernel(g_program, "rms_norm", &err);
    k_silu = p_clCreateKernel(g_program, "silu", &err);
    k_gelu = p_clCreateKernel(g_program, "gelu", &err);
    k_rope = p_clCreateKernel(g_program, "rope", &err);
    k_softmax = p_clCreateKernel(g_program, "softmax", &err);
    k_topk = p_clCreateKernel(g_program, "topk", &err);

    return 1;
}

// --- 4. Tensor Ops Implementation ---

typedef struct {
    float* data; // Stores cl_mem cast to float*
    int* shape;
    int ndim;
    int size;
    int device_id;
} Tensor;

// NOTE: We assume Tensor* passed from Python already has allocated 'data' field used as cl_mem.
// But Python `create_tensor` expects us to return a Tensor*.

Tensor* create_tensor(int* shape, int ndim, int device_id) {
    if (!init_opencl()) return NULL;

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->size = 1;
    for (int i = 0; i < ndim; i++) t->size *= shape[i];
    t->device_id = device_id;

    cl_int err;
    cl_mem mem = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, t->size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[OpenCL] Alloc failed: %d\n", err);
        free(t->shape); free(t); return NULL;
    }

    t->data = (float*)mem; // Store handle in float* slot
    return t;
}

void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) p_clReleaseMemObject((cl_mem)t->data);
        free(t->shape);
        free(t);
    }
}

void tensor_load_data(Tensor* t, float* buffer, int size) {
    if (!g_queue) return;
    p_clEnqueueWriteBuffer(g_queue, (cl_mem)t->data, CL_TRUE, 0, size * sizeof(float), buffer, 0, NULL, NULL);
}

void tensor_get_data(Tensor* t, float* buffer, int size) {
    if (!g_queue) return;
    p_clEnqueueReadBuffer(g_queue, (cl_mem)t->data, CL_TRUE, 0, size * sizeof(float), buffer, 0, NULL, NULL);
}

void tensor_fill(Tensor* t, float value) {
    if (!k_fill) return;
    p_clSetKernelArg(k_fill, 0, sizeof(cl_mem), &t->data);
    p_clSetKernelArg(k_fill, 1, sizeof(float), &value);
    size_t work_size = t->size;
    p_clEnqueueNDRangeKernel(g_queue, k_fill, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_add) return;
    p_clSetKernelArg(k_add, 0, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_add, 1, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_add, 2, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_add, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_mul) return;
    p_clSetKernelArg(k_mul, 0, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_mul, 1, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_mul, 2, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_mul, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_matmul) return;
    // Dimensions: A[M, K], B[K, N] -> C[M, N]
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    p_clSetKernelArg(k_matmul, 0, sizeof(int), &M);
    p_clSetKernelArg(k_matmul, 1, sizeof(int), &N);
    p_clSetKernelArg(k_matmul, 2, sizeof(int), &K);
    p_clSetKernelArg(k_matmul, 3, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_matmul, 4, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_matmul, 5, sizeof(cl_mem), &out->data);

    size_t global_work[2] = {N, M}; // Col, Row
    p_clEnqueueNDRangeKernel(g_queue, k_matmul, 2, NULL, global_work, NULL, 0, NULL, NULL);
}

void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) {
    if (!k_rms_norm) return;
    // Input flattened [rows * hidden], Weight [hidden]
    int hidden = input->shape[input->ndim - 1];
    int rows = input->size / hidden;

    p_clSetKernelArg(k_rms_norm, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_rms_norm, 1, sizeof(cl_mem), &weight->data);
    p_clSetKernelArg(k_rms_norm, 2, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_rms_norm, 3, sizeof(float), &eps);
    p_clSetKernelArg(k_rms_norm, 4, sizeof(int), &hidden);

    size_t global_work = rows;
    p_clEnqueueNDRangeKernel(g_queue, k_rms_norm, 1, NULL, &global_work, NULL, 0, NULL, NULL);
}

void tensor_silu(Tensor* input, Tensor* out) {
    if (!k_silu) return;
    p_clSetKernelArg(k_silu, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_silu, 1, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_silu, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

void tensor_gelu(Tensor* input, Tensor* out) {
    if (!k_gelu) return;
    p_clSetKernelArg(k_gelu, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_gelu, 1, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_gelu, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) {
    if (!k_rope) return;
    int dim = q->shape[q->ndim - 1];
    int total_tokens = q->size / dim;

    p_clSetKernelArg(k_rope, 0, sizeof(cl_mem), &q->data);
    p_clSetKernelArg(k_rope, 1, sizeof(cl_mem), &k->data);
    p_clSetKernelArg(k_rope, 2, sizeof(cl_mem), &cos->data);
    p_clSetKernelArg(k_rope, 3, sizeof(cl_mem), &sin->data);
    p_clSetKernelArg(k_rope, 4, sizeof(cl_mem), &out_q->data);
    p_clSetKernelArg(k_rope, 5, sizeof(cl_mem), &out_k->data);
    p_clSetKernelArg(k_rope, 6, sizeof(int), &dim);
    p_clSetKernelArg(k_rope, 7, sizeof(int), &total_tokens);

    size_t work_size = total_tokens * (dim / 2); // One thread per pair
    p_clEnqueueNDRangeKernel(g_queue, k_rope, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

void tensor_softmax(Tensor* input, Tensor* out) {
    if (!k_softmax) return;
    int cols = input->shape[input->ndim - 1];
    int rows = input->size / cols;

    p_clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_softmax, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_softmax, 2, sizeof(int), &cols);

    size_t global_work = rows;
    p_clEnqueueNDRangeKernel(g_queue, k_softmax, 1, NULL, &global_work, NULL, 0, NULL, NULL);
}

void tensor_topk(Tensor* input, int k, Tensor* out_val, Tensor* out_idx) {
    if (!k_topk) return;
    int cols = input->shape[input->ndim - 1];
    int rows = input->size / cols;

    p_clSetKernelArg(k_topk, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_topk, 1, sizeof(int), &cols);
    p_clSetKernelArg(k_topk, 2, sizeof(int), &k);
    p_clSetKernelArg(k_topk, 3, sizeof(cl_mem), &out_val->data);
    p_clSetKernelArg(k_topk, 4, sizeof(cl_mem), &out_idx->data);

    size_t global_work = rows;
    p_clEnqueueNDRangeKernel(g_queue, k_topk, 1, NULL, &global_work, NULL, 0, NULL, NULL);
}

// Add stub/implementations for other ops to match CPU backend symbols if needed
// For now, this covers the requested improvements.
