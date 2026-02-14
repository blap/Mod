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

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

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
static cl_kernel k_fused_attn = NULL;
static cl_kernel k_matmul_transposed = NULL;
static cl_kernel k_linear = NULL;
static cl_kernel k_swiglu = NULL;
static cl_kernel k_fused_gate_up_swiglu = NULL;
static cl_kernel k_argmax = NULL;
static cl_kernel k_embed = NULL;
static cl_kernel k_cat = NULL;
static cl_kernel k_slice = NULL;
static cl_kernel k_slice_device = NULL;
static cl_kernel k_set_slice = NULL;
static cl_kernel k_set_slice_device = NULL;
static cl_kernel k_permute = NULL;
static cl_kernel k_paged_attn = NULL;
static cl_kernel k_fused_split_rope = NULL;
static cl_kernel k_precompute_freqs = NULL;
static cl_kernel k_count_value = NULL;
static cl_kernel k_gather_by_value = NULL;
static cl_kernel k_scatter_add_by_index = NULL;
static cl_kernel k_deltanet_recurrence = NULL;
static cl_kernel k_conv2d = NULL;
static cl_kernel k_dequantize = NULL;
static cl_kernel k_fused_add_rms_norm = NULL;
static cl_kernel k_gemv = NULL;
static cl_kernel k_fused_add_mul = NULL;
static cl_kernel k_matmul_image = NULL;
static cl_kernel k_verify_tokens = NULL;
static cl_kernel k_fp32_to_bf16 = NULL;
static cl_kernel k_bf16_to_fp32 = NULL;
static cl_kernel k_matmul_fp16 = NULL;

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
static PTR_clEnqueueCopyBuffer p_clEnqueueCopyBuffer = NULL;
static PTR_clGetProgramInfo p_clGetProgramInfo = NULL;
static PTR_clCreateProgramWithBinary p_clCreateProgramWithBinary = NULL;
static PTR_clReleaseProgram p_clReleaseProgram = NULL;
static PTR_clWaitForEvents p_clWaitForEvents = NULL;
static PTR_clReleaseEvent p_clReleaseEvent = NULL;
static PTR_clGetEventProfilingInfo p_clGetEventProfilingInfo = NULL;
static PTR_clEnqueueFillBuffer p_clEnqueueFillBuffer = NULL;

// --- 2. Kernel Source Code ---

const char* KERNEL_SOURCE =
"#ifdef cl_khr_fp16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void add_fp16(__global const half* a, __global const half* b, __global half* out) {\n"
"    int idx = get_global_id(0);\n"
"    out[idx] = a[idx] + b[idx];\n"
"}\n"
"__kernel void matmul_fp16(const int M, const int N, const int K, __global const half* A, __global const half* B, __global half* C) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < M && col < N) {\n"
"        half sum = 0.0h;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row*K + k] * B[k*N + col];\n"
"        }\n"
"        C[row*N + col] = sum;\n"
"    }\n"
"}\n"
"#endif\n"
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
"__kernel __attribute__((reqd_work_group_size(32, 32, 1))) void matmul(const int M, const int N, const int K,\n"
"                     __global const float* A,\n"
"                     __global const float* B,\n"
"                     __global float* C) {\n"
"    const int row = get_local_id(1);\n"
"    const int col = get_local_id(0);\n"
"    const int globalRow = get_global_id(1);\n"
"    const int globalCol = get_global_id(0);\n"
"    __local float Asub[32][32];\n"
"    __local float Bsub[32][32];\n"
"    float acc = 0.0f;\n"
"    const int numTiles = (K + 31) / 32;\n"
"    for (int t = 0; t < numTiles; t++) {\n"
"        const int tiledRow = globalRow;\n"
"        const int tiledCol = t * 32 + col;\n"
"        const int tiledRowB = t * 32 + row;\n"
"        const int tiledColB = globalCol;\n"
"        float a_val = 0.0f;\n"
"        float b_val = 0.0f;\n"
"        if (tiledRow < M && tiledCol < K) a_val = A[tiledRow * K + tiledCol];\n"
"        if (tiledRowB < K && tiledColB < N) b_val = B[tiledRowB * N + tiledColB];\n"
"        Asub[row][col] = a_val;\n"
"        Bsub[row][col] = b_val;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        #pragma unroll\n"
"        for (int k = 0; k < 32; k++) {\n"
"            acc += Asub[row][k] * Bsub[k][col];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (globalRow < M && globalCol < N)\n"
"        C[globalRow * N + globalCol] = acc;\n"
"}\n"
"__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void gemv(const int K, const int N, \n"
"                   __global const float* A, __global const float* B, __global float* C) {\n"
"    int col = get_global_id(0);\n"
"    if (col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[k] * B[k * N + col];\n"
"        }\n"
"        C[col] = sum;\n"
"    }\n"
"}\n"
"__kernel void matmul_image(const int M, const int N, const int K,\n"
"                           __global const float* A,\n"
"                           __read_only image2d_t B_img,\n"
"                           __global float* C) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"        for (int k = 0; k < K; k++) {\n"
"            float4 val = read_imagef(B_img, smp, (int2)(col, k));\n"
"            sum += A[row*K + k] * val.x;\n"
"        }\n"
"        C[row*N + col] = sum;\n"
"    }\n"
"}\n"
"__kernel void fused_add_mul(__global const float* a, __global const float* b, __global const float* c, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    out[idx] = a[idx] + b[idx] * c[idx];\n"
"}\n"
"__kernel void rms_norm(__global const float* x, __global const float* w, __global float* out, float eps, int size) {\n"
"    int row = get_global_id(0);\n"
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
"    int row = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    int local_size = get_local_size(0);\n"
"    int offset = row * cols;\n"
"    float max_val = -1e20f;\n"
"    for (int i = tid; i < cols; i += local_size) {\n"
"        max_val = fmax(max_val, x[offset + i]);\n"
"    }\n"
"    __local float s_data[256];\n"
"    s_data[tid] = max_val;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int stride = local_size/2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) s_data[tid] = fmax(s_data[tid], s_data[tid + stride]);\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    float global_max = s_data[0];\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < cols; i += local_size) {\n"
"        sum += exp(x[offset + i] - global_max);\n"
"    }\n"
"    s_data[tid] = sum;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int stride = local_size/2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) s_data[tid] += s_data[tid + stride];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    float global_sum = s_data[0];\n"
"    float inv_sum = 1.0f / (global_sum + 1e-6f);\n"
"    for (int i = tid; i < cols; i += local_size) {\n"
"        out[offset + i] = exp(x[offset + i] - global_max) * inv_sum;\n"
"    }\n"
"}\n"
"__kernel void topk(__global const float* input, int cols, int k,\n"
"                   __global float* out_val, __global float* out_idx) {\n"
"    int row = get_global_id(0);\n"
"    int offset = row * cols;\n"
"    int out_offset = row * k;\n"
"    for (int i=0; i<k; i++) {\n"
"        out_val[out_offset + i] = -1e20f;\n"
"        out_idx[out_offset + i] = -1.0f;\n"
"    }\n"
"    for (int i=0; i<cols; i++) {\n"
"        float val = input[offset + i];\n"
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
"}\n"
"__kernel void fused_attention(\n"
"    __global const float* Q, __global const float* K, __global const float* V,\n"
"    __global float* O, float scale, int seq_len, int head_dim, int total_heads) {\n"
"    int bh = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    int d_offset = tid;\n"
"    if (d_offset >= head_dim) return;\n"
"    float q_val = Q[bh * head_dim + d_offset];\n"
"    float o_val = 0.0f;\n"
"    float m = -1e20f;\n"
"    float l = 0.0f;\n"
"    for (int t = 0; t < seq_len; t++) {\n"
"        __local float s_dot[256];\n"
"        float k_val = K[(bh * seq_len + t) * head_dim + d_offset];\n"
"        s_dot[tid] = q_val * k_val;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {\n"
"            if (tid < stride) s_dot[tid] += s_dot[tid + stride];\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"        }\n"
"        float score = s_dot[0] * scale;\n"
"        float m_prev = m;\n"
"        m = max(m_prev, score);\n"
"        float exp_val = exp(score - m);\n"
"        float correction = exp(m_prev - m);\n"
"        l = l * correction + exp_val;\n"
"        float v_val = V[(bh * seq_len + t) * head_dim + d_offset];\n"
"        o_val = o_val * correction + v_val * exp_val;\n"
"    }\n"
"    O[bh * head_dim + d_offset] = o_val / l;\n"
"}\n"
"__kernel void matmul_transposed(const int M, const int N, const int K,\n"
"                                __global const float* A,\n"
"                                __global const float* B,\n"
"                                __global float* C) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k=0; k<K; k++) {\n"
"            sum += A[row*K + k] * B[col*K + k];\n"
"        }\n"
"        C[row*N + col] = sum;\n"
"    }\n"
"}\n"
"__kernel void linear(const int rows, const int K, const int N,\n"
"                     __global const float* input,\n"
"                     __global const float* weight,\n"
"                     __global const float* bias,\n"
"                     __global float* out) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < rows && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k=0; k<K; k++) {\n"
"            sum += input[row*K + k] * weight[col*K + k];\n"
"        }\n"
"        if (bias) sum += bias[col];\n"
"        out[row*N + col] = sum;\n"
"    }\n"
"}\n"
"__kernel void swiglu(__global const float* gate, __global const float* up, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    float g = gate[idx];\n"
"    float u = up[idx];\n"
"    float sg = g / (1.0f + exp(-g));\n"
"    out[idx] = sg * u;\n"
"}\n"
"__kernel void fused_gate_up_swiglu(__global const float* gate_up, __global float* out, int hidden) {\n"
"    int idx = get_global_id(0);\n"
"    int row = idx / hidden;\n"
"    int col = idx % hidden;\n"
"    int in_idx = row * (2 * hidden) + col;\n"
"    float g = gate_up[in_idx];\n"
"    float u = gate_up[in_idx + hidden];\n"
"    float sg = g / (1.0f + exp(-g));\n"
"    out[idx] = sg * u;\n"
"}\n"
"__kernel void argmax(__global const float* input, __global float* out, int cols) {\n"
"    int row = get_global_id(0);\n"
"    int offset = row * cols;\n"
"    float max_val = -1e20f;\n"
"    int max_idx = 0;\n"
"    for(int i=0; i<cols; i++) {\n"
"        float v = input[offset + i];\n"
"        if(v > max_val) { max_val = v; max_idx = i; }\n"
"    }\n"
"    out[row] = (float)max_idx;\n"
"}\n"
"__kernel void embed(__global const float* weight, __global const float* indices, __global float* out, int hidden) {\n"
"    int idx = get_global_id(0);\n"
"    int token_idx = idx / hidden;\n"
"    int dim_idx = idx % hidden;\n"
"    int emb_idx = (int)indices[token_idx];\n"
"    out[idx] = weight[emb_idx * hidden + dim_idx];\n"
"}\n"
"__kernel void slice(__global const float* input, __global float* out, int ndim, \n"
"                    __global const int* start_indices, __global const int* h_in, __global const int* h_out) {\n"
"    int idx = get_global_id(0);\n"
"    int temp = idx;\n"
"    int in_offset = 0;\n"
"    for(int i=0; i<ndim; i++) {\n"
"        int c = temp / h_out[i];\n"
"        temp %= h_out[i];\n"
"        in_offset += (start_indices[i] + c) * h_in[i];\n"
"    }\n"
"    out[idx] = input[in_offset];\n"
"}\n"
"__kernel void slice_device(__global const float* input, __global float* out, int ndim, \n"
"                    __global const float* start_indices_float, __global const int* h_in, __global const int* h_out) {\n"
"    int idx = get_global_id(0);\n"
"    int temp = idx;\n"
"    int in_offset = 0;\n"
"    for(int i=0; i<ndim; i++) {\n"
"        int c = temp / h_out[i];\n"
"        temp %= h_out[i];\n"
"        in_offset += ((int)start_indices_float[i] + c) * h_in[i];\n"
"    }\n"
"    out[idx] = input[in_offset];\n"
"}\n"
"__kernel void set_slice(__global float* dst, __global const float* src, int ndim, \n"
"                        __global const int* start_indices, __global const int* h_dst, __global const int* h_src) {\n"
"    int idx = get_global_id(0);\n"
"    int temp = idx;\n"
"    int dst_offset = 0;\n"
"    for(int i=0; i<ndim; i++) {\n"
"        int c = temp / h_src[i];\n"
"        temp %= h_src[i];\n"
"        dst_offset += (start_indices[i] + c) * h_dst[i];\n"
"    }\n"
"    dst[dst_offset] = src[idx];\n"
"}\n"
"__kernel void set_slice_device(__global float* dst, __global const float* src, int ndim, \n"
"                        __global const float* start_indices_float, __global const int* h_dst, __global const int* h_src) {\n"
"    int idx = get_global_id(0);\n"
"    int temp = idx;\n"
"    int dst_offset = 0;\n"
"    for(int i=0; i<ndim; i++) {\n"
"        int c = temp / h_src[i];\n"
"        temp %= h_src[i];\n"
"        dst_offset += ((int)start_indices_float[i] + c) * h_dst[i];\n"
"    }\n"
"    dst[dst_offset] = src[idx];\n"
"}\n"
"__kernel void permute(__global const float* input, __global float* out, int ndim, \n"
"                      __global const int* dims, __global const int* h_in, __global const int* h_out) {\n"
"    int idx = get_global_id(0);\n"
"    int temp = idx;\n"
"    int in_offset = 0;\n"
"    for(int i=0; i<ndim; i++) {\n"
"        int c = temp / h_out[i];\n"
"        temp %= h_out[i];\n"
"        in_offset += c * h_in[dims[i]];\n"
"    }\n"
"    out[idx] = input[in_offset];\n"
"}\n"
"__kernel void precompute_freqs(__global float* cos_out, __global float* sin_out, \n"
"                               int end, int half, float theta) {\n"
"    int idx = get_global_id(0);\n"
"    if(idx < end*half) {\n"
"        int i = idx / half;\n"
"        int j = idx % half;\n"
"        float freq = 1.0f / pow(theta, (float)(2*j) / (half*2));\n"
"        float val = i * freq;\n"
"        cos_out[idx] = cos(val);\n"
"        sin_out[idx] = sin(val);\n"
"    }\n"
"}\n"
"__kernel void fused_split_rope(__global const float* qkv, __global const float* cos_t, __global const float* sin_t,\n"
"                               __global float* q_out, __global float* k_out, __global float* v_out,\n"
"                               int heads, int head_dim, int total_tokens) {\n"
"    int idx = get_global_id(0);\n"
"    int hidden = heads * head_dim;\n"
"    int total_elements = total_tokens * hidden;\n"
"    if (idx < total_elements) {\n"
"        int token_idx = idx / hidden;\n"
"        int dim_idx = idx % hidden;\n"
"        int d = dim_idx % head_dim;\n"
"        int half_dim = head_dim / 2;\n"
"        float q_val = qkv[token_idx * 3 * hidden + dim_idx];\n"
"        float k_val = qkv[token_idx * 3 * hidden + hidden + dim_idx];\n"
"        float v_val = qkv[token_idx * 3 * hidden + 2 * hidden + dim_idx];\n"
"        float c = 1.0f, s = 0.0f;\n"
"        int rot_idx = token_idx * half_dim + (d % half_dim);\n"
"        c = cos_t[rot_idx];\n"
"        s = sin_t[rot_idx];\n"
"        float val_r, val_i;\n"
"        if (d < half_dim) {\n"
"             val_r = q_val;\n"
"             float q_val_i = qkv[token_idx * 3 * hidden + dim_idx + half_dim];\n"
"             q_val = val_r * c - q_val_i * s;\n"
"             val_r = k_val;\n"
"             float k_val_i = qkv[token_idx * 3 * hidden + hidden + dim_idx + half_dim];\n"
"             k_val = val_r * c - k_val_i * s;\n"
"        } else {\n"
"             val_i = q_val;\n"
"             float q_val_r = qkv[token_idx * 3 * hidden + dim_idx - half_dim];\n"
"             q_val = q_val_r * s + val_i * c;\n"
"             val_i = k_val;\n"
"             float k_val_r = qkv[token_idx * 3 * hidden + hidden + dim_idx - half_dim];\n"
"             k_val = k_val_r * s + val_i * c;\n"
"        }\n"
"        q_out[idx] = q_val;\n"
"        k_out[idx] = k_val;\n"
"        v_out[idx] = v_val;\n"
"    }\n"
"}\n"
"__kernel void paged_attention(\n"
"    __global const float* Q, __global const float* K_cache, __global const float* V_cache,\n"
"    __global const int* block_tables, __global const int* context_lens,\n"
"    __global float* Out, float scale, int page_size, int max_blocks, int head_dim) {\n"
"    int b = get_group_id(0);\n"
"    int h = get_group_id(1);\n"
"    int tid = get_local_id(0);\n"
"    int heads = get_num_groups(1);\n"
"    if (tid >= head_dim) return;\n"
"    float q_val = Q[(b * heads + h) * head_dim + tid];\n"
"    float sum_score = 0.0f;\n"
"    float max_score = -1e20f;\n"
"    float acc_o = 0.0f;\n"
"    int seq_len = context_lens[b];\n"
"    int num_pages = (seq_len + page_size - 1) / page_size;\n"
"    for (int p = 0; p < num_pages; p++) {\n"
"        int block_idx = block_tables[b * max_blocks + p];\n"
"        int num_tokens = (p == num_pages - 1) ? (seq_len - p * page_size) : page_size;\n"
"        for (int t = 0; t < num_tokens; t++) {\n"
"            int k_offset = ((block_idx * page_size + t) * heads + h) * head_dim;\n"
"            __local float s_dot[256];\n"
"            s_dot[tid] = q_val * K_cache[k_offset + tid];\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"            for (int s = head_dim/2; s > 0; s >>= 1) {\n"
"                if (tid < s) s_dot[tid] += s_dot[tid + s];\n"
"                barrier(CLK_LOCAL_MEM_FENCE);\n"
"            }\n"
"            float score = s_dot[0] * scale;\n"
"            float m_prev = max_score;\n"
"            max_score = max(max_score, score);\n"
"            float exp_val = exp(score - max_score);\n"
"            float alpha = exp(m_prev - max_score);\n"
"            sum_score = sum_score * alpha + exp_val;\n"
"            int v_offset = k_offset;\n"
"            float v_val = V_cache[v_offset + tid];\n"
"            acc_o = acc_o * alpha + v_val * exp_val;\n"
"        }\n"
"    }\n"
"    Out[(b * heads + h) * head_dim + tid] = acc_o / sum_score;\n"
"}\n"
"__kernel void count_value(__global const float* data, int size, float value, __global int* out_count) {\n"
"    int idx = get_global_id(0);\n"
"    int match = 0;\n"
"    if (idx < size) {\n"
"        if (fabs(data[idx] - value) < 1e-6) match = 1;\n"
"    }\n"
"    if (match) atomic_add(out_count, 1);\n"
"}\n"
"__kernel void gather_by_value(__global const float* input, __global const float* indices, int size, float value, \n"
"                              __global float* out_data, __global float* out_indices, int hidden_size, \n"
"                              __global int* g_counter) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < size) {\n"
"        if (fabs(indices[idx] - value) < 1e-6) {\n"
"            int pos = atomic_add(g_counter, 1);\n"
"            out_indices[pos] = (float)idx;\n"
"            int src_base = idx * hidden_size;\n"
"            int dst_base = pos * hidden_size;\n"
"            for (int j = 0; j < hidden_size; j++) {\n"
"                out_data[dst_base + j] = input[src_base + j];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void scatter_add_by_index(__global float* out, __global const float* src, __global const float* indices,\n"
"                                   int count, int hidden_size, int total_rows) {\n"
"    int idx = get_global_id(0);\n"
"    int total_elements = count * hidden_size;\n"
"    if (idx < total_elements) {\n"
"        int row_idx = idx / hidden_size;\n"
"        int col_idx = idx % hidden_size;\n"
"        int target_row = (int)indices[row_idx];\n"
"        if (target_row >= 0 && target_row < total_rows) {\n"
"            float val = src[idx];\n"
"            __global int* out_int = (__global int*)out;\n"
"            int offset = target_row * hidden_size + col_idx;\n"
"            int old = out_int[offset];\n"
"            int assumed;\n"
"            do {\n"
"                assumed = old;\n"
"                float sum_val = as_float(assumed) + val;\n"
"                old = atomic_cmpxchg(out_int + offset, assumed, as_int(sum_val));\n"
"            } while (assumed != old);\n"
"        }\n"
"    }\n"
"}\n"
"__kernel void deltanet_recurrence(__global const float* q, __global const float* k, __global const float* v, __global const float* beta,\n"
"                                  __global float* state, __global float* out, int B, int S, int H, int D) {\n"
"    int b = get_group_id(0);\n"
"    int h = get_group_id(1);\n"
"    int tid = get_local_id(0);\n"
"    __global float* my_state = state + (b*H + h) * D * D;\n"
"    for (int t = 0; t < S; t++) {\n"
"        int offset = ((b*S + t)*H + h);\n"
"        int q_off = offset * D;\n"
"        int k_off = offset * D;\n"
"        int v_off = offset * D;\n"
"        int out_off = offset * D;\n"
"        float b_val = beta[offset];\n"
"        for (int i = tid; i < D*D; i += get_local_size(0)) {\n"
"            int r = i / D;\n"
"            int c = i % D;\n"
"            float kv = k[k_off + r] * v[v_off + c];\n"
"            my_state[i] = b_val * my_state[i] + kv;\n"
"        }\n"
"        barrier(CLK_GLOBAL_MEM_FENCE);\n"
"        if (tid < D) {\n"
"            float sum = 0.0f;\n"
"            for (int i = 0; i < D; i++) {\n"
"                sum += q[q_off + i] * my_state[i*D + tid];\n"
"            }\n"
"            out[out_off + tid] = sum;\n"
"        }\n"
"        barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    }\n"
"}\n"
"__kernel void conv2d_naive(__global const float* input, __global const float* weight, __global const float* bias, __global float* out,\n"
"                           int C_in, int H_in, int W_in, int C_out, int H_out, int W_out, \n"
"                           int KH, int KW, int stride, int padding) {\n"
"    int idx = get_global_id(0);\n"
"    int w_out_idx = idx % W_out;\n"
"    int temp = idx / W_out;\n"
"    int h_out_idx = temp % H_out;\n"
"    temp /= H_out;\n"
"    int c_out_idx = temp % C_out;\n"
"    int n_idx = temp / C_out;\n"
"    float sum = 0.0f;\n"
"    int h_in_base = h_out_idx * stride - padding;\n"
"    int w_in_base = w_out_idx * stride - padding;\n"
"    for(int c=0; c<C_in; c++) {\n"
"        for(int i=0; i<KH; i++) {\n"
"            for(int j=0; j<KW; j++) {\n"
"                int h = h_in_base + i;\n"
"                int w = w_in_base + j;\n"
"                if (h >= 0 && h < H_in && w >= 0 && w < W_in) {\n"
"                    int in_off = ((n_idx * C_in + c) * H_in + h) * W_in + w;\n"
"                    int w_off = ((c_out_idx * C_in + c) * KH + i) * KW + j;\n"
"                    sum += input[in_off] * weight[w_off];\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    if (bias) sum += bias[c_out_idx];\n"
"    out[idx] = sum;\n"
"}\n"
"__kernel void dequantize(__global const char* input, __global const float* scale, __global float* out, int hidden) {\n"
"    int idx = get_global_id(0);\n"
"    float s = scale[0]; \n"
"    out[idx] = (float)input[idx] * s;\n"
"}\n"
"__kernel void fused_add_rms_norm(__global float* x, __global const float* residual, \n"
"                                 __global const float* weight, __global float* out, \n"
"                                 float eps, int size) {\n"
"    int row = get_global_id(0);\n"
"    int offset = row * size;\n"
"    float sum_sq = 0.0f;\n"
"    for(int i=0; i<size; i++) {\n"
"        float val = x[offset + i] + residual[offset + i];\n"
"        x[offset + i] = val;\n"
"        sum_sq += val * val;\n"
"    }\n"
"    float scale = rsqrt(sum_sq / size + eps);\n"
"    for(int i=0; i<size; i++) {\n"
"        out[offset + i] = x[offset + i] * scale * weight[i];\n"
"    }\n"
"}\n"
"__kernel void verify_tokens(__global const int* draft, __global const int* target, __global int* out_count, int len) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx == 0) {\n"
"        int matches = 0;\n"
"        for (int i = 0; i < len; i++) {\n"
"            if (draft[i] == target[i]) matches++;\n"
"            else break;\n"
"        }\n"
"        *out_count = matches;\n"
"    }\n"
"}\n"
"__kernel void fp32_to_bf16(__global const float* in, __global ushort* out) {\n"
"    int idx = get_global_id(0);\n"
"    uint f = as_uint(in[idx]);\n"
"    out[idx] = (ushort)(f >> 16);\n"
"}\n"
"__kernel void bf16_to_fp32(__global const ushort* in, __global float* out) {\n"
"    int idx = get_global_id(0);\n"
"    uint f = ((uint)in[idx]) << 16;\n"
"    out[idx] = as_float(f);\n"
"}\n";

// --- 3. Helper Functions ---

#define CHECK_CL(err, msg) if(err != CL_SUCCESS) { printf("[OpenCL Error] %s: %d\n", msg, err); return; }
#define LOAD_SYM(name) p_##name = (PTR_##name)GET_PROC(g_cl_lib, #name); if(!p_##name) { printf("Failed to load symbol %s\n", #name); return 0; }

int init_opencl() {
    if (g_ctx) return 1;

#ifdef _WIN32
    g_cl_lib = LOAD_LIB("OpenCL.dll");
#else
    g_cl_lib = LOAD_LIB("libOpenCL.so");
    if(!g_cl_lib) g_cl_lib = LOAD_LIB("libOpenCL.so.1");
#endif

    if (!g_cl_lib) { printf("Failed to load OpenCL library\n"); return 0; }

    LOAD_SYM(clGetPlatformIDs); LOAD_SYM(clGetPlatformInfo); LOAD_SYM(clGetDeviceIDs);
    LOAD_SYM(clCreateContext); LOAD_SYM(clCreateCommandQueue); LOAD_SYM(clCreateBuffer);
    LOAD_SYM(clReleaseMemObject); LOAD_SYM(clEnqueueWriteBuffer); LOAD_SYM(clEnqueueReadBuffer);
    LOAD_SYM(clCreateProgramWithSource); LOAD_SYM(clBuildProgram); LOAD_SYM(clCreateKernel);
    LOAD_SYM(clSetKernelArg); LOAD_SYM(clEnqueueNDRangeKernel); LOAD_SYM(clFinish);
    LOAD_SYM(clGetProgramBuildInfo); LOAD_SYM(clEnqueueCopyBuffer); LOAD_SYM(clGetProgramInfo);
    LOAD_SYM(clCreateProgramWithBinary); LOAD_SYM(clReleaseProgram);
    LOAD_SYM(clWaitForEvents); LOAD_SYM(clReleaseEvent); LOAD_SYM(clGetEventProfilingInfo);
    p_clEnqueueFillBuffer = (PTR_clEnqueueFillBuffer)GET_PROC(g_cl_lib, "clEnqueueFillBuffer");

    cl_uint num_platforms = 0;
    p_clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) return 0;

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    p_clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_platform_id selected_plat = platforms[0];
    char buffer[128];
    for (cl_uint i = 0; i < num_platforms; i++) {
        p_clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 128, buffer, NULL);
        if (strstr(buffer, VENDOR_FILTER) || strstr(buffer, "Advanced Micro Devices") || strstr(buffer, "Intel")) {
             selected_plat = platforms[i];
             break;
        }
    }
    free(platforms);

    cl_device_id device;
    cl_uint num_devices;
    if (p_clGetDeviceIDs(selected_plat, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) != CL_SUCCESS) {
        printf("[OpenCL] No GPU found.\n");
        return 0;
    }

    cl_int err;
    g_ctx = p_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    g_queue = p_clCreateCommandQueue(g_ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // Build Program (Cached)
    const char* options = "-cl-mad-enable -cl-fast-relaxed-math";
    char cache_path[512];
    const char* home = getenv("HOME");
    if (home) snprintf(cache_path, 512, "%s/.inference_pio/cache/kernel_opencl.bin", home);
    else snprintf(cache_path, 512, "/tmp/inference_pio_kernel_opencl.bin");

    FILE* f = fopen(cache_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        size_t bin_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        unsigned char* bin = (unsigned char*)malloc(bin_size);
        if (fread(bin, 1, bin_size, f) == bin_size) {
            cl_int bin_status;
            const size_t lengths[1] = { bin_size };
            const unsigned char* bins[1] = { bin };
            g_program = p_clCreateProgramWithBinary(g_ctx, 1, &device, lengths, bins, &bin_status, &err);
            if (err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                err = p_clBuildProgram(g_program, 1, &device, options, NULL, NULL);
                if (err != CL_SUCCESS) {
                    p_clReleaseProgram(g_program); g_program = NULL;
                }
            } else {
                if(g_program) p_clReleaseProgram(g_program); g_program = NULL;
            }
        }
        free(bin);
        fclose(f);
    }

    if (!g_program) {
        g_program = p_clCreateProgramWithSource(g_ctx, 1, &KERNEL_SOURCE, NULL, &err);
        err = p_clBuildProgram(g_program, 1, &device, options, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t len;
            p_clGetProgramBuildInfo(g_program, device, 0x1183, 0, NULL, &len);
            char* log = (char*)malloc(len);
            p_clGetProgramBuildInfo(g_program, device, 0x1183, len, log, NULL);
            printf("[OpenCL] Build Error:\n%s\n", log);
            free(log);
            return 0;
        }
        size_t bin_size;
        p_clGetProgramInfo(g_program, 0x1165, sizeof(size_t), &bin_size, NULL);
        if (bin_size > 0) {
            unsigned char* bin = (unsigned char*)malloc(bin_size);
            unsigned char* bins[1] = { bin };
            p_clGetProgramInfo(g_program, 0x1166, sizeof(unsigned char*), bins, NULL);
            f = fopen(cache_path, "wb");
            if (!f && home) {
                 snprintf(cache_path, 512, "/tmp/inference_pio_kernel_opencl.bin");
                 f = fopen(cache_path, "wb");
            }
            if (f) { fwrite(bin, 1, bin_size, f); fclose(f); }
            free(bin);
        }
    }

    // Create Kernels
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
    k_fused_attn = p_clCreateKernel(g_program, "fused_attention", &err);
    k_matmul_transposed = p_clCreateKernel(g_program, "matmul_transposed", &err);
    k_linear = p_clCreateKernel(g_program, "linear", &err);
    k_swiglu = p_clCreateKernel(g_program, "swiglu", &err);
    k_fused_gate_up_swiglu = p_clCreateKernel(g_program, "fused_gate_up_swiglu", &err);
    k_argmax = p_clCreateKernel(g_program, "argmax", &err);
    k_embed = p_clCreateKernel(g_program, "embed", &err);
    k_slice = p_clCreateKernel(g_program, "slice", &err);
    k_slice_device = p_clCreateKernel(g_program, "slice_device", &err);
    k_set_slice = p_clCreateKernel(g_program, "set_slice", &err);
    k_set_slice_device = p_clCreateKernel(g_program, "set_slice_device", &err);
    k_permute = p_clCreateKernel(g_program, "permute", &err);
    k_gemv = p_clCreateKernel(g_program, "gemv", &err);
    k_fused_add_mul = p_clCreateKernel(g_program, "fused_add_mul", &err);
    k_precompute_freqs = p_clCreateKernel(g_program, "precompute_freqs", &err);
    k_fused_split_rope = p_clCreateKernel(g_program, "fused_split_rope", &err);
    k_paged_attn = p_clCreateKernel(g_program, "paged_attention", &err);
    k_count_value = p_clCreateKernel(g_program, "count_value", &err);
    k_gather_by_value = p_clCreateKernel(g_program, "gather_by_value", &err);
    k_scatter_add_by_index = p_clCreateKernel(g_program, "scatter_add_by_index", &err);
    k_deltanet_recurrence = p_clCreateKernel(g_program, "deltanet_recurrence", &err);
    k_conv2d = p_clCreateKernel(g_program, "conv2d_naive", &err);
    k_dequantize = p_clCreateKernel(g_program, "dequantize", &err);
    k_fused_add_rms_norm = p_clCreateKernel(g_program, "fused_add_rms_norm", &err);
    k_matmul_image = p_clCreateKernel(g_program, "matmul_image", &err);
    k_verify_tokens = p_clCreateKernel(g_program, "verify_tokens", &err);
    k_fp32_to_bf16 = p_clCreateKernel(g_program, "fp32_to_bf16", &err);
    k_bf16_to_fp32 = p_clCreateKernel(g_program, "bf16_to_fp32", &err);
    k_matmul_fp16 = p_clCreateKernel(g_program, "matmul_fp16", &err);

    return 1;
}

// --- 4. Tensor Ops Implementation ---

typedef struct {
    float* data; // cl_mem
    int* shape;
    int ndim;
    int size;
    int device_id;
} Tensor;

EXPORT Tensor* create_tensor(int* shape, int ndim, int device_id) {
    if (!init_opencl()) return NULL;
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim; t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->size = 1; for (int i = 0; i < ndim; i++) t->size *= shape[i];
    t->device_id = device_id;

    cl_int err;
    cl_mem mem = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, t->size * sizeof(float), NULL, &err);
    t->data = (float*)mem;
    return t;
}

EXPORT Tensor* create_tensor_zerocopy(int* shape, int ndim, void* host_ptr) {
    if (!init_opencl()) return NULL;
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim; t->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->size = 1; for (int i = 0; i < ndim; i++) t->size *= shape[i];
    t->device_id = 0;

    cl_int err;
    cl_mem mem = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, t->size * sizeof(float), host_ptr, &err);
    if(err != CL_SUCCESS) {
        mem = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, t->size * sizeof(float), NULL, &err);
    }
    t->data = (float*)mem;
    return t;
}

EXPORT void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) p_clReleaseMemObject((cl_mem)t->data);
        free(t->shape); free(t);
    }
}

EXPORT void tensor_load_data(Tensor* t, float* buffer, int size) {
    if (!g_queue) return;
    p_clEnqueueWriteBuffer(g_queue, (cl_mem)t->data, CL_TRUE, 0, size * sizeof(float), buffer, 0, NULL, NULL);
}

EXPORT void tensor_get_data(Tensor* t, float* buffer, int size) {
    if (!g_queue) return;
    p_clEnqueueReadBuffer(g_queue, (cl_mem)t->data, CL_TRUE, 0, size * sizeof(float), buffer, 0, NULL, NULL);
}

EXPORT void tensor_fill(Tensor* t, float value) {
    if (!k_fill) return;
    p_clSetKernelArg(k_fill, 0, sizeof(cl_mem), &t->data);
    p_clSetKernelArg(k_fill, 1, sizeof(float), &value);
    size_t work_size = t->size;
    p_clEnqueueNDRangeKernel(g_queue, k_fill, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

EXPORT void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_add) return;
    p_clSetKernelArg(k_add, 0, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_add, 1, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_add, 2, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_add, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

EXPORT void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_mul) return;
    p_clSetKernelArg(k_mul, 0, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_mul, 1, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_mul, 2, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_mul, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

EXPORT void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_matmul) return;
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    p_clSetKernelArg(k_matmul, 0, sizeof(int), &M);
    p_clSetKernelArg(k_matmul, 1, sizeof(int), &N);
    p_clSetKernelArg(k_matmul, 2, sizeof(int), &K);
    p_clSetKernelArg(k_matmul, 3, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_matmul, 4, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_matmul, 5, sizeof(cl_mem), &out->data);

    if (M == 1 && k_gemv) {
        p_clSetKernelArg(k_gemv, 0, sizeof(int), &K);
        p_clSetKernelArg(k_gemv, 1, sizeof(int), &N);
        p_clSetKernelArg(k_gemv, 2, sizeof(cl_mem), &a->data);
        p_clSetKernelArg(k_gemv, 3, sizeof(cl_mem), &b->data);
        p_clSetKernelArg(k_gemv, 4, sizeof(cl_mem), &out->data);
        size_t work = N;
        p_clEnqueueNDRangeKernel(g_queue, k_gemv, 1, NULL, &work, NULL, 0, NULL, NULL);
    } else {
        size_t local_work[2] = {32, 32};
        size_t global_work[2] = {
            (size_t)ceil((double)N / 32.0) * 32,
            (size_t)ceil((double)M / 32.0) * 32
        };
        p_clEnqueueNDRangeKernel(g_queue, k_matmul, 2, NULL, global_work, local_work, 0, NULL, NULL);
    }
}

EXPORT void tensor_matmul_fp16(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_matmul_fp16) return;
    int M = a->shape[a->ndim-2];
    int K = a->shape[a->ndim-1];
    int N = b->shape[b->ndim-1];
    p_clSetKernelArg(k_matmul_fp16, 0, sizeof(int), &M);
    p_clSetKernelArg(k_matmul_fp16, 1, sizeof(int), &N);
    p_clSetKernelArg(k_matmul_fp16, 2, sizeof(int), &K);
    p_clSetKernelArg(k_matmul_fp16, 3, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_matmul_fp16, 4, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_matmul_fp16, 5, sizeof(cl_mem), &out->data);
    size_t global[2] = {N, M};
    p_clEnqueueNDRangeKernel(g_queue, k_matmul_fp16, 2, NULL, global, NULL, 0, NULL, NULL);
}

EXPORT void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) {
    if (!k_rms_norm) return;
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

EXPORT void tensor_silu(Tensor* input, Tensor* out) {
    if (!k_silu) return;
    p_clSetKernelArg(k_silu, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_silu, 1, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_silu, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

EXPORT void tensor_gelu(Tensor* input, Tensor* out) {
    if (!k_gelu) return;
    p_clSetKernelArg(k_gelu, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_gelu, 1, sizeof(cl_mem), &out->data);
    size_t work_size = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_gelu, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

EXPORT void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) {
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
    size_t work_size = total_tokens * (dim / 2);
    p_clEnqueueNDRangeKernel(g_queue, k_rope, 1, NULL, &work_size, NULL, 0, NULL, NULL);
}

EXPORT void tensor_softmax(Tensor* input, Tensor* out) {
    if (!k_softmax) return;
    int cols = input->shape[input->ndim - 1];
    int rows = input->size / cols;
    p_clSetKernelArg(k_softmax, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_softmax, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_softmax, 2, sizeof(int), &cols);
    size_t local_work = 256;
    if (cols < 256) local_work = 64;
    size_t global_work = rows * local_work;
    p_clEnqueueNDRangeKernel(g_queue, k_softmax, 1, NULL, &global_work, &local_work, 0, NULL, NULL);
}

EXPORT void tensor_topk(Tensor* input, int k, Tensor* out_val, Tensor* out_idx) {
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

EXPORT void tensor_scaled_dot_product_attention(Tensor* q, Tensor* k, Tensor* v, Tensor* out, float scale) {
    if (!k_fused_attn) return;
    int batch = q->shape[0];
    int heads = q->shape[1];
    int head_dim = q->shape[2];
    int seq_len = k->shape[1];
    int total_heads = batch * heads;
    p_clSetKernelArg(k_fused_attn, 0, sizeof(cl_mem), &q->data);
    p_clSetKernelArg(k_fused_attn, 1, sizeof(cl_mem), &k->data);
    p_clSetKernelArg(k_fused_attn, 2, sizeof(cl_mem), &v->data);
    p_clSetKernelArg(k_fused_attn, 3, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_fused_attn, 4, sizeof(float), &scale);
    p_clSetKernelArg(k_fused_attn, 5, sizeof(int), &seq_len);
    p_clSetKernelArg(k_fused_attn, 6, sizeof(int), &head_dim);
    p_clSetKernelArg(k_fused_attn, 7, sizeof(int), &total_heads);
    size_t local_work = head_dim;
    if (local_work > 256) local_work = 256;
    size_t global_work = total_heads * local_work;
    p_clEnqueueNDRangeKernel(g_queue, k_fused_attn, 1, NULL, &global_work, &local_work, 0, NULL, NULL);
}

EXPORT void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_matmul_transposed) return;
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 2];
    p_clSetKernelArg(k_matmul_transposed, 0, sizeof(int), &M);
    p_clSetKernelArg(k_matmul_transposed, 1, sizeof(int), &N);
    p_clSetKernelArg(k_matmul_transposed, 2, sizeof(int), &K);
    p_clSetKernelArg(k_matmul_transposed, 3, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_matmul_transposed, 4, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_matmul_transposed, 5, sizeof(cl_mem), &out->data);
    size_t global_work[2] = {N, M};
    p_clEnqueueNDRangeKernel(g_queue, k_matmul_transposed, 2, NULL, global_work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) {
    if (!k_linear) return;
    int K = input->shape[input->ndim - 1];
    int rows = input->size / K;
    int N = weight->shape[0];
    p_clSetKernelArg(k_linear, 0, sizeof(int), &rows);
    p_clSetKernelArg(k_linear, 1, sizeof(int), &K);
    p_clSetKernelArg(k_linear, 2, sizeof(int), &N);
    p_clSetKernelArg(k_linear, 3, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_linear, 4, sizeof(cl_mem), &weight->data);
    cl_mem bias_mem = bias ? (cl_mem)bias->data : NULL;
    p_clSetKernelArg(k_linear, 5, sizeof(cl_mem), &bias_mem);
    p_clSetKernelArg(k_linear, 6, sizeof(cl_mem), &out->data);
    size_t global_work[2] = {N, rows};
    p_clEnqueueNDRangeKernel(g_queue, k_linear, 2, NULL, global_work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_swiglu(Tensor* gate, Tensor* up, Tensor* out) {
    if (!k_swiglu) return;
    p_clSetKernelArg(k_swiglu, 0, sizeof(cl_mem), &gate->data);
    p_clSetKernelArg(k_swiglu, 1, sizeof(cl_mem), &up->data);
    p_clSetKernelArg(k_swiglu, 2, sizeof(cl_mem), &out->data);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_swiglu, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_fused_gate_up_swiglu(Tensor* gate_up, Tensor* out) {
    if (!k_fused_gate_up_swiglu) return;
    int hidden = out->shape[out->ndim - 1];
    p_clSetKernelArg(k_fused_gate_up_swiglu, 0, sizeof(cl_mem), &gate_up->data);
    p_clSetKernelArg(k_fused_gate_up_swiglu, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_fused_gate_up_swiglu, 2, sizeof(int), &hidden);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_fused_gate_up_swiglu, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_argmax(Tensor* input, Tensor* out) {
    if (!k_argmax) return;
    int cols = input->shape[input->ndim - 1];
    int rows = input->size / cols;
    p_clSetKernelArg(k_argmax, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_argmax, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_argmax, 2, sizeof(int), &cols);
    size_t work = rows;
    p_clEnqueueNDRangeKernel(g_queue, k_argmax, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_embed(Tensor* weight, Tensor* indices, Tensor* out) {
    if (!k_embed) return;
    int hidden = weight->shape[1];
    p_clSetKernelArg(k_embed, 0, sizeof(cl_mem), &weight->data);
    p_clSetKernelArg(k_embed, 1, sizeof(cl_mem), &indices->data);
    p_clSetKernelArg(k_embed, 2, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_embed, 3, sizeof(int), &hidden);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_embed, 1, NULL, &work, NULL, 0, NULL, NULL);
}

void compute_strides(int* shape, int ndim, int* strides) {
    strides[ndim-1] = 1;
    for(int i=ndim-2; i>=0; i--) strides[i] = strides[i+1] * shape[i+1];
}

EXPORT void tensor_slice(Tensor* input, Tensor* out, int* start_indices, int* slice_shapes) {
    if (!k_slice) return;
    int ndim = input->ndim;
    int h_in[8], h_out[8]; // Max ndim 8
    compute_strides(input->shape, ndim, h_in);
    compute_strides(out->shape, ndim, h_out);
    cl_int err;
    cl_mem d_start = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, start_indices, &err);
    cl_mem d_hin = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_in, &err);
    cl_mem d_hout = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_out, &err);
    p_clSetKernelArg(k_slice, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_slice, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_slice, 2, sizeof(int), &ndim);
    p_clSetKernelArg(k_slice, 3, sizeof(cl_mem), &d_start);
    p_clSetKernelArg(k_slice, 4, sizeof(cl_mem), &d_hin);
    p_clSetKernelArg(k_slice, 5, sizeof(cl_mem), &d_hout);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_slice, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clFinish(g_queue);
    p_clReleaseMemObject(d_start);
    p_clReleaseMemObject(d_hin);
    p_clReleaseMemObject(d_hout);
}

EXPORT void tensor_slice_device(Tensor* input, Tensor* out, Tensor* start_indices) {
    if (!k_slice_device) return;
    int ndim = input->ndim;
    int h_in[8], h_out[8];
    compute_strides(input->shape, ndim, h_in);
    compute_strides(out->shape, ndim, h_out);
    cl_int err;
    cl_mem d_hin = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_in, &err);
    cl_mem d_hout = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_out, &err);
    p_clSetKernelArg(k_slice_device, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_slice_device, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_slice_device, 2, sizeof(int), &ndim);
    p_clSetKernelArg(k_slice_device, 3, sizeof(cl_mem), &start_indices->data);
    p_clSetKernelArg(k_slice_device, 4, sizeof(cl_mem), &d_hin);
    p_clSetKernelArg(k_slice_device, 5, sizeof(cl_mem), &d_hout);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_slice_device, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clFinish(g_queue);
    p_clReleaseMemObject(d_hin);
    p_clReleaseMemObject(d_hout);
}

EXPORT void tensor_set_slice(Tensor* dst, Tensor* src, int* start_indices) {
    if (!k_set_slice) return;
    int ndim = dst->ndim;
    int h_dst[8], h_src[8];
    compute_strides(dst->shape, ndim, h_dst);
    compute_strides(src->shape, ndim, h_src);
    cl_int err;
    cl_mem d_start = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, start_indices, &err);
    cl_mem d_hdst = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_dst, &err);
    cl_mem d_hsrc = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_src, &err);
    p_clSetKernelArg(k_set_slice, 0, sizeof(cl_mem), &dst->data);
    p_clSetKernelArg(k_set_slice, 1, sizeof(cl_mem), &src->data);
    p_clSetKernelArg(k_set_slice, 2, sizeof(int), &ndim);
    p_clSetKernelArg(k_set_slice, 3, sizeof(cl_mem), &d_start);
    p_clSetKernelArg(k_set_slice, 4, sizeof(cl_mem), &d_hdst);
    p_clSetKernelArg(k_set_slice, 5, sizeof(cl_mem), &d_hsrc);
    size_t work = src->size;
    p_clEnqueueNDRangeKernel(g_queue, k_set_slice, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clFinish(g_queue);
    p_clReleaseMemObject(d_start);
    p_clReleaseMemObject(d_hdst);
    p_clReleaseMemObject(d_hsrc);
}

EXPORT void tensor_set_slice_device(Tensor* dst, Tensor* src, Tensor* start_indices) {
    if (!k_set_slice_device) return;
    int ndim = dst->ndim;
    int h_dst[8], h_src[8];
    compute_strides(dst->shape, ndim, h_dst);
    compute_strides(src->shape, ndim, h_src);
    cl_int err;
    cl_mem d_hdst = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_dst, &err);
    cl_mem d_hsrc = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_src, &err);
    p_clSetKernelArg(k_set_slice_device, 0, sizeof(cl_mem), &dst->data);
    p_clSetKernelArg(k_set_slice_device, 1, sizeof(cl_mem), &src->data);
    p_clSetKernelArg(k_set_slice_device, 2, sizeof(int), &ndim);
    p_clSetKernelArg(k_set_slice_device, 3, sizeof(cl_mem), &start_indices->data);
    p_clSetKernelArg(k_set_slice_device, 4, sizeof(cl_mem), &d_hdst);
    p_clSetKernelArg(k_set_slice_device, 5, sizeof(cl_mem), &d_hsrc);
    size_t work = src->size;
    p_clEnqueueNDRangeKernel(g_queue, k_set_slice_device, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clFinish(g_queue);
    p_clReleaseMemObject(d_hdst);
    p_clReleaseMemObject(d_hsrc);
}

EXPORT void tensor_permute(Tensor* input, Tensor* out, int* dims) {
    if (!k_permute) return;
    int ndim = input->ndim;
    int h_in[8], h_out[8];
    compute_strides(input->shape, ndim, h_in);
    compute_strides(out->shape, ndim, h_out);
    cl_int err;
    cl_mem d_dims = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, dims, &err);
    cl_mem d_hin = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_in, &err);
    cl_mem d_hout = p_clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ndim*4, h_out, &err);
    p_clSetKernelArg(k_permute, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_permute, 1, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_permute, 2, sizeof(int), &ndim);
    p_clSetKernelArg(k_permute, 3, sizeof(cl_mem), &d_dims);
    p_clSetKernelArg(k_permute, 4, sizeof(cl_mem), &d_hin);
    p_clSetKernelArg(k_permute, 5, sizeof(cl_mem), &d_hout);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_permute, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clFinish(g_queue);
    p_clReleaseMemObject(d_dims);
    p_clReleaseMemObject(d_hin);
    p_clReleaseMemObject(d_hout);
}

EXPORT void tensor_reshape(Tensor* input, Tensor* out) {
    if (!g_queue) return;
    size_t size = input->size * sizeof(float);
    p_clEnqueueCopyBuffer(g_queue, (cl_mem)input->data, (cl_mem)out->data, 0, 0, size, 0, NULL, NULL);
}

EXPORT void tensor_precompute_freqs_cis(int dim, int end, float theta, Tensor* out_cos, Tensor* out_sin) {
    if (!k_precompute_freqs) return;
    int half = dim / 2;
    p_clSetKernelArg(k_precompute_freqs, 0, sizeof(cl_mem), &out_cos->data);
    p_clSetKernelArg(k_precompute_freqs, 1, sizeof(cl_mem), &out_sin->data);
    p_clSetKernelArg(k_precompute_freqs, 2, sizeof(int), &end);
    p_clSetKernelArg(k_precompute_freqs, 3, sizeof(int), &half);
    p_clSetKernelArg(k_precompute_freqs, 4, sizeof(float), &theta);
    size_t work = end * half;
    p_clEnqueueNDRangeKernel(g_queue, k_precompute_freqs, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_fused_split_rope(Tensor* qkv, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k, Tensor* out_v) {
    if (!k_fused_split_rope) return;
    int heads = out_q->shape[out_q->ndim - 2];
    int head_dim = out_q->shape[out_q->ndim - 1];
    int total_tokens = out_q->size / (heads * head_dim);
    p_clSetKernelArg(k_fused_split_rope, 0, sizeof(cl_mem), &qkv->data);
    p_clSetKernelArg(k_fused_split_rope, 1, sizeof(cl_mem), &cos->data);
    p_clSetKernelArg(k_fused_split_rope, 2, sizeof(cl_mem), &sin->data);
    p_clSetKernelArg(k_fused_split_rope, 3, sizeof(cl_mem), &out_q->data);
    p_clSetKernelArg(k_fused_split_rope, 4, sizeof(cl_mem), &out_k->data);
    p_clSetKernelArg(k_fused_split_rope, 5, sizeof(cl_mem), &out_v->data);
    p_clSetKernelArg(k_fused_split_rope, 6, sizeof(int), &heads);
    p_clSetKernelArg(k_fused_split_rope, 7, sizeof(int), &head_dim);
    p_clSetKernelArg(k_fused_split_rope, 8, sizeof(int), &total_tokens);
    size_t work = total_tokens * heads * head_dim;
    p_clEnqueueNDRangeKernel(g_queue, k_fused_split_rope, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_paged_attention(Tensor* q, Tensor* k_cache, Tensor* v_cache,
                            Tensor* block_tables, Tensor* context_lens,
                            Tensor* out, float scale, int page_size, int max_blocks, int head_dim) {
    if (!k_paged_attn) return;
    int batch = q->shape[0];
    int heads = q->shape[1];
    p_clSetKernelArg(k_paged_attn, 0, sizeof(cl_mem), &q->data);
    p_clSetKernelArg(k_paged_attn, 1, sizeof(cl_mem), &k_cache->data);
    p_clSetKernelArg(k_paged_attn, 2, sizeof(cl_mem), &v_cache->data);
    p_clSetKernelArg(k_paged_attn, 3, sizeof(cl_mem), &block_tables->data);
    p_clSetKernelArg(k_paged_attn, 4, sizeof(cl_mem), &context_lens->data);
    p_clSetKernelArg(k_paged_attn, 5, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_paged_attn, 6, sizeof(float), &scale);
    p_clSetKernelArg(k_paged_attn, 7, sizeof(int), &page_size);
    p_clSetKernelArg(k_paged_attn, 8, sizeof(int), &max_blocks);
    p_clSetKernelArg(k_paged_attn, 9, sizeof(int), &head_dim);
    size_t local_sz = head_dim;
    if (local_sz > 256) local_sz = 256;
    size_t g_work[2] = { batch * local_sz, heads };
    size_t l_work[2] = { local_sz, 1 };
    p_clEnqueueNDRangeKernel(g_queue, k_paged_attn, 2, NULL, g_work, l_work, 0, NULL, NULL);
}

EXPORT void tensor_count_value(Tensor* t, float value, int* count) {
    if (!k_count_value) return;
    cl_int err;
    cl_mem d_count = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    int zero = 0;
    p_clEnqueueWriteBuffer(g_queue, d_count, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
    p_clSetKernelArg(k_count_value, 0, sizeof(cl_mem), &t->data);
    p_clSetKernelArg(k_count_value, 1, sizeof(int), &t->size);
    p_clSetKernelArg(k_count_value, 2, sizeof(float), &value);
    p_clSetKernelArg(k_count_value, 3, sizeof(cl_mem), &d_count);
    size_t work = t->size;
    p_clEnqueueNDRangeKernel(g_queue, k_count_value, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clEnqueueReadBuffer(g_queue, d_count, CL_TRUE, 0, sizeof(int), count, 0, NULL, NULL);
    p_clReleaseMemObject(d_count);
}

EXPORT void tensor_gather_by_value(Tensor* input, Tensor* indices, float value, Tensor* out_data, Tensor* out_indices) {
    if (!k_gather_by_value) return;
    cl_int err;
    cl_mem d_counter = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    int zero = 0;
    p_clEnqueueWriteBuffer(g_queue, d_counter, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
    int hidden_size = input->shape[input->ndim - 1];
    int size = indices->size;
    p_clSetKernelArg(k_gather_by_value, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_gather_by_value, 1, sizeof(cl_mem), &indices->data);
    p_clSetKernelArg(k_gather_by_value, 2, sizeof(int), &size);
    p_clSetKernelArg(k_gather_by_value, 3, sizeof(float), &value);
    p_clSetKernelArg(k_gather_by_value, 4, sizeof(cl_mem), &out_data->data);
    p_clSetKernelArg(k_gather_by_value, 5, sizeof(cl_mem), &out_indices->data);
    p_clSetKernelArg(k_gather_by_value, 6, sizeof(int), &hidden_size);
    p_clSetKernelArg(k_gather_by_value, 7, sizeof(cl_mem), &d_counter);
    size_t work = size;
    p_clEnqueueNDRangeKernel(g_queue, k_gather_by_value, 1, NULL, &work, NULL, 0, NULL, NULL);
    p_clReleaseMemObject(d_counter);
}

EXPORT void tensor_scatter_add_by_index(Tensor* out, Tensor* src, Tensor* indices) {
    if (!k_scatter_add_by_index) return;
    int count = indices->size;
    int hidden_size = src->shape[src->ndim - 1];
    int total_rows = out->size / hidden_size;
    p_clSetKernelArg(k_scatter_add_by_index, 0, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_scatter_add_by_index, 1, sizeof(cl_mem), &src->data);
    p_clSetKernelArg(k_scatter_add_by_index, 2, sizeof(cl_mem), &indices->data);
    p_clSetKernelArg(k_scatter_add_by_index, 3, sizeof(int), &count);
    p_clSetKernelArg(k_scatter_add_by_index, 4, sizeof(int), &hidden_size);
    p_clSetKernelArg(k_scatter_add_by_index, 5, sizeof(int), &total_rows);
    size_t work = count * hidden_size;
    p_clEnqueueNDRangeKernel(g_queue, k_scatter_add_by_index, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding, int groups) {
    if (!k_conv2d) return;
    int N = input->shape[0];
    int C_in = input->shape[1];
    int H_in = input->shape[2];
    int W_in = input->shape[3];
    int C_out = weight->shape[0];
    int KH = weight->shape[2];
    int KW = weight->shape[3];
    int H_out = out->shape[2];
    int W_out = out->shape[3];
    p_clSetKernelArg(k_conv2d, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_conv2d, 1, sizeof(cl_mem), &weight->data);
    cl_mem b_mem = bias ? (cl_mem)bias->data : NULL;
    p_clSetKernelArg(k_conv2d, 2, sizeof(cl_mem), &b_mem);
    p_clSetKernelArg(k_conv2d, 3, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_conv2d, 4, sizeof(int), &C_in);
    p_clSetKernelArg(k_conv2d, 5, sizeof(int), &H_in);
    p_clSetKernelArg(k_conv2d, 6, sizeof(int), &W_in);
    p_clSetKernelArg(k_conv2d, 7, sizeof(int), &C_out);
    p_clSetKernelArg(k_conv2d, 8, sizeof(int), &H_out);
    p_clSetKernelArg(k_conv2d, 9, sizeof(int), &W_out);
    p_clSetKernelArg(k_conv2d, 10, sizeof(int), &KH);
    p_clSetKernelArg(k_conv2d, 11, sizeof(int), &KW);
    p_clSetKernelArg(k_conv2d, 12, sizeof(int), &stride);
    p_clSetKernelArg(k_conv2d, 13, sizeof(int), &padding);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_conv2d, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_dequantize(Tensor* input, Tensor* scale, Tensor* out) {
    if (!k_dequantize) return;
    int hidden = 1;
    p_clSetKernelArg(k_dequantize, 0, sizeof(cl_mem), &input->data);
    p_clSetKernelArg(k_dequantize, 1, sizeof(cl_mem), &scale->data);
    p_clSetKernelArg(k_dequantize, 2, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_dequantize, 3, sizeof(int), &hidden);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_dequantize, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_deltanet_recurrence(Tensor* q, Tensor* k, Tensor* v, Tensor* beta, Tensor* state, Tensor* out) {
    if (!k_deltanet_recurrence) return;
    int B = q->shape[0];
    int S = q->shape[1];
    int H = q->shape[2];
    int D = q->shape[3];
    p_clSetKernelArg(k_deltanet_recurrence, 0, sizeof(cl_mem), &q->data);
    p_clSetKernelArg(k_deltanet_recurrence, 1, sizeof(cl_mem), &k->data);
    p_clSetKernelArg(k_deltanet_recurrence, 2, sizeof(cl_mem), &v->data);
    p_clSetKernelArg(k_deltanet_recurrence, 3, sizeof(cl_mem), &beta->data);
    p_clSetKernelArg(k_deltanet_recurrence, 4, sizeof(cl_mem), &state->data);
    p_clSetKernelArg(k_deltanet_recurrence, 5, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_deltanet_recurrence, 6, sizeof(int), &B);
    p_clSetKernelArg(k_deltanet_recurrence, 7, sizeof(int), &S);
    p_clSetKernelArg(k_deltanet_recurrence, 8, sizeof(int), &H);
    p_clSetKernelArg(k_deltanet_recurrence, 9, sizeof(int), &D);
    size_t local_work[2] = { 256, 1 };
    size_t global_wk[2] = { (size_t)B * 256, (size_t)H };
    p_clEnqueueNDRangeKernel(g_queue, k_deltanet_recurrence, 2, NULL, global_wk, local_work, 0, NULL, NULL);
}

EXPORT void tensor_fused_add_mul(Tensor* a, Tensor* b, Tensor* c, Tensor* out) {
    if (!k_fused_add_mul) return;
    p_clSetKernelArg(k_fused_add_mul, 0, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_fused_add_mul, 1, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_fused_add_mul, 2, sizeof(cl_mem), &c->data);
    p_clSetKernelArg(k_fused_add_mul, 3, sizeof(cl_mem), &out->data);
    size_t work = out->size;
    p_clEnqueueNDRangeKernel(g_queue, k_fused_add_mul, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_fused_add_rms_norm(Tensor* x, Tensor* residual, Tensor* weight, Tensor* out, float eps) {
    if (!k_fused_add_rms_norm) return;
    int hidden = x->shape[x->ndim - 1];
    int rows = x->size / hidden;
    p_clSetKernelArg(k_fused_add_rms_norm, 0, sizeof(cl_mem), &x->data);
    p_clSetKernelArg(k_fused_add_rms_norm, 1, sizeof(cl_mem), &residual->data);
    p_clSetKernelArg(k_fused_add_rms_norm, 2, sizeof(cl_mem), &weight->data);
    p_clSetKernelArg(k_fused_add_rms_norm, 3, sizeof(cl_mem), &out->data);
    p_clSetKernelArg(k_fused_add_rms_norm, 4, sizeof(float), &eps);
    p_clSetKernelArg(k_fused_add_rms_norm, 5, sizeof(int), &hidden);
    size_t work = rows;
    p_clEnqueueNDRangeKernel(g_queue, k_fused_add_rms_norm, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_verify_tokens(Tensor* draft, Tensor* target, Tensor* out_count) {
    if (!k_verify_tokens) return;
    int len = draft->size;
    p_clSetKernelArg(k_verify_tokens, 0, sizeof(cl_mem), &draft->data);
    p_clSetKernelArg(k_verify_tokens, 1, sizeof(cl_mem), &target->data);
    p_clSetKernelArg(k_verify_tokens, 2, sizeof(cl_mem), &out_count->data);
    p_clSetKernelArg(k_verify_tokens, 3, sizeof(int), &len);
    size_t work = 1;
    p_clEnqueueNDRangeKernel(g_queue, k_verify_tokens, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_convert_fp32_bf16(Tensor* fp32, Tensor* bf16) {
    if (!k_fp32_to_bf16) return;
    p_clSetKernelArg(k_fp32_to_bf16, 0, sizeof(cl_mem), &fp32->data);
    p_clSetKernelArg(k_fp32_to_bf16, 1, sizeof(cl_mem), &bf16->data);
    size_t work = fp32->size;
    p_clEnqueueNDRangeKernel(g_queue, k_fp32_to_bf16, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_convert_bf16_fp32(Tensor* bf16, Tensor* fp32) {
    if (!k_bf16_to_fp32) return;
    p_clSetKernelArg(k_bf16_to_fp32, 0, sizeof(cl_mem), &bf16->data);
    p_clSetKernelArg(k_bf16_to_fp32, 1, sizeof(cl_mem), &fp32->data);
    size_t work = fp32->size;
    p_clEnqueueNDRangeKernel(g_queue, k_bf16_to_fp32, 1, NULL, &work, NULL, 0, NULL, NULL);
}

EXPORT void tensor_copy_p2p(Tensor* src, Tensor* dst) {
    if (!g_queue) return;
    size_t size = src->size;
    if ((size_t)dst->size < size) size = (size_t)dst->size;
    p_clEnqueueCopyBuffer(g_queue, (cl_mem)src->data, (cl_mem)dst->data, 0, 0, size * sizeof(float), 0, NULL, NULL);
}

EXPORT void tensor_copy_offset(Tensor* src, int src_offset, Tensor* dst, int dst_offset, int count) {
    if (!g_queue) return;
    p_clEnqueueCopyBuffer(g_queue, (cl_mem)src->data, (cl_mem)dst->data,
                          src_offset * sizeof(float), dst_offset * sizeof(float),
                          count * sizeof(float), 0, NULL, NULL);
}

// 39. Texture MatMul (Optimization)
EXPORT void tensor_matmul_image(Tensor* a, Tensor* b, Tensor* out) {
    if (!k_matmul_image) return;
    // B is assumed to be image2d_t (handled by higher level allocator/converter?)
    // For now, assume b->data IS a cl_mem image handle created elsewhere.
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    p_clSetKernelArg(k_matmul_image, 0, sizeof(int), &M);
    p_clSetKernelArg(k_matmul_image, 1, sizeof(int), &N);
    p_clSetKernelArg(k_matmul_image, 2, sizeof(int), &K);
    p_clSetKernelArg(k_matmul_image, 3, sizeof(cl_mem), &a->data);
    p_clSetKernelArg(k_matmul_image, 4, sizeof(cl_mem), &b->data);
    p_clSetKernelArg(k_matmul_image, 5, sizeof(cl_mem), &out->data);

    size_t global[2] = {N, M};
    p_clEnqueueNDRangeKernel(g_queue, k_matmul_image, 2, NULL, global, NULL, 0, NULL, NULL);
}

EXPORT void tensor_benchmark_matmul(int M, int N, int K, int trials, double* out_time_ms) {
    if (!g_queue || !k_matmul) return;
    cl_int err;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    cl_mem d_A = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, size_A, NULL, &err);
    cl_mem d_B = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, size_B, NULL, &err);
    cl_mem d_C = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, size_C, NULL, &err);
    if (err != CL_SUCCESS) {
        if(d_A) p_clReleaseMemObject(d_A);
        if(d_B) p_clReleaseMemObject(d_B);
        if(d_C) p_clReleaseMemObject(d_C);
        return;
    }
    float zero = 0.0f;
    if (k_fill) {
        p_clSetKernelArg(k_fill, 0, sizeof(cl_mem), &d_C);
        p_clSetKernelArg(k_fill, 1, sizeof(float), &zero);
        size_t fill_work = M * N;
        p_clEnqueueNDRangeKernel(g_queue, k_fill, 1, NULL, &fill_work, NULL, 0, NULL, NULL);
    }
    for(int i=0; i<3; i++) {
        p_clSetKernelArg(k_matmul, 0, sizeof(int), &M);
        p_clSetKernelArg(k_matmul, 1, sizeof(int), &N);
        p_clSetKernelArg(k_matmul, 2, sizeof(int), &K);
        p_clSetKernelArg(k_matmul, 3, sizeof(cl_mem), &d_A);
        p_clSetKernelArg(k_matmul, 4, sizeof(cl_mem), &d_B);
        p_clSetKernelArg(k_matmul, 5, sizeof(cl_mem), &d_C);
        size_t local_work[2] = {32, 32};
        size_t global_work[2] = {
            (size_t)ceil((double)N / 32.0) * 32,
            (size_t)ceil((double)M / 32.0) * 32
        };
        p_clEnqueueNDRangeKernel(g_queue, k_matmul, 2, NULL, global_work, local_work, 0, NULL, NULL);
    }
    p_clFinish(g_queue);
    double total_ms = 0.0;
    for(int i=0; i<trials; i++) {
        cl_event event;
        p_clSetKernelArg(k_matmul, 0, sizeof(int), &M);
        p_clSetKernelArg(k_matmul, 1, sizeof(int), &N);
        p_clSetKernelArg(k_matmul, 2, sizeof(int), &K);
        p_clSetKernelArg(k_matmul, 3, sizeof(cl_mem), &d_A);
        p_clSetKernelArg(k_matmul, 4, sizeof(cl_mem), &d_B);
        p_clSetKernelArg(k_matmul, 5, sizeof(cl_mem), &d_C);
        size_t local_work[2] = {32, 32};
        size_t global_work[2] = {
            (size_t)ceil((double)N / 32.0) * 32,
            (size_t)ceil((double)M / 32.0) * 32
        };
        p_clEnqueueNDRangeKernel(g_queue, k_matmul, 2, NULL, global_work, local_work, 0, NULL, &event);
        p_clWaitForEvents(1, &event);
        cl_ulong start = 0, end = 0;
        p_clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        p_clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        double ms = (double)(end - start) * 1e-6;
        total_ms += ms;
        p_clReleaseEvent(event);
    }
    *out_time_ms = total_ms / trials;
    p_clReleaseMemObject(d_A);
    p_clReleaseMemObject(d_B);
    p_clReleaseMemObject(d_C);
}

EXPORT void tensor_set_tuning_param(int param_id, int value) {
    // Placeholder for future tuning logic
    (void)param_id; (void)value;
}

EXPORT void tensor_cat(Tensor** tensors, int num_tensors, int axis, Tensor* out) {
    if (!g_queue) return;
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
            p_clEnqueueCopyBuffer(g_queue, (cl_mem)t->data, (cl_mem)out->data,
                                  src_offset * sizeof(float), dst_offset * sizeof(float),
                                  size * sizeof(float), 0, NULL, NULL);
        }
        offset_accum += dim;
    }
}
