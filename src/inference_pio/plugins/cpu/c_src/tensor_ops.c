#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <mm_malloc.h>
#define aligned_alloc(alignment, size) _mm_malloc(size, alignment)
#define aligned_free(ptr) _mm_free(ptr)
#endif
#include "../../common/tensor.h"

// Helper for tensor strides
void get_strides(const int* shape, int ndim, int* strides) {
    strides[ndim-1] = 1;
    for(int i=ndim-2; i>=0; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }
}

// Memory Management
Tensor* create_tensor(int* shape, int ndim, int device_id) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(sizeof(int) * ndim);
    t->size = 1;
    for(int i=0; i<ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    t->device_id = device_id;
    // Align to 32 bytes for AVX
    t->data = (float*)aligned_alloc(32, sizeof(float) * t->size);
    return t;
}

void free_tensor(Tensor* t) {
    if(t) {
        if(t->data) aligned_free(t->data);
        if(t->shape) free(t->shape);
        free(t);
    }
}

void tensor_fill(Tensor* t, float value) {
    #pragma omp parallel for simd
    for(int i=0; i<t->size; i++) t->data[i] = value;
}

void tensor_load_data(Tensor* t, float* buffer, int size) {
    if(size != t->size) { printf("Size mismatch load\n"); return; }
    memcpy(t->data, buffer, size * sizeof(float));
}

void tensor_get_data(Tensor* t, float* buffer, int size) {
    if(size != t->size) { printf("Size mismatch get\n"); return; }
    memcpy(buffer, t->data, size * sizeof(float));
}

// Math Ops
void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    #pragma omp parallel for simd
    for(int i=0; i<out->size; i++) out->data[i] = a->data[i] + b->data[i];
}

void tensor_mul(Tensor* a, Tensor* b, Tensor* out) {
    #pragma omp parallel for simd
    for(int i=0; i<out->size; i++) out->data[i] = a->data[i] * b->data[i];
}

void tensor_scale(Tensor* a, float scale, Tensor* out) {
    #pragma omp parallel for simd
    for(int i=0; i<out->size; i++) out->data[i] = a->data[i] * scale;
}

// Matmul: Simple O(N^3) with OpenMP tiling for cache
void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    // Assume 2D or Batched 2D (ND)
    // Flatten batch dims
    int m = a->shape[a->ndim-2];
    int k = a->shape[a->ndim-1];
    int n = b->shape[b->ndim-1];
    int batch_size = out->size / (m*n);

    #pragma omp parallel for
    for(int batch=0; batch<batch_size; batch++) {
        float* A = a->data + batch * m * k;
        float* B = b->data + batch * k * n;
        float* C = out->data + batch * m * n;

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                float sum = 0.0f;
                // Vectorizable loop
                #pragma omp simd reduction(+:sum)
                for(int p=0; p<k; p++) {
                    sum += A[i*k + p] * B[p*n + j];
                }
                C[i*n + j] = sum;
            }
        }
    }
}

void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out) {
    // B is [..., N, K] instead of [..., K, N]
    // A is [..., M, K]
    // Out is [..., M, N]
    int m = a->shape[a->ndim-2];
    int k = a->shape[a->ndim-1];
    int n = b->shape[b->ndim-2]; // Transposed B: [N, K]
    int batch_size = out->size / (m*n);

    #pragma omp parallel for
    for(int batch=0; batch<batch_size; batch++) {
        float* A = a->data + batch * m * k;
        float* B = b->data + batch * n * k; // B is stored as [N, K]
        float* C = out->data + batch * m * n;

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for(int p=0; p<k; p++) {
                    sum += A[i*k + p] * B[j*k + p]; // Contiguous access for B!
                }
                C[i*n + j] = sum;
            }
        }
    }
}

void tensor_linear(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out) {
    // Input: [Batch, In], Weight: [Out, In], Bias: [Out]
    // Out = Input * Weight^T + Bias
    // This is essentially matmul_transposed(input, weight) + bias

    tensor_matmul_transposed(input, weight, out);

    if(bias) {
        int last_dim = out->shape[out->ndim-1];
        int batch = out->size / last_dim;
        #pragma omp parallel for
        for(int i=0; i<batch; i++) {
            for(int j=0; j<last_dim; j++) {
                out->data[i*last_dim + j] += bias->data[j];
            }
        }
    }
}

// Activations
void tensor_silu(Tensor* input, Tensor* out) {
    #pragma omp parallel for simd
    for(int i=0; i<input->size; i++) {
        float x = input->data[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        out->data[i] = x * sigmoid;
    }
}

void tensor_gelu(Tensor* input, Tensor* out) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    #pragma omp parallel for simd
    for(int i=0; i<input->size; i++) {
        float x = input->data[i];
        float cdf = 0.5f * (1.0f + tanhf(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
        out->data[i] = x * cdf;
    }
}

void tensor_softmax(Tensor* input, Tensor* out) {
    int last_dim = input->shape[input->ndim-1];
    int batch = input->size / last_dim;

    #pragma omp parallel for
    for(int i=0; i<batch; i++) {
        float* in_ptr = input->data + i*last_dim;
        float* out_ptr = out->data + i*last_dim;

        float max_val = -1e9f;
        for(int j=0; j<last_dim; j++) if(in_ptr[j] > max_val) max_val = in_ptr[j];

        float sum = 0.0f;
        for(int j=0; j<last_dim; j++) {
            out_ptr[j] = expf(in_ptr[j] - max_val);
            sum += out_ptr[j];
        }

        float inv_sum = 1.0f / sum;
        for(int j=0; j<last_dim; j++) out_ptr[j] *= inv_sum;
    }
}

void tensor_rms_norm(Tensor* input, Tensor* weight, Tensor* out, float eps) {
    int last_dim = input->shape[input->ndim-1];
    int batch = input->size / last_dim;

    #pragma omp parallel for
    for(int i=0; i<batch; i++) {
        float* in_ptr = input->data + i*last_dim;
        float* out_ptr = out->data + i*last_dim;

        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for(int j=0; j<last_dim; j++) sum_sq += in_ptr[j] * in_ptr[j];

        float scale = 1.0f / sqrtf(sum_sq / last_dim + eps);

        #pragma omp simd
        for(int j=0; j<last_dim; j++) {
            out_ptr[j] = in_ptr[j] * scale * weight->data[j];
        }
    }
}

// RoPE
void tensor_rope(Tensor* q, Tensor* k, Tensor* cos, Tensor* sin, Tensor* out_q, Tensor* out_k) {
    // Shapes: [Batch, Seq, Heads, HeadDim]
    // Cos/Sin: [Seq, HeadDim/2] (Precomputed)
    // We assume Heads is dim -2, HeadDim is dim -1.
    // We iterate over Batch and Heads.

    int head_dim = q->shape[q->ndim-1];
    int heads = q->shape[q->ndim-2];
    int seq_len = q->shape[q->ndim-3];
    int batch = q->size / (seq_len * heads * head_dim);
    int half_dim = head_dim / 2;

    #pragma omp parallel for collapse(3)
    for(int b=0; b<batch; b++) {
        for(int s=0; s<seq_len; s++) {
            for(int h=0; h<heads; h++) {
                float* q_ptr = q->data + ((b*seq_len + s)*heads + h)*head_dim;
                float* k_ptr = k->data + ((b*seq_len + s)*heads + h)*head_dim;
                float* oq_ptr = out_q->data + ((b*seq_len + s)*heads + h)*head_dim;
                float* ok_ptr = out_k->data + ((b*seq_len + s)*heads + h)*head_dim;

                float* c_ptr = cos->data + s*half_dim; // [Seq, Half]
                float* s_ptr = sin->data + s*half_dim;

                for(int i=0; i<half_dim; i++) {
                    float q_r = q_ptr[i];
                    float q_i = q_ptr[i + half_dim];
                    float k_r = k_ptr[i];
                    float k_i = k_ptr[i + half_dim];

                    float cs = c_ptr[i];
                    float sn = s_ptr[i];

                    oq_ptr[i] = q_r * cs - q_i * sn;
                    oq_ptr[i + half_dim] = q_r * sn + q_i * cs;

                    ok_ptr[i] = k_r * cs - k_i * sn;
                    ok_ptr[i + half_dim] = k_r * sn + k_i * cs;
                }
            }
        }
    }
}

// Conv2d (Naive)
void tensor_conv2d(Tensor* input, Tensor* weight, Tensor* bias, Tensor* out, int stride, int padding, int groups) {
    int N = input->shape[0];
    int Cin = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];

    int Cout = weight->shape[0];
    int KH = weight->shape[2];
    int KW = weight->shape[3];

    int Hout = out->shape[2];
    int Wout = out->shape[3];

    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;

    #pragma omp parallel for collapse(2)
    for(int n=0; n<N; n++) {
        for(int c_out=0; c_out<Cout; c_out++) {
            int g = c_out / Cout_per_group;
            float b_val = bias ? bias->data[c_out] : 0.0f;

            for(int h_out=0; h_out<Hout; h_out++) {
                for(int w_out=0; w_out<Wout; w_out++) {
                    float sum = 0.0f;
                    int h_in_start = h_out * stride - padding;
                    int w_in_start = w_out * stride - padding;

                    for(int c_in=0; c_in<Cin_per_group; c_in++) {
                        int actual_c_in = g * Cin_per_group + c_in;
                        for(int kh=0; kh<KH; kh++) {
                            for(int kw=0; kw<KW; kw++) {
                                int h_in = h_in_start + kh;
                                int w_in = w_in_start + kw;

                                if(h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    float val = input->data[((n*Cin + actual_c_in)*H + h_in)*W + w_in];
                                    float w_val = weight->data[((c_out*Cin_per_group + c_in)*KH + kh)*KW + kw];
                                    sum += val * w_val;
                                }
                            }
                        }
                    }
                    out->data[((n*Cout + c_out)*Hout + h_out)*Wout + w_out] = sum + b_val;
                }
            }
        }
    }
}

// Permute
void tensor_permute(Tensor* input, Tensor* out, int* dims) {
    int ndim = input->ndim;
    int* in_strides = (int*)malloc(sizeof(int) * ndim);
    int* out_strides = (int*)malloc(sizeof(int) * ndim);

    get_strides(input->shape, ndim, in_strides);
    get_strides(out->shape, ndim, out_strides); // Assuming contiguous out

    #pragma omp parallel for
    for(int i=0; i<out->size; i++) {
        int temp = i;
        int in_offset = 0;
        for(int d=0; d<ndim; d++) {
            int coord = temp / out_strides[d];
            temp %= out_strides[d];
            in_offset += coord * in_strides[dims[d]];
        }
        out->data[i] = input->data[in_offset];
    }

    free(in_strides);
    free(out_strides);
}

// Scaled Dot Product Attention (Fused)
void tensor_scaled_dot_product_attention(Tensor* q, Tensor* k, Tensor* v, Tensor* out, float scale) {
    int batch = q->shape[0];
    int seq = q->shape[1];
    int heads = q->shape[2];
    int head_dim = q->shape[3];

    #pragma omp parallel for collapse(2)
    for(int b=0; b<batch; b++) {
        for(int h=0; h<heads; h++) {
            float* scores = (float*)malloc(sizeof(float) * seq * seq);

            // Q * K^T
            for(int i=0; i<seq; i++) {
                float* q_vec = q->data + ((b*seq + i)*heads + h)*head_dim;
                for(int j=0; j<seq; j++) {
                    float* k_vec = k->data + ((b*seq + j)*heads + h)*head_dim;
                    float sum = 0.0f;
                    #pragma omp simd reduction(+:sum)
                    for(int d=0; d<head_dim; d++) sum += q_vec[d] * k_vec[d];
                    scores[i*seq + j] = sum * scale;
                }
            }

            // Softmax
            for(int i=0; i<seq; i++) {
                float max_val = -1e9f;
                for(int j=0; j<seq; j++) if(scores[i*seq + j] > max_val) max_val = scores[i*seq + j];

                float sum = 0.0f;
                for(int j=0; j<seq; j++) {
                    scores[i*seq + j] = expf(scores[i*seq + j] - max_val);
                    sum += scores[i*seq + j];
                }
                float inv_sum = 1.0f / sum;
                for(int j=0; j<seq; j++) scores[i*seq + j] *= inv_sum;
            }

            // Score * V
            for(int i=0; i<seq; i++) {
                float* out_vec = out->data + ((b*seq + i)*heads + h)*head_dim;
                for(int d=0; d<head_dim; d++) out_vec[d] = 0.0f; // Init

                for(int j=0; j<seq; j++) {
                    float score = scores[i*seq + j];
                    float* v_vec = v->data + ((b*seq + j)*heads + h)*head_dim;
                    for(int d=0; d<head_dim; d++) {
                        out_vec[d] += score * v_vec[d];
                    }
                }
            }

            free(scores);
        }
    }
}

void tensor_swiglu(Tensor* gate, Tensor* up, Tensor* out) {
    #pragma omp parallel for simd
    for(int i=0; i<out->size; i++) {
        float g = gate->data[i];
        float u = up->data[i];
        float silu_g = g * (1.0f / (1.0f + expf(-g)));
        out->data[i] = silu_g * u;
    }
}

void tensor_fused_gate_up_swiglu(Tensor* gate_up, Tensor* out) {
    // Input: [..., 2*Hidden]
    // Output: [..., Hidden]
    int hidden = out->shape[out->ndim-1];
    int rows = out->size / hidden;
    int input_stride = hidden * 2;

    #pragma omp parallel for
    for(int i=0; i<rows; i++) {
        float* in_ptr = gate_up->data + i * input_stride;
        float* out_ptr = out->data + i * hidden;

        #pragma omp simd
        for(int j=0; j<hidden; j++) {
            float g = in_ptr[j]; // First half is gate (usually)
            float u = in_ptr[j + hidden]; // Second half is up
            float silu_g = g * (1.0f / (1.0f + expf(-g)));
            out_ptr[j] = silu_g * u;
        }
    }
}

void tensor_slice(Tensor* input, Tensor* out, int* start_indices, int* slice_shapes) {
    int ndim = input->ndim;
    int* in_strides = (int*)malloc(sizeof(int) * ndim);
    int* out_strides = (int*)malloc(sizeof(int) * ndim);
    get_strides(input->shape, ndim, in_strides);
    get_strides(out->shape, ndim, out_strides);

    #pragma omp parallel for
    for(int i=0; i<out->size; i++) {
        int temp = i;
        int in_offset = 0;
        for(int d=0; d<ndim; d++) {
            int coord = temp / out_strides[d];
            temp %= out_strides[d];
            in_offset += (start_indices[d] + coord) * in_strides[d];
        }
        out->data[i] = input->data[in_offset];
    }
    free(in_strides);
    free(out_strides);
}

void tensor_precompute_freqs_cis(int dim, int end, float theta, Tensor* out_cos, Tensor* out_sin) {
    #pragma omp parallel for
    for(int i=0; i<end; i++) {
        for(int j=0; j<dim/2; j++) {
            float freq = 1.0f / powf(theta, (float)(2*j) / dim);
            float val = i * freq;
            out_cos->data[i*(dim/2) + j] = cosf(val);
            out_sin->data[i*(dim/2) + j] = sinf(val);
        }
    }
}

void tensor_argmax(Tensor* input, Tensor* out) {
    int last_dim = input->shape[input->ndim-1];
    int batch = input->size / last_dim;
    #pragma omp parallel for
    for(int i=0; i<batch; i++) {
        float* in_ptr = input->data + i*last_dim;
        float max_val = -1e9f;
        int max_idx = 0;
        for(int j=0; j<last_dim; j++) {
            if(in_ptr[j] > max_val) {
                max_val = in_ptr[j];
                max_idx = j;
            }
        }
        out->data[i] = (float)max_idx;
    }
}

void tensor_embed(Tensor* weight, Tensor* indices, Tensor* out) {
    int embed_dim = weight->shape[1];
    #pragma omp parallel for
    for(int i=0; i<indices->size; i++) {
        int idx = (int)indices->data[i];
        if(idx < 0 || idx >= weight->shape[0]) idx = 0; // Safe
        memcpy(out->data + i*embed_dim, weight->data + idx*embed_dim, embed_dim * sizeof(float));
    }
}

void tensor_reshape(Tensor* input, Tensor* out) {
    if(input->size != out->size) return;
    memcpy(out->data, input->data, input->size * sizeof(float));
}

void tensor_gather(Tensor* input, Tensor* indices, Tensor* out, int axis) {
    int inner_size = 1;
    for(int i=axis+1; i<input->ndim; i++) inner_size *= input->shape[i];
    int dim_size = input->shape[axis];

    #pragma omp parallel for
    for(int i=0; i<indices->size; i++) {
        int idx = (int)indices->data[i];
        if(idx < 0 || idx >= dim_size) idx = 0;
        if(axis == 0) {
            float* src = input->data + idx * inner_size;
            float* dst = out->data + i * inner_size;
            memcpy(dst, src, inner_size * sizeof(float));
        }
    }
}

void tensor_scatter_add(Tensor* input, Tensor* indices, Tensor* src, int axis) {
    int inner_size = 1;
    for(int i=axis+1; i<input->ndim; i++) inner_size *= input->shape[i];

    if(axis == 0) {
        #pragma omp parallel for
        for(int i=0; i<indices->size; i++) {
            int idx = (int)indices->data[i];
            float* s = src->data + i * inner_size;
            float* d = input->data + idx * inner_size;
            for(int j=0; j<inner_size; j++) {
                #pragma omp atomic
                d[j] += s[j];
            }
        }
    }
}

void tensor_cat(Tensor** inputs, int count, int axis, Tensor* out) {
    int outer_size = 1;
    for(int i=0; i<axis; i++) outer_size *= out->shape[i];
    int inner_size = 1;
    for(int i=axis+1; i<out->ndim; i++) inner_size *= out->shape[i];

    int current_offset = 0;
    for(int k=0; k<count; k++) {
        Tensor* t = inputs[k];
        int dim_size = t->shape[axis];
        int block_size = dim_size * inner_size;

        #pragma omp parallel for
        for(int o=0; o<outer_size; o++) {
            float* src = t->data + o * block_size;
            float* dst = out->data + o * (out->shape[axis] * inner_size) + current_offset * inner_size;
            memcpy(dst, src, block_size * sizeof(float));
        }
        current_offset += dim_size;
    }
}

void tensor_topk(Tensor* input, int k, Tensor* out_values, Tensor* out_indices) {
    int last_dim = input->shape[input->ndim-1];
    int batch = input->size / last_dim;

    #pragma omp parallel for
    for(int b=0; b<batch; b++) {
        float* in_ptr = input->data + b*last_dim;
        float* val_ptr = out_values->data + b*k;
        float* idx_ptr = out_indices->data + b*k;

        for(int i=0; i<k; i++) {
            val_ptr[i] = -1e18f;
            idx_ptr[i] = -1.0f;
        }

        for(int i=0; i<last_dim; i++) {
            float val = in_ptr[i];
            int pos = -1;
            for(int j=0; j<k; j++) {
                if(val > val_ptr[j]) {
                    pos = j;
                    break;
                }
            }
            if(pos != -1) {
                for(int j=k-1; j>pos; j--) {
                    val_ptr[j] = val_ptr[j-1];
                    idx_ptr[j] = idx_ptr[j-1];
                }
                val_ptr[pos] = val;
                idx_ptr[pos] = (float)i;
            }
        }
    }
}

// MoE Primitives
void tensor_count_value(Tensor* t, float value, int* count) {
    int c = 0;
    #pragma omp parallel for reduction(+:c)
    for(int i=0; i<t->size; i++) {
        if(fabsf(t->data[i] - value) < 1e-6) {
            c++;
        }
    }
    *count = c;
}

void tensor_gather_by_value(Tensor* input, Tensor* indices, float value, Tensor* out_data, Tensor* out_indices) {
    int total_tokens = indices->size;
    int hidden_size = input->shape[input->ndim-1];

    int current_idx = 0;

    #pragma omp parallel for
    for(int i=0; i<total_tokens; i++) {
        if(fabsf(indices->data[i] - value) < 1e-6) {
            int pos;
            #pragma omp atomic capture
            pos = current_idx++;

            out_indices->data[pos] = (float)i;
            float* src = input->data + i * hidden_size;
            float* dst = out_data->data + pos * hidden_size;
            memcpy(dst, src, hidden_size * sizeof(float));
        }
    }
}

void tensor_scatter_add_by_index(Tensor* out, Tensor* src, Tensor* indices) {
    int count = indices->size;
    int hidden_size = src->shape[src->ndim-1];
    int total_rows = out->size / hidden_size;

    #pragma omp parallel for
    for(int i=0; i<count; i++) {
        int idx = (int)indices->data[i];
        if(idx >= 0 && idx < total_rows) {
             float* s = src->data + i * hidden_size;
             float* d = out->data + idx * hidden_size;
             for(int j=0; j<hidden_size; j++) {
                 // Atomic optional but good practice
                 #pragma omp atomic
                 d[j] += s[j];
             }
        }
    }
}

// DeltaNet Recurrence
void tensor_deltanet_recurrence(Tensor* q, Tensor* k, Tensor* v, Tensor* beta, Tensor* state, Tensor* out) {
    int B = q->shape[0];
    int S = q->shape[1];
    int H = q->shape[2];
    int D = q->shape[3];

    #pragma omp parallel for collapse(2)
    for(int b=0; b<B; b++) {
        for(int h=0; h<H; h++) {
            float* s_ptr = state->data + (b*H + h) * D * D;

            for(int t=0; t<S; t++) {
                int offset = ((b*S + t)*H + h);
                float* q_vec = q->data + offset * D;
                float* k_vec = k->data + offset * D;
                float* v_vec = v->data + offset * D;
                float* out_vec = out->data + offset * D;
                float b_val = beta->data[offset];

                // State Update
                for(int i=0; i<D; i++) {
                    for(int j=0; j<D; j++) {
                        float old_s = s_ptr[i*D + j];
                        float val = k_vec[i] * v_vec[j];
                        s_ptr[i*D + j] = b_val * old_s + val;
                    }
                }

                // Output
                for(int j=0; j<D; j++) {
                    float sum = 0.0f;
                    for(int i=0; i<D; i++) {
                        sum += q_vec[i] * s_ptr[i*D + j];
                    }
                    out_vec[j] = sum;
                }
            }
        }
    }
}
