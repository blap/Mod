#ifndef PYBIND_WRAPPER_H
#define PYBIND_WRAPPER_H

#include <torch/extension.h>

// Declaration of CUDA kernel launchers

void attention_forward_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output
);

void attention_backward_cuda_launcher(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v
);

void matmul_cuda_launcher(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output
);

// Additional kernel launchers for advanced optimizations

void block_sparse_attention_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    torch::Tensor block_mask
);

void memory_efficient_ops_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor weight,
    int op_type
);

void high_performance_matmul_cuda_launcher(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    bool use_tensor_cores
);

#endif // PYBIND_WRAPPER_H