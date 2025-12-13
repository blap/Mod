#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernel launchers
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

// Python bindings
torch::Tensor attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    // Validate input tensors
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Create output tensor with same shape as inputs
    auto output = torch::zeros_like(q);

    // Launch the optimized CUDA kernel
    attention_forward_cuda_launcher(q, k, v, output);

    return output;
}

std::vector<torch::Tensor> attention_backward(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output
) {
    // Validate input tensors
    TORCH_CHECK(grad_output.is_cuda(), "grad_output tensor must be on CUDA device");
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");

    // Create gradient tensors
    auto grad_q = torch::zeros_like(q);
    auto grad_k = torch::zeros_like(k);
    auto grad_v = torch::zeros_like(v);

    // Launch the optimized CUDA kernel
    attention_backward_cuda_launcher(
        grad_output, q, k, v, output, grad_q, grad_k, grad_v
    );

    return {grad_q, grad_k, grad_v};
}

torch::Tensor matmul_sm61(
    torch::Tensor a,
    torch::Tensor b
) {
    // Validate input tensors
    TORCH_CHECK(a.is_cuda(), "A tensor must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "B tensor must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Validate dimensions
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2D tensors are supported");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions are not compatible for multiplication");

    // Create output tensor
    auto output = torch::zeros({a.size(0), b.size(1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Launch the optimized CUDA kernel
    matmul_cuda_launcher(a, b, output);

    return output;
}

torch::Tensor block_sparse_attention_sm61(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor block_mask
) {
    // Validate input tensors
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");
    TORCH_CHECK(block_mask.is_cuda(), "Block mask tensor must be on CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(block_mask.dtype() == torch::kInt32, "Block mask should be int32");

    // Create output tensor with same shape as Q
    auto output = torch::zeros_like(q);

    // Launch the optimized block-sparse attention CUDA kernel
    block_sparse_attention_cuda_launcher(q, k, v, output, block_mask);

    return output;
}

torch::Tensor memory_efficient_ops_sm61(
    torch::Tensor input,
    torch::Tensor weight,
    int op_type
) {
    // Validate input tensors
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Create output tensor with same shape as input
    auto output = torch::zeros_like(input);

    // Launch the memory-efficient operations CUDA kernel
    memory_efficient_ops_cuda_launcher(input, output, weight, op_type);

    return output;
}

torch::Tensor high_performance_matmul_sm61(
    torch::Tensor a,
    torch::Tensor b,
    bool use_tensor_cores
) {
    // Validate input tensors
    TORCH_CHECK(a.is_cuda(), "A tensor must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "B tensor must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Validate dimensions
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2D tensors are supported");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions are not compatible for multiplication");

    // Create output tensor
    auto output = torch::zeros({a.size(0), b.size(1)}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Launch the high-performance matmul CUDA kernel
    high_performance_matmul_cuda_launcher(a, b, output, use_tensor_cores);

    return output;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "SM61 Optimized Attention Forward");
    m.def("attention_backward", &attention_backward, "SM61 Optimized Attention Backward");
    m.def("matmul_sm61", &matmul_sm61, "SM61 Optimized Matrix Multiplication");
    m.def("block_sparse_attention_sm61", &block_sparse_attention_sm61, "SM61 Optimized Block-Sparse Attention");
    m.def("memory_efficient_ops_sm61", &memory_efficient_ops_sm61, "SM61 Memory-Efficient Operations");
    m.def("high_performance_matmul_sm61", &high_performance_matmul_sm61, "SM61 High-Performance Matrix Multiplication");
}