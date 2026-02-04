#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernels
// Implementations would typically be in .cu files, but for this structure
// we'll keep the headers here and implementation details in .cu files

// ----------------------------------------------------------------------
// Gated DeltaNet Kernels
// ----------------------------------------------------------------------

torch::Tensor deltanet_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor beta,
    torch::Tensor initial_state
);

torch::Tensor deltanet_bwd_cuda(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor beta,
    torch::Tensor initial_state
);

// ----------------------------------------------------------------------
// Mixture of Experts (MoE) Kernels
// ----------------------------------------------------------------------

// Kernel for Top-K gating
std::tuple<torch::Tensor, torch::Tensor> moe_gating_cuda(
    torch::Tensor router_logits,
    int k,
    bool training
);

// Kernel for sparse expert computation (scatter/gather or permuted based)
torch::Tensor moe_dispatch_cuda(
    torch::Tensor hidden_states,
    torch::Tensor expert_indices,
    torch::Tensor expert_weights,
    std::vector<torch::Tensor> experts_weights_w1,
    std::vector<torch::Tensor> experts_weights_w2
);

// ----------------------------------------------------------------------
// Specialized Attention Kernels (GQA with specific head dim)
// ----------------------------------------------------------------------

torch::Tensor gated_attention_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate, // Specific to Gated Attention if applicable
    float scale
);

// ----------------------------------------------------------------------
// PyBind11 Definitions
// ----------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deltanet_fwd", &deltanet_fwd_cuda, "Gated DeltaNet Forward (CUDA)");
    m.def("deltanet_bwd", &deltanet_bwd_cuda, "Gated DeltaNet Backward (CUDA)");
    m.def("moe_gating", &moe_gating_cuda, "MoE Top-K Gating (CUDA)");
    m.def("moe_dispatch", &moe_dispatch_cuda, "MoE Sparse Dispatch (CUDA)");
    m.def("gated_attention_fwd", &gated_attention_fwd_cuda, "Gated Attention Forward (CUDA)");
}
