#include "pybind_wrapper.h"  // This will include the header for our functions
#include "attention_kernel.h"
#include "tensor_ops.h"
#include "block_sparse_attention.h"
#include <torch/extension.h>
#include <iostream>

// Implementation of the CUDA kernel launchers

void attention_forward_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output
) {
    // Get tensor dimensions
    const auto batch_size = q.size(0);
    const auto seq_len = q.size(1);
    const auto head_dim = q.size(2);
    const auto num_heads = 1; // Simplified for single head attention

    // Validate tensor properties before kernel launch
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Check for valid dimensions
    TORCH_CHECK(batch_size > 0 && seq_len > 0 && head_dim > 0,
                "Invalid tensor dimensions: batch_size=" + std::to_string(batch_size) +
                ", seq_len=" + std::to_string(seq_len) +
                ", head_dim=" + std::to_string(head_dim));

    // Get raw pointers to the data
    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Get the optimal kernel configuration for SM61
    KernelConfig config = get_attention_config(batch_size, seq_len, head_dim);

    // Validate kernel configuration
    TORCH_CHECK(config.grid.x > 0 && config.grid.y > 0 && config.block.x > 0 && config.block.y > 0,
                "Invalid kernel configuration: grid=(" + std::to_string(config.grid.x) + "," +
                std::to_string(config.grid.y) + "), block=(" + std::to_string(config.block.x) +
                "," + std::to_string(config.block.y) + ")");

    // Validate shared memory size
    if (config.shared_mem_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        std::cerr << "Warning: Requested shared memory (" << config.shared_mem_size
                  << ") exceeds SM61 limit (" << SM61_MAX_SHARED_MEMORY_PER_BLOCK << ")" << std::endl;
        // Use maximum allowed shared memory instead
        config.shared_mem_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK;
    }

    // Launch the attention kernel
    attention_kernel<<<config.grid, config.block, config.shared_mem_size>>>(
        q_ptr, k_ptr, v_ptr, output_ptr,
        batch_size, seq_len, head_dim, num_heads
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) +
                               " | Kernel config - grid=(" + std::to_string(config.grid.x) + "," +
                               std::to_string(config.grid.y) + "," + std::to_string(config.grid.z) +
                               "), block=(" + std::to_string(config.block.x) + "," +
                               std::to_string(config.block.y) + "," + std::to_string(config.block.z) +
                               "), shared_mem=" + std::to_string(config.shared_mem_size) +
                               " | Tensor shapes - Q=" + std::to_string(q.numel()) +
                               ", K=" + std::to_string(k.numel()) +
                               ", V=" + std::to_string(v.numel()) +
                               ", Output=" + std::to_string(output.numel());
        throw std::runtime_error(error_msg);
    }

    // Synchronize to ensure completion and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel execution failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

void attention_backward_cuda_launcher(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v
) {
    // In a full implementation, we would implement the backward pass
    // For now, we'll use PyTorch's autograd in the Python layer
    // This is a placeholder for the actual CUDA backward implementation
    TORCH_WARN("CUDA backward pass not fully implemented, using PyTorch autograd fallback");
}

void matmul_cuda_launcher(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output
) {
    // Get tensor dimensions
    const auto m = a.size(0);
    const auto k = a.size(1);
    const auto n = b.size(1);

    // Validate tensor properties before kernel launch
    TORCH_CHECK(a.is_cuda(), "A tensor must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "B tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Validate tensor dimensions
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2D tensors are supported");
    TORCH_CHECK(output.dim() == 2, "Output must be 2D tensor");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Matrix dimensions are not compatible for multiplication: A(" +
                std::to_string(a.size(0)) + "x" + std::to_string(a.size(1)) +
                ") * B(" + std::to_string(b.size(0)) + "x" + std::to_string(b.size(1)) +
                ") -> Expected output(" + std::to_string(a.size(0)) + "x" + std::to_string(b.size(1)) + ")");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "Output tensor has incorrect dimensions: expected(" +
                std::to_string(m) + "x" + std::to_string(n) +
                "), got(" + std::to_string(output.size(0)) + "x" + std::to_string(output.size(1)) + ")");

    // Check for valid dimensions
    TORCH_CHECK(m > 0 && k > 0 && n > 0,
                "Invalid tensor dimensions: m=" + std::to_string(m) +
                ", k=" + std::to_string(k) +
                ", n=" + std::to_string(n));

    // Get raw pointers to the data
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Get the optimal kernel configuration for SM61
    MatmulConfig config = get_matmul_config(m, n, k);

    // Validate kernel configuration
    TORCH_CHECK(config.grid.x > 0 && config.grid.y > 0 && config.block.x > 0 && config.block.y > 0,
                "Invalid kernel configuration: grid=(" + std::to_string(config.grid.x) + "," +
                std::to_string(config.grid.y) + "), block=(" + std::to_string(config.block.x) +
                "," + std::to_string(config.block.y) + ")");

    // Validate shared memory size
    if (config.shared_mem_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        std::cerr << "Warning: Requested shared memory (" << config.shared_mem_size
                  << ") exceeds SM61 limit (" << SM61_MAX_SHARED_MEMORY_PER_BLOCK << ")" << std::endl;
        // Use maximum allowed shared memory instead
        config.shared_mem_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK;
    }

    // Launch the matmul kernel
    matmul_kernel<<<config.grid, config.block, config.shared_mem_size>>>(
        a_ptr, b_ptr, output_ptr, m, n, k
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA matmul kernel launch failed: " + std::string(cudaGetErrorString(err)) +
                               " | Kernel config - grid=(" + std::to_string(config.grid.x) + "," +
                               std::to_string(config.grid.y) + "," + std::to_string(config.grid.z) +
                               "), block=(" + std::to_string(config.block.x) + "," +
                               std::to_string(config.block.y) + "," + std::to_string(config.block.z) +
                               "), shared_mem=" + std::to_string(config.shared_mem_size) +
                               " | Tensor shapes - A=" + std::to_string(a.numel()) +
                               ", B=" + std::to_string(b.numel()) +
                               ", Output=" + std::to_string(output.numel()) +
                               " | Dimensions - m=" + std::to_string(m) +
                               ", k=" + std::to_string(k) +
                               ", n=" + std::to_string(n);
        throw std::runtime_error(error_msg);
    }

    // Synchronize to ensure completion and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA matmul kernel execution failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

// Block-sparse attention launcher
void block_sparse_attention_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor output,
    torch::Tensor block_mask
) {
    // Get tensor dimensions
    const auto batch_size = q.size(0);
    const auto seq_len = q.size(1);
    const auto head_dim = q.size(2);
    const auto num_heads = q.size(2); // Assuming proper tensor layout: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]

    // Validate tensor properties before kernel launch
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(block_mask.is_cuda(), "Block mask tensor must be on CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(k.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(block_mask.dtype() == torch::kInt32, "Block mask should be int32");

    // Validate tensor dimensions
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Q, K, V tensors must be 4D");
    TORCH_CHECK(output.dim() == 4, "Output tensor must be 4D");
    TORCH_CHECK(block_mask.dim() == 2, "Block mask must be 2D");

    // Check that tensors have compatible shapes
    TORCH_CHECK(k.size(0) == batch_size && k.size(1) == seq_len && k.size(2) == num_heads && k.size(3) == head_dim,
                "K tensor has incompatible shape with Q tensor");
    TORCH_CHECK(v.size(0) == batch_size && v.size(1) == seq_len && v.size(2) == num_heads && v.size(3) == head_dim,
                "V tensor has incompatible shape with Q tensor");
    TORCH_CHECK(output.size(0) == batch_size && output.size(1) == seq_len && output.size(2) == num_heads && output.size(3) == head_dim,
                "Output tensor has incompatible shape with Q tensor");

    // Check for valid dimensions
    TORCH_CHECK(batch_size > 0 && seq_len > 0 && head_dim > 0 && num_heads > 0,
                "Invalid tensor dimensions: batch_size=" + std::to_string(batch_size) +
                ", seq_len=" + std::to_string(seq_len) +
                ", head_dim=" + std::to_string(head_dim) +
                ", num_heads=" + std::to_string(num_heads));

    // For this implementation, we'll assume the batch_size and num_heads are properly extracted
    // In a real implementation, the tensor shapes would need to be properly handled
    const int block_size = 64; // Default block size, could be passed as parameter

    // Validate block mask dimensions (assuming it's [num_blocks_q, num_blocks_k])
    const int num_blocks_q = (seq_len + block_size - 1) / block_size;  // Ceiling division
    const int num_blocks_k = num_blocks_q;
    TORCH_CHECK(block_mask.size(0) == num_blocks_q && block_mask.size(1) == num_blocks_k,
                "Block mask has incorrect dimensions: expected(" + std::to_string(num_blocks_q) +
                "x" + std::to_string(num_blocks_k) + "), got(" +
                std::to_string(block_mask.size(0)) + "x" + std::to_string(block_mask.size(1)) + ")");

    // Get raw pointers to the data
    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const int* mask_ptr = block_mask.data_ptr<int>();

    // Get the optimal kernel configuration for block-sparse attention on SM61
    BlockSparseConfig config = get_block_sparse_attention_config(
        batch_size, seq_len, head_dim, num_heads, block_size
    );

    // Validate kernel configuration
    TORCH_CHECK(config.grid.x > 0 && config.grid.y > 0 && config.block.x > 0 && config.block.y > 0,
                "Invalid kernel configuration: grid=(" + std::to_string(config.grid.x) + "," +
                std::to_string(config.grid.y) + "), block=(" + std::to_string(config.block.x) +
                "," + std::to_string(config.block.y) + ")");

    // Validate shared memory size
    if (config.shared_mem_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        std::cerr << "Warning: Requested shared memory (" << config.shared_mem_size
                  << ") exceeds SM61 limit (" << SM61_MAX_SHARED_MEMORY_PER_BLOCK << ")" << std::endl;
        // Use maximum allowed shared memory instead
        config.shared_mem_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK;
    }

    // Launch the block-sparse attention kernel
    block_sparse_attention_kernel<<<config.grid, config.block, config.shared_mem_size>>>(
        q_ptr, k_ptr, v_ptr, output_ptr, mask_ptr,
        batch_size, seq_len, head_dim, num_heads, block_size
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "Block-sparse attention CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) +
                               " | Kernel config - grid=(" + std::to_string(config.grid.x) + "," +
                               std::to_string(config.grid.y) + "," + std::to_string(config.grid.z) +
                               "), block=(" + std::to_string(config.block.x) + "," +
                               std::to_string(config.block.y) + "," + std::to_string(config.block.z) +
                               "), shared_mem=" + std::to_string(config.shared_mem_size) +
                               " | Tensor shapes - Q=" + std::to_string(q.numel()) +
                               ", K=" + std::to_string(k.numel()) +
                               ", V=" + std::to_string(v.numel()) +
                               ", Output=" + std::to_string(output.numel()) +
                               ", BlockMask=" + std::to_string(block_mask.numel()) +
                               " | Dimensions - batch_size=" + std::to_string(batch_size) +
                               ", seq_len=" + std::to_string(seq_len) +
                               ", head_dim=" + std::to_string(head_dim) +
                               ", num_heads=" + std::to_string(num_heads) +
                               ", block_size=" + std::to_string(block_size);
        throw std::runtime_error(error_msg);
    }

    // Synchronize to ensure completion and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string error_msg = "Block-sparse attention CUDA kernel execution failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

// Memory-efficient operations launcher
void memory_efficient_ops_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor weight,
    int op_type
) {
    // Get tensor dimensions
    const auto batch_size = input.size(0);
    const auto seq_len = input.size(1);
    const auto hidden_dim = input.size(2);

    // Validate tensor properties before kernel launch
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Validate tensor dimensions
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D [batch, seq, hidden]");
    TORCH_CHECK(output.dim() == 3, "Output tensor must be 3D [batch, seq, hidden]");
    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3, "Weight tensor must be 2D or 3D");

    // Check that tensors have compatible shapes
    TORCH_CHECK(output.size(0) == batch_size && output.size(1) == seq_len && output.size(2) == hidden_dim,
                "Output tensor has incompatible shape with input tensor");

    // For different op types, check weight compatibility
    if (op_type == 0) { // matmul
        TORCH_CHECK(weight.size(0) == hidden_dim, "For matmul op, weight first dim must match hidden_dim");
    } else if (op_type == 1) { // add
        TORCH_CHECK(weight.size(0) == batch_size && weight.size(1) == seq_len && weight.size(2) == hidden_dim,
                    "For add op, weight must match input dimensions");
    } else if (op_type == 2) { // mul
        TORCH_CHECK(weight.size(0) == batch_size && weight.size(1) == seq_len && weight.size(2) == hidden_dim,
                    "For mul op, weight must match input dimensions");
    } else if (op_type == 3) { // activation (e.g., silu)
        TORCH_CHECK(weight.size(0) == batch_size && weight.size(1) == seq_len && weight.size(2) == hidden_dim,
                    "For activation op, weight must match input dimensions");
    }

    // Check for valid dimensions
    TORCH_CHECK(batch_size > 0 && seq_len > 0 && hidden_dim > 0,
                "Invalid tensor dimensions: batch_size=" + std::to_string(batch_size) +
                ", seq_len=" + std::to_string(seq_len) +
                ", hidden_dim=" + std::to_string(hidden_dim));

    // Check op_type validity
    TORCH_CHECK(op_type >= 0 && op_type <= 3,
                "Invalid op_type: " + std::to_string(op_type) +
                ", must be between 0 and 3 (0=matmul, 1=add, 2=mul, 3=activation)");

    // Get raw pointers to the data
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();

    // Get the optimal kernel configuration for memory-efficient operations on SM61
    MemoryEfficientConfig config = get_memory_efficient_config(batch_size, seq_len, hidden_dim);

    // Validate kernel configuration
    TORCH_CHECK(config.grid.x > 0 && config.grid.y > 0 && config.block.x > 0 && config.block.y > 0,
                "Invalid kernel configuration: grid=(" + std::to_string(config.grid.x) + "," +
                std::to_string(config.grid.y) + "), block=(" + std::to_string(config.block.x) +
                "," + std::to_string(config.block.y) + ")");

    // Validate shared memory size
    if (config.shared_mem_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        std::cerr << "Warning: Requested shared memory (" << config.shared_mem_size
                  << ") exceeds SM61 limit (" << SM61_MAX_SHARED_MEMORY_PER_BLOCK << ")" << std::endl;
        // Use maximum allowed shared memory instead
        config.shared_mem_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK;
    }

    // Launch the memory-efficient operations kernel
    memory_efficient_ops_kernel<<<config.grid, config.block, config.shared_mem_size>>>(
        input_ptr, output_ptr, weight_ptr,
        batch_size, seq_len, hidden_dim, static_cast<OpType>(op_type)
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "Memory-efficient operations CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) +
                               " | Kernel config - grid=(" + std::to_string(config.grid.x) + "," +
                               std::to_string(config.grid.y) + "," + std::to_string(config.grid.z) +
                               "), block=(" + std::to_string(config.block.x) + "," +
                               std::to_string(config.block.y) + "," + std::to_string(config.block.z) +
                               "), shared_mem=" + std::to_string(config.shared_mem_size) +
                               " | Tensor shapes - Input=" + std::to_string(input.numel()) +
                               ", Weight=" + std::to_string(weight.numel()) +
                               ", Output=" + std::to_string(output.numel()) +
                               " | Dimensions - batch_size=" + std::to_string(batch_size) +
                               ", seq_len=" + std::to_string(seq_len) +
                               ", hidden_dim=" + std::to_string(hidden_dim) +
                               ", op_type=" + std::to_string(op_type);
        throw std::runtime_error(error_msg);
    }

    // Synchronize to ensure completion and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string error_msg = "Memory-efficient operations CUDA kernel execution failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}

// High-performance matmul launcher for SM61
void high_performance_matmul_cuda_launcher(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    bool use_tensor_cores  // Will be ignored for SM61
) {
    // Get tensor dimensions
    const auto m = a.size(0);
    const auto k = a.size(1);
    const auto n = b.size(1);

    // Validate tensor properties before kernel launch
    TORCH_CHECK(a.is_cuda(), "A tensor must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "B tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Only float32 tensors are supported");

    // Validate tensor dimensions
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2D tensors are supported");
    TORCH_CHECK(output.dim() == 2, "Output must be 2D tensor");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Matrix dimensions are not compatible for multiplication: A(" +
                std::to_string(a.size(0)) + "x" + std::to_string(a.size(1)) +
                ") * B(" + std::to_string(b.size(0)) + "x" + std::to_string(b.size(1)) +
                ") -> Expected output(" + std::to_string(a.size(0)) + "x" + std::to_string(b.size(1)) + ")");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "Output tensor has incorrect dimensions: expected(" +
                std::to_string(m) + "x" + std::to_string(n) +
                "), got(" + std::to_string(output.size(0)) + "x" + std::to_string(output.size(1)) + ")");

    // Check for valid dimensions
    TORCH_CHECK(m > 0 && k > 0 && n > 0,
                "Invalid tensor dimensions: m=" + std::to_string(m) +
                ", k=" + std::to_string(k) +
                ", n=" + std::to_string(n));

    // Get raw pointers to the data
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Get the optimal kernel configuration for high-performance matmul on SM61
    MatmulConfig config = get_high_performance_matmul_config(m, n, k);

    // Validate kernel configuration
    TORCH_CHECK(config.grid.x > 0 && config.grid.y > 0 && config.block.x > 0 && config.block.y > 0,
                "Invalid kernel configuration: grid=(" + std::to_string(config.grid.x) + "," +
                std::to_string(config.grid.y) + "), block=(" + std::to_string(config.block.x) +
                "," + std::to_string(config.block.y) + ")");

    // Validate shared memory size
    if (config.shared_mem_size > SM61_MAX_SHARED_MEMORY_PER_BLOCK) {
        std::cerr << "Warning: Requested shared memory (" << config.shared_mem_size
                  << ") exceeds SM61 limit (" << SM61_MAX_SHARED_MEMORY_PER_BLOCK << ")" << std::endl;
        // Use maximum allowed shared memory instead
        config.shared_mem_size = SM61_MAX_SHARED_MEMORY_PER_BLOCK;
    }

    // Launch the high-performance matmul kernel
    high_performance_matmul_kernel<<<config.grid, config.block, config.shared_mem_size>>>(
        a_ptr, b_ptr, output_ptr, m, n, k, use_tensor_cores
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "High-performance matmul CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)) +
                               " | Kernel config - grid=(" + std::to_string(config.grid.x) + "," +
                               std::to_string(config.grid.y) + "," + std::to_string(config.grid.z) +
                               "), block=(" + std::to_string(config.block.x) + "," +
                               std::to_string(config.block.y) + "," + std::to_string(config.block.z) +
                               "), shared_mem=" + std::to_string(config.shared_mem_size) +
                               " | Tensor shapes - A=" + std::to_string(a.numel()) +
                               ", B=" + std::to_string(b.numel()) +
                               ", Output=" + std::to_string(output.numel()) +
                               " | Dimensions - m=" + std::to_string(m) +
                               ", k=" + std::to_string(k) +
                               ", n=" + std::to_string(n) +
                               " | use_tensor_cores=" + std::to_string(use_tensor_cores);
        throw std::runtime_error(error_msg);
    }

    // Synchronize to ensure completion and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string error_msg = "High-performance matmul CUDA kernel execution failed: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
}