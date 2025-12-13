/*
 * PyBind11 Interface for SM61-Optimized CUDA Kernels
 * Connects optimized CUDA kernels with Python interface for Qwen3-VL model
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "sm61_optimized_kernels.h"

namespace py = pybind11;

// Helper function to convert PyTorch tensor to raw pointer
template<typename T>
T* get_raw_ptr(torch::Tensor& tensor) {
    return tensor.data_ptr<T>();
}

// Python wrapper for scaled dot product attention
torch::Tensor scaled_dot_product_attention_sm61(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    double dropout_p = 0.0,
    bool is_causal = false
) {
    // Validate input tensors
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(key.is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(value.is_cuda(), "Value tensor must be on CUDA device");
    
    TORCH_CHECK(query.dim() == 4, "Query must be 4D tensor [batch, num_heads, seq_len, head_dim]");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D tensor [batch, num_heads, seq_len, head_dim]");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D tensor [batch, num_heads, seq_len, head_dim]");
    
    TORCH_CHECK(query.dtype() == key.dtype() && key.dtype() == value.dtype(),
                "All tensors must have the same dtype");
    
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_len = query.size(2);
    auto head_dim = query.size(3);
    
    TORCH_CHECK(key.size(0) == batch_size && key.size(1) == num_heads && 
                key.size(2) == seq_len && key.size(3) == head_dim,
                "Key tensor shape mismatch");
    TORCH_CHECK(value.size(0) == batch_size && value.size(1) == num_heads && 
                value.size(2) == seq_len && value.size(3) == head_dim,
                "Value tensor shape mismatch");
    
    // Create output tensor
    auto output = torch::empty_like(query);
    
    // Calculate scale factor
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Create configuration for SM61
    SM61AttentionConfig config(seq_len, head_dim);
    
    // Launch the appropriate kernel based on data type
    cudaError_t err;
    if (query.dtype() == torch::kFloat32) {
        err = launch_scaled_dot_product_attention_sm61<float>(
            get_raw_ptr<float>(query),
            get_raw_ptr<float>(key),
            get_raw_ptr<float>(value),
            get_raw_ptr<float>(output),
            scale_factor,
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            config
        );
    } else if (query.dtype() == torch::kFloat16) {
        err = launch_scaled_dot_product_attention_sm61<half>(
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(query)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(key)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(value)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(output)),
            scale_factor,
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            config
        );
    } else {
        throw std::runtime_error("Unsupported tensor data type. Only float32 and float16 are supported.");
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("SM61 scaled dot-product attention kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return output;
}

// Python wrapper for block-sparse attention
torch::Tensor block_sparse_attention_sm61(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_mask
) {
    // Validate input tensors
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(key.is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(value.is_cuda(), "Value tensor must be on CUDA device");
    TORCH_CHECK(block_mask.is_cuda(), "Block mask tensor must be on CUDA device");
    
    TORCH_CHECK(query.dim() == 4, "Query must be 4D tensor [batch, num_heads, seq_len, head_dim]");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D tensor [batch, num_heads, seq_len, head_dim]");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D tensor [batch, num_heads, seq_len, head_dim]");
    TORCH_CHECK(block_mask.dim() == 2, "Block mask must be 2D tensor");
    
    TORCH_CHECK(query.dtype() == key.dtype() && key.dtype() == value.dtype(),
                "Query, Key, and Value tensors must have the same dtype");
    TORCH_CHECK(block_mask.dtype() == torch::kInt32, "Block mask must be int32");
    
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_len = query.size(2);
    auto head_dim = query.size(3);
    
    // Calculate block size based on mask dimensions
    int num_blocks = block_mask.size(0); // Assuming square blocks
    int block_size = (seq_len + num_blocks - 1) / num_blocks;
    
    // Create output tensor
    auto output = torch::empty_like(query);
    
    // Create configuration for SM61
    SM61AttentionConfig config(seq_len, head_dim);
    
    // Launch the appropriate kernel based on data type
    cudaError_t err;
    if (query.dtype() == torch::kFloat32) {
        err = launch_block_sparse_attention_sm61<float>(
            get_raw_ptr<float>(query),
            get_raw_ptr<float>(key),
            get_raw_ptr<float>(value),
            get_raw_ptr<float>(output),
            get_raw_ptr<int>(block_mask),
            1.0f / std::sqrt(static_cast<float>(head_dim)),  // Scale factor
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            block_size,
            config
        );
    } else if (query.dtype() == torch::kFloat16) {
        err = launch_block_sparse_attention_sm61<half>(
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(query)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(key)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(value)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(output)),
            get_raw_ptr<int>(block_mask),
            1.0f / std::sqrt(static_cast<float>(head_dim)),  // Scale factor
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            static_cast<int>(num_heads),
            static_cast<int>(head_dim),
            block_size,
            config
        );
    } else {
        throw std::runtime_error("Unsupported tensor data type. Only float32 and float16 are supported.");
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("SM61 block-sparse attention kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return output;
}

// Python wrapper for high-performance matmul
torch::Tensor high_performance_matmul_sm61(
    torch::Tensor a,
    torch::Tensor b,
    bool use_tensor_cores = false  // SM61 doesn't have tensor cores, but keep for interface compatibility
) {
    // Validate input tensors
    TORCH_CHECK(a.is_cuda(), "Tensor A must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "Tensor B must be on CUDA device");
    
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Expected 2D tensors for matmul");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions are not compatible for multiplication");
    TORCH_CHECK(a.dtype() == b.dtype(), "Tensors A and B must have the same dtype");
    
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    
    // Create output tensor
    auto output = torch::empty({m, n}, torch::TensorOptions().dtype(a.dtype()).device(a.device()));
    
    // Create configuration for SM61
    SM61MatmulConfig config(m, n, k);
    
    // Launch the appropriate kernel based on data type
    cudaError_t err;
    if (a.dtype() == torch::kFloat32) {
        err = launch_high_performance_matmul_sm61<float>(
            get_raw_ptr<float>(a),
            get_raw_ptr<float>(b),
            get_raw_ptr<float>(output),
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            1.0f,  // alpha
            0.0f,  // beta
            config
        );
    } else if (a.dtype() == torch::kFloat16) {
        err = launch_high_performance_matmul_sm61<half>(
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(a)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(b)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(output)),
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            1.0f,  // alpha
            0.0f,  // beta
            config
        );
    } else {
        throw std::runtime_error("Unsupported tensor data type. Only float32 and float16 are supported.");
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("SM61 high-performance matmul kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return output;
}

// Python wrapper for memory-efficient operations
torch::Tensor memory_efficient_ops_sm61(
    torch::Tensor input,
    torch::Tensor weight,
    int op_type  // 0=matmul, 1=add, 2=mul, 3=activation
) {
    // Validate input tensors
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA device");
    
    TORCH_CHECK(input.dtype() == weight.dtype(), "Input and weight tensors must have the same dtype");
    
    // Create output tensor with same shape as input
    auto output = torch::empty_like(input);
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto hidden_dim = input.size(2);
    
    // Create configuration for SM61
    SM61MemoryCopyConfig config(input.numel());
    
    // Launch the appropriate kernel based on data type
    cudaError_t err;
    if (input.dtype() == torch::kFloat32) {
        err = launch_memory_efficient_ops_sm61<float>(
            get_raw_ptr<float>(input),
            get_raw_ptr<float>(output),
            get_raw_ptr<float>(weight),
            op_type,
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            static_cast<int>(hidden_dim),
            config
        );
    } else if (input.dtype() == torch::kFloat16) {
        err = launch_memory_efficient_ops_sm61<half>(
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(input)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(output)),
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(weight)),
            op_type,
            static_cast<int>(batch_size),
            static_cast<int>(seq_len),
            static_cast<int>(hidden_dim),
            config
        );
    } else {
        throw std::runtime_error("Unsupported tensor data type. Only float32 and float16 are supported.");
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("SM61 memory-efficient operations kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return output;
}

// Python wrapper for coalesced memory copy
torch::Tensor coalesced_copy_sm61(torch::Tensor input) {
    // Validate input tensor
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    
    // Create output tensor with same properties as input
    auto output = torch::empty_like(input);
    
    // Create configuration for SM61
    SM61MemoryCopyConfig config(input.numel());
    
    // Launch the appropriate kernel based on data type
    cudaError_t err;
    if (input.dtype() == torch::kFloat32) {
        err = launch_coalesced_memory_copy_sm61<float>(
            get_raw_ptr<float>(output),
            get_raw_ptr<float>(input),
            input.numel(),
            config
        );
    } else if (input.dtype() == torch::kFloat16) {
        err = launch_coalesced_memory_copy_sm61<half>(
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(output)),
            reinterpret_cast<const half*>(get_raw_ptr<at::Half>(input)),
            input.numel(),
            config
        );
    } else if (input.dtype() == torch::kInt32) {
        err = launch_coalesced_memory_copy_sm61<int32_t>(
            get_raw_ptr<int32_t>(output),
            get_raw_ptr<int32_t>(input),
            input.numel(),
            config
        );
    } else {
        throw std::runtime_error("Unsupported tensor data type for coalesced copy. Only float32, float16, and int32 are supported.");
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("SM61 coalesced memory copy kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return output;
}

// Python wrapper for transpose operation
torch::Tensor transpose_sm61(torch::Tensor input) {
    // Validate input tensor
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2-dimensional for transpose");
    
    auto rows = input.size(0);
    auto cols = input.size(1);
    
    // Create output tensor with swapped dimensions
    auto output = torch::empty({cols, rows}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Create configuration for SM61
    SM61TransposeConfig config(rows, cols);
    
    // Launch the appropriate kernel based on data type
    cudaError_t err;
    if (input.dtype() == torch::kFloat32) {
        err = launch_transpose_sm61<float>(
            get_raw_ptr<float>(output),
            get_raw_ptr<float>(input),
            static_cast<int>(rows),
            static_cast<int>(cols),
            config
        );
    } else if (input.dtype() == torch::kFloat16) {
        err = launch_transpose_sm61<half>(
            reinterpret_cast<half*>(get_raw_ptr<at::Half>(output)),
            reinterpret_cast<const half*>(get_raw_ptr<at::Half>(input)),
            static_cast<int>(rows),
            static_cast<int>(cols),
            config
        );
    } else {
        throw std::runtime_error("Unsupported tensor data type for transpose. Only float32 and float16 are supported.");
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error("SM61 transpose kernel failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return output;
}

// Python wrapper for SM61 memory pool
class PySM61MemoryPool {
public:
    std::unique_ptr<SM61MemoryPool> pool;
    
    PySM61MemoryPool(size_t size = 64 * 1024 * 1024) {  // Default 64MB
        pool = std::make_unique<SM61MemoryPool>(size);
    }
    
    torch::Tensor allocate_tensor(std::vector<int64_t> sizes, torch::ScalarType dtype) {
        size_t total_elements = 1;
        for (auto size : sizes) {
            total_elements *= size;
        }
        
        size_t element_size = 0;
        if (dtype == torch::kFloat32) {
            element_size = sizeof(float);
        } else if (dtype == torch::kFloat16) {
            element_size = sizeof(half);
        } else if (dtype == torch::kInt32) {
            element_size = sizeof(int32_t);
        } else {
            throw std::runtime_error("Unsupported tensor data type for memory pool allocation.");
        }
        
        size_t tensor_size = total_elements * element_size;
        
        void* ptr = pool->allocate(tensor_size);
        if (!ptr) {
            throw std::runtime_error("Failed to allocate memory from pool.");
        }
        
        // Create a tensor that will handle deallocation through a custom deleter
        auto deleter = [this, tensor_size](void* data) {
            pool->deallocate(data, tensor_size);
        };
        
        return torch::from_blob(ptr, sizes, torch::TensorOptions().dtype(dtype).device(torch::kCUDA), deleter);
    }
    
    py::dict get_stats() {
        auto stats = pool->get_stats();
        return py::dict(
            "total_size"_a = stats.total_size,
            "allocated"_a = stats.allocated,
            "free"_a = stats.free,
            "fragmentation"_a = stats.fragmentation,
            "num_free_blocks"_a = stats.num_free_blocks
        );
    }
    
    void clear() {
        pool->clear();
    }
    
    void defragment() {
        pool->defragment();
    }
};

// PYBIND11_MODULE macro to create the Python extension
PYBIND11_MODULE(sm61_cuda_kernels, m) {
    m.doc() = "SM61-optimized CUDA kernels for Qwen3-VL model";
    
    // Attention operations
    m.def("scaled_dot_product_attention_sm61", &scaled_dot_product_attention_sm61,
          "SM61-optimized scaled dot-product attention",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("dropout_p") = 0.0, py::arg("is_causal") = false);
          
    m.def("block_sparse_attention_sm61", &block_sparse_attention_sm61,
          "SM61-optimized block-sparse attention",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("block_mask"));
    
    // Matrix operations
    m.def("high_performance_matmul_sm61", &high_performance_matmul_sm61,
          "SM61-optimized high-performance matrix multiplication",
          py::arg("a"), py::arg("b"), py::arg("use_tensor_cores") = false);
    
    // Memory operations
    m.def("memory_efficient_ops_sm61", &memory_efficient_ops_sm61,
          "SM61-optimized memory-efficient operations",
          py::arg("input"), py::arg("weight"), py::arg("op_type"));
    
    m.def("coalesced_copy_sm61", &coalesced_copy_sm61,
          "SM61-optimized coalesced memory copy",
          py::arg("input"));
    
    m.def("transpose_sm61", &transpose_sm61,
          "SM61-optimized matrix transpose with bank conflict avoidance",
          py::arg("input"));
    
    // Memory pool
    py::class_<PySM61MemoryPool>(m, "SM61MemoryPool")
        .def(py::init<size_t>(), py::arg("size") = 64 * 1024 * 1024)
        .def("allocate_tensor", &PySM61MemoryPool::allocate_tensor,
             "Allocate a tensor using the SM61 memory pool",
             py::arg("sizes"), py::arg("dtype"))
        .def("get_stats", &PySM61MemoryPool::get_stats,
             "Get memory pool statistics")
        .def("clear", &PySM61MemoryPool::clear,
             "Clear the memory pool")
        .def("defragment", &PySM61MemoryPool::defragment,
             "Defragment the memory pool");
    
    // Export the CUDA availability check
    m.def("cuda_available", []() { return torch::cuda::is_available(); },
          "Check if CUDA is available");
    
    // Export hardware detection function
    m.def("get_sm61_hardware_info", []() {
        py::dict info;
        if (torch::cuda::is_available()) {
            auto device_count = torch::cuda::device_count();
            info["cuda_available"] = true;
            info["device_count"] = device_count;
            
            if (device_count > 0) {
                auto device_name = torch::cuda::get_device_name(0);
                auto capability = torch::cuda::get_device_capability(0);
                auto total_memory = torch::cuda::get_device_properties(0).total_memory;
                
                info["device_name"] = device_name;
                info["compute_capability"] = std::make_pair(capability.first, capability.second);
                info["total_memory_gb"] = static_cast<double>(total_memory) / (1024.0 * 1024.0 * 1024.0);
                
                // Check if this is likely an SM61 device
                bool is_sm61 = (capability.first == 6 && capability.second == 1);
                info["is_sm61"] = is_sm61;
            }
        } else {
            info["cuda_available"] = false;
        }
        return info;
    }, "Get SM61 hardware information");
}