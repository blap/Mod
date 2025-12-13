#include "memory_pool.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

SM61MemoryPool::SM61MemoryPool() : current_pool_size_(0), total_pool_size_(MAX_POOL_SIZE), total_allocated_(0), total_freed_(0) {
    // Initialize the memory pools
    // Check CUDA device status during initialization
    cudaError_t err = cudaSetDevice(0);  // Assuming device 0
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device during memory pool initialization: " + std::string(cudaGetErrorString(err)));
    }
}

SM61MemoryPool::SM61MemoryPool(size_t pool_size) : current_pool_size_(0), total_pool_size_(pool_size), total_allocated_(0), total_freed_(0) {
    // Initialize the memory pools with custom size
    // Check CUDA device status during initialization
    cudaError_t err = cudaSetDevice(0);  // Assuming device 0
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device during memory pool initialization: " + std::string(cudaGetErrorString(err)));
    }
}

SM61MemoryPool::~SM61MemoryPool() {
    clear();  // Clean up all allocated memory
}

void* SM61MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Validate size
    if (size == 0) {
        return nullptr;  // Can't allocate zero bytes
    }

    // Find a suitable free block in the appropriate pool
    std::list<Block>& pool = get_pool(size);

    // Look for a free block that's large enough
    for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (it->free.load() && it->size >= size) {
            // Found a suitable block
            it->free.store(false);
            current_pool_size_ -= it->size;
            total_allocated_ += size;
            return it->ptr;
        }
    }

    // No suitable block found, allocate new memory
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        // If allocation failed due to out of memory, try to clear cache and retry
        if (err == cudaErrorMemoryAllocation) {
            std::cerr << "Memory allocation failed, attempting to clear cache and retry..." << std::endl;

            // Synchronize to ensure all operations are complete
            cudaDeviceSynchronize();

            // Clear CUDA cache
            // Note: This is a workaround since we can't directly call PyTorch from here
            // In practice, this would trigger cache clearing in the higher-level code

            // Try to allocate again after potential cache clear
            err = cudaMalloc(&ptr, size);
        }

        if (err != cudaSuccess) {
            std::string error_msg = "CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)) +
                                   " | Requested size: " + std::to_string(size) + " bytes" +
                                   " | Current pool size: " + std::to_string(current_pool_size_) + " bytes" +
                                   " | Total allocated: " + std::to_string(total_allocated_) + " bytes";
            std::cerr << error_msg << std::endl;
            return nullptr;
        }
    }

    total_allocated_ += size;

    // Add the new block to the appropriate pool (marked as not free for now)
    Block new_block;
    new_block.ptr = ptr;
    new_block.size = size;
    new_block.free.store(false);
    pool.push_back(new_block);

    return ptr;
}

void SM61MemoryPool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Validate parameters
    if (ptr == nullptr) {
        std::cerr << "Warning: Attempting to deallocate null pointer" << std::endl;
        return;
    }

    if (size == 0) {
        std::cerr << "Warning: Attempting to deallocate zero-size block" << std::endl;
        return;
    }

    // Find the block in the pools
    std::list<Block>& pool = get_pool(size);

    for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (it->ptr == ptr && !it->free.load()) {
            // Mark as free instead of actually freeing
            it->free.store(true);
            current_pool_size_ += size;
            total_freed_ += size;

            // If pool is getting too large, actually free some memory
            if (current_pool_size_ > MAX_POOL_SIZE) {
                cleanup_pool();
            }
            return;
        }
    }

    // If we get here, the block wasn't found in the pool, so free it directly
    // This might happen if the block was allocated outside the pool
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "Warning: Direct cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void SM61MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    // Free all memory in all pools
    auto free_pool = [this](std::list<Block>& pool) {
        for (auto it = pool.begin(); it != pool.end(); ++it) {
            cudaError_t err = cudaFree(it->ptr);
            if (err != cudaSuccess) {
                std::cerr << "Warning: Failed to free pooled memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
        pool.clear();
    };

    free_pool(small_pool_);
    free_pool(medium_pool_);
    free_pool(large_pool_);

    current_pool_size_ = 0;
}

std::list<SM61MemoryPool::Block>& SM61MemoryPool::get_pool(size_t size) {
    if (size <= 64 * 1024) {  // 64KB
        return small_pool_;
    } else if (size <= 256 * 1024) {  // 256KB
        return medium_pool_;
    } else {
        return large_pool_;
    }
}

void SM61MemoryPool::cleanup_pool() {
    // Free some memory from the pools if we're over the limit
    auto cleanup = [this](std::list<Block>& pool) {
        for (auto it = pool.begin(); it != pool.end();) {
            if (it->free.load()) {
                cudaError_t err = cudaFree(it->ptr);
                if (err != cudaSuccess) {
                    std::cerr << "Warning: Failed to free pooled memory during cleanup: " << cudaGetErrorString(err) << std::endl;
                }
                current_pool_size_ -= it->size;
                it = pool.erase(it);
            } else {
                ++it;
            }
        }
    };

    cleanup(small_pool_);
    cleanup(medium_pool_);
    cleanup(large_pool_);
}

SM61MemoryPool::Stats SM61MemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    Stats stats;
    stats.total_size = total_pool_size_;
    stats.allocated = total_allocated_ - total_freed_;  // Current allocated = total allocated - total freed
    stats.free = current_pool_size_;

    // Calculate fragmentation
    size_t total_free_blocks = 0;
    size_t total_free_size = 0;

    auto count_free_blocks = [&total_free_blocks, &total_free_size](const std::list<Block>& pool) {
        for (const auto& block : pool) {
            if (block.free.load()) {
                total_free_blocks++;
                total_free_size += block.size;
            }
        }
    };

    count_free_blocks(small_pool_);
    count_free_blocks(medium_pool_);
    count_free_blocks(large_pool_);

    stats.num_free_blocks = total_free_blocks;
    stats.fragmentation = (total_free_size > 0) ?
        static_cast<double>(total_free_size - current_pool_size_) / static_cast<double>(total_free_size) : 0.0;

    return stats;
}

// Additional utility methods for error handling
bool SM61MemoryPool::is_device_accessible() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    if (device_count == 0) {
        std::cerr << "No CUDA devices available" << std::endl;
        return false;
    }

    // Check current device status
    int current_device = 0;
    err = cudaGetDevice(&current_device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get current CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Check device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, current_device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

size_t SM61MemoryPool::get_available_memory() const {
    size_t free_mem = 0;
    size_t total_mem = 0;

    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(err) << std::endl;
        return 0;  // Return 0 if we can't get the info
    }

    return free_mem;
}