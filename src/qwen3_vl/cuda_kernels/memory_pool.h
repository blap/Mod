#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <unordered_map>
#include <list>
#include <mutex>
#include <atomic>

// Memory pool optimized for SM61 architecture
class SM61MemoryPool {
public:
    struct Block {
        void* ptr;
        size_t size;
        std::atomic<bool> free;  // Use atomic for thread safety

        Block() : ptr(nullptr), size(0), free(true) {}
        Block(void* p, size_t s) : ptr(p), size(s), free(true) {}
    };

    SM61MemoryPool();
    SM61MemoryPool(size_t pool_size);
    ~SM61MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);

    void clear();

    // Get memory pool statistics
    struct Stats {
        size_t total_size;
        size_t allocated;
        size_t free;
        double fragmentation;
        size_t num_free_blocks;
    };
    Stats get_stats() const;

    // Synchronization functions for CPU-GPU operations
    void synchronize() const;
    void stream_synchronize(cudaStream_t stream = 0) const;

    // Additional utility methods for error handling and diagnostics
    bool is_device_accessible() const;
    size_t get_available_memory() const;

private:
    // Use multiple pools for different size ranges to reduce fragmentation
    std::list<Block> small_pool_;    // For allocations <= 64KB (half of SM61's shared mem per block)
    std::list<Block> medium_pool_;   // For allocations > 64KB and <= 256KB
    std::list<Block> large_pool_;    // For allocations > 256KB

    mutable std::mutex pool_mutex_;  // Use mutable for const methods

    // Helper function to determine which pool to use based on size
    std::list<Block>& get_pool(size_t size);

    // Maximum memory to keep in pool (in bytes)
    static constexpr size_t MAX_POOL_SIZE = 256 * 1024 * 1024;  // 256MB
    size_t current_pool_size_;
    size_t total_pool_size_;

    // Statistics for optimization
    size_t total_allocated_;
    size_t total_freed_;

    // Cleanup pool if it gets too large
    void cleanup_pool();
};

// Implementation of synchronization functions
inline void SM61MemoryPool::synchronize() const {
    cudaDeviceSynchronize();
}

inline void SM61MemoryPool::stream_synchronize(cudaStream_t stream) const {
    cudaStreamSynchronize(stream);
}

#endif // MEMORY_POOL_H