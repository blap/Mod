#ifndef MEMORY_POOL_SM61_CUH
#define MEMORY_POOL_SM61_CUH

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <atomic>

// Memory pool implementation optimized for SM61 architecture
// Designed to reduce memory allocation overhead and improve memory access patterns

struct MemoryBlock {
    void* ptr;
    size_t size;
    std::atomic<bool> allocated;  // Use atomic for thread safety
    size_t original_size;  // Actual requested size

    MemoryBlock() : ptr(nullptr), size(0), allocated(false), original_size(0) {}
    MemoryBlock(void* p, size_t s) : ptr(p), size(s), allocated(false), original_size(0) {}
};

class SM61MemoryPool {
private:
    char* pool_memory;
    size_t pool_size;
    std::vector<MemoryBlock> blocks;
    mutable std::mutex pool_mutex;  // For thread safety during block management

public:
    explicit SM61MemoryPool(size_t size = 64 * 1024 * 1024);  // Default 64MB
    ~SM61MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr, size_t original_size = 0);

    struct Stats {
        size_t total_size;
        size_t allocated;
        size_t free;
        float fragmentation;
        int num_free_blocks;
    };

    Stats get_stats() const;

private:
    void initialize_pool();
    size_t find_suitable_block(size_t size) const;
    void split_block(size_t block_idx, size_t size);
    void merge_free_blocks();  // Add function to merge adjacent free blocks
};

inline SM61MemoryPool::SM61MemoryPool(size_t size) : pool_size(size), pool_memory(nullptr) {
    // Allocate the main memory pool on device
    cudaError_t err = cudaMalloc(&pool_memory, pool_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate memory pool: " + std::string(cudaGetErrorString(err)));
    }
    initialize_pool();
}

inline SM61MemoryPool::~SM61MemoryPool() {
    if (pool_memory) {
        cudaFree(pool_memory);
    }
}

inline void SM61MemoryPool::initialize_pool() {
    // Initialize with a single large block
    MemoryBlock initial_block;
    initial_block.ptr = pool_memory;
    initial_block.size = pool_size;
    initial_block.allocated.store(false);
    blocks.push_back(initial_block);
}

inline void* SM61MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);

    // Find a suitable block
    size_t block_idx = find_suitable_block(size);
    if (block_idx == blocks.size()) {
        // No suitable block found
        return nullptr;
    }

    // Mark as allocated atomically to prevent race conditions
    blocks[block_idx].allocated.store(true);
    blocks[block_idx].original_size = size;

    return blocks[block_idx].ptr;
}

inline size_t SM61MemoryPool::find_suitable_block(size_t size) const {
    for (size_t i = 0; i < blocks.size(); i++) {
        // Check if block is not allocated and has sufficient size
        if (!blocks[i].allocated.load() && blocks[i].size >= size) {
            return i;
        }
    }
    return blocks.size();  // Not found
}

inline void SM61MemoryPool::deallocate(void* ptr, size_t original_size) {
    std::lock_guard<std::mutex> lock(pool_mutex);

    for (auto& block : blocks) {
        if (block.ptr == ptr && block.allocated.load()) {
            block.allocated.store(false);
            block.original_size = 0;
            return;
        }
    }
}

inline void SM61MemoryPool::merge_free_blocks() {
    // Implementation for merging adjacent free blocks to reduce fragmentation
    // This is a simplified implementation - a full implementation would be more complex
    std::lock_guard<std::mutex> lock(pool_mutex);

    // Sort blocks by address for merging adjacent blocks
    std::vector<MemoryBlock*> sorted_blocks;
    for (auto& block : blocks) {
        sorted_blocks.push_back(&block);
    }

    std::sort(sorted_blocks.begin(), sorted_blocks.end(),
              [](const MemoryBlock* a, const MemoryBlock* b) {
                  return a->ptr < b->ptr;
              });

    // Merge adjacent free blocks
    for (size_t i = 0; i < sorted_blocks.size() - 1; i++) {
        MemoryBlock* curr = sorted_blocks[i];
        MemoryBlock* next = sorted_blocks[i + 1];

        if (!curr->allocated.load() && !next->allocated.load()) {
            // Check if blocks are adjacent
            char* curr_end = static_cast<char*>(curr->ptr) + curr->size;
            if (curr_end == next->ptr) {
                // Merge blocks
                curr->size += next->size;
                next->size = 0;
                next->ptr = nullptr;
                next->allocated.store(true); // Mark as "merged" to avoid double processing
            }
        }
    }

    // Clean up merged blocks (remove zero-size blocks)
    blocks.erase(
        std::remove_if(blocks.begin(), blocks.end(),
            [](const MemoryBlock& block) {
                return block.size == 0 && block.ptr == nullptr;
            }),
        blocks.end());
}

inline SM61MemoryPool::Stats SM61MemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(pool_mutex);

    Stats stats = {};
    stats.total_size = pool_size;

    for (const auto& block : blocks) {
        if (block.allocated.load()) {
            stats.allocated += block.original_size;
        }
    }

    stats.free = stats.total_size - stats.allocated;
    stats.num_free_blocks = 0;

    for (const auto& block : blocks) {
        if (!block.allocated.load()) {
            stats.num_free_blocks++;
        }
    }

    if (blocks.size() > 0) {
        stats.fragmentation = (float)stats.num_free_blocks / blocks.size();
    } else {
        stats.fragmentation = 0.0f;
    }

    return stats;
}

#endif // MEMORY_POOL_SM61_CUH