#ifndef MEMORY_OPTIMIZATION_SM61_CUH
#define MEMORY_OPTIMIZATION_SM61_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Memory optimization structures and functions for SM61 architecture
struct MemoryPoolStats {
    size_t total_size;
    size_t allocated;
    size_t free;
    float fragmentation;
    int num_free_blocks;
};

class SM61MemoryPool {
private:
    char* memory_pool;
    size_t pool_size;
    bool* block_allocated;  // Track which blocks are allocated
    size_t* block_sizes;    // Track sizes of allocated blocks
    size_t num_blocks;
    size_t block_size;

public:
    SM61MemoryPool(size_t size = 64 * 1024 * 1024) : pool_size(size) {
        // Initialize with a reasonable block size
        block_size = 1024;  // 1KB blocks
        num_blocks = pool_size / block_size;
        
        // Allocate the memory pool
        cudaMalloc(&memory_pool, pool_size);
        
        // Allocate tracking arrays
        block_allocated = new bool[num_blocks]();
        block_sizes = new size_t[num_blocks]();
    }

    ~SM61MemoryPool() {
        if (memory_pool) {
            cudaFree(memory_pool);
        }
        delete[] block_allocated;
        delete[] block_sizes;
    }

    void* allocate(size_t size) {
        // Find a free block that can accommodate the requested size
        size_t blocks_needed = (size + block_size - 1) / block_size;
        
        for (size_t i = 0; i < num_blocks; i++) {
            if (can_allocate_at(i, blocks_needed)) {
                // Mark blocks as allocated
                for (size_t j = i; j < i + blocks_needed && j < num_blocks; j++) {
                    block_allocated[j] = true;
                }
                block_sizes[i] = size;  // Store original requested size
                
                // Return pointer to the allocated block
                return memory_pool + (i * block_size);
            }
        }
        
        // Allocation failed
        return nullptr;
    }

    void deallocate(void* ptr, size_t size) {
        if (!ptr) return;
        
        // Calculate which block this pointer corresponds to
        size_t offset = (char*)ptr - memory_pool;
        size_t block_idx = offset / block_size;
        
        // Mark blocks as free
        size_t blocks_to_free = (size + block_size - 1) / block_size;
        for (size_t i = block_idx; i < block_idx + blocks_to_free && i < num_blocks; i++) {
            if (i < num_blocks) {
                block_allocated[i] = false;
                block_sizes[i] = 0;
            }
        }
    }

    MemoryPoolStats get_stats() {
        size_t allocated_bytes = 0;
        int free_blocks = 0;
        
        for (size_t i = 0; i < num_blocks; i++) {
            if (block_allocated[i]) {
                allocated_bytes += block_size;
            } else {
                free_blocks++;
            }
        }
        
        MemoryPoolStats stats;
        stats.total_size = pool_size;
        stats.allocated = allocated_bytes;
        stats.free = pool_size - allocated_bytes;
        stats.fragmentation = (float)free_blocks / num_blocks;
        stats.num_free_blocks = free_blocks;
        
        return stats;
    }

private:
    bool can_allocate_at(size_t start_idx, size_t blocks_needed) {
        if (start_idx + blocks_needed > num_blocks) return false;
        
        for (size_t i = start_idx; i < start_idx + blocks_needed; i++) {
            if (block_allocated[i]) return false;
        }
        return true;
    }
};

#endif // MEMORY_OPTIMIZATION_SM61_CUH