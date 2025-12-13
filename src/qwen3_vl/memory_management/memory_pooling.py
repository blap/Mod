"""
Memory Pooling System for Qwen3-VL Model

This module implements memory pooling classes that are expected by other modules.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryPool:
    """
    Memory pool for efficient tensor allocation and deallocation.
    """
    def __init__(self, pool_size: int = 1024*1024*1024):  # 1GB default
        self.pool_size = pool_size
        self.allocated_tensors: Dict[str, torch.Tensor] = {}
        self.free_blocks: List[tuple] = [(0, pool_size)]  # (start, size) of free blocks
        self.tensor_metadata: Dict[str, Dict[str, Any]] = {}
        
    def allocate_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        """
        Allocate a tensor from the memory pool.
        """
        # Calculate tensor size correctly: number of elements * size per element
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor([], dtype=dtype).element_size()
        tensor_size = num_elements * element_size

        # Find a suitable free block
        for i, (start, size) in enumerate(self.free_blocks):
            if size >= tensor_size:
                # Allocate from this block
                self.free_blocks.pop(i)

                # Create tensor
                tensor = torch.empty(shape, dtype=dtype, device=device)

                # Store allocation info
                tensor_id = id(tensor)
                self.allocated_tensors[str(tensor_id)] = tensor
                self.tensor_metadata[str(tensor_id)] = {
                    'shape': shape,
                    'dtype': dtype,
                    'device': device,
                    'size': tensor_size,
                    'start_addr': start
                }

                # Add remaining space back to free blocks if any
                if size > tensor_size:
                    self.free_blocks.append((start + tensor_size, size - tensor_size))

                return tensor

        # If no suitable block found, create tensor normally (outside pool)
        logger.warning(f"No suitable block found in pool for tensor of size {tensor_size}, creating outside pool")
        return torch.empty(shape, dtype=dtype, device=device)
    
    def deallocate_tensor(self, tensor_id: str):
        """
        Deallocate a tensor back to the memory pool.
        """
        if tensor_id in self.allocated_tensors:
            tensor = self.allocated_tensors[tensor_id]
            metadata = self.tensor_metadata[tensor_id]
            
            # Add the block back to free blocks
            start_addr = metadata['start_addr']
            size = metadata['size']
            self.free_blocks.append((start_addr, size))
            
            # Sort free blocks by start address
            self.free_blocks.sort(key=lambda x: x[0])
            
            # Clean up
            del self.allocated_tensors[tensor_id]
            del self.tensor_metadata[tensor_id]
    
    def get_pool_stats(self):
        """
        Get memory pool statistics.
        """
        allocated_size = sum(meta['size'] for meta in self.tensor_metadata.values())
        total_free = sum(size for _, size in self.free_blocks)
        
        return {
            'pool_size': self.pool_size,
            'allocated_size': allocated_size,
            'free_size': total_free,
            'utilization': allocated_size / self.pool_size if self.pool_size > 0 else 0,
            'num_allocated_tensors': len(self.allocated_tensors),
            'num_free_blocks': len(self.free_blocks)
        }

    def defragment(self):
        """
        Defragment the memory pool by combining adjacent free blocks.
        """
        # Sort free blocks by start address
        self.free_blocks.sort(key=lambda x: x[0])

        if not self.free_blocks:
            return

        # Combine adjacent blocks
        new_free_blocks = []
        current_start, current_size = self.free_blocks[0]

        for start, size in self.free_blocks[1:]:
            if current_start + current_size == start:
                # Adjacent blocks, merge them
                current_size += size
            else:
                # Non-adjacent block, add previous merged block to result
                new_free_blocks.append((current_start, current_size))
                current_start, current_size = start, size

        # Add the last block
        new_free_blocks.append((current_start, current_size))

        self.free_blocks = new_free_blocks


class BuddyAllocator:
    """
    Buddy memory allocation system for efficient memory management.
    """
    def __init__(self, total_size: int, min_block_size: int = 256):
        self.total_size = self._round_up_to_power_of_2(total_size)
        self.min_block_size = self._round_up_to_power_of_2(min_block_size)
        self.levels = int(self.total_size / self.min_block_size).bit_length()
        
        # Initialize free lists for each level
        self.free_lists = {i: [] for i in range(self.levels)}
        
        # Initially, the entire memory is one free block at the highest level
        self.free_lists[self.levels - 1].append((0, self.total_size))
        
        self.allocated_blocks = {}
        
    def _round_up_to_power_of_2(self, x: int) -> int:
        """Round up to the next power of 2."""
        if x == 0:
            return 1
        return 1 << (x - 1).bit_length()
    
    def _get_buddy(self, addr: int, size: int) -> int:
        """Get the buddy block address."""
        return addr ^ size
    
    def allocate(self, size: int) -> Optional[tuple]:
        """
        Allocate a block of memory of at least the requested size.
        Returns (address, actual_size) or None if allocation fails.
        """
        # Round up size to the next power of 2 and at least min_block_size
        alloc_size = self._round_up_to_power_of_2(max(size, self.min_block_size))
        level = int((alloc_size / self.min_block_size).bit_length()) - 1
        
        # Find a free block at or above the required level
        for current_level in range(level, self.levels):
            if self.free_lists[current_level]:
                # Found a block, allocate it
                addr, block_size = self.free_lists[current_level].pop()
                
                # Split the block down to the required size
                while current_level > level:
                    block_size //= 2
                    buddy_addr = addr + block_size
                    self.free_lists[current_level - 1].append((buddy_addr, block_size))
                    current_level -= 1
                
                self.allocated_blocks[addr] = (block_size, current_level)
                return addr, block_size
        
        return None
    
    def deallocate(self, addr: int, size: int):
        """
        Deallocate a previously allocated block.
        """
        if addr not in self.allocated_blocks:
            raise ValueError(f"Address {addr} not allocated")
        
        block_size, level = self.allocated_blocks[addr]
        if block_size != size:
            raise ValueError(f"Size mismatch: allocated {block_size}, deallocated {size}")
        
        # Add to free list and try to merge with buddies
        self._add_to_free_list(addr, level)
        
        # Clean up
        del self.allocated_blocks[addr]
    
    def _add_to_free_list(self, addr: int, level: int):
        """Add a block to the free list and try to merge with buddies."""
        size = self.min_block_size << level
        
        while level < self.levels - 1:
            buddy_addr = self._get_buddy(addr, size)
            
            # Check if buddy is free at the same level
            if (buddy_addr, size) in self.free_lists[level]:
                # Remove buddy from free list
                self.free_lists[level].remove((buddy_addr, size))
                
                # Merge with buddy (use the lower address)
                if addr > buddy_addr:
                    addr = buddy_addr
                
                size *= 2
                level += 1
            else:
                # Buddy not free, add current block to this level
                self.free_lists[level].append((addr, size))
                break
        else:
            # If we reached the highest level, add the block
            self.free_lists[level].append((addr, size))
    
    def get_stats(self):
        """Get buddy allocator statistics."""
        total_allocated = sum(size for size, _ in self.allocated_blocks.values())
        total_free = sum(size for level_list in self.free_lists.values() for _, size in level_list)
        
        return {
            'total_size': self.total_size,
            'total_allocated': total_allocated,
            'total_free': total_free,
            'utilization': total_allocated / self.total_size if self.total_size > 0 else 0,
            'num_allocated_blocks': len(self.allocated_blocks),
            'num_free_blocks': sum(len(blocks) for blocks in self.free_lists.values())
        }


class TensorCache:
    """
    Tensor cache for frequently used tensors.
    """
    def __init__(self, cache_size: int = 1024*1024*256):  # 256MB default
        self.cache_size = cache_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_order: List[str] = []  # For LRU
        self.tensor_sizes: Dict[str, int] = {}
        self.current_size = 0
        
    def put(self, key: str, tensor: torch.Tensor):
        """
        Put a tensor in the cache.
        """
        tensor_size = tensor.element_size() * tensor.nelement()
        
        if key in self.cache:
            # Update existing tensor
            old_size = self.tensor_sizes[key]
            self.current_size -= old_size
            self.cache_order.remove(key)
        elif self.current_size + tensor_size > self.cache_size:
            # Evict LRU items until we have enough space
            while self.cache_order and self.current_size + tensor_size > self.cache_size:
                lru_key = self.cache_order.pop(0)
                lru_size = self.tensor_sizes[lru_key]
                del self.cache[lru_key]
                del self.tensor_sizes[lru_key]
                self.current_size -= lru_size
        
        # Add new tensor
        self.cache[key] = tensor
        self.cache_order.append(key)
        self.tensor_sizes[key] = tensor_size
        self.current_size += tensor_size
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get a tensor from the cache.
        """
        if key in self.cache:
            # Move to end of list (most recently used)
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.cache[key]
        return None
    
    def evict(self, key: str):
        """
        Evict a tensor from the cache.
        """
        if key in self.cache:
            size = self.tensor_sizes[key]
            del self.cache[key]
            del self.tensor_sizes[key]
            self.cache_order.remove(key)
            self.current_size -= size
    
    def clear(self):
        """
        Clear the entire cache.
        """
        self.cache.clear()
        self.cache_order.clear()
        self.tensor_sizes.clear()
        self.current_size = 0
    
    def get_stats(self):
        """
        Get cache statistics.
        """
        return {
            'cache_size': self.cache_size,
            'current_size': self.current_size,
            'utilization': self.current_size / self.cache_size if self.cache_size > 0 else 0,
            'num_cached_tensors': len(self.cache)
        }


class PooledLinear(nn.Module):
    """
    Linear layer that uses memory pooling for efficient tensor allocation.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, memory_pool: Optional[MemoryPool] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.memory_pool = memory_pool or MemoryPool()
        
        # Allocate weight using memory pool
        self.weight = nn.Parameter(
            self.memory_pool.allocate_tensor((out_features, in_features), dtype=torch.float32)
        )
        
        if bias:
            self.bias = nn.Parameter(
                self.memory_pool.allocate_tensor((out_features,), dtype=torch.float32)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)


class PooledMLP(nn.Module):
    """
    MLP layer that uses memory pooling for efficient tensor allocation.
    """
    def __init__(self, intermediate_size: int, hidden_size: int, memory_pool: Optional[MemoryPool] = None):
        super().__init__()
        self.memory_pool = memory_pool or MemoryPool()
        
        self.fc1 = PooledLinear(hidden_size, intermediate_size, memory_pool=self.memory_pool)
        self.fc2 = PooledLinear(intermediate_size, hidden_size, memory_pool=self.memory_pool)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PooledAttention(nn.Module):
    """
    Attention layer that uses memory pooling for efficient tensor allocation.
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, memory_pool: Optional[MemoryPool] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.memory_pool = memory_pool or MemoryPool()
        
        self.query = PooledLinear(hidden_size, self.all_head_size, memory_pool=self.memory_pool)
        self.key = PooledLinear(hidden_size, self.all_head_size, memory_pool=self.memory_pool)
        self.value = PooledLinear(hidden_size, self.all_head_size, memory_pool=self.memory_pool)
        
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class PooledTransformerLayer(nn.Module):
    """
    Transformer layer that uses memory pooling for efficient tensor allocation.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int, memory_pool: Optional[MemoryPool] = None):
        super().__init__()
        self.memory_pool = memory_pool or MemoryPool()
        
        self.attention = PooledAttention(hidden_size, num_attention_heads, memory_pool=self.memory_pool)
        self.mlp = PooledMLP(intermediate_size, hidden_size, memory_pool=self.memory_pool)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention sublayer
        attention_output = self.attention(hidden_states)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        # MLP sublayer
        mlp_output = self.mlp(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + mlp_output)
        
        return hidden_states


# Global memory pool instance
_global_memory_pool = MemoryPool()

def get_global_memory_pool():
    """Get the global memory pool instance."""
    return _global_memory_pool

def set_global_memory_pool(pool: MemoryPool):
    """Set the global memory pool instance."""
    global _global_memory_pool
    _global_memory_pool = pool

def get_memory_pool():
    """Get memory pool (alias for global memory pool)."""
    return get_global_memory_pool()