"""
Paged KV Cache Implementation for PaddleOCR-VL-1.5

This module implements a Paged Key-Value Cache similar to vLLM, allowing for
non-contiguous memory allocation and dynamic block management.
"""

import torch
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BlockTable:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: List[int] = []

    def add_block(self, block_id: int):
        self.blocks.append(block_id)

    def get_physical_block_id(self, logical_page_idx: int) -> int:
        if logical_page_idx < len(self.blocks):
            return self.blocks[logical_page_idx]
        raise IndexError(f"Logical page index {logical_page_idx} out of bounds")

class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_num_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.dtype = dtype
        self.device = device

        # [num_layers, 2, max_num_blocks, block_size, num_heads, head_dim]
        # 2 represents Key and Value
        # Optimization: Store as [num_layers, 2, max_num_blocks, num_heads, head_dim, block_size] ?
        # Standard layout usually: [num_blocks, num_heads, head_dim/block_size, block_size]
        # Let's stick to a simple layout for this implementation:
        # K_cache: [num_layers, max_num_blocks, num_heads, head_dim, block_size]
        # V_cache: [num_layers, max_num_blocks, num_heads, head_dim, block_size]
        # Note: block_size at the end facilitates vectorization if needed,
        # but standard attention often wants [batch, heads, seq, dim].
        # For Paged Attention, we usually do [num_blocks, block_size, num_heads, head_dim]

        self.k_cache = [
            torch.zeros(
                (max_num_blocks, num_heads, block_size, head_dim),
                dtype=dtype,
                device=device
            ) for _ in range(num_layers)
        ]

        self.v_cache = [
            torch.zeros(
                (max_num_blocks, num_heads, block_size, head_dim),
                dtype=dtype,
                device=device
            ) for _ in range(num_layers)
        ]

        self.free_blocks = list(range(max_num_blocks))
        self.sequence_block_tables: Dict[int, BlockTable] = {} # seq_id -> BlockTable

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of memory: No free blocks available in KV Cache")
        return self.free_blocks.pop(0)

    def free_block(self, block_id: int):
        self.free_blocks.append(block_id)

    def initialize_sequence(self, seq_id: int):
        if seq_id in self.sequence_block_tables:
            logger.warning(f"Sequence {seq_id} already initialized. Resetting.")
            self.free_sequence(seq_id)

        block_table = BlockTable(self.block_size)
        # Allocate initial block
        first_block = self.allocate_block()
        block_table.add_block(first_block)
        self.sequence_block_tables[seq_id] = block_table

    def free_sequence(self, seq_id: int):
        if seq_id not in self.sequence_block_tables:
            return

        block_table = self.sequence_block_tables[seq_id]
        for block_id in block_table.blocks:
            self.free_block(block_id)
        del self.sequence_block_tables[seq_id]

    def append_token_kv(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor, token_position: int):
        """
        Append a single token's K and V to the cache for a specific sequence.
        k, v shape: [num_heads, head_dim]
        """
        if seq_id not in self.sequence_block_tables:
            raise ValueError(f"Sequence {seq_id} not initialized")

        block_table = self.sequence_block_tables[seq_id]
        logical_block_idx = token_position // self.block_size
        offset_in_block = token_position % self.block_size

        # Check if we need to allocate a new block
        if logical_block_idx >= len(block_table.blocks):
             new_block = self.allocate_block()
             block_table.add_block(new_block)

        physical_block_id = block_table.get_physical_block_id(logical_block_idx)

        # Write to cache
        # shape of k_cache[layer]: [max_blocks, num_heads, block_size, head_dim]
        self.k_cache[layer_idx][physical_block_id, :, offset_in_block, :] = k
        self.v_cache[layer_idx][physical_block_id, :, offset_in_block, :] = v

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_block_table_tensor(self, seq_id: int, max_len: int) -> torch.Tensor:
        """Returns block table as tensor for kernel usage"""
        if seq_id not in self.sequence_block_tables:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        blocks = self.sequence_block_tables[seq_id].blocks
        # Pad if necessary or just return used blocks
        return torch.tensor(blocks, dtype=torch.int32, device=self.device)
