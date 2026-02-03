"""
GLM-4.7 Paged KV Cache Implementation

This module implements a vLLM-style paged KV cache for efficient memory management
in the GLM-4.7 model. It follows the vLLM approach to divide the KV cache into
fixed-size pages managed by a page table.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Represents a single page in the paged KV cache."""

    page_id: int
    block_id: int
    is_allocated: bool = False
    ref_count: int = 0
    last_accessed: int = 0


class PagedKVCache:
    """
    Paged KV cache implementation following vLLM approach for GLM-4.7 model.

    This implementation divides the KV cache into fixed-size pages and manages
    them using a page table, which helps reduce memory fragmentation and allows
    for more efficient memory usage during inference.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_num_blocks: int = 1024,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Initialize the paged KV cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_num_blocks: Maximum number of blocks to allocate
            block_size: Size of each block/page
            dtype: Data type for tensors
            device: Device to store tensors on
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Initialize the physical cache blocks
        self.key_cache = torch.empty(
            (num_layers, max_num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.empty(
            (num_layers, max_num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Initialize page table and free list
        self.pages = [Page(i, i, False) for i in range(max_num_blocks)]
        self.free_list = list(range(max_num_blocks))
        self.logical_to_physical = {}  # Maps logical block IDs to physical block IDs
        self.access_counter = 0

        logger.info(
            f"Initialized PagedKVCache with {max_num_blocks} blocks of size {block_size}"
        )

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """
        Allocate physical blocks for the KV cache.

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated physical block IDs
        """
        if len(self.free_list) < num_blocks:
            raise RuntimeError(
                f"Not enough free blocks. Requested: {num_blocks}, Available: {len(self.free_list)}"
            )

        allocated_blocks = []
        for _ in range(num_blocks):
            block_id = self.free_list.pop()
            page = self.pages[block_id]
            page.is_allocated = True
            page.ref_count = 1
            allocated_blocks.append(block_id)

        return allocated_blocks

    def free_blocks(self, block_ids: List[int]):
        """
        Free previously allocated blocks.

        Args:
            block_ids: List of physical block IDs to free
        """
        for block_id in block_ids:
            page = self.pages[block_id]
            if page.ref_count > 0:
                page.ref_count -= 1
                if page.ref_count == 0:
                    page.is_allocated = False
                    self.free_list.append(block_id)

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        block_tables: List[List[int]],
        seq_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new keys and values to the paged KV cache.

        Args:
            key: New key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            value: New value tensor of shape (batch_size, seq_len, num_heads, head_dim)
            layer_idx: Index of the transformer layer
            block_tables: List of block tables for each sequence
            seq_lens: Length of each sequence

        Returns:
            Tuple of (cached_keys, cached_values) tensors
        """
        batch_size, seq_len, num_heads, head_dim = key.shape

        # Split sequences into blocks based on block_size
        for batch_idx in range(batch_size):
            seq_len_for_batch = seq_len
            num_blocks_needed = (
                seq_len_for_batch + self.block_size - 1
            ) // self.block_size

            # Allocate blocks if needed
            if len(block_tables[batch_idx]) < num_blocks_needed:
                new_blocks = self.allocate_blocks(
                    num_blocks_needed - len(block_tables[batch_idx])
                )
                block_tables[batch_idx].extend(new_blocks)

            # Copy key and value to the appropriate blocks
            for block_idx in range(num_blocks_needed):
                start_idx = block_idx * self.block_size
                end_idx = min(start_idx + self.block_size, seq_len_for_batch)

                physical_block_id = block_tables[batch_idx][block_idx]

                # Copy to key cache
                self.key_cache[
                    layer_idx, physical_block_id, 0 : (end_idx - start_idx), :, :
                ] = key[batch_idx, start_idx:end_idx, :, :]

                # Copy to value cache
                self.value_cache[
                    layer_idx, physical_block_id, 0 : (end_idx - start_idx), :, :
                ] = value[batch_idx, start_idx:end_idx, :, :]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_kv_cache(
        self, layer_idx: int, block_tables: List[List[int]], seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the KV cache for the given sequences.

        Args:
            layer_idx: Index of the transformer layer
            block_tables: List of block tables for each sequence
            seq_lens: Length of each sequence

        Returns:
            Tuple of (keys, values) tensors
        """
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens) if seq_lens else 0

        if max_seq_len == 0:
            # Return empty tensors if no sequences
            return (
                torch.empty(
                    0,
                    0,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.empty(
                    0,
                    0,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                ),
            )

        # Pre-allocate result tensors
        keys = torch.empty(
            batch_size,
            max_seq_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        values = torch.empty(
            batch_size,
            max_seq_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Fill the result tensors from the paged cache
        for batch_idx in range(batch_size):
            seq_len = seq_lens[batch_idx]
            num_blocks = (seq_len + self.block_size - 1) // self.block_size

            pos_in_seq = 0
            for block_idx in range(num_blocks):
                physical_block_id = block_tables[batch_idx][block_idx]

                # Calculate how many positions to copy from this block
                remaining_positions = seq_len - pos_in_seq
                positions_in_block = min(remaining_positions, self.block_size)

                # Copy from the physical block to the result tensor
                keys[batch_idx, pos_in_seq : pos_in_seq + positions_in_block, :, :] = (
                    self.key_cache[
                        layer_idx, physical_block_id, :positions_in_block, :, :
                    ]
                )
                values[
                    batch_idx, pos_in_seq : pos_in_seq + positions_in_block, :, :
                ] = self.value_cache[
                    layer_idx, physical_block_id, :positions_in_block, :, :
                ]

                pos_in_seq += positions_in_block

        return keys, values

    def reset(self):
        """Reset the cache by freeing all allocated blocks."""
        # Reset all pages to unallocated state
        for page in self.pages:
            page.is_allocated = False
            page.ref_count = 0
            page.last_accessed = 0

        # Reset free list to include all blocks
        self.free_list = list(range(self.max_num_blocks))

        logger.info("PagedKVCache reset")


class GLM47PagedAttentionCore(nn.Module):
    """
    Core paged attention implementation for GLM-4.7 model.

    This module handles the core attention computation using paged KV cache.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_num_blocks: int = 1024,
        sliding_window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.sliding_window_size = sliding_window_size
        self.dtype = dtype
        self.device = device

        # Initialize the paged KV cache
        self.kv_cache = PagedKVCache(
            num_layers=1,  # We'll handle multiple layers at the model level
            num_heads=num_heads,
            head_dim=head_dim,
            max_num_blocks=max_num_blocks,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_tables: List[List[int]],
        seq_lens: List[int],
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for paged attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            key: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            value: Value tensor of shape (batch_size, seq_len, num_heads, head_dim)
            block_tables: List of block tables for each sequence
            seq_lens: Length of each sequence
            layer_idx: Index of the transformer layer
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings

        Returns:
            Output tensor of shape (batch_size, seq_len, num_heads, head_dim)
        """
        batch_size, seq_len, num_heads, head_dim = query.shape

        # Append new keys and values to the paged cache
        self.kv_cache.append(key, value, layer_idx, block_tables, seq_lens)

        # Retrieve the full KV cache for attention computation
        cached_keys, cached_values = self.kv_cache.get_kv_cache(
            layer_idx, block_tables, seq_lens
        )

        # Compute attention scores
        # query: (batch_size, seq_len, num_heads, head_dim)
        # cached_keys: (batch_size, cached_seq_len, num_heads, head_dim)
        # Result: (batch_size, num_heads, seq_len, cached_seq_len)
        attn_weights = torch.matmul(
            query.transpose(1, 2),  # (batch_size, num_heads, seq_len, head_dim)
            cached_keys.transpose(1, 2).transpose(
                -1, -2
            ),  # (batch_size, num_heads, head_dim, cached_seq_len)
        ) / (self.head_dim**0.5)

        # Apply sliding window attention if configured
        if self.sliding_window_size is not None:
            attn_weights = self._apply_sliding_window(
                attn_weights, seq_len, cached_keys.size(1)
            )

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )

        # Apply attention to values
        # attn_weights: (batch_size, num_heads, seq_len, cached_seq_len)
        # cached_values: (batch_size, cached_seq_len, num_heads, head_dim)
        # Result: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(
            attn_weights,
            cached_values.transpose(
                1, 2
            ),  # (batch_size, num_heads, cached_seq_len, head_dim)
        )

        # Transpose back to (batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2)

        return output

    def _apply_sliding_window(
        self, attn_weights: torch.Tensor, query_seq_len: int, key_seq_len: int
    ) -> torch.Tensor:
        """
        Apply sliding window attention mask.

        Args:
            attn_weights: Attention weights tensor
            query_seq_len: Length of query sequence
            key_seq_len: Length of key sequence

        Returns:
            Attention weights with sliding window applied
        """
        # Create a sliding window mask
        batch_size, num_heads, q_len, k_len = attn_weights.shape

        # Create a mask that zeros out attention outside the sliding window
        # For each position i in the query, only attend to positions [i - window, i + window]
        window_mask = torch.ones_like(attn_weights, dtype=torch.bool)

        for i in range(q_len):
            start_idx = max(0, i + k_len - q_len - self.sliding_window_size // 2)
            end_idx = min(k_len, i + k_len - q_len + self.sliding_window_size // 2 + 1)

            # Zero out positions outside the window
            window_mask[:, :, i, :start_idx] = False
            if end_idx < k_len:
                window_mask[:, :, i, end_idx:] = False

        # Apply the mask by setting masked positions to negative infinity
        attn_weights = attn_weights.masked_fill(
            ~window_mask, torch.finfo(attn_weights.dtype).min
        )

        return attn_weights


def create_paged_kv_cache(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_num_blocks: int = 1024,
    block_size: int = 16,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> PagedKVCache:
    """
    Factory function to create a paged KV cache.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        max_num_blocks: Maximum number of blocks to allocate
        block_size: Size of each block
        dtype: Data type for tensors
        device: Device to store tensors on

    Returns:
        PagedKVCache instance
    """
    return PagedKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_num_blocks=max_num_blocks,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )


__all__ = ["PagedKVCache", "GLM47PagedAttentionCore", "create_paged_kv_cache", "Page"]
