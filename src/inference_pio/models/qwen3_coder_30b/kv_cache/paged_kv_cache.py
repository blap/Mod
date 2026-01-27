"""
Qwen3-Coder-30B Paged KV Cache Implementation

This module implements a vLLM-style paged KV cache for efficient memory management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

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
    Paged KV cache implementation for Qwen3-Coder-30B model.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_num_blocks: int = 2048,  # Larger default for 30B model
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        self.key_cache = torch.empty(
            (num_layers, max_num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.value_cache = torch.empty(
            (num_layers, max_num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

        self.pages = [Page(i, i, False) for i in range(max_num_blocks)]
        self.free_list = list(range(max_num_blocks))

        logger.info(f"Initialized Qwen3-Coder PagedKVCache with {max_num_blocks} blocks")

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        if len(self.free_list) < num_blocks:
            raise RuntimeError(f"Not enough free blocks. Requested: {num_blocks}, Available: {len(self.free_list)}")

        allocated_blocks = []
        for _ in range(num_blocks):
            block_id = self.free_list.pop()
            page = self.pages[block_id]
            page.is_allocated = True
            page.ref_count = 1
            allocated_blocks.append(block_id)
        return allocated_blocks

    def free_blocks(self, block_ids: List[int]):
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
        seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_heads, head_dim = key.shape

        for batch_idx in range(batch_size):
            current_total_len = seq_lens[batch_idx]
            num_new_tokens = seq_len
            start_pos = current_total_len - num_new_tokens

            tokens_processed = 0
            while tokens_processed < num_new_tokens:
                logical_pos = start_pos + tokens_processed
                block_idx = logical_pos // self.block_size
                block_offset = logical_pos % self.block_size

                if block_idx >= len(block_tables[batch_idx]):
                     raise IndexError(f"Block table too short for sequence length {current_total_len}")

                physical_block_id = block_tables[batch_idx][block_idx]
                num_to_copy = min(num_new_tokens - tokens_processed, self.block_size - block_offset)

                self.key_cache[
                    layer_idx,
                    physical_block_id,
                    block_offset : block_offset + num_to_copy
                ] = key[batch_idx, tokens_processed : tokens_processed + num_to_copy]

                self.value_cache[
                    layer_idx,
                    physical_block_id,
                    block_offset : block_offset + num_to_copy
                ] = value[batch_idx, tokens_processed : tokens_processed + num_to_copy]

                tokens_processed += num_to_copy

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_kv_cache(
        self,
        layer_idx: int,
        block_tables: List[List[int]],
        seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens) if seq_lens else 0

        if max_seq_len == 0:
            return (
                torch.empty(0, 0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device),
                torch.empty(0, 0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
            )

        keys = torch.zeros(
            batch_size, max_seq_len, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        values = torch.zeros(
            batch_size, max_seq_len, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device
        )

        for batch_idx in range(batch_size):
            seq_len = seq_lens[batch_idx]
            num_blocks = (seq_len + self.block_size - 1) // self.block_size

            pos_in_seq = 0
            for block_idx in range(num_blocks):
                physical_block_id = block_tables[batch_idx][block_idx]

                remaining_positions = seq_len - pos_in_seq
                positions_in_block = min(remaining_positions, self.block_size)

                keys[batch_idx, pos_in_seq:pos_in_seq + positions_in_block] = \
                    self.key_cache[layer_idx, physical_block_id, :positions_in_block]
                values[batch_idx, pos_in_seq:pos_in_seq + positions_in_block] = \
                    self.value_cache[layer_idx, physical_block_id, :positions_in_block]

                pos_in_seq += positions_in_block

        return keys, values

    def reset(self):
        for page in self.pages:
            page.is_allocated = False
            page.ref_count = 0
        self.free_list = list(range(self.max_num_blocks))
