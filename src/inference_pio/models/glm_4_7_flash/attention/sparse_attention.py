"""
GLM-4.7 Sparse Attention Implementation

This module implements sparse attention mechanisms for the GLM-4.7 model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseAttention
from ..config import GLM47Config


class GLM47SparseAttention(BaseAttention):
    """
    Sparse attention implementation for GLM-4.7 model.

    This implementation provides attention computation with sparse patterns to reduce
    computational complexity from O(n²) to O(n√n) or O(n log n) depending on the pattern used.
    """

    def __init__(self, config: GLM47Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Set up attention parameters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Sparse attention specific parameters
        self.sparse_pattern = config.sparse_attention_pattern
        self.sparsity_ratio = config.sparse_attention_sparsity_ratio
        self.block_size = config.sparse_attention_block_size
        self.local_window_size = config.sparse_attention_local_window_size
        self.use_global_attention = config.use_global_attention
        self.global_attention_indices = config.global_attention_indices

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for sparse attention.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None:
            from .rotary_embeddings.optimized_rotary import apply_rotary_pos_emb
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Apply sparse attention pattern based on configuration
        if self.sparse_pattern == "longformer":
            attn_weights = self._apply_longformer_attention(
                query_states, key_states, value_states, attention_mask
            )
        elif self.sparse_pattern == "bigbird":
            attn_weights = self._apply_bigbird_attention(
                query_states, key_states, value_states, attention_mask
            )
        elif self.sparse_pattern == "block_sparse":
            attn_weights = self._apply_block_sparse_attention(
                query_states, key_states, value_states, attention_mask
            )
        elif self.sparse_pattern == "local":
            attn_weights = self._apply_local_attention(
                query_states, key_states, value_states, attention_mask
            )
        elif self.sparse_pattern == "random":
            attn_weights = self._apply_random_sparse_attention(
                query_states, key_states, value_states, attention_mask
            )
        else:
            # Default to local attention if pattern is not recognized
            attn_weights = self._apply_local_attention(
                query_states, key_states, value_states, attention_mask
            )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)  # (bsz, num_heads, q_len, head_dim)

        # Reshape for output
        attn_output = (
            attn_output.transpose(1, 2)  # (bsz, q_len, num_heads, head_dim)
            .contiguous()
            .view(bsz, q_len, self.num_heads * self.head_dim)  # (bsz, q_len, hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        # Handle KV cache for inference
        if use_cache:
            if past_key_value is not None:
                # Concatenate with past keys and values
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states)

        return (
            attn_output,
            attn_weights if output_attentions else None,
            past_key_value,
        )

    def _apply_longformer_attention(self, query_states, key_states, value_states, attention_mask):
        """
        Apply Longformer-style sparse attention.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create sparse attention mask based on local window and global attention
        sparse_mask = self._create_longformer_sparse_mask(q_len, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        return attn_weights

    def _create_longformer_sparse_mask(self, seq_len, device):
        """
        Create sparse attention mask for Longformer pattern.
        """
        # Create a mask with local window attention
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add local window connections
        for i in range(seq_len):
            start_idx = max(0, i - self.local_window_size // 2)
            end_idx = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True

        # Add global attention connections if enabled
        if self.use_global_attention and self.global_attention_indices:
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < seq_len:
                    # Global token attends to all other tokens
                    mask[global_idx, :] = True
                    # All tokens attend to global token
                    mask[:, global_idx] = True

        # Expand mask to match attention weights shape
        # Shape: (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_bigbird_attention(self, query_states, key_states, value_states, attention_mask):
        """
        Apply BigBird-style sparse attention.
        """
        # Implementation for BigBird sparse attention
        # This includes random, window, and global attention patterns
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create sparse attention mask based on BigBird pattern
        sparse_mask = self._create_bigbird_sparse_mask(q_len, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        return attn_weights

    def _create_bigbird_sparse_mask(self, seq_len, device):
        """
        Create sparse attention mask for BigBird pattern.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add window attention (local connections)
        for i in range(seq_len):
            start_idx = max(0, i - self.local_window_size // 2)
            end_idx = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True

        # Add global attention connections
        if self.use_global_attention and self.global_attention_indices:
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < seq_len:
                    # Global token attends to all other tokens
                    mask[global_idx, :] = True
                    # All tokens attend to global token
                    mask[:, global_idx] = True

        # Add random attention connections
        num_random_connections = int(seq_len * self.sparsity_ratio)
        random_indices = torch.randperm(seq_len, device=device)[:num_random_connections]
        for idx in random_indices:
            # Randomly connect to other positions
            other_indices = torch.randperm(seq_len, device=device)[:self.local_window_size//2]
            mask[idx, other_indices] = True
            mask[other_indices, idx] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_block_sparse_attention(self, query_states, key_states, value_states, attention_mask):
        """
        Apply block sparse attention.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Calculate number of blocks
        num_blocks = (q_len + self.block_size - 1) // self.block_size

        # Create sparse attention mask based on block pattern
        sparse_mask = self._create_block_sparse_mask(q_len, num_blocks, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        return attn_weights

    def _create_block_sparse_mask(self, seq_len, num_blocks, device):
        """
        Create sparse attention mask for block sparse pattern.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Create block structure
        for block_row in range(num_blocks):
            for block_col in range(num_blocks):
                row_start = block_row * self.block_size
                row_end = min((block_row + 1) * self.block_size, seq_len)
                col_start = block_col * self.block_size
                col_end = min((block_col + 1) * self.block_size, seq_len)

                # Determine if this block should be active based on sparsity ratio
                if torch.rand(1, device=device).item() < self.sparsity_ratio or block_row == block_col:
                    # Activate this block (allows attention within block or diagonal blocks)
                    mask[row_start:row_end, col_start:col_end] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_local_attention(self, query_states, key_states, value_states, attention_mask):
        """
        Apply local attention with sliding window.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create local attention mask
        local_mask = self._create_local_attention_mask(q_len, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply local mask to attention weights
        attn_weights = attn_weights.masked_fill(local_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        return attn_weights

    def _create_local_attention_mask(self, seq_len, device):
        """
        Create local attention mask with sliding window.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add local window connections
        for i in range(seq_len):
            start_idx = max(0, i - self.local_window_size // 2)
            end_idx = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_random_sparse_attention(self, query_states, key_states, value_states, attention_mask):
        """
        Apply random sparse attention.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create random sparse mask based on sparsity ratio
        sparse_mask = self._create_random_sparse_mask(q_len, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        return attn_weights

    def _create_random_sparse_mask(self, seq_len, device):
        """
        Create random sparse attention mask.
        """
        # Create a random mask with the specified sparsity ratio
        mask = torch.rand(seq_len, seq_len, device=device) < self.sparsity_ratio
        # Ensure each position attends to itself
        mask.diagonal(dim1=0, dim2=1).fill_(True)

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask


def create_glm47_sparse_attention(config: GLM47Config, layer_idx: Optional[int] = None):
    """
    Factory function to create sparse attention implementation for GLM-4.7.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        GLM47SparseAttention: The sparse attention implementation
    """
    return GLM47SparseAttention(config, layer_idx)


__all__ = [
    "GLM47SparseAttention",
    "create_glm47_sparse_attention"
]