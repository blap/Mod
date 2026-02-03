"""
Sparse Attention Implementation for Inference-PIO System

This module provides an implementation of sparse attention mechanisms for the Inference-PIO system.
Sparse attention reduces memory usage and computation by limiting attention connections to a subset
of tokens rather than allowing full connectivity.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    """
    Sparse Attention implementation with optimizations for memory efficiency.
    This attention mechanism limits attention connections to a sparse pattern,
    significantly reducing memory usage and computation for long sequences.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: Optional[int] = None,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
        sparse_pattern: str = "longformer",  # Options: "longformer", "bigbird", "block_sparse", "local", "random"
        sparsity_ratio: float = 0.25,
        block_size: int = 64,
        local_window_size: int = 128,
        use_global_attention: bool = True,
        global_attention_indices: Optional[list] = None,
    ):
        """
        Initialize Sparse Attention.

        Args:
            config: Model configuration
            layer_idx: Index of the transformer layer
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
            sparse_pattern: Type of sparse attention pattern
            sparsity_ratio: Ratio of attention connections to keep (0.0 to 1.0)
            block_size: Size of blocks for block sparse attention
            local_window_size: Size of local attention window
            use_global_attention: Whether to use global attention tokens
            global_attention_indices: Indices of tokens that attend globally
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        self.sparse_pattern = sparse_pattern
        self.sparsity_ratio = sparsity_ratio
        self.block_size = block_size
        self.local_window_size = local_window_size
        self.use_global_attention = use_global_attention
        self.global_attention_indices = global_attention_indices or [
            0
        ]  # Default to first token as global

        # Calculate dimensions
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads

        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size: {self.hidden_size}, num_attention_heads: {self.num_attention_heads})"
            )

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=bias
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to model specifications."""
        # Initialize query, key, and value projections
        std = self.hidden_size**-0.5
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std)

        # Initialize biases if present
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass for Sparse Attention.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position IDs
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            cache_position: Cache position IDs

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Project query, key, and value
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Create sparse attention mask based on pattern
        sparse_mask = self._create_sparse_attention_mask(
            q_len, device=hidden_states.device
        )

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (
            self.head_dim**0.5
        )

        # Apply sparse mask to attention weights
        attn_weights.masked_fill_(sparse_mask == 0, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if configured
        if self.attention_dropout > 0.0:
            attn_weights = F.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, self.num_attention_heads * self.head_dim)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

    def _create_sparse_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create sparse attention mask based on the specified pattern.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Sparse attention mask tensor of shape (1, 1, seq_len, seq_len)
        """
        # Create base mask based on pattern
        if self.sparse_pattern == "longformer":
            mask = self._create_longformer_mask(seq_len, device)
        elif self.sparse_pattern == "bigbird":
            mask = self._create_bigbird_mask(seq_len, device)
        elif self.sparse_pattern == "block_sparse":
            mask = self._create_block_sparse_mask(seq_len, device)
        elif self.sparse_pattern == "local":
            mask = self._create_local_attention_mask(seq_len, device)
        elif self.sparse_pattern == "random":
            mask = self._create_random_sparse_mask(seq_len, device)
        else:
            # Default to local attention
            mask = self._create_local_attention_mask(seq_len, device)

        # Apply causal mask if needed
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask.masked_fill_(causal_mask, 0)

        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)

    def _create_longformer_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create Longformer-style sparse attention mask with local and global attention.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Longformer sparse attention mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start:end] = True

        # Global attention for specified indices
        if self.use_global_attention:
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < seq_len:
                    mask[global_idx, :] = True  # Global query
                    mask[:, global_idx] = True  # Global key/value

        return mask

    def _create_bigbird_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create BigBird-style sparse attention mask with random, local, and global attention.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            BigBird sparse attention mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start:end] = True

        # Global attention for specified indices
        if self.use_global_attention:
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < seq_len:
                    mask[global_idx, :] = True  # Global query
                    mask[:, global_idx] = True  # Global key/value

        # Random attention connections based on sparsity ratio
        num_random_connections = int(seq_len * self.sparsity_ratio)
        for i in range(seq_len):
            random_indices = torch.randperm(seq_len, device=device)[
                :num_random_connections
            ]
            mask[i, random_indices] = True

        return mask

    def _create_block_sparse_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create block sparse attention mask.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Block sparse attention mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Calculate number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Determine which blocks to connect based on sparsity ratio
        num_connected_blocks = max(1, int(num_blocks * self.sparsity_ratio))

        for i in range(seq_len):
            block_idx = i // self.block_size
            # Connect to current block and nearby blocks
            for j in range(
                max(0, block_idx - num_connected_blocks // 2),
                min(num_blocks, block_idx + num_connected_blocks // 2 + 1),
            ):
                start_idx = j * self.block_size
                end_idx = min(seq_len, (j + 1) * self.block_size)
                mask[i, start_idx:end_idx] = True

        return mask

    def _create_local_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create local attention mask with sliding window.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Local attention mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start:end] = True

        return mask

    def _create_random_sparse_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create random sparse attention mask.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Random sparse attention mask
        """
        mask = torch.rand(seq_len, seq_len, device=device) < self.sparsity_ratio
        return mask


def create_sparse_attention(
    config: Any, layer_idx: Optional[int] = None, sparse_pattern: str = "longformer"
):
    """
    Factory function to create sparse attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)
        sparse_pattern: Type of sparse attention pattern

    Returns:
        SparseAttention: The sparse attention implementation
    """
    return SparseAttention(
        config=config,
        layer_idx=layer_idx,
        num_attention_heads=config.num_attention_heads,
        attention_dropout=getattr(config, "attention_dropout_prob", 0.0),
        bias=not getattr(config, "remove_bias_in_attention", False),
        is_causal=getattr(config, "is_causal", True),
        sparse_pattern=sparse_pattern,
        sparsity_ratio=getattr(config, "sparse_attention_sparsity_ratio", 0.25),
        block_size=getattr(config, "sparse_attention_block_size", 64),
        local_window_size=getattr(config, "sparse_attention_local_window_size", 128),
        use_global_attention=getattr(config, "use_global_attention", True),
        global_attention_indices=getattr(config, "global_attention_indices", [0]),
    )


def get_sparse_attention_class():
    """
    Get the SparseAttention class for dynamic instantiation.

    Returns:
        SparseAttention: The SparseAttention class
    """
    return SparseAttention


__all__ = ["SparseAttention", "create_sparse_attention", "get_sparse_attention_class"]
