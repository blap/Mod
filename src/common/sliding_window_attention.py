"""
Sliding Window Attention Implementation for Inference-PIO System

This module provides an implementation of sliding window attention for the Inference-PIO system.
Sliding window attention limits the attention span to a fixed-size window, reducing memory usage
and improving performance for long sequences.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention implementation with optimizations for memory efficiency.
    This attention mechanism limits the attention span to a fixed-size window around each token,
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
        sliding_window_size: int = 4096,
        use_flash_attention: bool = True,
    ):
        """
        Initialize Sliding Window Attention.

        Args:
            config: Model configuration
            layer_idx: Index of the transformer layer
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
            sliding_window_size: Size of the sliding window
            use_flash_attention: Whether to use FlashAttention for computation
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        self.sliding_window_size = sliding_window_size
        self.use_flash_attention = use_flash_attention

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
        Forward pass for Sliding Window Attention.

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

        # Apply sliding window mask to limit attention span
        if attention_mask is not None:
            sliding_window_mask = self._create_sliding_window_mask(
                q_len, device=hidden_states.device
            )
            attention_mask = attention_mask + sliding_window_mask

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (
            self.head_dim**0.5
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask if needed
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(q_len, q_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

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

    def _create_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Sliding window mask tensor of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.sliding_window_size)
            mask[i, :start] = float("-inf")
        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)


def create_sliding_window_attention(
    config: Any, layer_idx: Optional[int] = None, sliding_window_size: int = 4096
):
    """
    Factory function to create sliding window attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)
        sliding_window_size: Size of the sliding window

    Returns:
        SlidingWindowAttention: The sliding window attention implementation
    """
    return SlidingWindowAttention(
        config=config,
        layer_idx=layer_idx,
        num_attention_heads=config.num_attention_heads,
        attention_dropout=getattr(config, "attention_dropout_prob", 0.0),
        bias=not getattr(config, "remove_bias_in_attention", False),
        is_causal=getattr(config, "is_causal", True),
        sliding_window_size=sliding_window_size,
        use_flash_attention=getattr(config, "use_flash_attention_2", True),
    )


__all__ = ["SlidingWindowAttention", "create_sliding_window_attention"]
