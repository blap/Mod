"""
GLM-4.7 Sliding Window Attention Implementation

This module implements sliding window attention for the GLM-4.7 model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.inference_pio.common.attention.base_attention import BaseAttention
from ..config import GLM47FlashConfig


class GLM47SlidingWindowAttention(BaseAttention):
    """
    Sliding window attention implementation for GLM-4.7 model.

    This implementation limits the attention span to a fixed-size window,
    improving efficiency for long sequences while maintaining good performance.
    """

    def __init__(
        self,
        config: GLM47FlashConfig,
        layer_idx: Optional[int] = None,
        window_size: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.window_size = window_size

        # Set up attention parameters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
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
        Forward pass for sliding window attention.

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
        ).transpose(
            1, 2
        )  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, num_heads, q_len, head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, num_heads, q_len, head_dim)

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None:
            from .glm47_rotary_embeddings import (
                GLM47RotaryEmbedding,
                apply_rotary_pos_emb,
            )

            # Initialize rotary embeddings if not already done
            if not hasattr(self, "rotary_emb"):
                self.rotary_emb = GLM47RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            cos, sin = self.rotary_emb(value_states, position_ids)

            # Apply rotary embeddings to query and key states
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Apply sliding window attention
        # Create a sliding window mask
        sliding_window_mask = self._create_sliding_window_mask(
            q_len, device=hidden_states.device
        )

        # Compute attention scores with sliding window
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply sliding window mask
        attn_weights = attn_weights + sliding_window_mask

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply attention to values
        attn_output = torch.matmul(
            attn_weights, value_states
        )  # (bsz, num_heads, q_len, head_dim)

        # Reshape for output
        attn_output = (
            attn_output.transpose(1, 2)  # (bsz, q_len, num_heads, head_dim)
            .contiguous()
            .view(
                bsz, q_len, self.num_heads * self.head_dim
            )  # (bsz, q_len, hidden_size)
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

    def _create_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a sliding window mask for attention.

        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on

        Returns:
            Sliding window mask
        """
        # Create a causal mask with sliding window
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

        # Fill in the sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            mask[i, start : i + 1] = 0.0

        # Expand mask for multi-head attention
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask


def create_glm47_sliding_window_attention(
    config: GLM47FlashConfig, layer_idx: Optional[int] = None, window_size: int = 4096
) -> GLM47SlidingWindowAttention:
    """
    Factory function to create GLM-4.7 Sliding Window Attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer
        window_size: Size of the sliding window

    Returns:
        GLM47SlidingWindowAttention: The GLM-4.7 Sliding Window Attention implementation
    """
    return GLM47SlidingWindowAttention(config, layer_idx, window_size)


__all__ = ["GLM47SlidingWindowAttention", "create_glm47_sliding_window_attention"]
