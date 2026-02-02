"""
GLM-4.7 Sliding Window Attention Implementation

This module implements sliding window attention for the GLM-4.7 model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseAttention
from ..config import GLM47Config


class GLM47SlidingWindowAttention(BaseAttention):
    """
    Sliding window attention implementation for GLM-4.7 model.

    This implementation limits the attention context to a fixed window size,
    reducing memory usage from O(n²) to O(n*w) where w is the window size.
    This is particularly valuable for processing long sequences.
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

        # Sliding window specific parameters
        self.window_size = config.sliding_window_size
        self.use_causal_mask = config.use_causal_mask
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
            from .rotary_embeddings.optimized_rotary import apply_rotary_pos_emb

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Apply sliding window attention
        attn_weights = self._apply_sliding_window_attention(
            query_states, key_states, value_states, attention_mask
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

    def _apply_sliding_window_attention(
        self, query_states, key_states, value_states, attention_mask
    ):
        """
        Apply sliding window attention mechanism.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Compute attention scores
        # For sliding window, we only compute attention within the window
        attn_weights = torch.full(
            (bsz, num_heads, q_len, q_len),
            float("-inf"),
            dtype=query_states.dtype,
            device=query_states.device,
        )

        # Apply sliding window attention
        for i in range(q_len):
            # Define the window boundaries for position i
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(q_len, i + self.window_size // 2 + 1)

            # Compute attention scores only within the window
            window_query = query_states[
                :, :, i : i + 1, :
            ]  # (bsz, num_heads, 1, head_dim)
            window_keys = key_states[
                :, :, start_idx:end_idx, :
            ]  # (bsz, num_heads, window_size, head_dim)

            # Calculate attention scores for this window
            window_attn_scores = torch.matmul(
                window_query, window_keys.transpose(2, 3)
            ) / math.sqrt(
                head_dim
            )  # (bsz, num_heads, 1, window_size)

            # Place the computed scores in the appropriate position in attn_weights
            attn_weights[:, :, i, start_idx:end_idx] = window_attn_scores.squeeze(2)

        # Apply causal mask if needed
        if self.use_causal_mask:
            causal_mask = torch.tril(
                torch.ones(q_len, q_len, dtype=torch.bool, device=query_states.device)
            )
            # Expand causal mask to match attention weights shape
            causal_mask = (
                causal_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(bsz, num_heads, q_len, q_len)
            )
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

        # Apply global attention if enabled
        if self.use_global_attention and self.global_attention_indices:
            # For global attention tokens, allow attention to all positions
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < q_len:
                    # Reset the attention scores for global token to allow attention to all positions
                    # First, compute full attention for the global token
                    global_query = query_states[
                        :, :, global_idx : global_idx + 1, :
                    ]  # (bsz, num_heads, 1, head_dim)
                    all_keys = key_states  # (bsz, num_heads, q_len, head_dim)

                    global_attn_scores = torch.matmul(
                        global_query, all_keys.transpose(2, 3)
                    ) / math.sqrt(
                        head_dim
                    )  # (bsz, num_heads, 1, q_len)

                    # Place the global attention scores
                    attn_weights[:, :, global_idx, :] = global_attn_scores.squeeze(2)

                    # Also make sure all tokens attend to the global token
                    all_queries = query_states  # (bsz, num_heads, q_len, head_dim)
                    global_key = key_states[
                        :, :, global_idx : global_idx + 1, :
                    ]  # (bsz, num_heads, 1, head_dim)

                    global_attn_scores_reverse = torch.matmul(
                        all_queries, global_key.transpose(2, 3)
                    ) / math.sqrt(
                        head_dim
                    )  # (bsz, num_heads, q_len, 1)

                    attn_weights[:, :, :, global_idx] = (
                        global_attn_scores_reverse.squeeze(3)
                    )

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        return attn_weights


def create_glm47_sliding_window_attention(
    config: GLM47Config, layer_idx: Optional[int] = None
):
    """
    Factory function to create sliding window attention implementation for GLM-4.7.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        GLM47SlidingWindowAttention: The sliding window attention implementation
    """
    return GLM47SlidingWindowAttention(config, layer_idx)


__all__ = ["GLM47SlidingWindowAttention", "create_glm47_sliding_window_attention"]
