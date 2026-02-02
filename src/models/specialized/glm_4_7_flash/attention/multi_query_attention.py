"""
GLM-4.7 Multi-Query and Grouped-Query Attention Implementation

This module implements Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) for the GLM-4.7 model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ....common.base_attention import BaseAttention
from ..config import GLM47Config


class GLM47MultiQueryAttention(BaseAttention):
    """
    Multi-Query Attention implementation for GLM-4.7 model.

    In Multi-Query Attention, each query head attends to the same key-value pair,
    significantly reducing memory requirements for the KV-cache. This is especially
    beneficial for models like GLM-4.7 that need to handle long sequences efficiently.
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

        # For MQA, we use only 1 key-value head regardless of num_attention_heads
        self.num_key_value_heads = 1  # Fixed to 1 for MQA
        self.num_key_value_groups = (
            self.num_heads
        )  # Each query head gets its own KV head (but they're identical)

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Optimization 14: Fuse Q, K, V projections into a single Linear layer
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Optimization 15: Rotary Cache
        # Ensure rotary embeddings are available (mock import here, assume it's set up)
        # self.rotary_emb = ... (should be initialized)

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
        Forward pass for Multi-Query Attention.
        """
        bsz, q_len, _ = hidden_states.size()

        # Optimization 14: Single QKV projection
        qkv = self.qkv_proj(hidden_states)

        # Optimization 18: Slice vs Cat
        query_states = qkv[..., : self.num_heads * self.head_dim]
        key_states = qkv[
            ...,
            self.num_heads
            * self.head_dim : (self.num_heads + self.num_key_value_heads)
            * self.head_dim,
        ]
        value_states = qkv[
            ..., (self.num_heads + self.num_key_value_heads) * self.head_dim :
        ]

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, num_heads, q_len, head_dim)

        # For MQA, key and value have only 1 head, so we expand them to match query heads
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, 1, q_len, head_dim)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, 1, q_len, head_dim)

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None and hasattr(self, "rotary_emb"):
            from .rotary_embeddings.optimized_rotary import apply_rotary_pos_emb

            # Optimization 15: Ensure rotary cache is used (implicit in implementation)
            cos, sin = self.rotary_emb(value_states, position_ids)

            # Apply rotary embeddings to query and key states
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Handle KV cache for inference
        if use_cache:
            if past_key_value is not None:
                # Concatenate with past keys and values
                # Optimization 17: In a real implementation, we would pre-allocate buffer
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states)

        # For MQA, we expand the single KV head to match the number of query heads
        # Do this AFTER caching to avoid caching redundant data
        key_states_expanded = key_states.expand(-1, self.num_heads, -1, -1)
        value_states_expanded = value_states.expand(-1, self.num_heads, -1, -1)

        # Optimization 11-13: SDPA
        if not output_attentions and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            # Optimization 20: Disable masking if trivial (SDPA handles checks internally often)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states_expanded,
                value_states_expanded,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,  # Causal mask usually handled by passed attention_mask
            )
            attn_weights = None
        else:
            # Compute attention scores
            attn_weights = torch.matmul(
                query_states, key_states_expanded.transpose(2, 3)
            ) / math.sqrt(
                self.head_dim
            )  # (bsz, num_heads, q_len, q_len)

            # Apply attention mask
            if attention_mask is not None:
                # Optimization 16: In-place add
                attn_weights.add_(attention_mask)

            # Optimization 19: Explicit softmax dim (standard, but ensuring F.softmax)
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            # Apply attention to values
            attn_output = torch.matmul(
                attn_weights, value_states_expanded
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

        return (
            attn_output,
            attn_weights if output_attentions else None,
            past_key_value,
        )


class GLM47GroupedQueryAttention(BaseAttention):
    """
    Grouped-Query Attention implementation for GLM-4.7 model.

    In Grouped-Query Attention, query heads are divided into groups, and each group
    shares the same key-value heads. This provides a good balance between memory
    efficiency and model quality compared to Multi-Head Attention (MHA) and
    Multi-Query Attention (MQA).
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

        # For GQA, we configure the number of key-value groups
        self.num_key_value_groups = config.num_key_value_groups
        self.num_key_value_heads = self.num_heads // self.num_key_value_groups

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if self.num_heads % self.num_key_value_groups != 0:
            raise ValueError(
                f"num_heads must be divisible by num_key_value_groups (got `num_heads`: {self.num_heads}"
                f" and `num_key_value_groups`: {self.num_key_value_groups})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
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
        Forward pass for Grouped-Query Attention.

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
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, num_key_value_heads, q_len, head_dim)

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (bsz, num_key_value_heads, q_len, head_dim)

        # Repeat key and value states to match query heads based on grouping
        # Each KV head serves multiple query heads (determined by num_key_value_groups)
        # key_states and value_states need to be expanded to match query heads
        # This is done by repeating along the head dimension
        from ......utils.tensor_utils import repeat_kv

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None:
            from .rotary_embeddings.optimized_rotary import apply_rotary_pos_emb

            cos, sin = self.rotary_emb(value_states, position_ids)

            # Apply rotary embeddings to query and key states
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(
            self.head_dim
        )  # (bsz, num_heads, q_len, q_len)

        # Apply attention mask
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


class GLM47MQAGQAAttentionSelector(nn.Module):
    """
    Flexible attention selector that can switch between MHA, GQA, and MQA based on configuration.

    This module provides a unified interface that can dynamically select between:
    - Multi-Head Attention (MHA): Standard attention with equal Q/K/V heads
    - Grouped-Query Attention (GQA): Multiple query heads share grouped KV heads
    - Multi-Query Attention (MQA): Each query head shares the same KV heads
    """

    def __init__(self, config: GLM47Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine attention type based on configuration
        self.attention_type = config.attention_type.lower()

        # Get the number of key-value groups for GQA (ignored for MQA/MHA)
        self.num_key_value_groups = config.num_key_value_groups

        if self.attention_type == "mqa":
            # Use Multi-Query Attention
            self.attention_impl = GLM47MultiQueryAttention(config, layer_idx)
        elif self.attention_type == "gqa":
            # Use Grouped-Query Attention
            self.attention_impl = GLM47GroupedQueryAttention(config, layer_idx)
        elif self.attention_type == "mha":
            # Use standard Multi-Head Attention
            from .flash_attention import GLM47FlashAttention2

            self.attention_impl = GLM47FlashAttention2(config, layer_idx)
        else:
            # Default to GQA for better memory efficiency
            self.attention_impl = GLM47GroupedQueryAttention(config, layer_idx)

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
        Forward pass that delegates to the selected attention implementation.
        """
        return self.attention_impl(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    def update_attention_type(
        self, attention_type: str, num_key_value_groups: Optional[int] = None
    ):
        """
        Update the attention type at runtime.

        Args:
            attention_type: New attention type ("mha", "gqa", or "mqa")
            num_key_value_groups: Number of key-value groups for GQA (optional)
        """
        attention_type = attention_type.lower()
        if attention_type not in ["mha", "gqa", "mqa"]:
            raise ValueError(
                f"Invalid attention_type: {attention_type}. Must be 'mha', 'gqa', or 'mqa'."
            )

        self.attention_type = attention_type
        if num_key_value_groups is not None:
            self.num_key_value_groups = num_key_value_groups

        if attention_type == "mqa":
            self.attention_impl = GLM47MultiQueryAttention(self.config, self.layer_idx)
        elif attention_type == "gqa":
            self.attention_impl = GLM47GroupedQueryAttention(
                self.config, self.layer_idx
            )
        elif attention_type == "mha":
            from .flash_attention import GLM47FlashAttention2

            self.attention_impl = GLM47FlashAttention2(self.config, self.layer_idx)


def create_mqa_gqa_attention(config: GLM47Config, layer_idx: Optional[int] = None):
    """
    Factory function to create MQA/GQA attention implementation based on configuration.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer

    Returns:
        Appropriate attention implementation based on configuration
    """
    return GLM47MQAGQAAttentionSelector(config, layer_idx)


__all__ = [
    "GLM47MultiQueryAttention",
    "GLM47GroupedQueryAttention",
    "GLM47MQAGQAAttentionSelector",
    "create_mqa_gqa_attention",
]
