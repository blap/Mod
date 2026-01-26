"""
Multi-Query and Grouped-Query Attention Implementation for Inference-PIO System

This module provides implementations for Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
for the Inference-PIO system. These attention mechanisms are designed to reduce memory usage and
improve inference efficiency, especially for large models.
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention implementation with optimizations for memory efficiency.
    In MQA, there is only one key-value head shared across all query heads.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: Optional[int] = None,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,  # MQA: only 1 KV head
        attention_dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096
    ):
        """
        Initialize Multi-Query Attention.

        Args:
            config: Model configuration
            layer_idx: Index of the transformer layer
            num_attention_heads: Number of query attention heads
            num_key_value_heads: Number of key-value attention heads (should be 1 for MQA)
            attention_dropout: Dropout rate for attention
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
            use_sliding_window: Whether to use sliding window attention
            sliding_window_size: Size of the sliding window
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size

        # Validate that this is indeed MQA (single KV head)
        if num_key_value_heads != 1:
            raise ValueError("MultiQueryAttention requires num_key_value_heads=1")

        # Calculate dimensions
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_head_dim = self.hidden_size // self.num_key_value_heads  # Fixed: should be self.num_key_value_heads

        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size: {self.hidden_size}, num_attention_heads: {self.num_attention_heads})"
            )

        # Projections - MQA has separate query projections but shared key-value
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to model specifications."""
        # Initialize query, key, and value projections
        std = self.hidden_size ** -0.5
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Multi-Query Attention.

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
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.kv_head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.kv_head_dim).transpose(1, 2)

        # Repeat key and value states to match number of query heads
        key_states = key_states.repeat(1, 1, self.num_attention_heads // self.num_key_value_heads, 1)
        value_states = value_states.repeat(1, 1, self.num_attention_heads // self.num_key_value_heads, 1)

        # Apply sliding window if enabled
        if self.use_sliding_window and attention_mask is not None:
            # Create sliding window mask
            sliding_window_mask = self._create_sliding_window_mask(q_len, device=hidden_states.device)
            attention_mask = attention_mask + sliding_window_mask

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask if needed
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply dropout if configured
        if self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Sliding window mask tensor
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.sliding_window_size)
            mask[i, :start] = float("-inf")
        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention implementation with optimizations for memory efficiency.
    In GQA, query heads are grouped with key-value heads (e.g., 8 query heads with 2 KV heads).
    """

    def __init__(
        self,
        config: Any,
        layer_idx: Optional[int] = None,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,  # GQA: multiple KV heads but fewer than query heads
        attention_dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096
    ):
        """
        Initialize Grouped-Query Attention.

        Args:
            config: Model configuration
            layer_idx: Index of the transformer layer
            num_attention_heads: Number of query attention heads
            num_key_value_heads: Number of key-value attention heads (should be < num_attention_heads for GQA)
            attention_dropout: Dropout rate for attention
            bias: Whether to use bias in projections
            is_causal: Whether to apply causal masking
            use_sliding_window: Whether to use sliding window attention
            sliding_window_size: Size of the sliding window
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size

        # Validate that this is indeed GQA (fewer KV heads than query heads)
        if num_key_value_heads >= num_attention_heads:
            raise ValueError("GroupedQueryAttention requires num_key_value_heads < num_attention_heads")

        # Calculate dimensions
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_head_dim = self.hidden_size // self.num_key_value_heads

        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size: {self.hidden_size}, num_attention_heads: {self.num_attention_heads})"
            )

        # Calculate grouping factor
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Projections - GQA has separate query projections but grouped key-value
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kv_head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to model specifications."""
        # Initialize query, key, and value projections
        std = self.hidden_size ** -0.5
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Grouped-Query Attention.

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
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.kv_head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.kv_head_dim).transpose(1, 2)

        # Repeat key and value states to match number of query heads (grouped replication)
        key_states = torch.repeat_interleave(key_states, dim=2, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=2, repeats=self.num_key_value_groups)

        # Apply sliding window if enabled
        if self.use_sliding_window and attention_mask is not None:
            # Create sliding window mask
            sliding_window_mask = self._create_sliding_window_mask(q_len, device=hidden_states.device)
            attention_mask = attention_mask + sliding_window_mask

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask if needed
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply dropout if configured
        if self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device for the mask

        Returns:
            Sliding window mask tensor
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.sliding_window_size)
            mask[i, :start] = float("-inf")
        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)


def create_mqa_gqa_attention(config: Any, layer_idx: Optional[int] = None, 
                           attention_type: str = "gqa", num_key_value_heads: Optional[int] = None):
    """
    Factory function to create Multi-Query or Grouped-Query Attention implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)
        attention_type: Type of attention ("mqa" or "gqa")
        num_key_value_heads: Number of key-value heads (if None, defaults based on attention_type)

    Returns:
        MultiQueryAttention or GroupedQueryAttention: The attention implementation
    """
    if attention_type.lower() == "mqa":
        # For MQA, we use 1 key-value head
        n_kv_heads = 1 if num_key_value_heads is None else num_key_value_heads
        return MultiQueryAttention(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=n_kv_heads,
            attention_dropout=getattr(config, 'attention_dropout_prob', 0.0),
            bias=not getattr(config, 'remove_bias_in_attention', False),
            is_causal=getattr(config, 'is_causal', True),
            use_sliding_window=getattr(config, 'use_sliding_window_attention', False),
            sliding_window_size=getattr(config, 'sliding_window_size', 4096)
        )
    elif attention_type.lower() == "gqa":
        # For GQA, we use multiple key-value heads but fewer than query heads
        n_kv_heads = num_key_value_heads or getattr(config, 'num_key_value_heads', config.num_attention_heads // 4)
        return GroupedQueryAttention(
            config=config,
            layer_idx=layer_idx,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=n_kv_heads,
            attention_dropout=getattr(config, 'attention_dropout_prob', 0.0),
            bias=not getattr(config, 'remove_bias_in_attention', False),
            is_causal=getattr(config, 'is_causal', True),
            use_sliding_window=getattr(config, 'use_sliding_window_attention', False),
            sliding_window_size=getattr(config, 'sliding_window_size', 4096)
        )
    else:
        raise ValueError(f"Invalid attention type: {attention_type}. Use 'mqa' or 'gqa'.")


__all__ = [
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "create_mqa_gqa_attention"
]