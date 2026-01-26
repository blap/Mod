"""
Base Attention Module for Inference-PIO System

This module defines the base attention mechanisms for the Inference-PIO system.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseAttention(nn.Module):
    """
    Base attention class that provides common functionality for attention mechanisms.
    """
    
    def __init__(self):
        super().__init__()
        self.scaling = 1.0
        self.dropout = 0.0

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        Reshape the tensor for multi-head attention.

        Args:
            tensor: Input tensor
            seq_len: Sequence length
            bsz: Batch size

        Returns:
            Reshaped tensor
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _masked_softmax(self, attention_scores: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """
        Apply softmax with attention mask.

        Args:
            attention_scores: Attention scores
            attention_mask: Attention mask

        Returns:
            Softmax applied attention weights
        """
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention weights
        return torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(attention_scores.dtype)

    def _apply_dropout(self, attention_probs: torch.Tensor):
        """
        Apply dropout to attention probabilities.

        Args:
            attention_probs: Attention probabilities

        Returns:
            Attention probabilities with dropout applied
        """
        if self.dropout > 0.0:
            return torch.nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)
        return attention_probs


class BaseMultiHeadAttention(BaseAttention):
    """
    Base multi-head attention implementation.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Initialize projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for padded elements in key
            attention_mask: General attention mask
            need_weights: Whether to return attention weights
            attn_mask: Attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # Apply projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        
        # Scale query
        q = q * self.scaling
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (bsz, num_heads, tgt_len, src_len)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights += attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )
        
        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply dropout if configured
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (bsz, num_heads, tgt_len, head_dim)
        
        # Reshape to combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights if need_weights else None


class BaseCausalAttention(BaseAttention):
    """
    Base causal attention implementation for autoregressive models.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Initialize projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for causal attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Attention mask
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # Apply projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        
        # Scale query
        q = q * self.scaling
        
        # Handle past key-value states for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            src_len = k.size(2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (bsz, num_heads, tgt_len, src_len)
        
        # Apply causal mask to prevent attending to future tokens
        causal_mask = torch.tril(
            torch.ones((tgt_len, src_len), dtype=torch.bool, device=attn_weights.device)
        ).view(1, 1, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply dropout if configured
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (bsz, num_heads, tgt_len, head_dim)
        
        # Reshape to combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        # Prepare past key-value states for caching
        past_key_value = (k, v) if use_cache else None
        
        return attn_output, attn_weights if output_attentions else None, past_key_value


__all__ = [
    "BaseAttention",
    "BaseMultiHeadAttention",
    "BaseCausalAttention"
]