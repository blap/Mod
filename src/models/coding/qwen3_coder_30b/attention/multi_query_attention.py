"""
Multi-Query Attention implementation for Qwen3-Coder-30B model.
Optimized for code generation tasks with single key-value per head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention implementation optimized for Qwen3-Coder-30B.
    Uses a single key-value pair per head to reduce memory usage while maintaining 
    performance for code generation tasks.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        scale_factor: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale_factor = scale_factor or self.head_dim ** -0.5
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim: {self.embed_dim}, "
                f"num_heads: {num_heads})"
            )
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)  # Single KV per head
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)  # Single KV per head
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Multi-Query Attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            attn_mask: Attention mask of shape (seq_len, seq_len) or (batch_size * num_heads, seq_len, seq_len)
            key_padding_mask: Key padding mask of shape (batch_size, seq_len)
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim) and optional attention weights
        """
        B, T, C = query.size()
        
        # Project queries, keys, values
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, T, 1, self.head_dim)  # Single KV per head
        v = self.v_proj(value).view(B, T, 1, self.head_dim)  # Single KV per head
        
        # Expand K and V to match number of query heads
        k = k.expand(-1, -1, self.num_heads, -1)  # (B, T, num_heads, head_dim)
        v = v.expand(-1, -1, self.num_heads, -1)  # (B, T, num_heads, head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        # Scale queries
        q = q * self.scale_factor
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, T, T)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask_expanded, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Apply attention dropout
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(output)
        
        if need_weights:
            # Average attention weights across heads if requested
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        else:
            return output, None


class MultiQueryAttentionConfig:
    """Configuration class for Multi-Query Attention."""
    
    def __init__(
        self,
        use_multi_query_attention: bool = True,
        mqa_attention_dropout: float = 0.1,
        mqa_attention_bias: bool = True,
        mqa_scale_factor: Optional[float] = None,
        mqa_num_heads: int = 32,
    ):
        self.use_multi_query_attention = use_multi_query_attention
        self.mqa_attention_dropout = mqa_attention_dropout
        self.mqa_attention_bias = mqa_attention_bias
        self.mqa_scale_factor = mqa_scale_factor
        self.mqa_num_heads = mqa_num_heads


def create_mqa_layer(config: MultiQueryAttentionConfig, embed_dim: int) -> MultiQueryAttention:
    """Factory function to create a Multi-Query Attention layer based on configuration."""
    return MultiQueryAttention(
        embed_dim=embed_dim,
        num_heads=config.mqa_num_heads,
        dropout=config.mqa_attention_dropout,
        bias=config.mqa_attention_bias,
        scale_factor=config.mqa_scale_factor,
        attention_dropout=config.mqa_attention_dropout,
    )