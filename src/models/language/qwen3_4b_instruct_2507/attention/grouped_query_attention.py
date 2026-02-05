"""
Grouped Query Attention implementation for Qwen3-4B-Instruct-2507 model.
Optimized for instruction-following tasks with grouped query processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention implementation optimized for Qwen3-4B-Instruct-2507.
    Reduces memory usage by grouping queries while maintaining accuracy for instruction tasks.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_groups: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        scale_factor: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups or num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.head_dim * self.num_kv_groups
        self.scale_factor = scale_factor or self.head_dim ** -0.5
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim: {self.embed_dim}, "
                f"num_heads: {num_heads})"
            )
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=bias)
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
        Forward pass for Grouped Query Attention.
        
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
        k = self.k_proj(key).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(value).view(B, T, self.num_kv_groups, self.head_dim)
        
        # Expand K and V to match number of query heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_groups, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_groups, dim=2)
        
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


class GroupedQueryAttentionConfig:
    """Configuration class for Grouped Query Attention."""
    
    def __init__(
        self,
        use_grouped_query_attention: bool = True,
        gqa_num_kv_groups: Optional[int] = None,
        gqa_attention_dropout: float = 0.1,
        gqa_attention_bias: bool = True,
        gqa_scale_factor: Optional[float] = None,
        gqa_num_heads: int = 32,
    ):
        self.use_grouped_query_attention = use_grouped_query_attention
        self.gqa_num_kv_groups = gqa_num_kv_groups
        self.gqa_attention_dropout = gqa_attention_dropout
        self.gqa_attention_bias = gqa_attention_bias
        self.gqa_scale_factor = gqa_scale_factor
        self.gqa_num_heads = gqa_num_heads


def create_gqa_layer(config: GroupedQueryAttentionConfig, embed_dim: int) -> GroupedQueryAttention:
    """Factory function to create a Grouped Query Attention layer based on configuration."""
    return GroupedQueryAttention(
        embed_dim=embed_dim,
        num_heads=config.gqa_num_heads,
        num_kv_groups=config.gqa_num_kv_groups,
        dropout=config.gqa_attention_dropout,
        bias=config.gqa_attention_bias,
        scale_factor=config.gqa_scale_factor,
        attention_dropout=config.gqa_attention_dropout,
    )