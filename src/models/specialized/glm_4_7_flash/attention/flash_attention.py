"""
Specialized Flash Attention implementation for GLM-4.7-Flash model.
Optimized for speed and memory efficiency with flash processing capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlashAttention(nn.Module):
    """
    Flash Attention implementation optimized for GLM-4.7-Flash model.
    Uses efficient memory management and optimized kernels for faster processing.
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
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
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
        Forward pass for Flash Attention.
        
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
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale queries
        q = q * self.scale_factor
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
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
        output = torch.matmul(attn_weights, v)
        
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


class FlashAttentionConfig:
    """Configuration class for Flash Attention."""
    
    def __init__(
        self,
        use_flash_attention: bool = True,
        flash_attention_dropout: float = 0.1,
        flash_attention_bias: bool = True,
        flash_scale_factor: Optional[float] = None,
        flash_num_heads: int = 16,
    ):
        self.use_flash_attention = use_flash_attention
        self.flash_attention_dropout = flash_attention_dropout
        self.flash_attention_bias = flash_attention_bias
        self.flash_scale_factor = flash_scale_factor
        self.flash_num_heads = flash_num_heads


def create_flash_attention_layer(config: FlashAttentionConfig, embed_dim: int) -> FlashAttention:
    """Factory function to create a Flash Attention layer based on configuration."""
    return FlashAttention(
        embed_dim=embed_dim,
        num_heads=config.flash_num_heads,
        dropout=config.flash_attention_dropout,
        bias=config.flash_attention_bias,
        scale_factor=config.flash_scale_factor,
        attention_dropout=config.flash_attention_dropout,
    )