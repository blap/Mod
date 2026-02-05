"""
Sparse Attention implementation for Qwen3-0.6B model.
Optimized for lightweight processing with sparse attention patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseAttention(nn.Module):
    """
    Sparse Attention implementation optimized for Qwen3-0.6B.
    Uses sparse attention patterns to reduce computational complexity while maintaining performance.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        local_window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = True,
        scale_factor: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.local_window_size = local_window_size
        self.scale_factor = scale_factor or self.head_dim ** -0.5
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim: {self.embed_dim}, "
                f"num_heads: {num_heads})"
            )
        
        # Projections
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
        Forward pass for Sparse Attention.
        
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
        
        # Compute sparse attention
        output = self._sparse_attention_forward(q, k, v, T)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(output)
        
        if need_weights:
            # Return None for attention weights since sparse attention doesn't compute full weights
            return output, None
        else:
            return output, None
    
    def _sparse_attention_forward(self, q, k, v, seq_len):
        """
        Compute sparse attention by processing blocks and local windows.
        """
        B, H, T, D = q.shape
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Process in blocks
        for i in range(0, T, self.block_size):
            end_i = min(i + self.block_size, T)
            
            # Get query block
            q_block = q[:, :, i:end_i, :]
            
            # Local attention: attend to nearby tokens within window
            start_idx = max(0, i - self.local_window_size // 2)
            end_idx = min(T, i + self.local_window_size // 2 + self.block_size)
            
            k_local = k[:, :, start_idx:end_idx, :]
            v_local = v[:, :, start_idx:end_idx, :]
            
            # Compute attention for this block
            attn_scores = torch.matmul(q_block, k_local.transpose(-2, -1))
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_probs = self.attention_dropout(attn_probs)
            
            # Apply attention to values
            output_block = torch.matmul(attn_probs, v_local)
            output[:, :, i:end_i, :] = output_block
        
        return output


class SparseAttentionConfig:
    """Configuration class for Sparse Attention."""
    
    def __init__(
        self,
        use_sparse_attention: bool = True,
        sparse_block_size: int = 64,
        sparse_local_window_size: int = 128,
        sparse_attention_dropout: float = 0.1,
        sparse_attention_bias: bool = True,
        sparse_scale_factor: Optional[float] = None,
        sparse_num_heads: int = 16,
    ):
        self.use_sparse_attention = use_sparse_attention
        self.sparse_block_size = sparse_block_size
        self.sparse_local_window_size = sparse_local_window_size
        self.sparse_attention_dropout = sparse_attention_dropout
        self.sparse_attention_bias = sparse_attention_bias
        self.sparse_scale_factor = sparse_scale_factor
        self.sparse_num_heads = sparse_num_heads


def create_sparse_attention_layer(config: SparseAttentionConfig, embed_dim: int) -> SparseAttention:
    """Factory function to create a Sparse Attention layer based on configuration."""
    return SparseAttention(
        embed_dim=embed_dim,
        num_heads=config.sparse_num_heads,
        block_size=config.sparse_block_size,
        local_window_size=config.sparse_local_window_size,
        dropout=config.sparse_attention_dropout,
        bias=config.sparse_attention_bias,
        scale_factor=config.sparse_scale_factor,
        attention_dropout=config.sparse_attention_dropout,
    )