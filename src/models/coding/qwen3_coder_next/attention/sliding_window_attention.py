"""
Sliding Window Attention implementation for Qwen3-Coder-Next model.
Optimized for next-generation code generation with sliding window attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention implementation optimized for Qwen3-Coder-Next.
    Uses a sliding window approach to limit attention to recent tokens, reducing 
    computational complexity while maintaining context for code generation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
        scale_factor: Optional[float] = None,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
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
        Forward pass for Sliding Window Attention.
        
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
        
        # Compute sliding window attention
        output = self._sliding_window_attention(q, k, v, T)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(output)
        
        if need_weights:
            # Return None for attention weights since sliding window doesn't compute full weights
            return output, None
        else:
            return output, None
    
    def _sliding_window_attention(self, q, k, v, seq_len):
        """
        Compute sliding window attention by limiting attention to a window around each token.
        """
        B, H, T, D = q.shape

        # Initialize output
        output = torch.zeros_like(q)

        # Process each position with its sliding window
        for i in range(T):
            # Determine the window bounds for position i
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(T, i + self.window_size // 2 + 1)

            # Extract the window for this position
            q_window = q[:, :, i:i+1, :]  # Shape: (B, H, 1, D)
            k_window = k[:, :, start_idx:end_idx, :]  # Shape: (B, H, actual_window_size, D)
            v_window = v[:, :, start_idx:end_idx, :]  # Shape: (B, H, actual_window_size, D)

            # Compute attention scores for this window
            attn_scores = torch.matmul(q_window, k_window.transpose(-2, -1))  # (B, H, 1, actual_window_size)

            # Apply causal masking within the window to prevent attending to future tokens
            window_len = end_idx - start_idx
            if window_len > 0:
                # Create causal mask for position i: only attend to positions <= i
                # Since we're computing attention for position i specifically,
                # we only need to mask out positions > i within the current window
                causal_mask = torch.zeros(1, window_len, device=q.device, dtype=torch.bool)

                for col in range(window_len):
                    actual_col_pos = start_idx + col  # Actual sequence position of the key
                    if actual_col_pos <= i:  # Only attend to positions up to current token
                        causal_mask[0, col] = True

                attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Apply softmax and dropout
                attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_weights = self.attention_dropout(attn_weights)

                # Apply attention to values
                attended_values = torch.matmul(attn_weights, v_window)  # (B, H, 1, D)
                output[:, :, i:i+1, :] = attended_values

        return output


class SlidingWindowAttentionConfig:
    """Configuration class for Sliding Window Attention."""
    
    def __init__(
        self,
        use_sliding_window_attention: bool = True,
        sliding_window_size: int = 1024,
        sliding_attention_dropout: float = 0.1,
        sliding_attention_bias: bool = True,
        sliding_scale_factor: Optional[float] = None,
        sliding_num_heads: int = 32,
    ):
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.sliding_attention_dropout = sliding_attention_dropout
        self.sliding_attention_bias = sliding_attention_bias
        self.sliding_scale_factor = sliding_scale_factor
        self.sliding_num_heads = sliding_num_heads


def create_sliding_window_attention_layer(config: SlidingWindowAttentionConfig, embed_dim: int) -> SlidingWindowAttention:
    """Factory function to create a Sliding Window Attention layer based on configuration."""
    return SlidingWindowAttention(
        embed_dim=embed_dim,
        num_heads=config.sliding_num_heads,
        window_size=config.sliding_window_size,
        dropout=config.sliding_attention_dropout,
        bias=config.sliding_attention_bias,
        scale_factor=config.sliding_scale_factor,
        attention_dropout=config.sliding_attention_dropout,
    )