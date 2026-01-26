"""
Vision-Specific Attention Mechanisms for Inference-PIO System

This module provides attention mechanisms specifically optimized for visual data processing.
These mechanisms are designed to efficiently handle patch-based visual representations
and incorporate spatial relationships in attention computations.
"""

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionPatchAttention(nn.Module):
    """
    Attention mechanism optimized for visual patch processing.
    
    This attention mechanism is designed to efficiently process visual patches
    by incorporating spatial relationships and optimizing for the typical
    characteristics of visual data.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        patch_size: int = 14,
        image_size: int = 224,
        dropout: float = 0.0,
        bias: bool = True,
        use_relative_position: bool = True,
        use_patch_convolution: bool = True
    ):
        """
        Initialize the vision patch attention module.

        Args:
            embed_dim: Total model dimension
            num_heads: Number of attention heads
            patch_size: Size of each visual patch
            image_size: Size of the input image (assumed square)
            dropout: Dropout rate
            bias: Whether to use bias in projections
            use_relative_position: Whether to use relative position embeddings
            use_patch_convolution: Whether to use convolutional operations on patches
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_position = use_relative_position
        self.use_patch_convolution = use_patch_convolution

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize dropout
        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

        # Relative position embeddings if enabled
        if use_relative_position:
            self._init_relative_position_embeddings()

        # Patch convolution if enabled
        if use_patch_convolution:
            self.patch_conv = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim  # Depthwise convolution
            )

        # Spatial position embeddings
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def _init_relative_position_embeddings(self):
        """
        Initialize relative position embeddings for spatial relationships.
        """
        # Calculate max positions based on image dimensions
        num_patches_per_side = self.image_size // self.patch_size
        max_positions = num_patches_per_side
        
        # Create relative position embeddings
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_positions - 1) * (2 * max_positions - 1), self.num_heads)
        )
        
        # Get relative position indices
        coords_h = torch.arange(max_positions)
        coords_w = torch.arange(max_positions)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, H, W
        coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
        relative_coords[:, :, 0] += max_positions - 1  # shift to start from 0
        relative_coords[:, :, 1] += max_positions - 1
        relative_coords[:, :, 0] *= 2 * max_positions - 1
        relative_position_index = relative_coords.sum(-1)  # H*W, H*W
        
        self.register_buffer("relative_position_index", relative_position_index)

    def _calculate_spatial_attention(self, attention_weights: torch.Tensor, 
                                   height: int, width: int) -> torch.Tensor:
        """
        Apply spatial attention based on relative positions.
        
        Args:
            attention_weights: Original attention weights
            height: Height of the spatial grid
            width: Width of the spatial grid
            
        Returns:
            Updated attention weights with spatial consideration
        """
        if self.use_relative_position and hasattr(self, 'relative_position_bias_table'):
            # Get relative position bias
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(height * width, height * width, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attention_weights = attention_weights + relative_position_bias.unsqueeze(0)

        return attention_weights

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for vision patch attention.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        B, N, C = x.shape
        
        # Add class token if not present
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :N+1, :]
        
        # Apply patch convolution if enabled
        if self.use_patch_convolution:
            # Reshape to image-like format for convolution
            H = W = int(math.sqrt(N))  # Assuming square patches
            if H * W == N:
                x_patches = x[:, 1:, :]  # Exclude class token
                x_patches = x_patches.transpose(-2, -1).reshape(B, C, H, W)
                
                # Apply convolution
                x_conv = self.patch_conv(x_patches)
                x_conv = x_conv.reshape(B, C, H * W).transpose(-2, -1)
                
                # Combine with original patches
                x_updated = torch.cat([x[:, :1, :], x_conv], dim=1)
            else:
                # If not square, skip convolution
                x_updated = x
        else:
            x_updated = x

        # Apply projections
        q = self.q_proj(x_updated)
        k = self.k_proj(x_updated)
        v = self.v_proj(x_updated)

        # Reshape for multi-head attention
        q = q.view(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N+1, head_dim)
        k = k.view(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N+1, head_dim)
        v = v.view(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N+1, head_dim)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, N+1, N+1)

        # Apply spatial attention if enabled
        if self.use_relative_position:
            attn_weights = self._calculate_spatial_attention(
                attn_weights, 
                int(math.sqrt(N)), 
                int(math.sqrt(N))
            )

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Apply dropout if configured
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, N+1, head_dim)

        # Reshape to combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N + 1, C)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights if need_weights else None


class SparseVisionAttention(nn.Module):
    """
    Sparse attention mechanism specifically designed for visual data.
    
    This attention mechanism applies sparsity patterns that are particularly
    effective for visual data, considering spatial locality and reducing
    computational complexity.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sparsity_ratio: float = 0.25,
        local_window_size: int = 7,
        use_global_tokens: bool = True,
        num_global_tokens: int = 16,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize the sparse vision attention module.

        Args:
            embed_dim: Total model dimension
            num_heads: Number of attention heads
            sparsity_ratio: Ratio of tokens to attend to (0.0 to 1.0)
            local_window_size: Size of local attention windows
            use_global_tokens: Whether to use global tokens for long-range attention
            num_global_tokens: Number of global tokens to maintain
            dropout: Dropout rate
            bias: Whether to use bias in projections
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        self.local_window_size = local_window_size
        self.use_global_tokens = use_global_tokens
        self.num_global_tokens = num_global_tokens
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize dropout
        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

        # Global tokens if enabled
        if use_global_tokens:
            self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, embed_dim))

    def _create_sparse_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create sparse attention mask for visual data based on spatial locality.
        
        Args:
            seq_len: Length of the sequence (excluding global tokens if applicable)
            device: Device to create the mask on
            
        Returns:
            Sparse attention mask
        """
        # Calculate spatial dimensions assuming square layout
        spatial_len = int(math.sqrt(seq_len))
        if spatial_len * spatial_len != seq_len:
            # If not perfect square, use rectangular approximation
            spatial_len = math.ceil(math.sqrt(seq_len))
        
        # Create a mask with local window connections
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            # Calculate 2D coordinates
            row, col = i // spatial_len, i % spatial_len
            
            # Define local window boundaries
            start_row = max(0, row - self.local_window_size // 2)
            end_row = min(spatial_len, row + self.local_window_size // 2 + 1)
            start_col = max(0, col - self.local_window_size // 2)
            end_col = min(spatial_len, col + self.local_window_size // 2 + 1)
            
            # Mark local window as connected
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    j = r * spatial_len + c
                    if 0 <= j < seq_len:
                        mask[i, j] = True
        
        # Expand mask to match attention weights shape: (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)
        
        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for sparse vision attention.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        B, N, C = x.shape
        
        # Add global tokens if enabled
        if self.use_global_tokens:
            global_tokens = self.global_tokens.expand(B, -1, -1)
            x_with_globals = torch.cat([global_tokens, x], dim=1)
            N_total = N + self.num_global_tokens
        else:
            x_with_globals = x
            N_total = N

        # Apply projections
        q = self.q_proj(x_with_globals)
        k = self.k_proj(x_with_globals)
        v = self.v_proj(x_with_globals)

        # Reshape for multi-head attention
        q = q.view(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_total, head_dim)
        k = k.view(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_total, head_dim)
        v = v.view(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_total, head_dim)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, N_total, N_total)

        # Create and apply sparse attention mask
        sparse_mask = self._create_sparse_attention_mask(N, x.device)
        
        # Expand sparse mask to include global tokens if they exist
        if self.use_global_tokens:
            full_sparse_mask = torch.ones(N_total, N_total, dtype=torch.bool, device=x.device)
            # Allow global tokens to attend to everything
            full_sparse_mask[:self.num_global_tokens, :] = True
            full_sparse_mask[:, :self.num_global_tokens] = True
            # Apply local sparsity to regular patches
            full_sparse_mask[self.num_global_tokens:, self.num_global_tokens:] = sparse_mask[0, 0, :, :]
            sparse_mask = full_sparse_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Apply dropout if configured
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, N_total, head_dim)

        # Reshape to combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_total, C)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Return only the original patches if global tokens were added
        if self.use_global_tokens:
            attn_output = attn_output[:, self.num_global_tokens:, :]

        return attn_output, attn_weights if need_weights else None


class EfficientVisionAttention(nn.Module):
    """
    Efficient attention mechanism optimized for visual data processing.
    
    This attention mechanism uses techniques like linear attention and
    kernel-based approximations to achieve O(n) complexity instead of O(n^2).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_linear_attention: bool = True,
        low_rank_dim: int = 64
    ):
        """
        Initialize the efficient vision attention module.

        Args:
            embed_dim: Total model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in projections
            use_linear_attention: Whether to use linear attention approximation
            low_rank_dim: Dimension for low-rank approximations
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_linear_attention = use_linear_attention
        self.low_rank_dim = min(low_rank_dim, self.head_dim)

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize dropout
        self.dropout_module = nn.Dropout(dropout) if dropout > 0.0 else None

        # Low-rank projections for efficient computation
        if use_linear_attention:
            self.q_low_rank = nn.Linear(embed_dim, self.low_rank_dim * num_heads, bias=False)
            self.k_low_rank = nn.Linear(embed_dim, self.low_rank_dim * num_heads, bias=False)
            self.v_low_rank = nn.Linear(embed_dim, self.low_rank_dim * num_heads, bias=False)

    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute linear attention: phi(Q) @ [phi(K)^T @ V], where phi is a feature map.
        Using ReLU as the feature map for efficiency.
        
        Args:
            q: Query tensor of shape (B, num_heads, seq_len, head_dim)
            k: Key tensor of shape (B, num_heads, seq_len, head_dim)  
            v: Value tensor of shape (B, num_heads, seq_len, head_dim)
            
        Returns:
            Output tensor of shape (B, num_heads, seq_len, head_dim)
        """
        # Apply feature map (ReLU) to Q, K
        q_prime = F.relu(q)
        k_prime = F.relu(k)
        
        # Compute K^T @ V
        kv = torch.einsum("bhnd,bnhd->bhnd", k_prime, v)
        
        # Compute Q @ (K^T @ V)
        output = torch.einsum("bhnd,bhnd->bhdn", q_prime, kv)
        
        return output.transpose(-2, -1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for efficient vision attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        B, N, C = x.shape

        if self.use_linear_attention:
            # Use low-rank projections for efficiency
            q_low = self.q_low_rank(x).view(B, N, self.num_heads, self.low_rank_dim).transpose(1, 2)
            k_low = self.k_low_rank(x).view(B, N, self.num_heads, self.low_rank_dim).transpose(1, 2)
            v_low = self.v_low_rank(x).view(B, N, self.num_heads, self.low_rank_dim).transpose(1, 2)

            # Apply linear attention
            attn_output = self._linear_attention(q_low, k_low, v_low)
            
            # Reshape to combine heads
            attn_output = attn_output.contiguous().view(B, N, C)
            
            # Apply output projection
            attn_output = self.out_proj(attn_output)
            
            # Return with None for attention weights since linear attention doesn't produce interpretable weights
            return attn_output, None
        else:
            # Standard attention for comparison
            q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Apply softmax
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)

            # Apply dropout if configured
            if self.dropout_module is not None:
                attn_weights = self.dropout_module(attn_weights)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)

            # Reshape to combine heads
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

            # Apply output projection
            attn_output = self.out_proj(attn_output)

            return attn_output, attn_weights if need_weights else None


def create_vision_patch_attention(
    embed_dim: int,
    num_heads: int,
    patch_size: int = 14,
    image_size: int = 224,
    dropout: float = 0.0,
    bias: bool = True,
    use_relative_position: bool = True,
    use_patch_convolution: bool = True
) -> VisionPatchAttention:
    """
    Factory function to create vision patch attention.

    Args:
        embed_dim: Total model dimension
        num_heads: Number of attention heads
        patch_size: Size of each visual patch
        image_size: Size of the input image (assumed square)
        dropout: Dropout rate
        bias: Whether to use bias in projections
        use_relative_position: Whether to use relative position embeddings
        use_patch_convolution: Whether to use convolutional operations on patches

    Returns:
        VisionPatchAttention: The vision patch attention implementation
    """
    return VisionPatchAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        image_size=image_size,
        dropout=dropout,
        bias=bias,
        use_relative_position=use_relative_position,
        use_patch_convolution=use_patch_convolution
    )


def create_sparse_vision_attention(
    embed_dim: int,
    num_heads: int,
    sparsity_ratio: float = 0.25,
    local_window_size: int = 7,
    use_global_tokens: bool = True,
    num_global_tokens: int = 16,
    dropout: float = 0.0,
    bias: bool = True
) -> SparseVisionAttention:
    """
    Factory function to create sparse vision attention.

    Args:
        embed_dim: Total model dimension
        num_heads: Number of attention heads
        sparsity_ratio: Ratio of tokens to attend to (0.0 to 1.0)
        local_window_size: Size of local attention windows
        use_global_tokens: Whether to use global tokens for long-range attention
        num_global_tokens: Number of global tokens to maintain
        dropout: Dropout rate
        bias: Whether to use bias in projections

    Returns:
        SparseVisionAttention: The sparse vision attention implementation
    """
    return SparseVisionAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparsity_ratio=sparsity_ratio,
        local_window_size=local_window_size,
        use_global_tokens=use_global_tokens,
        num_global_tokens=num_global_tokens,
        dropout=dropout,
        bias=bias
    )


def create_efficient_vision_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    bias: bool = True,
    use_linear_attention: bool = True,
    low_rank_dim: int = 64
) -> EfficientVisionAttention:
    """
    Factory function to create efficient vision attention.

    Args:
        embed_dim: Total model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Whether to use bias in projections
        use_linear_attention: Whether to use linear attention approximation
        low_rank_dim: Dimension for low-rank approximations

    Returns:
        EfficientVisionAttention: The efficient vision attention implementation
    """
    return EfficientVisionAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        use_linear_attention=use_linear_attention,
        low_rank_dim=low_rank_dim
    )


__all__ = [
    "VisionPatchAttention",
    "SparseVisionAttention", 
    "EfficientVisionAttention",
    "create_vision_patch_attention",
    "create_sparse_vision_attention",
    "create_efficient_vision_attention"
]