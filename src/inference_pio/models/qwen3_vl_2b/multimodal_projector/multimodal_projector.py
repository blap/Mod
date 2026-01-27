"""
Qwen3-VL-2B Multimodal Projector Implementation - Self-Contained Version

This module implements optimized multimodal projection layers specifically for the Qwen3-VL-2B model.
These layers map between different modalities (vision and language) with Qwen3-VL-2B specific optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class Qwen3VL2BProjectionLayer(nn.Module):
    """
    Qwen3-VL-2B specific projection layer for multimodal fusion.
    This layer is optimized for mapping between vision and language representations
    with Qwen3-VL-2B specific parameters and architecture considerations.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        intermediate_dim: Optional[int] = None,
        use_bias: bool = True,
        activation: str = "silu",  # Qwen3-VL-2B uses SiLU activation
        dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_residual: bool = True,
        use_low_rank: bool = False,
        low_rank_dim: Optional[int] = None,
        use_group_norm: bool = False,
        num_groups: int = 32,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8
    ):
        """
        Initialize the Qwen3-VL-2B projection layer.

        Args:
            vision_dim: Dimension of the vision input
            language_dim: Dimension of the language input
            intermediate_dim: Intermediate dimension for the projection (defaults to average of vision_dim and language_dim)
            use_bias: Whether to use bias in the projection
            activation: Activation function to use ('silu', 'gelu', 'relu', 'linear')
            dropout_prob: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            use_residual: Whether to use residual connections
            use_low_rank: Whether to use low-rank projection for memory efficiency
            low_rank_dim: Dimension for low-rank projection (only used if use_low_rank is True)
            use_group_norm: Whether to use group normalization instead of layer normalization
            num_groups: Number of groups for group normalization
            use_cross_attention: Whether to include cross-attention between modalities
            num_attention_heads: Number of attention heads for cross-attention
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.use_residual = use_residual
        self.use_low_rank = use_low_rank
        self.use_group_norm = use_group_norm

        # Determine intermediate dimension
        if intermediate_dim is None:
            intermediate_dim = (vision_dim + language_dim) // 2

        self.intermediate_dim = intermediate_dim

        # Create projection layers
        if use_low_rank and low_rank_dim is not None:
            # Low-rank projection for memory efficiency
            self.vision_projection = nn.Sequential(
                nn.Linear(vision_dim, low_rank_dim, bias=use_bias),
                nn.Linear(low_rank_dim, language_dim, bias=use_bias)
            )
            self.language_projection = nn.Sequential(
                nn.Linear(language_dim, low_rank_dim, bias=use_bias),
                nn.Linear(low_rank_dim, vision_dim, bias=use_bias)
            )
        else:
            # Standard projection layers
            self.vision_projection = nn.Linear(vision_dim, language_dim, bias=use_bias)
            self.language_projection = nn.Linear(language_dim, vision_dim, bias=use_bias)

        # Create intermediate projection for multimodal fusion
        self.multimodal_projection = nn.Linear(vision_dim + language_dim, intermediate_dim, bias=use_bias)

        # Create output projection
        self.output_projection = nn.Linear(intermediate_dim, language_dim, bias=use_bias)

        # Create normalization layer
        if use_group_norm:
            self.norm = nn.GroupNorm(num_groups, intermediate_dim)
        else:
            self.norm = nn.LayerNorm(intermediate_dim, eps=layer_norm_eps)

        # Create activation function
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        # Create dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Cross-attention for vision-language interaction if enabled
        if use_cross_attention:
            self.vision_to_language_attn = nn.MultiheadAttention(
                embed_dim=language_dim,
                num_heads=num_attention_heads,
                dropout=dropout_prob,
                bias=use_bias,
                batch_first=True
            )
            self.language_to_vision_attn = nn.MultiheadAttention(
                embed_dim=vision_dim,
                num_heads=num_attention_heads,
                dropout=dropout_prob,
                bias=use_bias,
                batch_first=True
            )

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights according to Qwen3-VL-2B specifications.
        """
        # Initialize vision projection weights
        if self.use_low_rank:
            # Initialize each layer in the sequential projection
            for layer in self.vision_projection:
                if isinstance(layer, nn.Linear):
                    std = self.vision_projection[0].in_features ** -0.5
                    nn.init.normal_(layer.weight, mean=0.0, std=std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        else:
            std = self.vision_projection.in_features ** -0.5
            nn.init.normal_(self.vision_projection.weight, mean=0.0, std=std)
            if self.vision_projection.bias is not None:
                nn.init.zeros_(self.vision_projection.bias)

        # Initialize language projection weights
        if self.use_low_rank:
            for layer in self.language_projection:
                if isinstance(layer, nn.Linear):
                    std = self.language_projection[0].in_features ** -0.5
                    nn.init.normal_(layer.weight, mean=0.0, std=std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        else:
            std = self.language_projection.in_features ** -0.5
            nn.init.normal_(self.language_projection.weight, mean=0.0, std=std)
            if self.language_projection.bias is not None:
                nn.init.zeros_(self.language_projection.bias)

        # Initialize multimodal projection weights
        std = self.multimodal_projection.in_features ** -0.5
        nn.init.normal_(self.multimodal_projection.weight, mean=0.0, std=std)
        if self.multimodal_projection.bias is not None:
            nn.init.zeros_(self.multimodal_projection.bias)

        # Initialize output projection weights
        std = self.output_projection.in_features ** -0.5
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

        # Initialize cross-attention weights if enabled
        if hasattr(self, 'vision_to_language_attn'):
            # Initialize attention weights
            std = self.language_projection.in_features ** -0.5
            nn.init.normal_(self.vision_to_language_attn.in_proj_weight, mean=0.0, std=std)
            nn.init.normal_(self.vision_to_language_attn.out_proj.weight, mean=0.0, std=std)
            
            std = self.vision_projection.in_features ** -0.5
            nn.init.normal_(self.language_to_vision_attn.in_proj_weight, mean=0.0, std=std)
            nn.init.normal_(self.language_to_vision_attn.out_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Qwen3-VL-2B specific multimodal projection layer.

        Args:
            vision_features: Vision features tensor of shape (batch, vision_seq_len, vision_dim)
            language_features: Language features tensor of shape (batch, lang_seq_len, language_dim)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (projected_vision, projected_language, fused_features)
        """
        batch_size, vision_seq_len, _ = vision_features.shape
        _, lang_seq_len, _ = language_features.shape

        # Project vision features to language space
        projected_vision = self.vision_projection(vision_features)

        # Project language features to vision space
        projected_language = self.language_projection(language_features)

        # Pad sequences to same length if needed for fusion operations
        max_len = max(vision_seq_len, lang_seq_len)

        if vision_seq_len < max_len:
            vision_padded = F.pad(projected_vision, (0, 0, 0, max_len - vision_seq_len), value=0)
        else:
            vision_padded = projected_vision

        if lang_seq_len < max_len:
            language_padded = F.pad(projected_language, (0, 0, 0, max_len - lang_seq_len), value=0)
        else:
            language_padded = projected_language

        # Optimization 5: Split Linear - Avoid concatenation overhead
        # Instead of torch.cat([v, l], dim=-1) -> Linear(v_dim + l_dim, out)
        # We do: Linear(v, out_partial) + Linear(l, out_partial)
        # This avoids creating the large intermediate concatenated tensor

        # Split weights and bias
        w_v, w_l = self.multimodal_projection.weight[:, :self.vision_dim], self.multimodal_projection.weight[:, self.vision_dim:]

        # Calculate parts
        multimodal_features = F.linear(vision_padded, w_v)
        multimodal_features += F.linear(language_padded, w_l)

        # Add bias if exists
        if self.multimodal_projection.bias is not None:
            multimodal_features += self.multimodal_projection.bias

        # Apply normalization
        if self.use_group_norm:
            # Reshape for group norm: (batch_size, channels, height, width)
            # We'll treat the sequence dimension as "height" and feature dimension as "channels"
            multimodal_features = multimodal_features.transpose(1, 2)  # (batch, features, seq)
            multimodal_features = self.norm(multimodal_features)
            multimodal_features = multimodal_features.transpose(1, 2)  # Back to (batch, seq, features)
        else:
            multimodal_features = self.norm(multimodal_features)

        # Apply activation
        activated_features = self.activation(multimodal_features)

        # Apply dropout
        activated_features = self.dropout(activated_features)

        # Apply output projection to get final language-aligned features
        output_features = self.output_projection(activated_features)

        # Apply cross-attention if enabled
        if hasattr(self, 'vision_to_language_attn') and hasattr(self, 'language_to_vision_attn'):
            # Cross-attention: vision attending to language
            vision_attended_to_language, _ = self.vision_to_language_attn(
                query=projected_vision,
                key=projected_language,
                value=projected_language,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )

            # Cross-attention: language attending to vision
            language_attended_to_vision, _ = self.language_to_vision_attn(
                query=projected_language,
                key=projected_vision,
                value=projected_vision,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )

            # Combine attended features with projected features
            projected_vision = projected_vision + vision_attended_to_language
            projected_language = projected_language + language_attended_to_vision

        # Truncate back to original sequence lengths
        projected_vision = projected_vision[:, :vision_seq_len, :]
        projected_language = projected_language[:, :lang_seq_len, :]
        output_features = output_features[:, :max(vision_seq_len, lang_seq_len), :]

        return projected_vision, projected_language, output_features


class Qwen3VL2BMultiModalProjector(nn.Module):
    """
    Qwen3-VL-2B specific multimodal projector with multiple projection layers.

    This projector combines multiple projection techniques for enhanced vision-language alignment.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        num_layers: int = 2,
        intermediate_dim: Optional[int] = None,
        use_bias: bool = True,
        activation: str = "silu",  # Qwen3-VL-2B uses SiLU activation
        dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_residual: bool = True,
        use_low_rank: bool = False,
        low_rank_dim: Optional[int] = None,
        use_group_norm: bool = False,
        num_groups: int = 32,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8
    ):
        """
        Initialize the Qwen3-VL-2B multimodal projector.

        Args:
            vision_dim: Dimension of the vision input
            language_dim: Dimension of the language input
            num_layers: Number of projection layers
            intermediate_dim: Intermediate dimension for the projection
            use_bias: Whether to use bias in the projection
            activation: Activation function to use
            dropout_prob: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            use_residual: Whether to use residual connections
            use_low_rank: Whether to use low-rank projection for memory efficiency
            low_rank_dim: Dimension for low-rank projection (only used if use_low_rank is True)
            use_group_norm: Whether to use group normalization instead of layer normalization
            num_groups: Number of groups for group normalization
            use_cross_attention: Whether to include cross-attention between modalities
            num_attention_heads: Number of attention heads for cross-attention
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention

        # Create projection layers
        self.projection_layers = nn.ModuleList([
            Qwen3VL2BProjectionLayer(
                vision_dim=vision_dim,
                language_dim=language_dim,
                intermediate_dim=intermediate_dim,
                use_bias=use_bias,
                activation=activation,
                dropout_prob=dropout_prob,
                layer_norm_eps=layer_norm_eps,
                use_residual=use_residual,
                use_low_rank=use_low_rank,
                low_rank_dim=low_rank_dim,
                use_group_norm=use_group_norm,
                num_groups=num_groups,
                use_cross_attention=use_cross_attention,
                num_attention_heads=num_attention_heads
            ) for _ in range(num_layers)
        ])

        # Final output projection
        self.final_projection = nn.Linear(language_dim, language_dim, bias=use_bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights according to Qwen3-VL-2B specifications.
        """
        # Initialize final projection weights
        std = self.final_projection.in_features ** -0.5
        nn.init.normal_(self.final_projection.weight, mean=0.0, std=std)
        if self.final_projection.bias is not None:
            nn.init.zeros_(self.final_projection.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Qwen3-VL-2B multimodal projector.

        Args:
            vision_features: Vision features tensor of shape (batch, vision_seq_len, vision_dim)
            language_features: Language features tensor of shape (batch, lang_seq_len, language_dim)
            attention_mask: Optional attention mask

        Returns:
            Projected features tensor of shape (batch, lang_seq_len, language_dim)
        """
        # Process through projection layers
        proj_vision = vision_features
        proj_language = language_features

        for layer in self.projection_layers:
            proj_vision, proj_language, fused_features = layer(
                proj_vision, proj_language, attention_mask
            )

        # Apply final projection
        output = self.final_projection(proj_language)

        return output


class Qwen3VL2BVisionLanguageProjector(nn.Module):
    """
    Qwen3-VL-2B specific vision-language projector with specialized attention mechanisms.

    This projector is designed specifically for the Qwen3-VL-2B model's architecture
    with optimized attention and projection mechanisms for vision-language alignment.
    """

    def __init__(
        self,
        config: Any,  # Qwen3VL2BConfig
        use_flash_attention: bool = True,
        use_sparse_attention: bool = False,
        sparse_attention_density: float = 0.5,
        use_conv_projection: bool = True,
        conv_kernel_size: int = 3,
        use_mlp_fusion: bool = True,
        mlp_expansion_ratio: float = 2.0
    ):
        """
        Initialize the Qwen3-VL-2B vision-language projector.

        Args:
            config: Qwen3VL2BConfig object
            use_flash_attention: Whether to use FlashAttention for efficiency
            use_sparse_attention: Whether to use sparse attention for efficiency
            sparse_attention_density: Density for sparse attention (0.0 to 1.0)
            use_conv_projection: Whether to use convolutional projection
            conv_kernel_size: Kernel size for convolutional projection
            use_mlp_fusion: Whether to use MLP-based fusion
            mlp_expansion_ratio: Expansion ratio for MLP fusion layers
        """
        super().__init__()

        self.config = config
        self.use_flash_attention = use_flash_attention
        self.use_sparse_attention = use_sparse_attention
        self.use_conv_projection = use_conv_projection
        self.use_mlp_fusion = use_mlp_fusion

        # Get dimensions from config
        vision_dim = getattr(config, 'vision_hidden_size', 1024)
        language_dim = getattr(config, 'hidden_size', 2048)
        intermediate_dim = int(language_dim * mlp_expansion_ratio)

        # Create vision projection layer
        if use_conv_projection:
            self.vision_projection = nn.Conv2d(
                in_channels=vision_dim,
                out_channels=language_dim,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2,
                bias=False
            )
        else:
            self.vision_projection = nn.Linear(vision_dim, language_dim, bias=False)

        # Create language projection layer
        self.language_projection = nn.Linear(language_dim, language_dim, bias=False)

        # Create attention layer
        if use_sparse_attention:
            try:
                from .sparse_attention import SparseAttention
                self.attention = SparseAttention(
                    embed_dim=language_dim,
                    num_heads=getattr(config, 'num_attention_heads', 16),
                    dropout=getattr(config, 'attention_dropout', 0.1),
                    bias=getattr(config, 'attention_bias', False),
                    sparse_attention_density=sparse_attention_density
                )
            except ImportError:
                # Fallback to standard attention if sparse attention is not available
                self.attention = nn.MultiheadAttention(
                    embed_dim=language_dim,
                    num_heads=getattr(config, 'num_attention_heads', 16),
                    dropout=getattr(config, 'attention_dropout', 0.1),
                    bias=getattr(config, 'attention_bias', False),
                    batch_first=True
                )
        elif use_flash_attention:
            try:
                from .flash_attention_2 import FlashAttention2
                self.attention = FlashAttention2(
                    embed_dim=language_dim,
                    num_heads=getattr(config, 'num_attention_heads', 16),
                    dropout=getattr(config, 'attention_dropout', 0.1),
                    bias=getattr(config, 'attention_bias', False),
                    is_causal=True
                )
            except ImportError:
                # Fallback to standard attention if flash attention is not available
                self.attention = nn.MultiheadAttention(
                    embed_dim=language_dim,
                    num_heads=getattr(config, 'num_attention_heads', 16),
                    dropout=getattr(config, 'attention_dropout', 0.1),
                    bias=getattr(config, 'attention_bias', False),
                    batch_first=True
                )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=language_dim,
                num_heads=getattr(config, 'num_attention_heads', 16),
                dropout=getattr(config, 'attention_dropout', 0.1),
                bias=getattr(config, 'attention_bias', False),
                batch_first=True
            )

        # Create MLP fusion layer if enabled
        if use_mlp_fusion:
            self.mlp_fusion = nn.Sequential(
                nn.Linear(language_dim, intermediate_dim, bias=False),
                nn.SiLU(),  # Qwen3-VL-2B uses SiLU activation
                nn.Dropout(getattr(config, 'mlp_dropout', 0.1)),
                nn.Linear(intermediate_dim, language_dim, bias=False)
            )

        # Create final layer normalization
        self.layer_norm = nn.LayerNorm(
            language_dim,
            eps=getattr(config, 'layer_norm_eps', 1e-6)
        )

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights according to Qwen3-VL-2B specifications.
        """
        # Initialize vision projection weights
        if isinstance(self.vision_projection, nn.Conv2d):
            nn.init.xavier_uniform_(self.vision_projection.weight)
        else:
            nn.init.xavier_uniform_(self.vision_projection.weight)

        # Initialize language projection weights
        nn.init.xavier_uniform_(self.language_projection.weight)

        # Initialize MLP fusion weights if enabled
        if self.use_mlp_fusion:
            nn.init.xavier_uniform_(self.mlp_fusion[0].weight)
            nn.init.xavier_uniform_(self.mlp_fusion[3].weight)

        # Initialize layer norm weights
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the Qwen3-VL-2B vision-language projector.

        Args:
            vision_features: Vision features tensor of shape (batch, num_patches, vision_dim)
                            or (batch, channels, height, width) if using conv projection
            language_features: Language features tensor of shape (batch, seq_len, language_dim)
            attention_mask: Optional attention mask

        Returns:
            Projected features tensor of shape (batch, seq_len, language_dim)
        """
        batch_size, seq_len, language_dim = language_features.shape

        # Process vision features
        if self.use_conv_projection and len(vision_features.shape) == 4:
            # Apply convolution to vision features (batch, channels, height, width)
            vision_proj = self.vision_projection(vision_features)
            # Flatten to (batch, seq_len, language_dim)
            vision_proj = vision_proj.flatten(2).transpose(1, 2)
        else:
            # Apply linear projection to vision features (batch, seq_len, vision_dim)
            vision_proj = self.vision_projection(vision_features)

        # Process language features
        language_proj = self.language_projection(language_features)

        # Pad features to same sequence length if needed
        vision_seq_len = vision_proj.size(1)
        language_seq_len = language_proj.size(1)
        max_len = max(vision_seq_len, language_seq_len)

        if vision_seq_len < max_len:
            vision_padded = F.pad(vision_proj, (0, 0, 0, max_len - vision_seq_len), value=0)
        else:
            vision_padded = vision_proj

        if language_seq_len < max_len:
            language_padded = F.pad(language_proj, (0, 0, 0, max_len - language_seq_len), value=0)
        else:
            language_padded = language_proj

        # Apply attention mechanism
        if isinstance(self.attention, nn.MultiheadAttention):
            # Standard attention expects (seq_len, batch, embed_dim) when batch_first=False
            # But we're using batch_first=True
            attended_features, _ = self.attention(
                query=language_padded,
                key=vision_padded,
                value=vision_padded,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        else:
            # FlashAttention or SparseAttention
            attended_features = self.attention(
                language_padded,
                vision_padded,
                attention_mask=attention_mask
            ) if attention_mask is not None else self.attention(language_padded, vision_padded)

        # Apply MLP fusion if enabled
        if self.use_mlp_fusion:
            mlp_output = self.mlp_fusion(attended_features)
            # Optimization 6: In-place Residual
            fused_features = attended_features.add_(mlp_output)
        else:
            fused_features = attended_features

        # Apply layer normalization
        output = self.layer_norm(fused_features)

        # Truncate back to original language sequence length
        output = output[:, :language_seq_len, :]

        return output


def create_qwen3_vl_projection_layer(
    config: Any,  # Qwen3VL2BConfig
    layer_idx: int = 0
) -> Qwen3VL2BProjectionLayer:
    """
    Factory function to create a Qwen3-VL-2B projection layer.

    Args:
        config: Qwen3VL2BConfig object
        layer_idx: Index of the layer (for layer-specific configurations)

    Returns:
        Qwen3VL2BProjectionLayer: The created projection layer
    """
    vision_dim = getattr(config, 'vision_hidden_size', 1024)
    language_dim = getattr(config, 'hidden_size', 2048)
    intermediate_dim = getattr(config, 'intermediate_size', 5504)

    return Qwen3VL2BProjectionLayer(
        vision_dim=vision_dim,
        language_dim=language_dim,
        intermediate_dim=intermediate_dim,
        use_bias=getattr(config, 'use_bias_in_projection', True),
        activation=getattr(config, 'projection_activation', 'silu'),  # Qwen3-VL-2B uses SiLU
        dropout_prob=getattr(config, 'projection_dropout', 0.1),
        layer_norm_eps=getattr(config, 'layer_norm_eps', 1e-6),
        use_residual=getattr(config, 'use_residual_in_projection', True),
        use_low_rank=getattr(config, 'use_low_rank_projection', False),
        low_rank_dim=getattr(config, 'low_rank_projection_dim', None),
        use_group_norm=getattr(config, 'use_group_norm_in_projection', False),
        num_groups=getattr(config, 'group_norm_num_groups', 32),
        use_cross_attention=getattr(config, 'use_cross_attention_in_projection', True),
        num_attention_heads=getattr(config, 'num_projection_attention_heads', 8)
    )


def create_qwen3_vl_multimodal_projector(
    config: Any,  # Qwen3VL2BConfig
    layer_idx: int = 0
) -> Qwen3VL2BMultiModalProjector:
    """
    Factory function to create a Qwen3-VL-2B multimodal projector.

    Args:
        config: Qwen3VL2BConfig object
        layer_idx: Index of the layer (for layer-specific configurations)

    Returns:
        Qwen3VL2BMultiModalProjector: The created multimodal projector
    """
    vision_dim = getattr(config, 'vision_hidden_size', 1024)
    language_dim = getattr(config, 'hidden_size', 2048)
    num_layers = getattr(config, 'num_projection_layers', 2)

    return Qwen3VL2BMultiModalProjector(
        vision_dim=vision_dim,
        language_dim=language_dim,
        num_layers=num_layers,
        intermediate_dim=getattr(config, 'intermediate_size', 5504),
        use_bias=getattr(config, 'use_bias_in_projection', True),
        activation=getattr(config, 'projection_activation', 'silu'),  # Qwen3-VL-2B uses SiLU
        dropout_prob=getattr(config, 'projection_dropout', 0.1),
        layer_norm_eps=getattr(config, 'layer_norm_eps', 1e-6),
        use_residual=getattr(config, 'use_residual_in_projection', True),
        use_low_rank=getattr(config, 'use_low_rank_projection', False),
        low_rank_dim=getattr(config, 'low_rank_projection_dim', None),
        use_group_norm=getattr(config, 'use_group_norm_in_projection', False),
        num_groups=getattr(config, 'group_norm_num_groups', 32),
        use_cross_attention=getattr(config, 'use_cross_attention_in_projection', True),
        num_attention_heads=getattr(config, 'num_projection_attention_heads', 8)
    )


def create_qwen3_vl_vision_language_projector(
    config: Any,  # Qwen3VL2BConfig
    layer_idx: int = 0
) -> Qwen3VL2BVisionLanguageProjector:
    """
    Factory function to create a Qwen3-VL-2B vision-language projector.

    Args:
        config: Qwen3VL2BConfig object
        layer_idx: Index of the layer (for layer-specific configurations)

    Returns:
        Qwen3VL2BVisionLanguageProjector: The created vision-language projector
    """
    return Qwen3VL2BVisionLanguageProjector(
        config=config,
        use_flash_attention=getattr(config, 'use_flash_attention_2', True),
        use_sparse_attention=getattr(config, 'use_sparse_attention', False),
        sparse_attention_density=getattr(config, 'sparse_attention_density', 0.5),
        use_conv_projection=getattr(config, 'use_conv_projection', True),
        conv_kernel_size=getattr(config, 'conv_projection_kernel_size', 3),
        use_mlp_fusion=getattr(config, 'use_mlp_fusion', True),
        mlp_expansion_ratio=getattr(config, 'mlp_expansion_ratio', 2.0)
    )


def apply_qwen3_vl_projection_optimizations(
    model: nn.Module,
    config: Any  # Qwen3VL2BConfig
) -> nn.Module:
    """
    Apply Qwen3-VL-2B specific projection optimizations to the model.

    Args:
        model: The Qwen3-VL-2B model to optimize
        config: Qwen3VL2BConfig object

    Returns:
        Model with projection optimizations applied
    """
    logger.info("Applying Qwen3-VL-2B specific projection optimizations...")

    # Identify and replace projection layers in the model
    for name, module in model.named_modules():
        if 'projector' in name.lower() or 'projection' in name.lower():
            if isinstance(module, nn.Linear) and module.in_features != module.out_features:
                # This is likely a projection layer, replace with Qwen3-VL-2B optimized version
                vision_dim = module.in_features
                language_dim = module.out_features

                new_projection = create_qwen3_vl_projection_layer(config)
                
                # Find parent module and replace the child
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent_module = model
                if parent_name:
                    for parent_part in parent_name.split('.'):
                        parent_module = getattr(parent_module, parent_part)

                setattr(parent_module, child_name, new_projection)
                logger.info(f"Replaced projection layer '{name}' with Qwen3-VL-2B optimized version")

    logger.info("Qwen3-VL-2B projection optimizations applied successfully")
    return model


__all__ = [
    "Qwen3VL2BProjectionLayer",
    "Qwen3VL2BMultiModalProjector",
    "Qwen3VL2BVisionLanguageProjector",
    "create_qwen3_vl_projection_layer",
    "create_qwen3_vl_multimodal_projector",
    "create_qwen3_vl_vision_language_projector",
    "apply_qwen3_vl_projection_optimizations"
]