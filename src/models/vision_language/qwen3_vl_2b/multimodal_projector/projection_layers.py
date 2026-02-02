"""
Qwen3-VL-2B Projection Layer Implementation

This module implements optimized projection layers for the Qwen3-VL-2B model.
These layers are specifically designed for mapping between vision and language representations
in the multimodal fusion process.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import logging as transformers_logging

logger = transformers_logging.get_logger(__name__)


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
        activation: str = "gelu",
        dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_residual: bool = True,
        use_low_rank: bool = False,
        low_rank_dim: Optional[int] = None,
        use_group_norm: bool = False,
        num_groups: int = 32,
    ):
        """
        Initialize the Qwen3-VL-2B projection layer.

        Args:
            vision_dim: Dimension of the vision input
            language_dim: Dimension of the language input
            intermediate_dim: Intermediate dimension for the projection (defaults to average of vision_dim and language_dim)
            use_bias: Whether to use bias in the projection
            activation: Activation function to use ('gelu', 'relu', 'swish', 'linear')
            dropout_prob: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            use_residual: Whether to use residual connections
            use_low_rank: Whether to use low-rank projection for memory efficiency
            low_rank_dim: Dimension for low-rank projection (only used if use_low_rank is True)
            use_group_norm: Whether to use group normalization instead of layer normalization
            num_groups: Number of groups for group normalization
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
                nn.Linear(low_rank_dim, language_dim, bias=use_bias),
            )
            self.language_projection = nn.Sequential(
                nn.Linear(language_dim, low_rank_dim, bias=use_bias),
                nn.Linear(low_rank_dim, vision_dim, bias=use_bias),
            )
        else:
            # Standard projection layers
            self.vision_projection = nn.Linear(vision_dim, language_dim, bias=use_bias)
            self.language_projection = nn.Linear(
                language_dim, vision_dim, bias=use_bias
            )

        # Create intermediate projection for multimodal fusion
        self.multimodal_projection = nn.Linear(
            vision_dim + language_dim, intermediate_dim, bias=use_bias
        )

        # Create output projection
        self.output_projection = nn.Linear(
            intermediate_dim, language_dim, bias=use_bias
        )

        # Create normalization layer
        if use_group_norm:
            self.norm = nn.GroupNorm(num_groups, intermediate_dim)
        else:
            self.norm = nn.LayerNorm(intermediate_dim, eps=layer_norm_eps)

        # Create activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

        # Create dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights according to Qwen3-VL-2B specifications.
        """
        # Initialize vision projection weights
        if self.use_low_rank:
            nn.init.xavier_uniform_(self.vision_projection[0].weight)
            nn.init.xavier_uniform_(self.vision_projection[1].weight)
            if (
                hasattr(self.vision_projection[0], "bias")
                and self.vision_projection[0].bias is not None
            ):
                nn.init.zeros_(self.vision_projection[0].bias)
            if (
                hasattr(self.vision_projection[1], "bias")
                and self.vision_projection[1].bias is not None
            ):
                nn.init.zeros_(self.vision_projection[1].bias)
        else:
            nn.init.xavier_uniform_(self.vision_projection.weight)
            if (
                hasattr(self.vision_projection, "bias")
                and self.vision_projection.bias is not None
            ):
                nn.init.zeros_(self.vision_projection.bias)

        # Initialize language projection weights
        if self.use_low_rank:
            nn.init.xavier_uniform_(self.language_projection[0].weight)
            nn.init.xavier_uniform_(self.language_projection[1].weight)
            if (
                hasattr(self.language_projection[0], "bias")
                and self.language_projection[0].bias is not None
            ):
                nn.init.zeros_(self.language_projection[0].bias)
            if (
                hasattr(self.language_projection[1], "bias")
                and self.language_projection[1].bias is not None
            ):
                nn.init.zeros_(self.language_projection[1].bias)
        else:
            nn.init.xavier_uniform_(self.language_projection.weight)
            if (
                hasattr(self.language_projection, "bias")
                and self.language_projection.bias is not None
            ):
                nn.init.zeros_(self.language_projection.bias)

        # Initialize multimodal projection weights
        nn.init.xavier_uniform_(self.multimodal_projection.weight)
        if self.multimodal_projection.bias is not None:
            nn.init.zeros_(self.multimodal_projection.bias)

        # Initialize output projection weights
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Qwen3-VL-2B projection layer.

        Args:
            vision_features: Vision features tensor of shape (batch_size, seq_len_vision, vision_dim)
            language_features: Language features tensor of shape (batch_size, seq_len_language, language_dim)
            attention_mask: Attention mask for language features (optional)

        Returns:
            Tuple of (projected_vision, projected_language, fused_features)
        """
        batch_size, seq_len_vision, _ = vision_features.shape
        _, seq_len_language, _ = language_features.shape

        # Project vision features to language space
        projected_vision = self.vision_projection(vision_features)

        # Project language features to vision space
        projected_language = self.language_projection(language_features)

        # Pad features to same sequence length if needed
        if seq_len_vision != seq_len_language:
            max_seq_len = max(seq_len_vision, seq_len_language)
            if seq_len_vision < max_seq_len:
                # Pad vision features
                pad_size = max_seq_len - seq_len_vision
                projected_vision = torch.cat(
                    [
                        projected_vision,
                        torch.zeros(
                            batch_size,
                            pad_size,
                            projected_vision.shape[-1],
                            dtype=projected_vision.dtype,
                            device=projected_vision.device,
                        ),
                    ],
                    dim=1,
                )
            if seq_len_language < max_seq_len:
                # Pad language features
                pad_size = max_seq_len - seq_len_language
                projected_language = torch.cat(
                    [
                        projected_language,
                        torch.zeros(
                            batch_size,
                            pad_size,
                            projected_language.shape[-1],
                            dtype=projected_language.dtype,
                            device=projected_language.device,
                        ),
                    ],
                    dim=1,
                )

        # Concatenate projected features for multimodal fusion
        concatenated_features = torch.cat(
            [projected_vision, projected_language], dim=-1
        )

        # Apply multimodal projection
        multimodal_features = self.multimodal_projection(concatenated_features)

        # Apply normalization
        if self.use_group_norm:
            # Reshape for group norm: (batch_size, channels, height, width)
            # We'll treat the sequence dimension as "height" and feature dimension as "channels"
            multimodal_features = multimodal_features.permute(
                0, 2, 1
            )  # (batch, features, seq)
            multimodal_features = self.norm(multimodal_features)
            multimodal_features = multimodal_features.permute(
                0, 2, 1
            )  # Back to (batch, seq, features)
        else:
            multimodal_features = self.norm(multimodal_features)

        # Apply activation
        activated_features = self.activation(multimodal_features)

        # Apply dropout
        activated_features = self.dropout(activated_features)

        # Apply output projection to get final language-aligned features
        output_features = self.output_projection(activated_features)

        # Truncate back to original sequence lengths if padded
        if seq_len_vision != seq_len_language:
            output_features = output_features[
                :, : max(seq_len_vision, seq_len_language), :
            ]
            projected_vision = projected_vision[:, :seq_len_vision, :]
            projected_language = projected_language[:, :seq_len_language, :]

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
        activation: str = "gelu",
        dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_residual: bool = True,
        use_low_rank: bool = False,
        low_rank_dim: Optional[int] = None,
        use_group_norm: bool = False,
        num_groups: int = 32,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
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
            low_rank_dim: Dimension for low-rank projection
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
        self.projection_layers = nn.ModuleList(
            [
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
                )
                for _ in range(num_layers)
            ]
        )

        # Create cross-attention layers if enabled
        if use_cross_attention:
            self.vision_to_language_attn = nn.MultiheadAttention(
                embed_dim=language_dim,
                num_heads=num_attention_heads,
                dropout=dropout_prob,
                bias=use_bias,
                batch_first=True,
            )
            self.language_to_vision_attn = nn.MultiheadAttention(
                embed_dim=vision_dim,
                num_heads=num_attention_heads,
                dropout=dropout_prob,
                bias=use_bias,
                batch_first=True,
            )

        # Final output projection
        self.final_projection = nn.Linear(language_dim, language_dim, bias=use_bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights according to Qwen3-VL-2B specifications.
        """
        # Initialize final projection weights
        nn.init.xavier_uniform_(self.final_projection.weight)
        if self.final_projection.bias is not None:
            nn.init.zeros_(self.final_projection.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Qwen3-VL-2B multimodal projector.

        Args:
            vision_features: Vision features tensor of shape (batch_size, seq_len_vision, vision_dim)
            language_features: Language features tensor of shape (batch_size, seq_len_language, language_dim)
            attention_mask: Attention mask for language features (optional)

        Returns:
            Projected features tensor of shape (batch_size, seq_len_language, language_dim)
        """
        # Process through projection layers
        proj_vision, proj_language, fused_features = self.projection_layers[0](
            vision_features, language_features
        )

        # Process through remaining layers
        for layer in self.projection_layers[1:]:
            # Use the output of previous layer as input to next layer
            proj_vision, proj_language, fused_features = layer(
                proj_vision, proj_language
            )

        # Apply cross-attention if enabled
        if self.use_cross_attention:
            # Cross-attention: language attending to vision
            lang_attended_to_vision, _ = self.vision_to_language_attn(
                query=proj_language,
                key=proj_vision,
                value=proj_vision,
                key_padding_mask=(
                    ~attention_mask.bool() if attention_mask is not None else None
                ),
            )

            # Cross-attention: vision attending to language
            vision_attended_to_lang, _ = self.language_to_vision_attn(
                query=proj_vision,
                key=proj_language,
                value=proj_language,
                key_padding_mask=(
                    ~attention_mask.bool() if attention_mask is not None else None
                ),
            )

            # Combine attended features with fused features
            combined_features = fused_features + lang_attended_to_vision

            # Apply final projection
            output = self.final_projection(combined_features)
        else:
            # Apply final projection directly
            output = self.final_projection(fused_features)

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
        mlp_expansion_ratio: float = 2.0,
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
        vision_dim = getattr(config, "vision_hidden_size", 1024)
        language_dim = getattr(config, "hidden_size", 2048)
        intermediate_dim = int(language_dim * mlp_expansion_ratio)

        # Create vision projection layer
        if use_conv_projection:
            self.vision_projection = nn.Conv2d(
                in_channels=vision_dim,
                out_channels=language_dim,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2,
                bias=False,
            )
        else:
            self.vision_projection = nn.Linear(vision_dim, language_dim, bias=False)

        # Create language projection layer
        self.language_projection = nn.Linear(language_dim, language_dim, bias=False)

        # Create attention layer
        if use_sparse_attention:
            try:
                from ...common.sparse_attention import SparseAttention

                self.attention = SparseAttention(
                    embed_dim=language_dim,
                    num_heads=getattr(config, "num_attention_heads", 16),
                    dropout=getattr(config, "attention_dropout", 0.1),
                    bias=getattr(config, "attention_bias", False),
                    sparse_attention_density=sparse_attention_density,
                )
            except ImportError:
                # Fallback to standard attention if sparse attention is not available
                self.attention = nn.MultiheadAttention(
                    embed_dim=language_dim,
                    num_heads=getattr(config, "num_attention_heads", 16),
                    dropout=getattr(config, "attention_dropout", 0.1),
                    bias=getattr(config, "attention_bias", False),
                    batch_first=True,
                )
        elif use_flash_attention:
            try:
                from ...common.flash_attention_2 import FlashAttention2

                self.attention = FlashAttention2(
                    embed_dim=language_dim,
                    num_heads=getattr(config, "num_attention_heads", 16),
                    dropout=getattr(config, "attention_dropout", 0.1),
                    bias=getattr(config, "attention_bias", False),
                    is_causal=True,
                )
            except ImportError:
                # Fallback to standard attention if flash attention is not available
                self.attention = nn.MultiheadAttention(
                    embed_dim=language_dim,
                    num_heads=getattr(config, "num_attention_heads", 16),
                    dropout=getattr(config, "attention_dropout", 0.1),
                    bias=getattr(config, "attention_bias", False),
                    batch_first=True,
                )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=language_dim,
                num_heads=getattr(config, "num_attention_heads", 16),
                dropout=getattr(config, "attention_dropout", 0.1),
                bias=getattr(config, "attention_bias", False),
                batch_first=True,
            )

        # Create MLP fusion layer if enabled
        if use_mlp_fusion:
            self.mlp_fusion = nn.Sequential(
                nn.Linear(language_dim, intermediate_dim, bias=False),
                nn.GELU(),
                nn.Dropout(getattr(config, "mlp_dropout", 0.1)),
                nn.Linear(intermediate_dim, language_dim, bias=False),
            )

        # Create final layer normalization
        self.layer_norm = nn.LayerNorm(
            language_dim, eps=getattr(config, "layer_norm_eps", 1e-6)
        )

        # Initialize weights
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Qwen3-VL-2B vision-language projector.

        Args:
            vision_features: Vision features tensor of shape (batch_size, num_patches, vision_dim)
                            or (batch_size, channels, height, width) if using conv projection
            language_features: Language features tensor of shape (batch_size, seq_len, language_dim)
            attention_mask: Attention mask for language features (optional)

        Returns:
            Projected features tensor of shape (batch_size, seq_len, language_dim)
        """
        # Project vision features to language space
        if self.use_conv_projection and len(vision_features.shape) == 4:
            # Apply convolution to vision features (batch, channels, height, width)
            batch_size, channels, height, width = vision_features.shape
            projected_vision = self.vision_projection(vision_features)
            # Flatten to (batch, seq_len, language_dim)
            projected_vision = projected_vision.view(
                batch_size, -1, projected_vision.shape[-1]
            )
        else:
            # Apply linear projection to vision features (batch, seq_len, vision_dim)
            projected_vision = self.vision_projection(vision_features)

        # Project language features
        projected_language = self.language_projection(language_features)

        # Pad features to same sequence length if needed
        if projected_vision.shape[1] != projected_language.shape[1]:
            max_seq_len = max(projected_vision.shape[1], projected_language.shape[1])
            if projected_vision.shape[1] < max_seq_len:
                # Pad vision features
                pad_size = max_seq_len - projected_vision.shape[1]
                projected_vision = torch.cat(
                    [
                        projected_vision,
                        torch.zeros(
                            projected_vision.shape[0],
                            pad_size,
                            projected_vision.shape[-1],
                            dtype=projected_vision.dtype,
                            device=projected_vision.device,
                        ),
                    ],
                    dim=1,
                )
            if projected_language.shape[1] < max_seq_len:
                # Pad language features
                pad_size = max_seq_len - projected_language.shape[1]
                projected_language = torch.cat(
                    [
                        projected_language,
                        torch.zeros(
                            projected_language.shape[0],
                            pad_size,
                            projected_language.shape[-1],
                            dtype=projected_language.dtype,
                            device=projected_language.device,
                        ),
                    ],
                    dim=1,
                )

        # Combine vision and language features
        combined_features = projected_vision + projected_language

        # Apply attention mechanism
        if isinstance(self.attention, nn.MultiheadAttention):
            # Standard attention expects (seq_len, batch, embed_dim) when batch_first=False
            # But we're using batch_first=True
            attended_features, _ = self.attention(
                query=combined_features,
                key=combined_features,
                value=combined_features,
                key_padding_mask=(
                    ~attention_mask.bool() if attention_mask is not None else None
                ),
            )
        else:
            # FlashAttention or SparseAttention
            attended_features = (
                self.attention(combined_features, attention_mask=attention_mask)
                if attention_mask is not None
                else self.attention(combined_features)
            )

        # Apply MLP fusion if enabled
        if self.use_mlp_fusion:
            mlp_output = self.mlp_fusion(attended_features)
            fused_features = attended_features + mlp_output
        else:
            fused_features = attended_features

        # Apply layer normalization
        output = self.layer_norm(fused_features)

        # Truncate back to original sequence length
        output = output[:, : language_features.shape[1], :]

        return output


def create_qwen3_vl_projection_layer(
    config: Any, layer_idx: int = 0  # Qwen3VL2BConfig
) -> Qwen3VL2BProjectionLayer:
    """
    Factory function to create a Qwen3-VL-2B projection layer.

    Args:
        config: Qwen3VL2BConfig object
        layer_idx: Index of the layer (for layer-specific configurations)

    Returns:
        Qwen3VL2BProjectionLayer: The created projection layer
    """
    vision_dim = getattr(config, "vision_hidden_size", 1024)
    language_dim = getattr(config, "hidden_size", 2048)
    intermediate_dim = getattr(config, "intermediate_size", 5504)

    return Qwen3VL2BProjectionLayer(
        vision_dim=vision_dim,
        language_dim=language_dim,
        intermediate_dim=intermediate_dim,
        use_bias=getattr(config, "use_bias_in_projection", True),
        activation=getattr(config, "projection_activation", "gelu"),
        dropout_prob=getattr(config, "projection_dropout", 0.1),
        layer_norm_eps=getattr(config, "layer_norm_eps", 1e-6),
        use_residual=getattr(config, "use_residual_in_projection", True),
        use_low_rank=getattr(config, "use_low_rank_projection", False),
        low_rank_dim=getattr(config, "low_rank_projection_dim", None),
        use_group_norm=getattr(config, "use_group_norm_in_projection", False),
        num_groups=getattr(config, "group_norm_num_groups", 32),
    )


def create_qwen3_vl_multimodal_projector(
    config: Any, layer_idx: int = 0  # Qwen3VL2BConfig
) -> Qwen3VL2BMultiModalProjector:
    """
    Factory function to create a Qwen3-VL-2B multimodal projector.

    Args:
        config: Qwen3VL2BConfig object
        layer_idx: Index of the layer (for layer-specific configurations)

    Returns:
        Qwen3VL2BMultiModalProjector: The created multimodal projector
    """
    vision_dim = getattr(config, "vision_hidden_size", 1024)
    language_dim = getattr(config, "hidden_size", 2048)
    num_layers = getattr(config, "num_projection_layers", 2)

    return Qwen3VL2BMultiModalProjector(
        vision_dim=vision_dim,
        language_dim=language_dim,
        num_layers=num_layers,
        intermediate_dim=getattr(config, "intermediate_size", 5504),
        use_bias=getattr(config, "use_bias_in_projection", True),
        activation=getattr(config, "projection_activation", "gelu"),
        dropout_prob=getattr(config, "projection_dropout", 0.1),
        layer_norm_eps=getattr(config, "layer_norm_eps", 1e-6),
        use_residual=getattr(config, "use_residual_in_projection", True),
        use_low_rank=getattr(config, "use_low_rank_projection", False),
        low_rank_dim=getattr(config, "low_rank_projection_dim", None),
        use_group_norm=getattr(config, "use_group_norm_in_projection", False),
        num_groups=getattr(config, "group_norm_num_groups", 32),
        use_cross_attention=getattr(config, "use_cross_attention_in_projection", True),
        num_attention_heads=getattr(config, "num_projection_attention_heads", 8),
    )


def apply_qwen3_vl_projection_optimizations(
    model: nn.Module, config: Any  # Qwen3VL2BConfig
) -> nn.Module:
    """
    Apply Qwen3-VL-2B specific projection optimizations to the model.

    Args:
        model: The Qwen3-VL-2B model to optimize
        config: Qwen3VL2BConfig object

    Returns:
        nn.Module: The optimized model
    """
    logger.info("Applying Qwen3-VL-2B specific projection optimizations...")

    # Identify and replace projection layers in the model
    for name, module in model.named_modules():
        if "projector" in name.lower() or "projection" in name.lower():
            # Replace with Qwen3-VL-2B optimized projection layer
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                vision_dim = module.in_features
                language_dim = module.out_features

                new_projection = Qwen3VL2BProjectionLayer(
                    vision_dim=vision_dim,
                    language_dim=language_dim,
                    intermediate_dim=getattr(config, "intermediate_size", 5504),
                    use_bias=getattr(config, "use_bias_in_projection", True),
                    activation=getattr(config, "projection_activation", "gelu"),
                    dropout_prob=getattr(config, "projection_dropout", 0.1),
                    layer_norm_eps=getattr(config, "layer_norm_eps", 1e-6),
                    use_residual=getattr(config, "use_residual_in_projection", True),
                    use_low_rank=getattr(config, "use_low_rank_projection", False),
                    low_rank_dim=getattr(config, "low_rank_projection_dim", None),
                    use_group_norm=getattr(
                        config, "use_group_norm_in_projection", False
                    ),
                    num_groups=getattr(config, "group_norm_num_groups", 32),
                )

                # Find parent module and replace the child
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent_module = model
                if parent_name:
                    for parent_part in parent_name.split("."):
                        parent_module = getattr(parent_module, parent_part)

                setattr(parent_module, child_name, new_projection)
                logger.info(
                    f"Replaced projection layer '{name}' with Qwen3-VL-2B optimized version"
                )

    logger.info("Qwen3-VL-2B projection optimizations applied successfully")
    return model
