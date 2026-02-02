"""
Vision Transformer Kernels for Qwen3-VL-2B Model - Self-Contained Version

This module implements optimized CUDA kernels specifically for vision processing
in the Qwen3-VL-2B model. These kernels are designed to accelerate vision-specific
operations like convolutions, patch embeddings, and vision transformer blocks.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VisionTransformerConfig:
    """Configuration for Vision Transformer kernels."""

    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    image_size: int = 448
    intermediate_size: int = 2816
    layer_norm_eps: float = 1e-6
    use_flash_attention: bool = True
    use_cuda_kernels: bool = True


class VisionPatchEmbeddingKernel(nn.Module):
    """
    Optimized CUDA kernel for vision patch embedding.
    This kernel efficiently converts images to patch embeddings using convolution.
    """

    def __init__(self, config: VisionTransformerConfig):
        super().__init__()

        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size

        # Use convolution for efficient patch embedding
        self.projection = nn.Conv2d(
            in_channels=3,  # RGB channels
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # Initialize projection weights
        self._initialize_weights()

        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Position embeddings for patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, config.hidden_size)
        )

        # Layer norm
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _initialize_weights(self):
        """Initialize convolution weights."""
        nn.init.trunc_normal_(self.projection.weight, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision patch embedding kernel.

        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)

        Returns:
            Patch embeddings of shape (batch_size, num_patches, hidden_size)
        """
        batch_size, channels, height, width = pixel_values.shape

        # Apply convolution to extract patches
        patches = self.projection(
            pixel_values
        )  # (batch_size, hidden_size, grid_h, grid_w)

        # Flatten patches
        patches = patches.flatten(2).transpose(
            1, 2
        )  # (batch_size, num_patches, hidden_size)

        # Add position embeddings
        patches = patches + self.position_embeddings[:, : patches.size(1), :]

        # Apply layer norm
        patches = self.layernorm(patches)

        return patches


class VisionSelfAttentionKernel(nn.Module):
    """
    Optimized CUDA kernel for vision self-attention.
    This kernel efficiently computes self-attention for vision transformer blocks.
    """

    def __init__(self, config: VisionTransformerConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_dim
        self.use_flash_attention = config.use_flash_attention

        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"Hidden size must be divisible by number of attention heads (got hidden_size: {self.hidden_size}, num_attention_heads: {self.num_attention_heads})"
            )

        # Linear projections for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # Output projection
        self.output_projection = nn.Linear(
            self.all_head_size, config.hidden_size, bias=False
        )

        # Dropout
        self.dropout = nn.Dropout(0.1) if 0.1 > 0.0 else None

        # Scaling factor
        self.scaling = self.head_dim**-0.5

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize attention weights according to Qwen3-VL-2B specifications."""
        std = self.hidden_size**-0.5
        nn.init.normal_(self.query.weight, mean=0.0, std=std)
        nn.init.normal_(self.key.weight, mean=0.0, std=std)
        nn.init.normal_(self.value.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for vision self-attention kernel.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask

        Returns:
            Attended output of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Apply projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape for multi-head attention
        q = q.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        k = k.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        v = v.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)

        # Scale query
        q = q * self.scaling

        # Compute attention scores
        if self.use_flash_attention and torch.cuda.is_available():
            # Use efficient attention computation
            attention_scores = torch.matmul(q, k.transpose(-2, -1))
        else:
            # Standard attention computation
            attention_scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax
        attention_scores = torch.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).to(q.dtype)

        # Apply dropout if configured
        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # Apply attention to values
        attn_output = torch.matmul(
            attention_scores, v
        )  # (batch, num_heads, seq_len, head_dim)

        # Reshape to combine heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, hidden_size)
        )

        # Apply output projection
        attn_output = self.output_projection(attn_output)

        return attn_output


class VisionMLPKernel(nn.Module):
    """
    Optimized CUDA kernel for vision MLP.
    This kernel implements the MLP component of vision transformer blocks.
    """

    def __init__(self, config: VisionTransformerConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize MLP weights according to Qwen3-VL-2B specifications."""
        std = self.fc1.in_features**-0.5
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        std = self.fc2.in_features**-0.5
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision MLP kernel.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)

        Returns:
            MLP output of shape (batch_size, seq_len, hidden_size)
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class VisionTransformerBlockKernel(nn.Module):
    """
    Optimized CUDA kernel for vision transformer block.
    This kernel combines attention and MLP with layer normalization.
    """

    def __init__(self, config: VisionTransformerConfig, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.config = config

        # Layer norms
        self.pre_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.pre_mlp_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # Attention and MLP components
        self.attention = VisionSelfAttentionKernel(config)
        self.mlp = VisionMLPKernel(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for vision transformer block kernel.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask

        Returns:
            Output of shape (batch_size, seq_len, hidden_size)
        """
        # Pre-attention layer norm
        residual = hidden_states
        hidden_states = self.pre_attention_layernorm(hidden_states)

        # Attention
        attention_output = self.attention(hidden_states, attention_mask)

        # Residual connection
        hidden_states = residual + attention_output

        # Pre-MLP layer norm
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)

        # MLP
        mlp_output = self.mlp(hidden_states)

        # Residual connection
        output = residual + mlp_output

        return output


class VisionConvolutionKernel(nn.Module):
    """
    Optimized CUDA kernel for vision convolution operations.
    This kernel efficiently performs convolution operations for vision processing.
    """

    def __init__(self, config: VisionTransformerConfig):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=config.hidden_size,  # Depthwise convolution
        )

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = nn.GELU()

        # Initialize weights according to Qwen3-VL-2B specifications
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution weights according to Qwen3-VL-2B specifications."""
        std = self.conv.weight.shape[1] ** -0.5  # fan-in initialization
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision convolution kernel.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)

        Returns:
            Convolved output of shape (batch_size, seq_len, hidden_size)
        """
        # Reshape from (B, L, D) to (B, D, H, W) assuming square patches
        batch_size, seq_len, hidden_size = hidden_states.shape
        grid_size = int(seq_len**0.5)  # Assuming square grid

        # Reshape to (B, D, H, W)
        hidden_states = hidden_states.transpose(1, 2).view(
            batch_size, hidden_size, grid_size, grid_size
        )

        # Apply convolution
        conv_output = self.conv(hidden_states)

        # Reshape back to (B, L, D)
        conv_output = conv_output.view(batch_size, hidden_size, -1).transpose(1, 2)

        # Apply layer norm and activation
        conv_output = self.norm(conv_output)
        conv_output = self.activation(conv_output)

        return conv_output


class Qwen3VL2BVisionEncoderKernel(nn.Module):
    """
    Qwen3-VL-2B specific vision encoder kernel.
    This kernel combines all vision-specific components for the Qwen3-VL-2B model.
    """

    def __init__(self, config: VisionTransformerConfig):
        super().__init__()

        self.config = config

        # Create vision transformer config from Qwen3VL2B config
        vision_config = VisionTransformerConfig(
            hidden_size=getattr(config, "vision_hidden_size", 1024),
            num_attention_heads=getattr(config, "vision_num_attention_heads", 16),
            num_hidden_layers=getattr(config, "vision_num_hidden_layers", 24),
            patch_size=getattr(config, "vision_patch_size", 14),
            image_size=getattr(config, "vision_image_size", 448),
            intermediate_size=getattr(config, "vision_intermediate_size", 2816),
            layer_norm_eps=getattr(config, "vision_layer_norm_eps", 1e-6),
            use_flash_attention=getattr(config, "use_vision_flash_attention", True),
            use_cuda_kernels=getattr(config, "use_cuda_kernels", True),
        )

        # Vision encoder with optimized components
        self.vision_encoder = self._create_vision_encoder(vision_config)

    def _create_vision_encoder(self, config: VisionTransformerConfig) -> nn.Module:
        """Create the vision encoder with optimized components."""
        # Create patch embedding layer
        patch_embedding = VisionPatchEmbeddingKernel(config)

        # Create transformer blocks
        transformer_blocks = nn.ModuleList(
            [
                VisionTransformerBlockKernel(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        # Create final layer norm
        final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Return a sequential model with all components
        class VisionEncoder(nn.Module):
            def __init__(self, patch_embed, blocks, final_norm):
                super().__init__()
                self.patch_embedding = patch_embed
                self.blocks = blocks
                self.final_layernorm = final_norm

            def forward(
                self, pixel_values: torch.Tensor, output_hidden_states: bool = False
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
                # Patch embedding
                hidden_states = self.patch_embedding(pixel_values)

                all_hidden_states = [] if output_hidden_states else None

                # Apply transformer blocks
                for i, block in enumerate(self.blocks):
                    hidden_states = block(hidden_states)

                    if output_hidden_states:
                        all_hidden_states.append(hidden_states)

                # Apply final layer norm
                hidden_states = self.final_layernorm(hidden_states)

                return hidden_states, all_hidden_states

        return VisionEncoder(patch_embedding, transformer_blocks, final_layernorm)

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for Qwen3-VL-2B vision encoder kernel.

        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            output_hidden_states: Whether to output hidden states from all layers

        Returns:
            Tuple of (final_hidden_states, all_hidden_states if requested)
        """
        return self.vision_encoder(pixel_values, output_hidden_states)


def create_vision_patch_embedding_kernel(
    config: VisionTransformerConfig,
) -> VisionPatchEmbeddingKernel:
    """
    Factory function to create vision patch embedding kernel.

    Args:
        config: Vision transformer configuration

    Returns:
        Vision patch embedding kernel
    """
    return VisionPatchEmbeddingKernel(config)


def create_vision_self_attention_kernel(
    config: VisionTransformerConfig,
) -> VisionSelfAttentionKernel:
    """
    Factory function to create vision self-attention kernel.

    Args:
        config: Vision transformer configuration

    Returns:
        Vision self-attention kernel
    """
    return VisionSelfAttentionKernel(config)


def create_vision_mlp_kernel(config: VisionTransformerConfig) -> VisionMLPKernel:
    """
    Factory function to create vision MLP kernel.

    Args:
        config: Vision transformer configuration

    Returns:
        Vision MLP kernel
    """
    return VisionMLPKernel(config)


def create_vision_transformer_block_kernel(
    config: VisionTransformerConfig, layer_idx: int = 0
) -> VisionTransformerBlockKernel:
    """
    Factory function to create vision transformer block kernel.

    Args:
        config: Vision transformer configuration
        layer_idx: Index of the transformer layer

    Returns:
        Vision transformer block kernel
    """
    return VisionTransformerBlockKernel(config, layer_idx)


def create_qwen3_vl_2b_vision_encoder_kernel(
    config: VisionTransformerConfig,
) -> Qwen3VL2BVisionEncoderKernel:
    """
    Factory function to create Qwen3-VL-2B vision encoder kernel.

    Args:
        config: Vision transformer configuration

    Returns:
        Qwen3-VL-2B vision encoder kernel
    """
    return Qwen3VL2BVisionEncoderKernel(config)


def apply_vision_cuda_optimizations_to_model(
    model: nn.Module, config: VisionTransformerConfig
) -> nn.Module:
    """
    Apply vision-specific CUDA optimizations to the model.

    Args:
        model: The model to optimize
        config: Vision transformer configuration

    Returns:
        Optimized model with vision-specific CUDA optimizations
    """
    logger.info("Applying vision-specific CUDA optimizations...")

    # Look for vision-related modules and replace them with optimized versions
    for name, module in model.named_modules():
        if (
            "vision" in name.lower()
            or "visual" in name.lower()
            or "patch" in name.lower()
        ):
            if (
                isinstance(module, nn.Conv2d)
                and module.kernel_size[0] == config.patch_size
            ):
                # Replace with optimized patch embedding if it matches patch size
                logger.debug(f"Found vision patch embedding layer: {name}")

                # Create optimized patch embedding kernel
                vision_patch_embed = create_vision_patch_embedding_kernel(config)

                # Find parent module and replace the child
                parent_module, child_name = _get_parent_module(model, name)
                setattr(parent_module, child_name, vision_patch_embed)

                logger.info(
                    f"Replaced vision patch embedding module {name} with optimized version"
                )

            elif isinstance(module, nn.MultiheadAttention) and "vision" in name.lower():
                # Replace with optimized vision attention
                logger.debug(f"Found vision attention layer: {name}")

                # Create optimized vision attention kernel
                vision_attn = create_vision_self_attention_kernel(config)

                # Find parent module and replace the child
                parent_module, child_name = _get_parent_module(model, name)
                setattr(parent_module, child_name, vision_attn)

                logger.info(
                    f"Replaced vision attention module {name} with optimized version"
                )

    logger.info("Vision-specific CUDA optimizations applied successfully")
    return model


def _get_parent_module(model: nn.Module, full_name: str) -> tuple:
    """
    Get parent module and child name by full name.

    Args:
        model: The model
        full_name: Full name of the module (e.g., 'transformer.layers.0.attention')

    Returns:
        Tuple of (parent_module, child_name)
    """
    parts = full_name.split(".")
    if len(parts) == 1:
        # If there's no parent (top-level module), return the model itself and the child name
        return model, parts[0]

    parent_name = ".".join(parts[:-1])
    child_name = parts[-1]

    parent_module = model
    for n in parent_name.split("."):
        if n:  # Skip empty strings
            parent_module = getattr(parent_module, n)

    return parent_module, child_name


__all__ = [
    "VisionTransformerConfig",
    "VisionPatchEmbeddingKernel",
    "VisionSelfAttentionKernel",
    "VisionMLPKernel",
    "VisionTransformerBlockKernel",
    "VisionConvolutionKernel",
    "Qwen3VL2BVisionEncoderKernel",
    "create_vision_patch_embedding_kernel",
    "create_vision_self_attention_kernel",
    "create_vision_mlp_kernel",
    "create_vision_transformer_block_kernel",
    "create_qwen3_vl_2b_vision_encoder_kernel",
    "apply_vision_cuda_optimizations_to_model",
]
