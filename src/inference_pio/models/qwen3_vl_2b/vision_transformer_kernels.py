"""
Vision Transformer Kernels for Qwen3-VL-2B Model - Self-Contained Version
Dependency-Free
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...core.engine.backend import Module, Tensor, Linear, Conv2d, RMSNorm, GELU, scaled_dot_product_attention

logger = logging.getLogger(__name__)

@dataclass
class VisionTransformerConfig:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    image_size: int = 448
    intermediate_size: int = 2816
    layer_norm_eps: float = 1e-6
    use_flash_attention: bool = True
    use_cuda_kernels: bool = True

class VisionPatchEmbeddingKernel(Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size

        self.projection = Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Position embedding: [1, num_patches, hidden_size]
        self.position_embeddings = Tensor([1, self.num_patches, config.hidden_size])
        self.register_parameter("position_embeddings", self.position_embeddings)

        self.layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: [B, 3, H, W]
        patches = self.projection(pixel_values) # [B, Hidden, GridH, GridW]
        B = patches.shape[0]
        C = patches.shape[1]
        H = patches.shape[2]
        W = patches.shape[3]

        # [B, C, H, W] -> [B, H, W, C]
        patches = patches.permute([0, 2, 3, 1])

        # [B, H, W, C] -> [B, H*W, C]
        patches = patches.reshape([B, H*W, C])

        return patches

class VisionSelfAttentionKernel(Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output_projection = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Expects [B, Seq, Hidden]
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        out = scaled_dot_product_attention(q, k, v, scale=self.scale)

        return self.output_projection(out)

class VisionMLPKernel(Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = GELU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        h = self.fc1(hidden_states)
        h = self.activation(h)
        h = self.fc2(h)
        return h

class VisionTransformerBlockKernel(Module):
    def __init__(self, config: VisionTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.pre_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = VisionSelfAttentionKernel(config)
        self.mlp = VisionMLPKernel(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        h = self.pre_attention_layernorm(hidden_states)
        h = self.attention(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_mlp_layernorm(hidden_states)
        h = self.mlp(h)
        return residual + h

class Qwen3VL2BVisionEncoderKernel(Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.patch_embedding = VisionPatchEmbeddingKernel(config)
        self.blocks = []
        for i in range(config.num_hidden_layers):
            b = VisionTransformerBlockKernel(config, i)
            self.blocks.append(b)
            self._modules[f"block_{i}"] = b
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor):
        h = self.patch_embedding(pixel_values)
        # HACK: Assume patch_embed output is somehow [B, Seq, Hidden] or compatible
        for block in self.blocks:
            h = block(h)
        h = self.final_layernorm(h)
        return h

# Factories
def create_vision_patch_embedding_kernel(config): return VisionPatchEmbeddingKernel(config)
def create_vision_self_attention_kernel(config): return VisionSelfAttentionKernel(config)
def create_vision_mlp_kernel(config): return VisionMLPKernel(config)
def create_vision_transformer_block_kernel(config, idx=0): return VisionTransformerBlockKernel(config, idx)
def create_qwen3_vl_2b_vision_encoder_kernel(config): return Qwen3VL2BVisionEncoderKernel(config)

def apply_vision_cuda_optimizations_to_model(model, config):
    # Simplified placeholder since we don't have torch.nn.modules to inspect easily in the same way
    return model

__all__ = [
    "VisionTransformerConfig",
    "VisionPatchEmbeddingKernel",
    "VisionSelfAttentionKernel",
    "VisionMLPKernel",
    "VisionTransformerBlockKernel",
    "Qwen3VL2BVisionEncoderKernel"
]
