"""
Vision transformer component for Qwen3-VL
"""
import torch
import torch.nn as nn
from typing import Optional
from src.qwen3_vl.config.config import Qwen3VLConfig
from src.qwen3_vl.model_layers.layer_components import Qwen3VLVisionLayer


class Qwen3VLVisionTransformer(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Vision patch embedding
        self.embeddings = nn.Conv2d(
            in_channels=config.vision_num_channels,
            out_channels=config.vision_hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            bias=True
        )
        
        # Positional embeddings
        self.embed_positions = nn.Parameter(
            torch.randn((config.vision_image_size // config.vision_patch_size) ** 2 + 1, config.vision_hidden_size)
        )
        
        # Vision transformer layers
        self.layers = nn.ModuleList([
            Qwen3VLVisionLayer(config) 
            for _ in range(config.vision_num_hidden_layers)
        ])
        
        # Layer norm
        self.post_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        # Get batch size, channels, height, width
        batch_size, channels, height, width = pixel_values.shape
        
        # Embed patches
        patch_embeds = self.embeddings(pixel_values)
        batch_size, num_channels, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (batch_size, num_patches, hidden_size)
        
        # Add positional embeddings
        embeddings = patch_embeds + self.embed_positions[:patch_embeds.size(1)]
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Apply final layer norm
        hidden_states = self.post_layernorm(hidden_states)
        
        return hidden_states