"""
Multimodal projector component for Qwen3-VL
"""
import torch
import torch.nn as nn
from typing import Optional
from src.qwen3_vl.config.config import Qwen3VLConfig


class Qwen3VLMultimodalProjector(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Create projection layers
        self.linear_1 = nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states