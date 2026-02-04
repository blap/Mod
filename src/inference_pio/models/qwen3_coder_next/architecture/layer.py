"""
Qwen3-Coder-Next Hybrid Layer Block
"""

import torch
import torch.nn as nn
from .attention import GatedAttention
from .deltanet import GatedDeltaNet
from .moe import Qwen3CoderNextMoE

class Qwen3CoderNextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine Layer Type based on hybrid pattern
        # default pattern: 3 DeltaNet -> 1 Attention
        pattern_len = len(config.hybrid_block_pattern)
        layer_type = config.hybrid_block_pattern[layer_idx % pattern_len]
        self.layer_type = layer_type

        if layer_type == "attention":
            self.mixer = GatedAttention(config, layer_idx=layer_idx)
        else:
            self.mixer = GatedDeltaNet(config, layer_idx=layer_idx)

        self.moe = Qwen3CoderNextMoE(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        rotary_emb=None
    ):
        # Mixer Block (Attention or DeltaNet)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mixer_out, present_key_value = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **({"position_ids": position_ids, "rotary_emb": rotary_emb} if self.layer_type == "attention" else {})
        )

        hidden_states = residual + mixer_out

        # MoE Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
