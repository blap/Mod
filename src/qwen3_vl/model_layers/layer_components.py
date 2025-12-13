"""
Layer components for Qwen3-VL model
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.qwen3_vl.config.config import Qwen3VLConfig


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Replicate key and value tensors n_rep times along the head dimension.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3VLMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        if self.config.use_adapters:
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            down_input = gate_output * self.act_fn(up_output)

            # Apply adapter if present
            if hasattr(self, 'adapter') and self.adapter is not None:
                down_input = down_input + self.adapter(down_input)

            return self.down_proj(down_input)
        else:
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            down_input = gate_output * self.act_fn(up_output)
            return self.down_proj(down_input)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        from src.qwen3_vl.model_layers.attention_mechanisms import Qwen3VLAttention
        self.self_attn = Qwen3VLAttention(config, layer_idx)

        self.mlp = Qwen3VLMLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3VLVisionLayer(nn.Module):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

        from src.qwen3_vl.model_layers.attention_mechanisms import Qwen3VLVisionAttention
        self.self_attn = Qwen3VLVisionAttention(config)

        # Qwen3VLVisionMLP is already defined in this module
        self.mlp = Qwen3VLVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs