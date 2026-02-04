"""
Gated Attention Module for Qwen3-Coder-Next
"""

import torch
import torch.nn as nn
import math
from .rotary import apply_rotary_pos_emb

class GatedAttention(nn.Module):
    """
    Gated Attention with GQA (Grouped Query Attention)

    Specs:
    - 16 Query Heads
    - 2 KV Heads (GQA Factor 8)
    - Head Dim 256
    - RoPE Dim 64
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # 16
        self.num_key_value_heads = config.num_key_value_heads # 2
        self.head_dim = config.attention_head_dim # 256
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # Gate projection unique to "Gated Attention"
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        rotary_emb=None
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        gate_states = torch.sigmoid(self.gate_proj(hidden_states))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        gate_states = gate_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Reuse KV cache logic
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states)

        # Repeat KV heads for GQA
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)

        # Standard SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True
        )

        # Apply Gating
        attn_output = attn_output * gate_states

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        output = self.o_proj(attn_output)

        return output, past_key_value
