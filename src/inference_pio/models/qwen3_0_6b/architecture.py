"""
Qwen3-0.6B Architecture - C Backend
"""

import math
from typing import Optional, Tuple, List

from ...core.engine.layers import Module, Linear, Embedding, RMSNorm, ModuleList
from ...core.engine.tensor_ops import softmax, matmul, silu, apply_rotary_emb, precompute_freqs_cis
from ...core.engine.backend import Tensor

class Qwen3RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.cos, self.sin = precompute_freqs_cis(dim, max_position_embeddings, base)
        self.register_buffer("cos_cached", self.cos)
        self.register_buffer("sin_cached", self.sin)

    def forward(self, x, seq_len):
        return self.cos, self.sin # Simplified return

class Qwen3MLP(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # x: Tensor
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(gate.silu().mul(up))

class Qwen3Attention(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = Qwen3RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        # Simplified forward
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # RoPE not fully implemented in this loop for C backend, pass

        # Attn: Q @ K.T
        # Transpose logic is implicit in C-matmul or needs explicit call
        attn = q.matmul(k)
        attn = attn.softmax()
        out = attn.matmul(v)
        return self.o_proj(out), None, None

class Qwen3DecoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(hidden_states)
        hidden_states = residual.add(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual.add(hidden_states)

        return hidden_states, None

class Qwen3Model(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids=None, **kwargs):
        # input_ids: Tensor
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
        return self.norm(hidden_states), None

class Qwen3ForCausalLM(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, **kwargs):
        hidden_states, _ = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits, None

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        # Stub loop
        return input_ids
