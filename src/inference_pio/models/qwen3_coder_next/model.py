"""
Qwen3-Coder-Next Model Implementation (Dependency-Free)
"""

from typing import Optional, Tuple, Union, List, Dict, Any
import logging
import math

from ...common.config import CustomGenerationConfig
from ...core.engine import backend
from ...core.engine.backend import Tensor, Module, Linear, Embedding, RMSNorm

logger = logging.getLogger(__name__)

class Qwen3CoderNextModel(Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        for i in range(config.num_hidden_layers):
            layer = Qwen3CoderNextDecoderLayer(config)
            self.layers.append(layer)
            self._modules[f"layer_{i}"] = layer
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: Optional[Tensor] = None):
        if input_ids is None: raise ValueError("input_ids required")
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3CoderNextDecoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3CoderNextAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Qwen3CoderNextMLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3CoderNextAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Stub for RoPE inputs (functional flow)
        cos = Tensor(list(q.shape))
        sin = Tensor(list(q.shape))
        cos.fill(1.0)
        sin.fill(0.0)
        q, k = q.rope(k, cos, sin)

        # Attention Score: Q * K^T
        # Use matmul with transpose_b=True
        scores = q.matmul(k, transpose_b=True)

        # Scale
        scale_tensor = Tensor(list(scores.shape), device=scores.device)
        scale_tensor.fill(self.scale)
        scores = scores * scale_tensor

        attn_probs = scores.softmax()
        context = attn_probs.matmul(v)
        output = self.o_proj(context)
        return output

class Qwen3CoderNextMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x).silu()
        up = self.up_proj(x)
        merged = gate * up
        return self.down_proj(merged)
