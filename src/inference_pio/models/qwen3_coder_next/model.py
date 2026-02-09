"""
Qwen3-Coder-Next Model Implementation (Dependency-Free)
"""

from typing import Optional, Tuple, Union, List, Dict, Any
import logging
import math

from ...core.engine.backend import Tensor, Module, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention
from .config import Qwen3CoderNextConfig

logger = logging.getLogger(__name__)

class Qwen3CoderNextModel(Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

        # Precompute RoPE Cache (Global for model)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb_dim = config.attention_rope_dim if hasattr(config, 'attention_rope_dim') else head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_base = config.rope_theta

        # Create cache on device (default cpu, moved in .to())
        self.cos_cache, self.sin_cache = precompute_freqs_cis(self.rotary_emb_dim, self.max_position_embeddings, self.rope_base)
        self.register_buffer("cos_cache", self.cos_cache)
        self.register_buffer("sin_cache", self.sin_cache)

        self.layers = []
        for i in range(config.num_hidden_layers):
            layer = Qwen3CoderNextDecoderLayer(config, self.cos_cache, self.sin_cache)
            self.layers.append(layer)
            self._modules[f"layer_{i}"] = layer

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: Optional[Tensor] = None):
        if input_ids is None: raise ValueError("input_ids required")

        # Update device of cache if needed (naive check)
        if self.cos_cache.device != input_ids.device:
             self.cos_cache = self.cos_cache.to(input_ids.device)
             self.sin_cache = self.sin_cache.to(input_ids.device)
             for layer in self.layers:
                 layer.self_attn.cos_cache = self.cos_cache
                 layer.self_attn.sin_cache = self.sin_cache

        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3CoderNextDecoderLayer(Module):
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3CoderNextAttention(config, cos_cache, sin_cache)
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
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache references
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def forward(self, hidden_states: Tensor) -> Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape [B, S, Hidden] -> [B, S, Heads, HeadDim]
        B = q.shape[0]
        S = q.shape[1]
        H = q.shape[2]
        new_shape = [B, S, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # Apply RoPE
        seq_len = S
        start_indices = [0, 0]
        slice_shapes = [seq_len, self.cos_cache.shape[1]]

        cos = self.cos_cache.slice(start_indices, slice_shapes)
        sin = self.sin_cache.slice(start_indices, slice_shapes)

        q, k = q.rope(k, cos, sin)

        # Fused Attention
        context = scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Flatten
        context = context.reshape([B, S, H])
        output = self.o_proj(context)
        return output

class Qwen3CoderNextMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Fused SwiGLU
        merged = gate.swiglu(up)
        return self.down_proj(merged)
