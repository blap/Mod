"""
Qwen3-Coder-30B Model Implementation - Self-Contained Version
Dependency-Free using Custom Backend
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ...core.engine.backend import Module, Tensor, Linear, Embedding, RMSNorm, precompute_freqs_cis
from ...common.custom_components.tokenizer import CustomBPETokenizer
from .config import Qwen3Coder30BConfig

logger = logging.getLogger(__name__)

class Qwen3Coder30BModel(Module):
    """
    Qwen3-Coder-30B model implementation.
    """

    def __init__(self, config: Qwen3Coder30BConfig):
        super().__init__()
        self.config = config
        self._tokenizer = None

        # Initialize Architecture
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []

        # RoPE Cache
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos_cache, self.sin_cache = precompute_freqs_cis(head_dim, config.max_position_embeddings, config.rope_theta)

        for i in range(config.num_hidden_layers):
            layer = Qwen3Coder30BDecoderLayer(config, self.cos_cache, self.sin_cache)
            self.layers.append(layer)
            self._modules[f"layer_{i}"] = layer

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Helper components
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        try:
            # Assuming tokenizer files are at config.model_path or handled by factory
            # For now, simplistic init or lazy load
            pass
        except Exception as e:
            logger.warning(f"Tokenizer init warning: {e}")

    def get_tokenizer(self):
        if not self._tokenizer:
             # Lazy load
             # self._tokenizer = CustomBPETokenizer(...)
             pass
        return self._tokenizer

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3Coder30BDecoderLayer(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Qwen3Coder30BAttention(config, cos, sin)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Qwen3Coder30BMLP(config)

    def forward(self, x):
        h = self.input_layernorm(x)
        h = self.self_attn(h)
        x = x + h

        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = x + h
        return x

class Qwen3Coder30BAttention(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)

        self.cos = cos
        self.sin = sin
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # RoPE
        seq_len = q.shape[1]
        start = [0, 0]
        shape = [seq_len, self.cos.shape[1]]
        cos_slice = self.cos.slice(start, shape)
        sin_slice = self.sin.slice(start, shape)

        q, k = q.rope(k, cos_slice, sin_slice)

        # Attention
        scores = q.matmul(k, transpose_b=True)

        scale = Tensor(list(scores.shape), device=scores.device)
        scale.fill(self.scale)
        scores = scores * scale

        probs = scores.softmax()
        context = probs.matmul(v)

        return self.o_proj(context)

class Qwen3Coder30BMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

__all__ = ["Qwen3Coder30BModel"]
