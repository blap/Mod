"""
GLM-4.7-Flash Model Implementation - Self-Contained
"""

import logging
from typing import Any, Dict, List, Optional

from ...core.engine.backend import Module, Tensor, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention, cat
from ...common.custom_components.tokenizer import CustomBPETokenizer
from .config import GLM47FlashConfig

logger = logging.getLogger(__name__)

class GLM47FlashModel(Module):
    def __init__(self, config: GLM47FlashConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []

        dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = precompute_freqs_cis(dim, config.max_position_embeddings)

        # Config uses num_hidden_layers
        for i in range(config.num_hidden_layers):
            l = GLM47FlashLayer(config, self.cos, self.sin)
            self.layers.append(l)
            self._modules[f"layer_{i}"] = l

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.scheduler = None

    def forward(self, input_ids: Tensor):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)
            h = layer(h)
        h = self.final_layernorm(h)
        return h

    def generate(self, input_ids: Tensor, max_new_tokens: int = 10):
        current_ids = input_ids
        for _ in range(max_new_tokens):
            h = self.forward(current_ids)
            logits = self.lm_head(h)

            # Greedy decode last token
            # logits: [B, S, Vocab]
            vocab_size = logits.shape[2]
            last_logits = logits.slice([0, logits.shape[1]-1, 0], [1, 1, vocab_size])
            next_token = last_logits.argmax()

            current_ids = cat([current_ids, next_token], axis=1)
        return current_ids

class GLM47FlashLayer(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention = GLM47FlashAttention(config, cos, sin)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GLM47FlashMLP(config)

    def forward(self, x):
        h = self.input_layernorm(x)
        h = self.self_attention(h)
        x = x + h
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        return x + h

class GLM47FlashAttention(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.query_key_value = Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cos = cos
        self.sin = sin
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # QKV Proj
        qkv = self.query_key_value(x)

        # Split using backend slice logic
        # qkv: [B, Seq, 3*H]
        B, Seq, _ = qkv.shape
        H = self.hidden_size

        q = qkv.slice([0, 0, 0], [B, Seq, H])
        k = qkv.slice([0, 0, H], [B, Seq, H])
        v = qkv.slice([0, 0, 2*H], [B, Seq, H])

        # Reshape to 4D for Multi-Head Attention: [B, S, Heads, HeadDim]
        new_shape = [B, Seq, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # RoPE
        start = [0, 0]
        shape = [Seq, self.cos.shape[1]]
        c = self.cos.slice(start, shape)
        s = self.sin.slice(start, shape)

        # Apply RoPE (works on last dim HeadDim)
        q, k = q.rope(k, c, s)

        # Attention (Scaled Dot Product)
        # Handles correct batching over heads
        out = scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape back to [B, Seq, Hidden]
        out = out.reshape([B, Seq, H])

        return self.dense(out)

class GLM47FlashMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.dense_4h_to_h = Linear(config.hidden_size * 4, config.hidden_size, bias=False)

    def forward(self, x):
        h = self.dense_h_to_4h(x)
        h = h.gelu()
        return self.dense_4h_to_h(h)

__all__ = ["GLM47FlashModel"]
