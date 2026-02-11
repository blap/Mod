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

    def forward(self, input_ids: Tensor, past_key_values: Optional[List[Tensor]] = None, use_cache: bool = False, cache_position: int = 0):
        h = self.embed_tokens(input_ids)

        # Initialize empty cache if needed but usually passed from generate
        if use_cache and past_key_values is None:
             # Just a safety check, generate handles alloc
             past_key_values = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)

            pkv = past_key_values[i] if past_key_values else None
            h = layer(h, past_key_value=pkv, use_cache=use_cache, cache_position=cache_position)

        h = self.final_layernorm(h)
        return h

    def generate(self, input_ids: Tensor, max_new_tokens: int = 10):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_seq_len = seq_len + max_new_tokens

        # 1. Pre-allocate Static KV Cache: [Layers, 2(K,V), B, MaxSeq, Heads, HeadDim]
        device = input_ids.device
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        past_key_values = []

        for _ in range(len(self.layers)):
             k_cache = Tensor([batch_size, max_seq_len, self.config.num_attention_heads, head_dim], device=device)
             v_cache = Tensor([batch_size, max_seq_len, self.config.num_attention_heads, head_dim], device=device)
             k_cache.fill(0.0)
             v_cache.fill(0.0)
             past_key_values.append((k_cache, v_cache))

        current_ids = input_ids
        for step in range(max_new_tokens):
            curr_seq_len = current_ids.shape[1]
            if step == 0:
                model_input = current_ids
                cache_position = 0
            else:
                model_input = current_ids.slice([0, curr_seq_len-1], [batch_size, 1])
                cache_position = curr_seq_len - 1

            h = self.forward(model_input, past_key_values=past_key_values, use_cache=True, cache_position=cache_position)
            logits = self.lm_head(h)

            vocab_size = logits.shape[2]
            last_logits = logits.slice([0, logits.shape[1]-1, 0], [batch_size, 1, vocab_size])
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

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        h = self.input_layernorm(x)
        h = self.self_attention(h, past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position)
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

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        # QKV Proj
        qkv = self.query_key_value(x)

        # Split using backend slice logic
        B, Seq, _ = qkv.shape
        H = self.hidden_size

        q = qkv.slice([0, 0, 0], [B, Seq, H])
        k = qkv.slice([0, 0, H], [B, Seq, H])
        v = qkv.slice([0, 0, 2*H], [B, Seq, H])

        # Reshape: [B, S, Heads, HeadDim]
        new_shape = [B, Seq, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # RoPE
        # Use cache_position for correct offset
        start = [cache_position, 0]
        shape = [Seq, self.cos.shape[1]]
        c = self.cos.slice(start, shape)
        s = self.sin.slice(start, shape)

        q, k = q.rope(k, c, s)

        # KV Cache Update (In-Place Static)
        if use_cache and past_key_value is not None:
             k_cache, v_cache = past_key_value
             # Write to static buffer
             start_indices = [0, cache_position, 0, 0]
             k_cache.set_slice(k, start_indices)
             v_cache.set_slice(v, start_indices)

             # View valid context
             valid_len = cache_position + Seq
             k = k_cache.slice([0,0,0,0], [B, valid_len, self.num_heads, self.head_dim])
             v = v_cache.slice([0,0,0,0], [B, valid_len, self.num_heads, self.head_dim])

        # Attention
        out = scaled_dot_product_attention(q, k, v, scale=self.scale)
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
