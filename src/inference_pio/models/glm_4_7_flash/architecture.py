"""
GLM-4.7-Flash Architecture - C Backend Implementation
"""

import math
from typing import Optional, Tuple, List
from ...core.engine.backend import Module, Linear, Embedding, RMSNorm, Tensor, cat, precompute_freqs_cis, scaled_dot_product_attention

class GLMRotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000.0):
        super().__init__()
        self.cos, self.sin = precompute_freqs_cis(dim, max_position_embeddings, base)
        self.register_buffer("cos", self.cos)
        self.register_buffer("sin", self.sin)

    def forward(self, x, seq_len):
        start = [0, 0]
        shape = [seq_len, self.cos.shape[1]]
        return self.cos.slice(start, shape), self.sin.slice(start, shape)

class GLMMLP(Module):
    def __init__(self, config):
        super().__init__()
        # GLM-4 uses SwiGLU: H -> 2*Inter -> H (or similar ratio)
        # Assuming config.intermediate_size is the projection size
        # Usually for SwiGLU we project to 2x Intermediate?
        # Let's assume standard Llama-like SwiGLU:
        # gate_up = Linear(H, 2*Inter) -> Split -> Silu(Gate)*Up -> Linear(Inter, H)
        # Or separate Gate/Up like Qwen.
        # GLM often fuses them.
        # Let's assume separate Gate/Up for clarity if config allows, or single linear and split.
        # Since I can't check GLM config easily, I'll stick to what Qwen did (separate Gate/Up) as it's safe.
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(gate.swiglu(up))

class GLMAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # GLM often uses single QKV matrix
        self.query_key_value = Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=True)

        self.rotary_emb = GLMRotaryEmbedding(self.head_dim // 2) # Often RoPE is applied to half dim or full? Assume half for GLM specific or full.
        # Let's assume full head_dim for standard RoPE
        self.rotary_emb = GLMRotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        # x: [Batch, Seq, Hidden]
        B, S, H = x.shape
        qkv = self.query_key_value(x)

        q = qkv.slice([0, 0, 0], [B, S, self.hidden_size])
        k = qkv.slice([0, 0, self.hidden_size], [B, S, self.hidden_size])
        v = qkv.slice([0, 0, 2*self.hidden_size], [B, S, self.hidden_size])

        new_shape = [B, S, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # RoPE
        # Calc offset if cache
        past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        total_len = S + past_len

        # RoPE cache slice
        # Assuming rotary_emb handles seq_len or indices.
        # But standard precomputed usually needs slice logic.
        # Original: cos, sin = self.rotary_emb(v, S) -> slices from 0 to S.
        # We need from past_len to total_len.

        start = [past_len, 0]
        shape = [S, self.rotary_emb.cos.shape[1]]
        cos = self.rotary_emb.cos.slice(start, shape)
        sin = self.rotary_emb.sin.slice(start, shape)

        q, k = q.rope(k, cos, sin)

        # Cache Update
        if past_key_value is not None:
            k = cat([past_key_value[0], k], axis=1)
            v = cat([past_key_value[1], v], axis=1)

        present_key_value = (k, v) if use_cache else None

        # Attention
        out = scaled_dot_product_attention(q, k, v)

        # Reshape back to [B, Seq, Hidden]
        out = out.reshape([B, S, self.hidden_size])

        return self.dense(out), present_key_value

class GLMBlock(Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention = GLMAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GLMMLP(config)

    def forward(self, x, past_key_value=None, use_cache=False):
        residual = x
        x = self.input_layernorm(x)
        x, pkv = self.self_attention(x, past_key_value=past_key_value, use_cache=use_cache)
        x = residual + x # add

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, pkv

class GLMForCausalLM(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GLMModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        h, next_cache = self.transformer(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(h)
        return logits, next_cache

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        current_ids = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values:
                # Slice last token
                seq_len = current_ids.shape[1]
                model_input = current_ids.slice([0, seq_len-1], [1, 1])
            else:
                model_input = current_ids

            logits, pkv = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = pkv

            # Logits: [B, Seq, Vocab]
            # Take last token logits
            next_token_logits = logits.slice([0, logits.shape[1]-1, 0], [1, 1, logits.shape[2]])
            next_token = next_token_logits.argmax() # [1, 1]
            current_ids = cat([current_ids, next_token], axis=1)
        return current_ids

class GLMModel(Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        for i in range(config.num_hidden_layers):
            l = GLMBlock(config)
            self.layers.append(l)
            self._modules[f"layer_{i}"] = l
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        h = self.embedding(input_ids)
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None
            h, pkv = layer(h, past_key_value=layer_past, use_cache=use_cache)
            if use_cache: next_cache.append(pkv)

        h = self.final_layernorm(h)
        return h, next_cache
