"""
Qwen3-0.6B Architecture - C Backend (Real Implementation)
"""

from typing import Optional, Tuple, List
from ...core.engine.backend import Module, Linear, Embedding, RMSNorm, Tensor, cat, precompute_freqs_cis, scaled_dot_product_attention

class Qwen3RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000.0):
        super().__init__()
        self.cos, self.sin = precompute_freqs_cis(dim, max_position_embeddings, base)
        self.register_buffer("cos", self.cos)
        self.register_buffer("sin", self.sin)

    def forward(self, x, seq_len):
        start = [0, 0]
        shape = [seq_len, self.cos.shape[1]]
        return self.cos.slice(start, shape), self.sin.slice(start, shape)

class Qwen3MLP(Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
    def forward(self, x):
        fused = self.gate_up_proj(x)
        return self.down_proj(fused.fused_swiglu())

class Qwen3Attention(Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = Qwen3RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, cache_position=0):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape [B, S, Hidden] -> [B, S, Heads, HeadDim]
        B = q.shape[0]
        S = q.shape[1]
        H = q.shape[2]
        heads = H // self.head_dim
        new_shape = [B, S, heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # RoPE with correct offset
        start = [cache_position, 0]
        shape = [S, self.rotary_emb.cos.shape[1]]
        cos = self.rotary_emb.cos.slice(start, shape)
        sin = self.rotary_emb.sin.slice(start, shape)

        q, k = q.rope(k, cos, sin)

        if use_cache and past_key_value is not None:
             k_cache, v_cache = past_key_value
             start_indices = [0, cache_position, 0, 0]
             k_cache.set_slice(k, start_indices)
             v_cache.set_slice(v, start_indices)

             valid_len = cache_position + S
             k = k_cache.slice([0,0,0,0], [B, valid_len, heads, self.head_dim])
             v = v_cache.slice([0,0,0,0], [B, valid_len, heads, self.head_dim])

        current_cache = past_key_value if use_cache else None

        out = scaled_dot_product_attention(q, k, v)

        # Flatten back
        out = out.reshape([B, S, H])

        return self.o_proj(out), None, current_cache

class Qwen3DecoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, cache_position=0):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, _, pkv = self.self_attn(h, attention_mask, position_ids, past_key_value, use_cache, cache_position)

        if hasattr(residual, 'fused_add_rms_norm'):
            h_norm = residual.fused_add_rms_norm(h, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
            hidden_states = residual
        else:
            hidden_states = residual.add(h)
            h_norm = self.post_attention_layernorm(hidden_states)

        mlp_out = self.mlp(h_norm)
        hidden_states = hidden_states + mlp_out

        return hidden_states, pkv

class Qwen3Model(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        for i in range(config.num_hidden_layers):
             l = Qwen3DecoderLayer(config)
             self.layers.append(l)
             self._modules[f"layer_{i}"] = l
        self.norm = RMSNorm(config.hidden_size)
        self.scheduler = None

    def forward(self, input_ids, past_key_values=None, use_cache=None, cache_position=0):
        h = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
             past_key_values = [None] * len(self.layers)

        next_cache = past_key_values if use_cache else None

        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)

            past = past_key_values[i] if past_key_values else None
            h, pkv = layer(h, past_key_value=past, use_cache=use_cache, cache_position=cache_position)
            if use_cache: next_cache[i] = pkv
        return self.norm(h), next_cache

class Qwen3ForCausalLM(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=None, cache_position=0, **kwargs):
        h, pkv = self.model(input_ids, past_key_values, use_cache, cache_position)
        logits = self.lm_head(h)
        return logits, pkv

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_seq_len = seq_len + max_new_tokens

        # Static KV Cache Pre-allocation
        device = input_ids.device
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        past_key_values = []
        for _ in range(self.config.num_hidden_layers):
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

            logits, _ = self.forward(model_input, past_key_values=past_key_values, use_cache=True, cache_position=cache_position)

            vocab_size = logits.shape[2]
            last_token_logits = logits.slice([0, logits.shape[1]-1, 0], [batch_size, 1, vocab_size])
            next_token_tensor = last_token_logits.argmax()

            current_ids = cat([current_ids, next_token_tensor], axis=1)

        return current_ids