"""
Qwen3-0.6B Architecture - Numpy Backend
"""

import math
import numpy as np
from typing import Optional, Tuple, List

from ...core.engine.layers import Module, Linear, Embedding, RMSNorm, ModuleList
from ...core.engine.tensor_ops import softmax, matmul, silu, apply_rotary_emb, precompute_freqs_cis

class Qwen3RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute cos/sin
        self.cos, self.sin = precompute_freqs_cis(dim, max_position_embeddings, base)
        self.register_buffer("cos_cached", self.cos)
        self.register_buffer("sin_cached", self.sin)

    def forward(self, x, seq_len):
        # x is just for device context in torch, ignored here
        return self.cos[:seq_len], self.sin[:seq_len]

class Qwen3MLP(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen3Attention(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv_seq_len = key_states.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        # RoPE
        # We need slicing logic similar to torch
        # position_ids [batch, seq_len]
        # Simply take cos/sin based on length for now (assuming contiguous)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Apply RoPE needs careful broadcasting in numpy
        # cos: [seq_len, dim] -> [1, 1, seq_len, dim]
        # We need to slice cos/sin based on position_ids if not contiguous
        # Assuming simple causal for now: slice last q_len
        cos = cos[-q_len:].reshape(1, 1, q_len, self.head_dim)
        sin = sin[-q_len:].reshape(1, 1, q_len, self.head_dim)

        query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Concatenate along seq_len dimension (axis 2)
            key_states = np.concatenate([past_key_value[0], key_states], axis=2)
            value_states = np.concatenate([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        # key_states: [B, n_kv_heads, S, D] -> [B, n_heads, S, D]
        if self.num_key_value_groups > 1:
            key_states = np.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = np.repeat(value_states, self.num_key_value_groups, axis=1)

        # Attention
        # Q: [B, H, S, D], K.T: [B, H, D, S] -> [B, H, S, S]
        attn_weights = matmul(query_states, key_states.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = softmax(attn_weights, axis=-1)
        attn_output = matmul(attn_weights, value_states)

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


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

        hidden_states, _, present_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class Qwen3Model(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = 151643
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, inputs_embeds=None):
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Mask generation (simplified for inference)
        # If attention_mask is None, assume causal
        # In numpy, mask should be float with -inf
        if attention_mask is None:
            seq_len = hidden_states.shape[1]
            # Causal mask: lower triangular is 0, upper is -inf
            mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
            mask = np.triu(mask, k=1)
            # Expand to [1, 1, seq_len, seq_len]
            attention_mask = mask.reshape(1, 1, seq_len, seq_len)

        pkv = () if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None

            hidden_states, layer_pkv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                pkv = pkv + (layer_pkv,)

        hidden_states = self.norm(hidden_states)
        return hidden_states, pkv


class Qwen3ForCausalLM(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, inputs_embeds=None):
        hidden_states, pkv = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(hidden_states)
        return logits, pkv

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 50, **kwargs) -> np.ndarray:
        """
        Numpy-based generation loop.
        """
        generated_ids = input_ids.copy() # [batch, seq_len]
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values:
                # Use last token only
                model_inputs = generated_ids[:, -1:]
                # Position IDs would need update here
            else:
                model_inputs = generated_ids

            outputs, past_key_values = self.forward(input_ids=model_inputs, past_key_values=past_key_values, use_cache=True)

            next_token_logits = outputs[:, -1, :] # [batch, vocab]

            # Greedy/Sample
            if temperature == 0:
                next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)
            else:
                # Basic temp scaling
                next_token_logits = next_token_logits / temperature
                probs = softmax(next_token_logits)
                # Sampling logic (simplified to argmax for stability/speed in numpy unless explicitly requested)
                next_token = np.argmax(probs, axis=-1, keepdims=True)

            generated_ids = np.concatenate([generated_ids, next_token], axis=-1)

        return generated_ids

    def to(self, device):
        # No-op for numpy
        return self
