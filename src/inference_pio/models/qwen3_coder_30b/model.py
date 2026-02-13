"""
Qwen3-Coder-30B Model Implementation - Self-Contained Version
Dependency-Free using Custom Backend
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple

from ...core.engine.backend import Module, Tensor, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention, cat
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
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.scheduler = None

        # Helper components
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        try:
            # Assuming tokenizer files are at config.model_path or handled by factory
            pass
        except Exception as e:
            logger.warning(f"Tokenizer init warning: {e}")

    def get_tokenizer(self):
        return self._tokenizer

    def forward(self, input_ids: Tensor, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False, cache_position: int = 0) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
             past_key_values = [None] * len(self.layers)

        next_cache = past_key_values if use_cache else None

        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)

            past = past_key_values[i] if past_key_values else None
            hidden_states, pkv = layer(hidden_states, past_key_value=past, use_cache=use_cache, cache_position=cache_position)
            if use_cache:
                next_cache[i] = pkv

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache

    def generate(self, input_ids: Tensor, max_new_tokens: int = 10) -> Tensor:
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

            h, pkv = self.forward(model_input, past_key_values=past_key_values, use_cache=True, cache_position=cache_position)
            past_key_values = pkv # In-place update

            logits = self.lm_head(h)

            # Greedy decode last token
            B = logits.shape[0]
            S = logits.shape[1]
            V = logits.shape[2]

            # Slice last token: [B, 1, V]
            last_logits = logits.slice([0, S-1, 0], [B, 1, V])
            next_token = last_logits.argmax() # [B, 1]

            current_ids = cat([current_ids, next_token], axis=1)

        return current_ids

class Qwen3Coder30BDecoderLayer(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Qwen3Coder30BAttention(config, cos, sin)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Qwen3Coder30BMLP(config)

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        residual = x
        h = self.input_layernorm(x)
        h, pkv = self.self_attn(h, past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position)
        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = residual + h
        return x, pkv

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

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to 4D [B, S, Heads, HeadDim]
        B = q.shape[0]
        S = q.shape[1]
        new_shape = [B, S, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # RoPE
        start = [cache_position, 0]
        shape = [S, self.cos.shape[1]]
        cos_slice = self.cos.slice(start, shape)
        sin_slice = self.sin.slice(start, shape)

        q, k = q.rope(k, cos_slice, sin_slice)

        # KV Cache (Static)
        if use_cache and past_key_value is not None:
             k_cache, v_cache = past_key_value
             start_indices = [0, cache_position, 0, 0]
             k_cache.set_slice(k, start_indices)
             v_cache.set_slice(v, start_indices)

             valid_len = cache_position + S
             k = k_cache.slice([0,0,0,0], [B, valid_len, self.num_heads, self.head_dim])
             v = v_cache.slice([0,0,0,0], [B, valid_len, self.num_heads, self.head_dim])

        present_key_value = past_key_value if use_cache else None

        # Fused Attention
        context = scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Flatten back
        context = context.reshape([B, S, self.hidden_size])

        return self.o_proj(context), present_key_value

class Qwen3Coder30BMLP(Module):
    def __init__(self, config):
        super().__init__()
        # Use optimized fused projection
        self.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        fused = self.gate_up_proj(x)
        return self.down_proj(fused.fused_swiglu())

class Qwen3_Coder_30B_Model(Qwen3Coder30BModel):
    # Alias for plugin compatibility if needed
    pass

__all__ = ["Qwen3Coder30BModel", "Qwen3_Coder_30B_Model"]
