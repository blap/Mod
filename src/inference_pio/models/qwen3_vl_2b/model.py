"""
Qwen3-VL-2B Model Implementation - Self-Contained
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ...core.engine.backend import Module, Tensor, Linear, Embedding, RMSNorm, precompute_freqs_cis, cat, GELU, scaled_dot_product_attention
from .vision_transformer_kernels import Qwen3VL2BVisionEncoderKernel, VisionTransformerConfig

logger = logging.getLogger(__name__)

class Qwen3VL2BConfig:
    def __init__(self, **kwargs):
        self.hidden_size = 2048
        self.num_attention_heads = 16
        self.num_hidden_layers = 28
        self.vocab_size = 151936
        self.max_position_embeddings = 32768
        self.layer_norm_eps = 1e-6
        self.intermediate_size = 11008 # Default for 2B? Or 5504? Let's use standard value.
        # Vision
        self.vision_hidden_size = 1024
        self.vision_num_attention_heads = 16
        self.vision_num_hidden_layers = 24
        self.vision_patch_size = 14
        self.vision_image_size = 448
        self.vision_intermediate_size = 2816
        for k, v in kwargs.items(): setattr(self, k, v)

class Qwen3VL2BMultimodalProjector(Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = Linear(config.vision_hidden_size, config.hidden_size)
        self.activation = GELU()
        self.linear2 = Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class Qwen3VL2BModel(Module):
    def __init__(self, config: Qwen3VL2BConfig):
        super().__init__()
        self.config = config

        # Vision Encoder
        vis_conf = VisionTransformerConfig(
            hidden_size=config.vision_hidden_size,
            num_attention_heads=config.vision_num_attention_heads,
            num_hidden_layers=config.vision_num_hidden_layers,
            patch_size=config.vision_patch_size,
            image_size=config.vision_image_size,
            intermediate_size=config.vision_intermediate_size
        )
        self.visual = Qwen3VL2BVisionEncoderKernel(vis_conf)

        # Projector
        self.projector = Qwen3VL2BMultimodalProjector(config)

        # LLM
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = []

        # RoPE
        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = precompute_freqs_cis(head_dim, config.max_position_embeddings)

        for i in range(config.num_hidden_layers):
            l = Qwen3VL2BDecoderLayer(config, self.cos, self.sin)
            self.layers.append(l)
            self._modules[f"layer_{i}"] = l

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.scheduler = None

    def forward(self, input_ids: Tensor, pixel_values: Optional[Tensor] = None, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False, cache_position: int = 0):
        # Embed Text
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is not None and past_key_values is None:
            # 1. Encode Vision
            vis_features = self.visual(pixel_values) # [B, N_patches, VisDim]

            # 2. Project Vision to Text Dim
            vis_projected = self.projector(vis_features)

            # 3. Concatenate
            # Assume vision tokens come before text
            # cat([vis, text], axis=1)
            hidden_states = cat([vis_projected, hidden_states], axis=1)

        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_cache = past_key_values if use_cache else None

        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)

            past = past_key_values[i] if past_key_values else None
            hidden_states, pkv = layer(hidden_states, past_key_value=past, use_cache=use_cache, cache_position=cache_position)
            if use_cache and next_cache is not None:
                next_cache[i] = pkv

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache

    def generate(self, input_ids: Tensor, pixel_values: Optional[Tensor] = None, max_new_tokens: int = 10):
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

        # First step might involve vision
        for step in range(max_new_tokens):
            curr_seq_len = current_ids.shape[1]
            if step == 0:
                model_input = current_ids
                cache_position = 0
                step_pixels = pixel_values # Only used in first step for prefill
            else:
                model_input = current_ids.slice([0, curr_seq_len-1], [batch_size, 1])
                cache_position = curr_seq_len - 1
                step_pixels = None

            h, pkv = self.forward(model_input, pixel_values=step_pixels, past_key_values=past_key_values, use_cache=True, cache_position=cache_position)
            past_key_values = pkv # Technically redundant as it's modified in place or same list

            logits = self.lm_head(h)

            vocab_size = logits.shape[2]
            next_token_logits = logits.slice([0, logits.shape[1]-1, 0], [batch_size, 1, vocab_size])
            next_token = next_token_logits.argmax()

            current_ids = cat([current_ids, next_token], axis=1)

        return current_ids

class Qwen3VL2BDecoderLayer(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Qwen3VL2BAttention(config, cos, sin)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Qwen3VL2BMLP(config)

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        h = self.input_layernorm(x)
        h, pkv = self.self_attn(h, past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position)
        x = x + h
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        return x + h, pkv

class Qwen3VL2BAttention(Module):
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
        self.scale = self.head_dim ** -0.5

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to 4D for RoPE and SDPA: [B, S, Heads, HeadDim]
        B = q.shape[0]
        S = q.shape[1]
        new_shape = [B, S, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # RoPE
        start = [cache_position, 0]
        shape = [S, self.cos.shape[1]]
        c = self.cos.slice(start, shape)
        s = self.sin.slice(start, shape)

        q, k = q.rope(k, c, s)

        # Cache
        if use_cache and past_key_value is not None:
             k_cache, v_cache = past_key_value
             start_indices = [0, cache_position, 0, 0]
             k_cache.set_slice(k, start_indices)
             v_cache.set_slice(v, start_indices)

             valid_len = cache_position + S
             k = k_cache.slice([0,0,0,0], [B, valid_len, self.num_heads, self.head_dim])
             v = v_cache.slice([0,0,0,0], [B, valid_len, self.num_heads, self.head_dim])

        present_key_value = past_key_value if use_cache else None

        out = scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Flatten back to 3D for projection
        out = out.reshape([B, S, self.hidden_size])

        return self.o_proj(out), present_key_value

class Qwen3VL2BMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        fused = self.gate_up_proj(x)
        return self.down_proj(fused.fused_swiglu())

def create_qwen3_vl_2b_model(config): return Qwen3VL2BModel(config)
__all__ = ["Qwen3VL2BModel", "create_qwen3_vl_2b_model"]
