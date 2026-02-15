"""
Qwen3-Coder-30B Model Implementation - Self-Contained Version
Dependency-Free using Custom Backend
Supports GQA and MoE
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union, Tuple

from ...core.engine.backend import Module, Tensor, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention, cat
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer, CustomBPETokenizer
from .config import Qwen3Coder30BConfig

logger = logging.getLogger(__name__)

class Qwen3Coder30BModel(Module):
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

        self._initialize_model()

    def _resolve_model_path(self) -> str:
        """
        Resolves the model path with the following priority:
        1. H:/Qwen3-Coder-30B-A3B-Instruct
        2. Local Cache
        3. Download from HuggingFace
        """
        model_name = "Qwen3-Coder-30B-A3B-Instruct"
        hf_repo = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

        # 1. Check H: Drive
        h_drive_path = os.path.join("H:/", model_name)
        if os.path.exists(h_drive_path):
            logger.info(f"Found model on H: drive: {h_drive_path}")
            return h_drive_path

        # 2. Check Local Cache
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "_model_cache")
        local_path = os.path.join(cache_dir, model_name)

        if os.path.exists(local_path):
            if os.listdir(local_path):
                logger.info(f"Found model in local cache: {local_path}")
                return local_path

        # 3. Download
        logger.info(f"Model not found. Attempting to download {hf_repo} to {local_path}...")

        os.makedirs(cache_dir, exist_ok=True)

        # Check Disk Space (30B is Huge, ~60GB for fp16, ~18GB for int4)
        total, used, free = shutil.disk_usage(cache_dir)
        required_space = 20 * 1024 * 1024 * 1024 # Require 20GB minimum

        if free < required_space:
            raise RuntimeError(f"Insufficient disk space. Required: 20GB, Available: {free/1024/1024/1024:.2f}GB")

        try:
            logger.info("Cloning from HuggingFace...")
            subprocess.run(
                ["git", "clone", f"https://huggingface.co/{hf_repo}", local_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Download complete.")
            return local_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            raise RuntimeError(f"Failed to download model from {hf_repo}")

    def _initialize_model(self):
        logger.info("Initializing Qwen3-Coder-30B model...")

        try:
            model_path = self._resolve_model_path()
            CustomModelLoader.load_weights(self, model_path, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Model will use random initialization.")

        try:
            model_path = self._resolve_model_path()
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Text processing will be limited.")

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
            # Pass use_cache and cache_position to layer
            layer_out = layer(hidden_states, past_key_value=past, use_cache=use_cache, cache_position=cache_position)

            # Unpack layer output
            if isinstance(layer_out, tuple):
                hidden_states, pkv = layer_out
            else:
                hidden_states = layer_out
                pkv = None

            if use_cache and next_cache is not None:
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
             k_cache = Tensor([batch_size, max_seq_len, self.config.num_key_value_heads, head_dim], device=device)
             v_cache = Tensor([batch_size, max_seq_len, self.config.num_key_value_heads, head_dim], device=device)
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
            # past_key_values is updated in place via set_slice or replaced by pkv if re-allocated

            logits = self.lm_head(h)

            B = logits.shape[0]
            S = logits.shape[1]
            V = logits.shape[2]

            last_logits = logits.slice([0, S-1, 0], [B, 1, V])
            next_token = last_logits.argmax()

            current_ids = cat([current_ids, next_token], axis=1)

        return current_ids

class Qwen3Coder30BDecoderLayer(Module):
    def __init__(self, config, cos, sin):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Qwen3Coder30BAttention(config, cos, sin)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Replaced standard MLP with MoE
        self.mlp = Qwen3Coder30BMoE(config)

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
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.total_proj = self.q_size + 2 * self.kv_size

        self.qkv_proj = Linear(self.hidden_size, self.total_proj, bias=True)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)

        self.cos = cos
        self.sin = sin
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.group_size = self.num_heads // self.num_kv_heads

    def repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        if n_rep == 1: return x
        B, S, H_kv, D = x.shape
        head_tensors = []
        for h in range(H_kv):
            head = x.slice([0, 0, h, 0], [B, S, 1, D])
            for _ in range(n_rep):
                head_tensors.append(head)
        return cat(head_tensors, axis=2)

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0):
        qkv = self.qkv_proj(x)
        B, S, _ = qkv.shape

        q = qkv.slice([0, 0, 0], [B, S, self.q_size])
        k = qkv.slice([0, 0, self.q_size], [B, S, self.kv_size])
        v = qkv.slice([0, 0, self.q_size + self.kv_size], [B, S, self.kv_size])

        q = q.reshape([B, S, self.num_heads, self.head_dim])
        k = k.reshape([B, S, self.num_kv_heads, self.head_dim])
        v = v.reshape([B, S, self.num_kv_heads, self.head_dim])

        start = [cache_position, 0]
        shape = [S, self.cos.shape[1]]
        cos_slice = self.cos.slice(start, shape)
        sin_slice = self.sin.slice(start, shape)

        q, k = q.rope(k, cos_slice, sin_slice)

        if use_cache and past_key_value is not None:
             k_cache, v_cache = past_key_value
             start_indices = [0, cache_position, 0, 0]
             k_cache.set_slice(k, start_indices)
             v_cache.set_slice(v, start_indices)

             valid_len = cache_position + S
             k = k_cache.slice([0,0,0,0], [B, valid_len, self.num_kv_heads, self.head_dim])
             v = v_cache.slice([0,0,0,0], [B, valid_len, self.num_kv_heads, self.head_dim])

        present_key_value = past_key_value if use_cache else None

        if self.group_size > 1:
            k = self.repeat_kv(k, self.group_size)
            v = self.repeat_kv(v, self.group_size)

        context = scaled_dot_product_attention(q, k, v, scale=self.scale)
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

class Qwen3Coder30BMoE(Module):
    """
    Naive MoE Implementation for 30B (128 Experts, Top-8).
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.gate = Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = []
        for i in range(self.num_experts):
            e = Qwen3Coder30BMLP(config)
            self.experts.append(e)
            self._modules[f"expert_{i}"] = e

    def forward(self, x):
        B, S, H = x.shape
        router_logits = self.gate(x)
        router_probs = router_logits.softmax()

        # Flatten
        router_probs_flat = router_probs.reshape([B*S, self.num_experts])
        input_flat = x.reshape([B*S, H])

        top_k_weights, top_k_inds = router_probs_flat.topk(self.num_experts_per_tok)

        final_output = Tensor([B*S, H], device=x.device)
        final_output.fill(0.0)

        # Iterate over K
        for k in range(self.num_experts_per_tok):
            inds_slice = top_k_inds.slice([0, k], [B*S, 1])
            weights_slice = top_k_weights.slice([0, k], [B*S, 1])

            inds_vec = inds_slice.reshape([B*S])
            weights_vec = weights_slice.reshape([B*S, 1])

            # Simple loop over all experts (Very slow if num_experts is large, but correct)
            # Optimization: Only loop over unique experts in inds_vec
            indices_cpu = inds_vec.to_list()
            unique_experts = set(int(x) for x in indices_cpu)

            for expert_idx in unique_experts:
                # Gather tokens for this expert
                input_subset, original_indices = input_flat.gather_by_value(inds_vec, float(expert_idx))

                if input_subset is None: continue

                # Gather weights
                weight_subset, _ = weights_vec.gather_by_value(inds_vec, float(expert_idx))

                # Execute Expert
                expert_out = self.experts[expert_idx](input_subset)

                # Weighting
                weighted_out = expert_out * weight_subset

                # Scatter Add
                final_output.scatter_add_by_index(weighted_out, original_indices)

        return final_output.reshape([B, S, H])

class Qwen3_Coder_30B_Model(Qwen3Coder30BModel):
    pass

__all__ = ["Qwen3Coder30BModel", "Qwen3_Coder_30B_Model"]
