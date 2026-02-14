"""
Qwen3-Coder-Next Model Implementation (Dependency-Free)
"""

from typing import Optional, Tuple, Union, List, Dict, Any
import logging
import math
import os
import shutil
import subprocess
import sys

from ...core.engine.backend import Tensor, Module, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention, cat
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer, CustomBPETokenizer
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
        # Use config specific rope dim if available (64)
        self.rotary_emb_dim = config.attention_rope_dim if hasattr(config, 'attention_rope_dim') else head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_base = config.rope_theta

        # Create cache on device (default cpu, moved in .to())
        self.cos_cache, self.sin_cache = precompute_freqs_cis(self.rotary_emb_dim, self.max_position_embeddings, self.rope_base)
        self.register_buffer("cos_cache", self.cos_cache)
        self.register_buffer("sin_cache", self.sin_cache)

        self.layers = []
        for i in range(config.num_hidden_layers):
            layer = Qwen3CoderNextDecoderLayer(config, self.cos_cache, self.sin_cache, layer_idx=i)
            self.layers.append(layer)
            self._modules[f"layer_{i}"] = layer

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.scheduler = None # For Dynamic Offloading

    def forward(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False, cache_position: Union[int, Tensor] = 0, max_cache_len: int = 0):
        if input_ids is None: raise ValueError("input_ids required")

        # Update device of cache if needed (naive check)
        if self.cos_cache.device != input_ids.device:
             self.cos_cache = self.cos_cache.to(input_ids.device)
             self.sin_cache = self.sin_cache.to(input_ids.device)
             for layer in self.layers:
                 if hasattr(layer.self_attn, 'cos_cache'):
                     layer.self_attn.cos_cache = self.cos_cache
                     layer.self_attn.sin_cache = self.sin_cache

        hidden_states = self.embed_tokens(input_ids)

        # Init Cache (List of Nones if starting fresh)
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_cache = past_key_values if use_cache else None

        for i, layer in enumerate(self.layers):
            if self.scheduler:
                self.scheduler.check_migration_policy(i, layer, self.layers)

            target_device = layer.input_layernorm.weight.device
            if hidden_states.device != target_device:
                hidden_states = hidden_states.to(target_device)

            layer_past = past_key_values[i] if past_key_values is not None else None

            hidden_states, pkv = layer(
                hidden_states,
                past_key_value=layer_past,
                use_cache=use_cache,
                cache_position=cache_position,
                max_cache_len=max_cache_len
            )
            if use_cache and next_cache is not None:
                next_cache[i] = pkv

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache

class Qwen3CoderNextForCausalLM(Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = Qwen3CoderNextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tokenizer = None

        self._initialize_model()

    def _resolve_model_path(self) -> str:
        model_name = "Qwen3-Coder-Next"
        hf_repo = "Qwen/Qwen3-Coder-Next"
        h_drive_path = os.path.join("H:/", model_name)
        if os.path.exists(h_drive_path): return h_drive_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "_model_cache")
        local_path = os.path.join(cache_dir, model_name)
        if os.path.exists(local_path) and os.listdir(local_path): return local_path

        os.makedirs(cache_dir, exist_ok=True)
        total, used, free = shutil.disk_usage(cache_dir)
        required_space = 20 * 1024 * 1024 * 1024
        if free < required_space: raise RuntimeError(f"Insufficient disk space. Required: 20GB.")
        try:
            subprocess.run(["git", "clone", f"https://huggingface.co/{hf_repo}", local_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return local_path
        except subprocess.CalledProcessError:
            if os.path.exists(local_path): shutil.rmtree(local_path)
            raise RuntimeError(f"Failed to download model from {hf_repo}")

    def _initialize_model(self):
        logger.info("Initializing Qwen3-Coder-Next model...")
        try:
            model_path = self._resolve_model_path()
            CustomModelLoader.load_weights(self, model_path, device="cpu")
        except Exception:
            logger.warning(f"Failed to load weights. Model will use random initialization.")
        try:
            model_path = self._resolve_model_path()
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception:
            logger.warning(f"Failed to load tokenizer.")

    def forward(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False, cache_position: Union[int, Tensor] = 0, max_cache_len: int = 0):
        hidden_states, next_cache = self.model(input_ids, past_key_values, use_cache, cache_position, max_cache_len)
        logits = self.lm_head(hidden_states)
        return logits, next_cache

    def generate(self, input_ids: Tensor, max_new_tokens: int = 10, ngram_speculate: bool = False, **kwargs) -> Tensor:
        # Optimization 4: CUDA Graphs for Decoding
        from ...core.engine.backend import CUDAGraph, HAS_CUDA

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_seq_len = seq_len + max_new_tokens

        past_key_values = []
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        device = input_ids.device
        pattern = self.config.hybrid_block_pattern

        for i in range(self.config.num_hidden_layers):
            layer_type = pattern[i % len(pattern)]
            if layer_type == "deltanet":
                 d_head_dim = self.config.deltanet_head_dim
                 d_heads = self.config.deltanet_query_key_heads
                 state = Tensor([batch_size, d_heads, d_head_dim, d_head_dim], device=device)
                 state.fill(0.0)
                 past_key_values.append(state)
            else:
                 k_cache = Tensor([batch_size, max_seq_len, self.config.num_key_value_heads, self.config.attention_head_dim], device=device)
                 v_cache = Tensor([batch_size, max_seq_len, self.config.num_key_value_heads, self.config.attention_head_dim], device=device)
                 k_cache.fill(0.0)
                 v_cache.fill(0.0)
                 past_key_values.append((k_cache, v_cache))

        current_ids = input_ids

        # Graph State
        graph = None
        input_buffer = None
        cache_pos_tensor = None

        for step in range(max_new_tokens):
            curr_seq_len = current_ids.shape[1]

            if step == 0:
                # Prefill Phase
                model_input = current_ids
                logits, _ = self.forward(model_input, past_key_values=past_key_values, use_cache=True, cache_position=0, max_cache_len=max_seq_len)
            else:
                # Decoding Phase
                cache_position = curr_seq_len - 1

                # Update input buffer
                last_token = current_ids.slice([0, curr_seq_len-1], [batch_size, 1])

                if HAS_CUDA and batch_size == 1:
                    if input_buffer is None:
                        input_buffer = Tensor([batch_size, 1], device=input_ids.device)
                        cache_pos_tensor = Tensor([1], device=input_ids.device) # Scalar tensor

                    input_buffer.load(last_token.to_list())
                    cache_pos_tensor.fill(float(cache_position)) # Update position tensor on GPU

                    if graph is None and step == 1:
                        # Capture
                        graph = CUDAGraph()
                        graph.capture_begin()
                        logits, _ = self.forward(
                            input_buffer,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_pos_tensor, # Pass Tensor!
                            max_cache_len=max_seq_len
                        )
                        graph.capture_end()
                        graph.replay()
                    elif graph is not None:
                        # Replay
                        graph.replay()
                    else:
                        # Fallback (step != 1 if graph skipped?)
                        logits, _ = self.forward(input_buffer, past_key_values=past_key_values, use_cache=True, cache_position=cache_position, max_cache_len=max_seq_len)
                else:
                    # Normal path
                    logits, _ = self.forward(last_token, past_key_values=past_key_values, use_cache=True, cache_position=cache_position, max_cache_len=max_seq_len)

            vocab_size = logits.shape[2]
            last_logits = logits.slice([0, logits.shape[1]-1, 0], [batch_size, 1, vocab_size])
            next_token = last_logits.argmax()

            current_ids = cat([current_ids, next_token], axis=1)

        return current_ids

class Qwen3CoderNextDeltaNet(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.deltanet_query_key_heads
        self.value_heads = config.deltanet_value_heads
        self.head_dim = config.deltanet_head_dim

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.beta_proj = Linear(self.hidden_size, self.num_heads, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, past_key_value=None, use_cache=False, cache_position=0, max_cache_len=0):
        # x: [B, S, H]
        B = x.shape[0]
        S = x.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = self.beta_proj(x) # [B, S, Heads]

        # Reshape to [B, S, Heads, Dim]
        new_shape = [B, S, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # Beta activation (SiLU as proxy for Sigmoid)
        beta = beta.silu()

        state = past_key_value
        if state is None:
            # Init state [B, Heads, Dim, Dim]
            state = Tensor([B, self.num_heads, self.head_dim, self.head_dim], device=x.device)
            state.fill(0.0)

        # Run Kernel
        out = q.deltanet_recurrence(k, v, beta, state)

        # Flatten
        out = out.reshape([B, S, self.num_heads * self.head_dim])
        out = self.o_proj(out)

        return out, state if use_cache else None

class Qwen3CoderNextDecoderLayer(Module):
    def __init__(self, config, cos_cache, sin_cache, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Select layer type based on hybrid pattern
        pattern = config.hybrid_block_pattern
        layer_type = pattern[layer_idx % len(pattern)]

        if layer_type == "deltanet":
            self.self_attn = Qwen3CoderNextDeltaNet(config)
        else:
            self.self_attn = Qwen3CoderNextAttention(config, cos_cache, sin_cache)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        if hasattr(config, "num_experts") and config.num_experts > 1:
            self.mlp = Qwen3CoderNextMoE(config)
        else:
            self.mlp = Qwen3CoderNextMLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, past_key_value: Optional[Any] = None, use_cache: bool = False, cache_position: Union[int, Tensor] = 0, max_cache_len: int = 0) -> Tuple[Tensor, Optional[Any]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, pkv = self.self_attn(
            hidden_states,
            past_key_value if past_key_value is not None else None,
            use_cache=use_cache,
            cache_position=cache_position,
            max_cache_len=max_cache_len
        )

        if hasattr(residual, 'fused_add_rms_norm'):
             hidden_states = residual.fused_add_rms_norm(attn_out, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
             residual = residual
        else:
             residual = residual + attn_out
             hidden_states = self.post_attention_layernorm(residual)

        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states, pkv

class Qwen3CoderNextAttention(Module):
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim

        # Proj size: (16 * 256) + 2 * (2 * 256) = 4096 + 1024 = 5120
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.total_proj = self.q_size + 2 * self.kv_size

        self.qkv_proj = Linear(self.hidden_size, self.total_proj, bias=True)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
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

    def forward(self, hidden_states: Tensor, past_key_value: Optional[Tuple[Tensor, Tensor]] = None, use_cache: bool = False, cache_position: Union[int, Tensor] = 0, max_cache_len: int = 0) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        qkv = self.qkv_proj(hidden_states)
        B, S, _ = qkv.shape

        q = qkv.slice([0, 0, 0], [B, S, self.q_size])
        k = qkv.slice([0, 0, self.q_size], [B, S, self.kv_size])
        v = qkv.slice([0, 0, self.q_size + self.kv_size], [B, S, self.kv_size])

        q = q.reshape([B, S, self.num_heads, self.head_dim])
        k = k.reshape([B, S, self.num_kv_heads, self.head_dim])
        v = v.reshape([B, S, self.num_kv_heads, self.head_dim])

        if isinstance(cache_position, Tensor):
             zero_1 = Tensor([1], device=cache_position.device); zero_1.fill(0.0)
             slice_start = cat([cache_position, zero_1], axis=0)
             slice_shape = [S, self.cos_cache.shape[1]]
             cos = self.cos_cache.slice(slice_start, slice_shape)
             sin = self.sin_cache.slice(slice_start, slice_shape)
        else:
             slice_start = [cache_position, 0]
             slice_shape = [S, self.cos_cache.shape[1]]
             cos = self.cos_cache.slice(slice_start, slice_shape)
             sin = self.sin_cache.slice(slice_start, slice_shape)

        q, k = q.rope(k, cos, sin)

        if use_cache:
            if past_key_value is not None:
                B, S, H, D = k.shape
                if isinstance(cache_position, Tensor):
                     pass
                else:
                    start_indices = [0, cache_position, 0, 0]
                    past_key_value[0].set_slice(k, start_indices)
                    past_key_value[1].set_slice(v, start_indices)

                if isinstance(cache_position, int):
                    valid_len = cache_position + S
                    k = past_key_value[0].slice([0,0,0,0], [B, valid_len, H, D])
                    v = past_key_value[1].slice([0,0,0,0], [B, valid_len, H, D])

                present_key_value = past_key_value
            else:
                present_key_value = None
        else:
            present_key_value = None

        if self.group_size > 1:
            k = self.repeat_kv(k, self.group_size)
            v = self.repeat_kv(v, self.group_size)

        context = scaled_dot_product_attention(q, k, v, scale=self.scale)
        context = context.reshape([B, S, self.num_heads * self.head_dim])
        output = self.o_proj(context)
        return output, present_key_value

class Qwen3CoderNextMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        fused = self.gate_up_proj(x)
        merged = fused.fused_swiglu()
        return self.down_proj(merged)

class Qwen3CoderNextMoE(Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate = Linear(config.hidden_size, config.num_experts, bias=False)

        self.experts = []
        for i in range(self.num_experts):
            e = Qwen3CoderNextMLP(config)
            self.experts.append(e)
            self._modules[f"expert_{i}"] = e

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        S = x.shape[1]
        H = x.shape[2]

        router_logits = self.gate(x)
        router_probs = router_logits.softmax()
        router_probs_flat = router_probs.reshape([B*S, self.num_experts])
        input_flat = x.reshape([B*S, H])

        top_k_weights, top_k_inds = router_probs_flat.topk(self.num_experts_per_tok)
        final_output = Tensor([B*S, H], device=x.device)
        final_output.fill(0.0)

        for k in range(self.num_experts_per_tok):
            inds_slice = top_k_inds.slice([0, k], [B*S, 1])
            weights_slice = top_k_weights.slice([0, k], [B*S, 1])
            inds_vec = inds_slice.reshape([B*S])
            weights_vec = weights_slice.reshape([B*S, 1])
            indices_cpu = inds_vec.to_list()
            unique_experts = set(int(x) for x in indices_cpu)

            for expert_idx in unique_experts:
                input_subset, original_indices = input_flat.gather_by_value(inds_vec, float(expert_idx))
                if input_subset is None or input_subset.shape[0] == 0:
                    continue
                weight_subset, _ = weights_vec.gather_by_value(inds_vec, float(expert_idx))
                expert_output = self.experts[expert_idx](input_subset)
                weighted_output = expert_output * weight_subset
                final_output.scatter_add_by_index(weighted_output, original_indices)

        return final_output.reshape([B, S, H])
