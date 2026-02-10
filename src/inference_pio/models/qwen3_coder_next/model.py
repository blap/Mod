"""
Qwen3-Coder-Next Model Implementation (Dependency-Free)
"""

from typing import Optional, Tuple, Union, List, Dict, Any
import logging
import math

from ...core.engine.backend import Tensor, Module, Linear, Embedding, RMSNorm, precompute_freqs_cis, scaled_dot_product_attention, cat
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

    def forward(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False):
        if input_ids is None: raise ValueError("input_ids required")

        # Update device of cache if needed (naive check)
        if self.cos_cache.device != input_ids.device:
             self.cos_cache = self.cos_cache.to(input_ids.device)
             self.sin_cache = self.sin_cache.to(input_ids.device)
             for layer in self.layers:
                 layer.self_attn.cos_cache = self.cos_cache
                 layer.self_attn.sin_cache = self.sin_cache

        hidden_states = self.embed_tokens(input_ids)
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, pkv = layer(hidden_states, past_key_value=layer_past, use_cache=use_cache)
            if use_cache:
                next_cache.append(pkv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache

class Qwen3CoderNextForCausalLM(Module):
    def __init__(self, config: Any):
        super().__init__()
        self.model = Qwen3CoderNextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False):
        hidden_states, next_cache = self.model(input_ids, past_key_values, use_cache)
        logits = self.lm_head(hidden_states)
        return logits, next_cache

    def generate(self, input_ids: Tensor, max_new_tokens: int = 10, **kwargs) -> Tensor:
        current_ids = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values:
                # Slice input to last token
                seq_len = current_ids.shape[1]
                model_input = current_ids.slice([0, seq_len - 1], [1, 1])
            else:
                model_input = current_ids

            logits, pkv = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = pkv

            # Greedy
            next_token_logits = logits.slice([0, logits.shape[1]-1, 0], [1, 1, logits.shape[2]])
            next_token = next_token_logits.argmax()

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

    def forward(self, x, past_key_value=None, use_cache=False):
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

    def forward(self, hidden_states: Tensor, past_key_value: Optional[Any] = None, use_cache: bool = False) -> Tuple[Tensor, Optional[Any]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, pkv = self.self_attn(hidden_states, past_key_value if past_key_value is not None else None, use_cache=use_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, pkv

class Qwen3CoderNextAttention(Module):
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Fused QKV: 3 * Hidden
        self.qkv_proj = Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache references
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def forward(self, hidden_states: Tensor, past_key_value: Optional[Tuple[Tensor, Tensor]] = None, use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Fused Projection
        qkv = self.qkv_proj(hidden_states) # [B, S, 3*H]

        # Split (Slice) - Optimized backends can use strided access if supported, otherwise explicit slice
        B = qkv.shape[0]
        S = qkv.shape[1]
        H = self.hidden_size

        # Ideally, we should add `tensor_chunk` or `split` to backend for efficiency.
        # But `slice` is correct. We minimize python ops.
        # However, for RoPE, we need Q and K separated anyway.

        q = qkv.slice([0, 0, 0], [B, S, H])
        k = qkv.slice([0, 0, H], [B, S, H])
        v = qkv.slice([0, 0, 2*H], [B, S, H])

        # Reshape [B, S, Hidden] -> [B, S, Heads, HeadDim]
        S = q.shape[1]
        H = q.shape[2]
        new_shape = [B, S, self.num_heads, self.head_dim]
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # Apply RoPE
        if past_key_value is not None:
            past_len = past_key_value[0].shape[1]
        else:
            past_len = 0

        # RoPE indices: slice from past_len to current_len
        slice_start = [past_len, 0]
        slice_shape = [S, self.cos_cache.shape[1]]

        cos = self.cos_cache.slice(slice_start, slice_shape)
        sin = self.sin_cache.slice(slice_start, slice_shape)

        q, k = q.rope(k, cos, sin)

        # KV Cache Update
        if past_key_value is not None:
            k = cat([past_key_value[0], k], axis=1)
            v = cat([past_key_value[1], v], axis=1)

        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Fused Attention
        context = scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Flatten
        context = context.reshape([B, S, H])
        output = self.o_proj(context)
        return output, present_key_value

class Qwen3CoderNextMLP(Module):
    def __init__(self, config):
        super().__init__()
        # Fused Gate+Up Projection
        self.gate_up_proj = Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Fused Projection -> [B, S, 2*Inter]
        fused = self.gate_up_proj(x)
        # Fused SwiGLU Kernel
        merged = fused.fused_swiglu()
        return self.down_proj(merged)

class Qwen3CoderNextMoE(Module):
    """
    Real MoE implementation for Qwen3-Coder-Next using backend primitives.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok if hasattr(config, 'num_experts_per_tok') else 1
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate = Linear(config.hidden_size, config.num_experts, bias=False)

        # Experts
        self.experts = []
        for i in range(self.num_experts):
            e = Qwen3CoderNextMLP(config)
            self.experts.append(e)
            self._modules[f"expert_{i}"] = e

    def forward(self, x: Tensor) -> Tensor:
        # x: [Batch, Seq, Hidden]
        B = x.shape[0]
        S = x.shape[1]
        H = x.shape[2]

        # 1. Router Logits: [Batch, Seq, NumExperts]
        router_logits = self.gate(x)

        # 2. Softmax to get probabilities (weights)
        # We need to reshape to 2D for softmax in backend if it only supports [Batch, Classes] or check impl.
        # Backend softmax typically operates on last dim.
        router_probs = router_logits.softmax()

        # 3. TopK Selection
        # Backend TopK works on last dim.
        # router_probs: [B, S, NumExperts]
        # output: [B, S, K]

        # We process as flattened tokens for easier gathering
        router_probs_flat = router_probs.reshape([B*S, self.num_experts])
        input_flat = x.reshape([B*S, H])

        top_k_weights, top_k_inds = router_probs_flat.topk(self.num_experts_per_tok)

        # 4. Output Accumulator
        final_output = Tensor([B*S, H], device=x.device)
        final_output.fill(0.0)

        # 5. Loop over K choices
        for k in range(self.num_experts_per_tok):
            # Slice the k-th column of indices and weights
            # [B*S, K] -> [B*S, 1]
            inds_slice = top_k_inds.slice([0, k], [B*S, 1])
            weights_slice = top_k_weights.slice([0, k], [B*S, 1])

            # Reshape to [B*S] for gather_by_value
            inds_vec = inds_slice.reshape([B*S])
            weights_vec = weights_slice.reshape([B*S, 1])

            # Loop over experts to process
            # Note: In a fully optimized CUDA kernel, this would be one launch.
            # Here we loop over experts, which is standard for Python-based MoE if no specific kernel exists.

            # Optimization: Only iterate over experts that were actually selected
            # This avoids N_experts loop (often 64+) for every token, reducing to K loop (e.g. 2-8).
            # However, getting unique indices efficiently from Tensor is hard without copying to CPU.
            # But we must avoid copying to CPU for performance if running on GPU.
            #
            # If backend supports it, we could do `unique_indices = inds_vec.unique()`.
            # Without it, we fallback to iterating all experts OR blindly trying.
            # Given "No external dependencies" and minimal backend:
            # We will assume num_experts is large and sparse activation.

            # Since we cannot easily get unique indices on device without a kernel,
            # we will iterate over experts BUT we rely on `gather_by_value` being fast (returning 0 size quickly)
            # which is implemented in C/CUDA.

            # IMPROVEMENT: Use a set on CPU if we can afford the sync?
            # For 2048 tokens * 4 experts = 8192 indices. Copying 8k ints is cheap.
            # Let's try to optimize by reading indices to CPU, finding unique experts, and only looping those.

            indices_cpu = inds_vec.to_list()
            unique_experts = set(int(x) for x in indices_cpu)

            for expert_idx in unique_experts:
                # Gather tokens that selected this expert at this rank k
                # input_subset: [Count, H]
                # original_indices: [Count]
                input_subset, original_indices = input_flat.gather_by_value(inds_vec, float(expert_idx))

                if input_subset is None or input_subset.shape[0] == 0:
                    continue

                # Also gather the weights for these tokens
                # weight_subset: [Count, 1]
                weight_subset, _ = weights_vec.gather_by_value(inds_vec, float(expert_idx))

                # Run Expert
                expert_output = self.experts[expert_idx](input_subset)

                # Apply weighting: expert_output * weight_subset
                # expert_output: [Count, H]
                # weight_subset: [Count, 1]

                weighted_output = expert_output * weight_subset

                # Scatter Add back to final output
                final_output.scatter_add_by_index(weighted_output, original_indices)

        # 6. Reshape back to [B, S, H]
        return final_output.reshape([B, S, H])
