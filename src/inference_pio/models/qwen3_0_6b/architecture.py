"""
Qwen3-0.6B Self-Contained Architecture

This module implements the Qwen3 model architecture in pure PyTorch to serve as a
fallback when the installed transformers library does not support 'qwen3'.
Based on Qwen2/LLaMA architecture standards.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Import custom generation config
try:
    from ...common.config.generation_config import CustomGenerationConfig
except ImportError:
    class CustomGenerationConfig:
        def __init__(self, **kwargs):
            pass

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device, dtype=torch.float32):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = torch.repeat_interleave(
            key_states, dim=1, repeats=self.num_key_value_groups
        )
        value_states = torch.repeat_interleave(
            value_states, dim=1, repeats=self.num_key_value_groups
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast to fp32 for softmax stability
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = 151643  # Default Qwen pad token
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Simple causal mask generation if not provided
        if attention_mask is None:
            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
            else:
                batch_size, seq_len, _ = inputs_embeds.shape
                device = inputs_embeds.device

            attention_mask = torch.ones((batch_size, 1, 1, seq_len), device=device, dtype=torch.bool)
            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).expand(1, 1, seq_len, seq_len)
            attention_mask = attention_mask & causal_mask.bool()
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        # Convert attention mask to the format expected by attention layers
        # [bs, 1, tgt_seq_len, src_seq_len] -> [bs, 1, 1, src_seq_len] for causal
        if attention_mask.dim() == 2:
            # [bs, seq_len] -> [bs, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            # [bs, 1, seq_len] -> [bs, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(2)
        elif attention_mask.dim() == 4:
            # Already in the right format [bs, 1, seq_len, seq_len]
            pass

        # Convert to negative infinity for masked values
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        pkv = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if past_key_values:
                layer_past = past_key_values[i]
            else:
                layer_past = None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value=None, use_cache=False)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0] # handle return
            else:
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


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        inputs_embeds=None,
        **kwargs,
    ):
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

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Custom generation loop to remove dependency on transformers.generation.
        """
        # Ensure input_ids is on the correct device
        input_ids = input_ids.to(self.device)

        # Keep track of generated tokens
        generated_ids = input_ids.clone()

        # Initialize past_key_values
        past_key_values = None

        for _ in range(max_new_tokens):
            # If we have past_key_values, we only need to pass the last token
            if past_key_values:
                model_inputs = generated_ids[:, -1:]
                position_ids = torch.arange(
                    generated_ids.shape[1] - 1, generated_ids.shape[1],
                    device=self.device
                ).unsqueeze(0)
            else:
                model_inputs = generated_ids
                position_ids = None

            # Forward pass
            outputs, past_key_values = self.forward(
                input_ids=model_inputs,
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=position_ids
            )

            # Get logits for the last token
            next_token_logits = outputs[:, -1, :]

            # Apply repetition penalty if needed
            if repetition_penalty != 1.0:
                score = torch.gather(next_token_logits, 1, generated_ids)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                next_token_logits.scatter_(1, generated_ids, score)

            # Sampling logic
            if do_sample:
                # Temperature scaling
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

                # Top-P (Nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append next token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS (assuming standard EOS token ID for Qwen, can be configurable)
            # 151643 is pad/eos often, or 151645. We'll rely on calling code or config if strict stopping is needed.
            # Here we just generate max_new_tokens.

        return generated_ids

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.model.gradient_checkpointing = True

    @property
    def device(self):
        return next(self.parameters()).device
