"""
Qwen3-Coder-Next Model Implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from .config import Qwen3CoderNextConfig
from .architecture.layer import Qwen3CoderNextDecoderLayer
from .architecture.rotary import Qwen3CoderNextRotaryEmbedding

class Qwen3CoderNextModel(nn.Module):
    def __init__(self, config: Qwen3CoderNextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3CoderNextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.rotary_emb = Qwen3CoderNextRotaryEmbedding(
            config.attention_rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        pkv_len = 0
        if use_cache and past_key_values is not None:
            # Try to infer sequence length from past_key_values
            # 1. Try to find an Attention layer cache (tuple of tensors)
            found_len = False
            for layer_past in past_key_values:
                if layer_past is not None and isinstance(layer_past, tuple) and len(layer_past) > 0:
                    # Expecting (key, value) where key is [bsz, heads, seq_len, head_dim]
                    if hasattr(layer_past[0], 'shape'):
                        pkv_len = layer_past[0].shape[-2]
                        found_len = True
                        break

            # 2. If no attention layer found (unlikely in hybrid), or all None
            # We rely on the caller to provide correct position_ids for DeltaNet-only generation steps
            # if we can't infer it.
            if not found_len:
                 pass # Cannot infer, pkv_len remains 0 unless we track it externally

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            if pkv_len > 0:
                 position_ids = position_ids + pkv_len

        # Attention Mask Preparation
        if attention_mask is None:
             # Create causal mask
             # For hybrid model, we generally use causal mask for attention layers
             # DeltaNet handles causality via recurrence order/masking internally
             attention_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
             attention_mask = attention_mask.to(inputs_embeds.device)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient Checkpointing Logic
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past,
                    self.rotary_emb
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    rotary_emb=self.rotary_emb
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],) if len(layer_outputs) > 2 else (None,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }
