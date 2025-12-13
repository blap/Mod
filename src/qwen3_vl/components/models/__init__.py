"""
Models component for Qwen3-VL
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List
from src.qwen3_vl.core.config import Qwen3VLConfig


class Qwen3VLPreTrainedModel(nn.Module):
    config_class = Qwen3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLDecoderLayer", "Qwen3VLVisionLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config):
        super().__init__()
        self.config = config


class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize vision encoder
        from src.qwen3_vl.components.models.vision_transformer import Qwen3VLVisionTransformer
        from src.qwen3_vl.components.models.language_decoder import Qwen3VLDecoder
        from src.qwen3_vl.components.models.multimodal_projector import Qwen3VLMultimodalProjector
        self.vision_embed_tokens = Qwen3VLVisionTransformer(config)

        # Initialize language decoder
        self.model = Qwen3VLDecoder(config)  # Using model as the main decoder for consistency

        # Initialize multimodal projector
        self.multi_modal_projector = Qwen3VLMultimodalProjector(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        # Process image input if provided
        if pixel_values is not None:
            # Get vision embeddings
            vision_embeddings = self.vision_embed_tokens(pixel_values)
            
            # Project vision embeddings to language space
            vision_embeddings = self.multi_modal_projector(vision_embeddings)
            
            # Combine with text embeddings if both are provided
            if inputs_embeds is not None:
                # Concatenate vision and text embeddings
                inputs_embeds = torch.cat([vision_embeddings, inputs_embeds], dim=1)
            else:
                inputs_embeds = vision_embeddings

        # Forward through the decoder
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits

    def post_init(self):
        """
        A method executed at the end of each model initialization, to execute code that needs the model's modules
        to be initialized.
        """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


__all__ = ["Qwen3VLPreTrainedModel", "Qwen3VLForConditionalGeneration"]