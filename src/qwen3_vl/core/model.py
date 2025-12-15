"""
Qwen3-VL Model core implementation
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List
from src.qwen3_vl.core.config import Qwen3VLConfig
from src.qwen3_vl.model_layers.language_decoder import Qwen3VLDecoder
from src.qwen3_vl.model_layers.vision_transformer import Qwen3VLVisionTransformer
from src.qwen3_vl.model_layers.multimodal_projector import Qwen3VLMultimodalProjector


class Qwen3VLModel(nn.Module):
    """
    The Qwen3-VL Model transformer with multimodal capabilities
    """
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

        # Initialize vision encoder
        self.vision_embed_tokens = Qwen3VLVisionTransformer(config)

        # Initialize language decoder
        self.decoder = Qwen3VLDecoder(config)

        # Initialize multimodal projector
        self.multi_modal_projector = Qwen3VLMultimodalProjector(config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        # Process image input if provided
        if pixel_values is not None:
            # Get vision embeddings
            if hasattr(self, 'vision_embed_tokens'):
                vision_embeddings = self.vision_embed_tokens(pixel_values)
            else:
                # If vision embed tokens don't exist, we'll handle this differently
                # For now, we'll just continue without vision processing
                vision_embeddings = None

            if vision_embeddings is not None:
                # Project vision embeddings to language space
                if hasattr(self, 'multi_modal_projector'):
                    vision_embeddings = self.multi_modal_projector(vision_embeddings)

                # Combine with text embeddings if both are provided
                if inputs_embeds is not None:
                    # Concatenate vision and text embeddings
                    inputs_embeds = torch.cat([vision_embeddings, inputs_embeds], dim=1)
                else:
                    inputs_embeds = vision_embeddings

        # Forward through the decoder
        hidden_states = self.decoder(
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

        return hidden_states


def load_qwen3_vl_model(config: Qwen3VLConfig):
    """
    Load a Qwen3-VL model from configuration
    """
    model = Qwen3VLModel(config)
    return model
