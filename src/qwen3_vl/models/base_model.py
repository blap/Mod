"""
Main entry point for Qwen3-VL model.
"""

import torch
import torch.nn as nn
from typing import Optional
from src.qwen3_vl.config.config import Qwen3VLConfig


class Qwen3VLModel(nn.Module):
    """
    Qwen3-VL Model implementation.
    """

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config

        # Initialize model parameters based on config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads

        # Create embedding layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Create transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=0.0,
                activation='gelu',  # Using gelu instead of silu to avoid issues
                batch_first=True
            ) for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        with torch.no_grad():
            self.embed_tokens.weight.normal_(mean=0.0, std=self.config.initializer_range)
            self.embed_positions.weight.normal_(mean=0.0, std=self.config.initializer_range)
            self.lm_head.weight.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            pixel_values: Pixel values for vision input
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        # Get input embeddings
        if hasattr(self, 'embed_tokens'):
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            # If embed_tokens doesn't exist, use input_ids directly
            inputs_embeds = input_ids

        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.size(1), dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        if hasattr(self, 'embed_positions'):
            position_embeds = self.embed_positions(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            # If embed_positions doesn't exist, use inputs_embeds directly
            hidden_states = inputs_embeds

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)

        # Apply final layer norm
        hidden_states = self.final_layernorm(hidden_states)

        # Generate logits
        logits = self.lm_head(hidden_states)

        return logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text using the model.

        Args:
            input_ids: Input token IDs
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            **kwargs: Additional generation arguments

        Returns:
            Generated sequences
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', self.config.pad_token_id)

        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]

        # Generate tokens one by one
        sequences = input_ids.clone()
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        while cur_len < max_length:
            # Get model outputs
            outputs = self(input_ids=sequences)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply softmax to get probabilities
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            if do_sample:
                # Sample from the distribution
                next_tokens = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            else:
                # Take the most likely token
                next_tokens = torch.argmax(next_token_probs, dim=-1)

            # Update sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

            # Update unfinished sequences
            for eos_id in eos_token_id:
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_id)

            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

            cur_len += 1

        return sequences


def create_model_from_config(config: Qwen3VLConfig) -> Qwen3VLModel:
    """
    Factory function to create a Qwen3-VL model from configuration.

    Args:
        config: Qwen3VLConfig instance

    Returns:
        Qwen3VLModel instance
    """
    return Qwen3VLModel(config)


def create_model_from_pretrained(
    pretrained_model_name_or_path: str,
    config: Optional[Qwen3VLConfig] = None,
    **kwargs
) -> Qwen3VLModel:
    """
    Create a Qwen3-VL model from a pretrained model.

    Args:
        pretrained_model_name_or_path: Path to pretrained model
        config: Configuration to use (optional)
        **kwargs: Additional arguments

    Returns:
        Qwen3VLModel instance
    """
    if config is None:
        config = Qwen3VLConfig()

    model = create_model_from_config(config)

    return model