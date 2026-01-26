"""
Qwen3-VL-2B Model Adapter

This module provides a model-specific adapter for the Qwen3-VL-2B model.
The adapter handles model-specific architecture modifications and optimizations.
"""

import logging
from typing import Any, Dict
import torch
import torch.nn as nn

from ...common.model_adapter import BaseModelAdapter
from ...common.nas_controller import NASController

logger = logging.getLogger(__name__)


class Qwen3VL2BModelAdapter(BaseModelAdapter):
    """
    Model adapter for Qwen3-VL-2B model architecture.
    """

    def _capture_original_architecture(self) -> Dict[str, Any]:
        """Capture the original architecture of the Qwen3-VL-2B model."""
        arch_info = {}

        # Capture transformer layers info
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            arch_info['num_layers'] = len(self.model.transformer.layers)
            arch_info['layers'] = self.model.transformer.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            arch_info['num_layers'] = len(self.model.model.layers)
            arch_info['layers'] = self.model.model.layers
        else:
            arch_info['num_layers'] = 0
            arch_info['layers'] = []

        # Capture hidden size and attention heads info
        if hasattr(self.model.config, 'hidden_size'):
            arch_info['hidden_size'] = self.model.config.hidden_size
        if hasattr(self.model.config, 'num_attention_heads'):
            arch_info['num_attention_heads'] = self.model.config.num_attention_heads
        if hasattr(self.model.config, 'intermediate_size'):
            arch_info['intermediate_size'] = self.model.config.intermediate_size

        return arch_info

    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """Adapt the model depth based on the ratio for Qwen3-VL-2B."""
        if depth_ratio >= 1.0:
            return self.model  # No change needed

        num_layers = self.original_architecture.get('num_layers', 0)
        if num_layers == 0:
            logger.warning("Could not determine number of layers, skipping depth adaptation")
            return self.model

        target_layers = max(1, int(num_layers * depth_ratio))

        # Get the layers container
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers_container = self.model.transformer.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers_container = self.model.model.layers
        else:
            logger.warning("Could not find transformer layers, skipping depth adaptation")
            return self.model

        # Keep only the first target_layers
        selected_layers = layers_container[:target_layers]

        # Create a new module list with selected layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            self.model.transformer.layers = nn.ModuleList(selected_layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.model.model.layers = nn.ModuleList(selected_layers)

        logger.info(f"Qwen3-VL-2B depth adapted: {num_layers} -> {target_layers} layers")

        return self.model

    def adapt_width(self, width_ratio: float) -> nn.Module:
        """Adapt the model width based on the ratio for Qwen3-VL-2B."""
        if width_ratio >= 1.0:
            return self.model  # No change needed

        hidden_size = self.original_architecture.get('hidden_size')
        if not hidden_size:
            logger.warning("Could not determine hidden size, skipping width adaptation")
            return self.model

        target_hidden_size = max(64, int(hidden_size * width_ratio))  # Minimum size of 64

        # Modify the model's hidden size
        if hasattr(self.model.config, 'hidden_size'):
            self.model.config.hidden_size = target_hidden_size

        # Update embedding dimensions
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'word_embeddings'):
            emb = self.model.transformer.word_embeddings
            if hasattr(emb, 'weight') and emb.weight.shape[1] != target_hidden_size:
                # Create new embedding with target size
                new_emb = nn.Embedding(emb.num_embeddings, target_hidden_size)
                # Copy weights for the common part
                min_size = min(emb.embedding_dim, target_hidden_size)
                new_emb.weight.data[:, :min_size] = emb.weight.data[:, :min_size]
                self.model.transformer.word_embeddings = new_emb
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            emb = self.model.model.embed_tokens
            if hasattr(emb, 'weight') and emb.weight.shape[1] != target_hidden_size:
                new_emb = nn.Embedding(emb.num_embeddings, target_hidden_size)
                min_size = min(emb.embedding_dim, target_hidden_size)
                new_emb.weight.data[:, :min_size] = emb.weight.data[:, :min_size]
                self.model.model.embed_tokens = new_emb

        # Update layer norms and other components that depend on hidden size
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                if module.normalized_shape[0] != target_hidden_size:
                    new_norm = nn.LayerNorm(target_hidden_size)
                    # Copy weights for the common part
                    min_size = min(module.normalized_shape[0], target_hidden_size)
                    new_norm.weight.data[:min_size] = module.weight.data[:min_size]
                    new_norm.bias.data[:min_size] = module.bias.data[:min_size]
                    # Set the new layer norm in place
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]

                    if parent_name:
                        parent_module = dict(self.model.named_modules())[parent_name]
                        setattr(parent_module, child_name, new_norm)
                    else:
                        setattr(self.model, child_name, new_norm)

        logger.info(f"Qwen3-VL-2B width adapted: hidden_size {hidden_size} -> {target_hidden_size}")

        return self.model


__all__ = [
    "Qwen3VL2BModelAdapter"
]