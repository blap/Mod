"""
Model Adapter for Continuous NAS System

This module provides adapters for different model architectures to work with the
continuous NAS system, allowing dynamic adjustment of depth and width during inference.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .input_complexity_analyzer import InputComplexityAnalyzer
from .nas_controller import ContinuousNASController, NASConfig

logger = logging.getLogger(__name__)


class BaseModelAdapter:
    """
    Base class for model adapters that work with the NAS system.
    Each specific model type should inherit from this class.
    """

    def __init__(self, model: nn.Module, nas_controller: ContinuousNASController):
        self.model = model
        self.nas_controller = nas_controller
        self.original_architecture = self._capture_original_architecture()

    def _capture_original_architecture(self) -> Dict[str, Any]:
        """Capture the original architecture of the model."""
        # This should be overridden by subclasses
        return {}

    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """Adapt the model depth based on the ratio."""
        # This should be overridden by subclasses
        return self.model

    def adapt_width(self, width_ratio: float) -> nn.Module:
        """Adapt the model width based on the ratio."""
        # This should be overridden by subclasses
        return self.model

    def adapt_architecture(self, depth_ratio: float, width_ratio: float) -> nn.Module:
        """Adapt both depth and width of the model."""
        # First adapt depth
        adapted_model = self.adapt_depth(depth_ratio)
        # Then adapt width
        adapted_model = self.adapt_width(width_ratio)
        return adapted_model


class GLM47ModelAdapter(BaseModelAdapter):
    """
    Model adapter for GLM-4.7 model architecture.
    """

    def _capture_original_architecture(self) -> Dict[str, Any]:
        """Capture the original architecture of the GLM-4.7 model."""
        arch_info = {}

        # Capture transformer layers info
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            arch_info["num_layers"] = len(self.model.transformer.layers)
            arch_info["layers"] = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            arch_info["num_layers"] = len(self.model.model.layers)
            arch_info["layers"] = self.model.model.layers
        else:
            arch_info["num_layers"] = 0
            arch_info["layers"] = []

        # Capture hidden size and attention heads info
        if hasattr(self.model.config, "hidden_size"):
            arch_info["hidden_size"] = self.model.config.hidden_size
        if hasattr(self.model.config, "num_attention_heads"):
            arch_info["num_attention_heads"] = self.model.config.num_attention_heads
        if hasattr(self.model.config, "intermediate_size"):
            arch_info["intermediate_size"] = self.model.config.intermediate_size

        return arch_info

    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """Adapt the model depth based on the ratio for GLM-4.7."""
        if depth_ratio >= 1.0:
            return self.model  # No change needed

        num_layers = self.original_architecture.get("num_layers", 0)
        if num_layers == 0:
            logger.warning(
                "Could not determine number of layers, skipping depth adaptation"
            )
            return self.model

        target_layers = max(1, int(num_layers * depth_ratio))

        # Get the layers container
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            layers_container = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_container = self.model.model.layers
        else:
            logger.warning(
                "Could not find transformer layers, skipping depth adaptation"
            )
            return self.model

        # Keep only the first target_layers
        selected_layers = layers_container[:target_layers]

        # Create a new module list with selected layers
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            self.model.transformer.layers = nn.ModuleList(selected_layers)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers = nn.ModuleList(selected_layers)

        logger.info(f"GLM-4.7 depth adapted: {num_layers} -> {target_layers} layers")

        return self.model

    def adapt_width(self, width_ratio: float) -> nn.Module:
        """Adapt the model width based on the ratio for GLM-4.7."""
        if width_ratio >= 1.0:
            return self.model  # No change needed

        hidden_size = self.original_architecture.get("hidden_size")
        if not hidden_size:
            logger.warning("Could not determine hidden size, skipping width adaptation")
            return self.model

        target_hidden_size = max(
            64, int(hidden_size * width_ratio)
        )  # Minimum size of 64

        # Modify the model's hidden size
        if hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = target_hidden_size

        # Update embedding dimensions
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "word_embeddings"
        ):
            emb = self.model.transformer.word_embeddings
            if hasattr(emb, "weight") and emb.weight.shape[1] != target_hidden_size:
                # Create new embedding with target size
                new_emb = nn.Embedding(emb.num_embeddings, target_hidden_size)
                # Copy weights for the common part
                min_size = min(emb.embedding_dim, target_hidden_size)
                new_emb.weight.data[:, :min_size] = emb.weight.data[:, :min_size]
                self.model.transformer.word_embeddings = new_emb
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            emb = self.model.model.embed_tokens
            if hasattr(emb, "weight") and emb.weight.shape[1] != target_hidden_size:
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
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    if parent_name:
                        parent_module = dict(self.model.named_modules())[parent_name]
                        setattr(parent_module, child_name, new_norm)
                    else:
                        setattr(self.model, child_name, new_norm)

        logger.info(
            f"GLM-4.7 width adapted: hidden_size {hidden_size} -> {target_hidden_size}"
        )

        return self.model


class Qwen3Coder30BModelAdapter(BaseModelAdapter):
    """
    Model adapter for Qwen3-Coder-30B model architecture.
    """

    def _capture_original_architecture(self) -> Dict[str, Any]:
        """Capture the original architecture of the Qwen3-Coder-30B model."""
        arch_info = {}

        # Capture transformer layers info
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            arch_info["num_layers"] = len(self.model.transformer.layers)
            arch_info["layers"] = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            arch_info["num_layers"] = len(self.model.model.layers)
            arch_info["layers"] = self.model.model.layers
        else:
            arch_info["num_layers"] = 0
            arch_info["layers"] = []

        # Capture hidden size and attention heads info
        if hasattr(self.model.config, "hidden_size"):
            arch_info["hidden_size"] = self.model.config.hidden_size
        if hasattr(self.model.config, "num_attention_heads"):
            arch_info["num_attention_heads"] = self.model.config.num_attention_heads
        if hasattr(self.model.config, "intermediate_size"):
            arch_info["intermediate_size"] = self.model.config.intermediate_size

        return arch_info

    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """Adapt the model depth based on the ratio for Qwen3-Coder-30B."""
        if depth_ratio >= 1.0:
            return self.model  # No change needed

        num_layers = self.original_architecture.get("num_layers", 0)
        if num_layers == 0:
            logger.warning(
                "Could not determine number of layers, skipping depth adaptation"
            )
            return self.model

        target_layers = max(1, int(num_layers * depth_ratio))

        # Get the layers container
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            layers_container = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_container = self.model.model.layers
        else:
            logger.warning(
                "Could not find transformer layers, skipping depth adaptation"
            )
            return self.model

        # Keep only the first target_layers
        selected_layers = layers_container[:target_layers]

        # Create a new module list with selected layers
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            self.model.transformer.layers = nn.ModuleList(selected_layers)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers = nn.ModuleList(selected_layers)

        logger.info(
            f"Qwen3-Coder-30B depth adapted: {num_layers} -> {target_layers} layers"
        )

        return self.model

    def adapt_width(self, width_ratio: float) -> nn.Module:
        """Adapt the model width based on the ratio for Qwen3-Coder-30B."""
        if width_ratio >= 1.0:
            return self.model  # No change needed

        hidden_size = self.original_architecture.get("hidden_size")
        if not hidden_size:
            logger.warning("Could not determine hidden size, skipping width adaptation")
            return self.model

        target_hidden_size = max(
            64, int(hidden_size * width_ratio)
        )  # Minimum size of 64

        # Modify the model's hidden size
        if hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = target_hidden_size

        # Update embedding dimensions
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "word_embeddings"
        ):
            emb = self.model.transformer.word_embeddings
            if hasattr(emb, "weight") and emb.weight.shape[1] != target_hidden_size:
                # Create new embedding with target size
                new_emb = nn.Embedding(emb.num_embeddings, target_hidden_size)
                # Copy weights for the common part
                min_size = min(emb.embedding_dim, target_hidden_size)
                new_emb.weight.data[:, :min_size] = emb.weight.data[:, :min_size]
                self.model.transformer.word_embeddings = new_emb
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            emb = self.model.model.embed_tokens
            if hasattr(emb, "weight") and emb.weight.shape[1] != target_hidden_size:
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
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    if parent_name:
                        parent_module = dict(self.model.named_modules())[parent_name]
                        setattr(parent_module, child_name, new_norm)
                    else:
                        setattr(self.model, child_name, new_norm)

        logger.info(
            f"Qwen3-Coder-30B width adapted: hidden_size {hidden_size} -> {target_hidden_size}"
        )

        return self.model


class Qwen34BInstruct2507ModelAdapter(BaseModelAdapter):
    """
    Model adapter for Qwen3-4B-Instruct-2507 model architecture.
    """

    def __init__(self, model: nn.Module, nas_controller: ContinuousNASController):
        super().__init__(model, nas_controller)
        self.model_type = "qwen3_4b_instruct_2507"

    def _capture_original_architecture(self) -> Dict[str, Any]:
        """Capture the original architecture of the Qwen3-4B-Instruct-2507 model."""
        arch_info = {}

        # Capture transformer layers info
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            arch_info["num_layers"] = len(self.model.transformer.layers)
            arch_info["layers"] = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            arch_info["num_layers"] = len(self.model.model.layers)
            arch_info["layers"] = self.model.model.layers
        else:
            arch_info["num_layers"] = 0
            arch_info["layers"] = []

        # Capture hidden size and attention heads info
        if hasattr(self.model.config, "hidden_size"):
            arch_info["hidden_size"] = self.model.config.hidden_size
        if hasattr(self.model.config, "num_attention_heads"):
            arch_info["num_attention_heads"] = self.model.config.num_attention_heads
        if hasattr(self.model.config, "intermediate_size"):
            arch_info["intermediate_size"] = self.model.config.intermediate_size

        return arch_info

    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """Adapt the model depth based on the ratio for Qwen3-4B-Instruct-2507."""
        if depth_ratio >= 1.0:
            return self.model  # No change needed

        num_layers = self.original_architecture.get("num_layers", 0)
        if num_layers == 0:
            logger.warning(
                "Could not determine number of layers, skipping depth adaptation"
            )
            return self.model

        target_layers = max(1, int(num_layers * depth_ratio))

        # Get the layers container
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            layers_container = self.model.transformer.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_container = self.model.model.layers
        else:
            logger.warning(
                "Could not find transformer layers, skipping depth adaptation"
            )
            return self.model

        # Keep only the first target_layers
        selected_layers = layers_container[:target_layers]

        # Create a new module list with selected layers
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "layers"
        ):
            self.model.transformer.layers = nn.ModuleList(selected_layers)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers = nn.ModuleList(selected_layers)

        logger.info(
            f"Qwen3-4B-Instruct-2507 depth adapted: {num_layers} -> {target_layers} layers"
        )

        return self.model

    def adapt_width(self, width_ratio: float) -> nn.Module:
        """Adapt the model width based on the ratio for Qwen3-4B-Instruct-2507."""
        if width_ratio >= 1.0:
            return self.model  # No change needed

        hidden_size = self.original_architecture.get("hidden_size")
        if not hidden_size:
            logger.warning("Could not determine hidden size, skipping width adaptation")
            return self.model

        target_hidden_size = max(
            64, int(hidden_size * width_ratio)
        )  # Minimum size of 64

        # Modify the model's hidden size
        if hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = target_hidden_size

        # Update embedding dimensions
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "word_embeddings"
        ):
            emb = self.model.transformer.word_embeddings
            if hasattr(emb, "weight") and emb.weight.shape[1] != target_hidden_size:
                # Create new embedding with target size
                new_emb = nn.Embedding(emb.num_embeddings, target_hidden_size)
                # Copy weights for the common part
                min_size = min(emb.embedding_dim, target_hidden_size)
                new_emb.weight.data[:, :min_size] = emb.weight.data[:, :min_size]
                self.model.transformer.word_embeddings = new_emb
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            emb = self.model.model.embed_tokens
            if hasattr(emb, "weight") and emb.weight.shape[1] != target_hidden_size:
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
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    if parent_name:
                        parent_module = dict(self.model.named_modules())[parent_name]
                        setattr(parent_module, child_name, new_norm)
                    else:
                        setattr(self.model, child_name, new_norm)

        logger.info(
            f"Qwen3-4B-Instruct-2507 width adapted: hidden_size {hidden_size} -> {target_hidden_size}"
        )

        return self.model


def get_model_adapter(
    model: nn.Module, nas_controller: ContinuousNASController
) -> BaseModelAdapter:
    """
    Factory function to get the appropriate model adapter based on the model type.
    """
    # Try to identify the model type based on its attributes
    model_str = str(type(model)).lower()

    if "glm" in model_str and "4" in model_str:
        return GLM47ModelAdapter(model, nas_controller)
    elif "qwen3" in model_str and "4b" in model_str:
        return Qwen34BInstruct2507ModelAdapter(model, nas_controller)
    elif "qwen3" in model_str and ("coder" in model_str or "30b" in model_str.lower()):
        return Qwen3Coder30BModelAdapter(model, nas_controller)
    elif "qwen3" in model_str and ("vl" in model_str or "2b" in model_str.lower()):
        # Import the Qwen3-VL adapter from its specific plugin directory
        from ..models.qwen3_vl_2b.qwen3_vl_2b_model_adapter import Qwen3VL2BModelAdapter

        return Qwen3VL2BModelAdapter(model, nas_controller)
    else:
        # Default to a basic adapter if model type cannot be identified
        logger.warning(f"Unknown model type '{model_str}', using basic adapter")
        return BaseModelAdapter(model, nas_controller)


__all__ = [
    "BaseModelAdapter",
    "GLM47ModelAdapter",
    "Qwen34BInstruct2507ModelAdapter",
    "Qwen3Coder30BModelAdapter",
    "get_model_adapter",
]
