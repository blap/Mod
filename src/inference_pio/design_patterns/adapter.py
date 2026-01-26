"""
Adapter Pattern Implementation for Inference-PIO

This module implements the Adapter pattern for integrating different model architectures.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..common.nas_controller import ContinuousNASController, NASConfig
from ..common.input_complexity_analyzer import InputComplexityAnalyzer


logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_architecture = self._capture_original_architecture()
    
    @abstractmethod
    def _capture_original_architecture(self) -> Dict[str, Any]:
        """
        Capture the original architecture of the model.
        
        Returns:
            Dictionary containing architecture information
        """
        pass
    
    @abstractmethod
    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """
        Adapt the model depth based on the ratio.
        
        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
        pass
    
    @abstractmethod
    def adapt_width(self, width_ratio: float) -> nn.Module:
        """
        Adapt the model width based on the ratio.
        
        Args:
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
        pass
    
    def adapt_architecture(self, depth_ratio: float, width_ratio: float) -> nn.Module:
        """
        Adapt both depth and width of the model.
        
        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
        # First adapt depth
        adapted_model = self.adapt_depth(depth_ratio)
        # Then adapt width
        adapted_model = self.adapt_width(width_ratio)
        return adapted_model


class GLM47ModelAdapter(ModelAdapter):
    """
    Model adapter for GLM-4.7 model architecture.
    """
    
    def _capture_original_architecture(self) -> Dict[str, Any]:
        """
        Capture the original architecture of the GLM-4.7 model.

        Returns:
            Dictionary containing architecture information
        """
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
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            arch_info['hidden_size'] = self.model.config.hidden_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_attention_heads'):
            arch_info['num_attention_heads'] = self.model.config.num_attention_heads
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'intermediate_size'):
            arch_info['intermediate_size'] = self.model.config.intermediate_size

        return arch_info
    
    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """
        Adapt the model depth based on the ratio for GLM-4.7.

        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)

        Returns:
            Adapted model
        """
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
        else:
            logger.warning("Could not update layers, model structure not compatible")
            return self.model

        logger.info(f"GLM-4.7 depth adapted: {num_layers} -> {target_layers} layers")

        return self.model
    
    def adapt_width(self, width_ratio: float) -> nn.Module:
        """
        Adapt the model width based on the ratio for GLM-4.7.

        Args:
            width_ratio: Ratio to adjust width (0.0 to 1.0)

        Returns:
            Adapted model
        """
        if width_ratio >= 1.0:
            return self.model  # No change needed

        hidden_size = self.original_architecture.get('hidden_size')
        if not hidden_size:
            logger.warning("Could not determine hidden size, skipping width adaptation")
            return self.model

        target_hidden_size = max(64, int(hidden_size * width_ratio))  # Minimum size of 64

        # Modify the model's hidden size if config exists
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
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
        else:
            logger.warning("Could not find embedding layers to adapt, skipping embedding width adaptation")

        # Update layer norms and other components that depend on hidden size
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.LayerNorm):
                    if hasattr(module, 'normalized_shape') and module.normalized_shape[0] != target_hidden_size:
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
        except Exception as e:
            logger.warning(f"Error during layer norm adaptation: {e}")

        logger.info(f"GLM-4.7 width adapted: hidden_size {hidden_size} -> {target_hidden_size}")

        return self.model


class Qwen34BInstruct2507ModelAdapter(ModelAdapter):
    """
    Model adapter for Qwen3-4B-Instruct-2507 model architecture.
    """
    
    def _capture_original_architecture(self) -> Dict[str, Any]:
        """
        Capture the original architecture of the Qwen3-4B-Instruct-2507 model.

        Returns:
            Dictionary containing architecture information
        """
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
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            arch_info['hidden_size'] = self.model.config.hidden_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_attention_heads'):
            arch_info['num_attention_heads'] = self.model.config.num_attention_heads
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'intermediate_size'):
            arch_info['intermediate_size'] = self.model.config.intermediate_size

        return arch_info
    
    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """
        Adapt the model depth based on the ratio for Qwen3-4B-Instruct-2507.
        
        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
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
        
        logger.info(f"Qwen3-4B-Instruct-2507 depth adapted: {num_layers} -> {target_layers} layers")
        
        return self.model
    
    def adapt_width(self, width_ratio: float) -> nn.Module:
        """
        Adapt the model width based on the ratio for Qwen3-4B-Instruct-2507.
        
        Args:
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
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
        
        logger.info(f"Qwen3-4B-Instruct-2507 width adapted: hidden_size {hidden_size} -> {target_hidden_size}")
        
        return self.model


class Qwen3Coder30BModelAdapter(ModelAdapter):
    """
    Model adapter for Qwen3-Coder-30B model architecture.
    """
    
    def _capture_original_architecture(self) -> Dict[str, Any]:
        """
        Capture the original architecture of the Qwen3-Coder-30B model.

        Returns:
            Dictionary containing architecture information
        """
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
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            arch_info['hidden_size'] = self.model.config.hidden_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_attention_heads'):
            arch_info['num_attention_heads'] = self.model.config.num_attention_heads
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'intermediate_size'):
            arch_info['intermediate_size'] = self.model.config.intermediate_size

        return arch_info
    
    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """
        Adapt the model depth based on the ratio for Qwen3-Coder-30B.
        
        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
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
        
        logger.info(f"Qwen3-Coder-30B depth adapted: {num_layers} -> {target_layers} layers")
        
        return self.model
    
    def adapt_width(self, width_ratio: float) -> nn.Module:
        """
        Adapt the model width based on the ratio for Qwen3-Coder-30B.
        
        Args:
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
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
        
        logger.info(f"Qwen3-Coder-30B width adapted: hidden_size {hidden_size} -> {target_hidden_size}")
        
        return self.model


class Qwen3VL2BModelAdapter(ModelAdapter):
    """
    Model adapter for Qwen3-VL-2B model architecture.
    """
    
    def _capture_original_architecture(self) -> Dict[str, Any]:
        """
        Capture the original architecture of the Qwen3-VL-2B model.

        Returns:
            Dictionary containing architecture information
        """
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
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            arch_info['hidden_size'] = self.model.config.hidden_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_attention_heads'):
            arch_info['num_attention_heads'] = self.model.config.num_attention_heads
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'intermediate_size'):
            arch_info['intermediate_size'] = self.model.config.intermediate_size

        return arch_info
    
    def adapt_depth(self, depth_ratio: float) -> nn.Module:
        """
        Adapt the model depth based on the ratio for Qwen3-VL-2B.
        
        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
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
        """
        Adapt the model width based on the ratio for Qwen3-VL-2B.
        
        Args:
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
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


class ModelAdapterSelector:
    """
    Selector class that chooses the appropriate model adapter based on the model type.
    """
    
    def __init__(self):
        self.adapters: Dict[str, type] = {
            'glm_4_7_flash': GLM47ModelAdapter,
            'qwen3_4b_instruct_2507': Qwen34BInstruct2507ModelAdapter,
            'qwen3_coder_30b': Qwen3Coder30BModelAdapter,
            'qwen3_vl_2b': Qwen3VL2BModelAdapter,
        }
    
    def select_adapter(self, model_type: str, model: nn.Module) -> ModelAdapter:
        """
        Select the appropriate model adapter based on the model type.
        
        Args:
            model_type: Type of model ('glm_4_7', 'qwen3_4b_instruct_2507', etc.)
            model: The model to adapt
            
        Returns:
            Appropriate model adapter instance
        """
        adapter_class = self.adapters.get(model_type.lower())
        if adapter_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return adapter_class(model)
    
    def adapt_model(self, model_type: str, model: nn.Module, 
                   depth_ratio: float = 1.0, width_ratio: float = 1.0) -> nn.Module:
        """
        Adapt the model using the appropriate adapter.
        
        Args:
            model_type: Type of model to adapt
            model: The model to adapt
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
        adapter = self.select_adapter(model_type, model)
        return adapter.adapt_architecture(depth_ratio, width_ratio)


class ModelIntegrationAdapter:
    """
    Adapter that integrates different model interfaces to work with a common interface.
    """
    
    def __init__(self, model: nn.Module, model_type: str):
        self.model = model
        self.model_type = model_type.lower()
        self.adapter_selector = ModelAdapterSelector()
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the adapted model.
        
        Args:
            *args: Arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            Model output
        """
        # Different models may have different forward signatures
        if self.model_type in ['glm_4_7_flash', 'qwen3_4b_instruct_2507', 'qwen3_coder_30b']:
            # Text models typically take input_ids and attention_mask
            return self.model(*args, **kwargs)
        elif self.model_type == 'qwen3_vl_2b':
            # Vision-language models may take pixel_values as well
            return self.model(*args, **kwargs)
        else:
            # Default to standard forward pass
            return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        Generation method adapted for different model types.
        
        Args:
            *args: Arguments to pass to the model's generate method
            **kwargs: Keyword arguments to pass to the model's generate method
            
        Returns:
            Generated output
        """
        if hasattr(self.model, 'generate'):
            return self.model.generate(*args, **kwargs)
        else:
            raise NotImplementedError(f"Generate method not implemented for model type: {self.model_type}")
    
    def adapt_architecture(self, depth_ratio: float = 1.0, width_ratio: float = 1.0) -> nn.Module:
        """
        Adapt the model architecture using the appropriate adapter.
        
        Args:
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model
        """
        return self.adapter_selector.adapt_model(
            self.model_type, self.model, depth_ratio, width_ratio
        )


__all__ = [
    'ModelAdapter',
    'GLM47ModelAdapter',
    'Qwen34BInstruct2507ModelAdapter',
    'Qwen3Coder30BModelAdapter',
    'Qwen3VL2BModelAdapter',
    'ModelAdapterSelector',
    'ModelIntegrationAdapter'
]