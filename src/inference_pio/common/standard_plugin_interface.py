"""
Standard Plugin Interface for Inference-PIO System

This module defines the standardized interfaces for all plugins in the Inference-PIO system.
All plugins must implement these interfaces to ensure consistent behavior and interoperability.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, Tuple
from enum import Enum

import torch
import torch.nn as nn


class PluginType(Enum):
    """
    Enum for different types of plugins in the Inference-PIO system.
    """
    ATTENTION = "attention"
    MEMORY_MANAGER = "memory_manager"
    OPTIMIZATION = "optimization"
    HARDWARE = "hardware"
    PERFORMANCE = "performance"
    MODEL_COMPONENT = "model_component"
    TRAINING_STRATEGY = "training_strategy"
    INFERENCE_STRATEGY = "inference_strategy"
    DATA_PROCESSOR = "data_processor"
    METRIC = "metric"
    TUNING_STRATEGY = "tuning_strategy"
    KV_CACHE = "kv_cache"


class PluginMetadata:
    """
    Standardized metadata for a plugin containing essential information.
    """

    def __init__(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        plugin_type: PluginType,
        dependencies: List[str],
        compatibility: Dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
        model_architecture: str = "",
        model_size: str = "",
        required_memory_gb: float = 0.0,
        supported_modalities: List[str] = None,
        license: str = "",
        tags: List[str] = None,
        model_family: str = "",
        num_parameters: int = 0,
        test_coverage: float = 0.0,
        validation_passed: bool = False
    ):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.plugin_type = plugin_type
        self.dependencies = dependencies
        self.compatibility = compatibility
        self.created_at = created_at
        self.updated_at = updated_at
        self.model_architecture = model_architecture
        self.model_size = model_size
        self.required_memory_gb = required_memory_gb
        self.supported_modalities = supported_modalities or []
        self.license = license
        self.tags = tags or []
        self.model_family = model_family
        self.num_parameters = num_parameters
        self.test_coverage = test_coverage
        self.validation_passed = validation_passed


class StandardPluginInterface(ABC):
    """
    Standard interface that all plugins must implement.
    """

    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.is_loaded = False
        self.is_active = False
        self._initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the plugin with the provided parameters.

        Args:
            **kwargs: Additional initialization parameters

        Returns:
            True if initialization was successful, False otherwise
        """

    @abstractmethod
    def load_model(self, config: Any = None) -> nn.Module:
        """
        Load the model with the given configuration.

        Args:
            config: Model configuration (optional)

        Returns:
            Loaded model instance
        """

    @abstractmethod
    def infer(self, data: Any) -> Any:
        """
        Perform inference on the given data.

        Args:
            data: Input data for inference

        Returns:
            Inference results
        """

    @abstractmethod
    def cleanup(self) -> bool:
        """
        Clean up resources used by the plugin.

        Returns:
            True if cleanup was successful, False otherwise
        """

    @abstractmethod
    def supports_config(self, config: Any) -> bool:
        """
        Check if this plugin supports the given configuration.

        Args:
            config: Configuration to check

        Returns:
            True if the configuration is supported, False otherwise
        """


class ModelPluginInterface(StandardPluginInterface):
    """
    Interface for model plugins in the Inference-PIO system.
    Extends the standard plugin interface with model-specific methods.
    """

    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        if metadata.plugin_type != PluginType.MODEL_COMPONENT:
            raise ValueError(
                f"Plugin type must be MODEL_COMPONENT, got {metadata.plugin_type}"
            )

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> Any:
        """
        Tokenize the given text.

        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters

        Returns:
            Tokenized result
        """

    @abstractmethod
    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional decoding parameters

        Returns:
            Decoded text
        """

    @abstractmethod
    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt: Text generation prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """

    def chat_completion(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024, **kwargs) -> str:
        """
        Perform chat completion with the model.

        Args:
            messages: List of message dictionaries with role and content
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        # Default implementation that formats messages and calls generate_text
        formatted_prompt = self._format_chat_messages(messages)
        return self.generate_text(formatted_prompt, max_new_tokens=max_new_tokens, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for the model.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Formatted prompt string
        """
        # Default implementation - subclasses should override for model-specific formatting
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        return formatted


logger = logging.getLogger(__name__)


__all__ = [
    "PluginType",
    "PluginMetadata",
    "StandardPluginInterface",
    "ModelPluginInterface",
    "logger"
]