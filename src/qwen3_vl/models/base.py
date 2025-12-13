"""
Base model classes and interfaces for the Flexible Model System.

This module implements a comprehensive architecture for managing multiple models dynamically,
with support for different model types, adapters, loading/unloading, and hardware abstraction.
"""


import abc
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
import logging
from transformers import AutoModel, AutoConfig, AutoTokenizer


# Custom exceptions
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when there's an error loading a model."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class ModelValidationError(ModelError):
    """Raised when model configuration validation fails."""
    pass


class IModel(abc.ABC):
    """
    Interface defining the core functionality that all models must implement.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Allow the model to be called directly."""
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass

    @abc.abstractmethod
    def train(self, mode: bool = True):
        """Set the model to training mode."""
        pass

    @abc.abstractmethod
    def eval(self):
        """Set the model to evaluation mode."""
        pass

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        pass

    @abc.abstractmethod
    def get_device(self) -> torch.device:
        """Get the device the model is currently on."""
        pass

    @abc.abstractmethod
    def to_device(self, device: Union[str, torch.device]) -> 'IModel':
        """Move the model to the specified device."""
        pass


class ModelAdapter(IModel):
    """
    Base adapter class that adapts different model types to the unified interface.
    """

    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None,
                 tokenizer=None):
        self.model = model
        self.config = config or {}
        self.tokenizer = tokenizer
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __call__(self, *args, **kwargs):
        """Call the model directly."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self.config

    def get_device(self) -> torch.device:
        """Get the device the model is currently on."""
        # Get the device of the first parameter
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # If no parameters, assume CPU
            return torch.device('cpu')

    def to_device(self, device: Union[str, torch.device]) -> 'ModelAdapter':
        """Move the model to the specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.model = self.model.to(device)
        return self

    def get_tokenizer(self):
        """Get the tokenizer associated with the model."""
        return self.tokenizer

    def save(self, save_path: str):
        """Save the model to the specified path."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, save_path)

    def load(self, load_path: str):
        """Load the model from the specified path."""
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', self.config)


class BaseModelAdapter(ModelAdapter):
    """
    Base implementation of model adapter that can be extended for specific model types.
    """

    def load_model(self, model_path: str, config: Optional[Dict[str, Any]] = None) -> 'BaseModelAdapter':
        """
        Load a model from the specified path.

        Args:
            model_path: Path to the model
            config: Configuration for the model

        Returns:
            New instance of the adapter with the loaded model
        """
        raise RuntimeError("BaseModelAdapter.load_model() is an abstract method that must be "
                         "implemented by subclasses. Use a specific model adapter instead.")


class HuggingFaceAdapter(BaseModelAdapter):
    """
    Adapter for Hugging Face models.
    """

    def load_model(self, model_path: str, config: Optional[Dict[str, Any]] = None) -> 'HuggingFaceAdapter':
        """
        Load a Hugging Face model from the specified path.

        Args:
            model_path: Path to the Hugging Face model
            config: Configuration for the model

        Returns:
            New instance of HuggingFaceAdapter with the loaded model
        """
        try:
            # Load configuration
            model_config = AutoConfig.from_pretrained(model_path, **(config or {}))

            # Load model
            model = AutoModel.from_pretrained(model_path, config=model_config)

            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = None  # Tokenizer might not be available

            return HuggingFaceAdapter(model, config, tokenizer)

        except Exception as e:
            raise ModelLoadError(f"Failed to load Hugging Face model from {model_path}: {str(e)}")


class Qwen3VLAdapter(BaseModelAdapter):
    """
    Adapter for Qwen3-VL models.
    """

    def load_model(self, model_path: str, config: Optional[Dict[str, Any]] = None) -> 'Qwen3VLAdapter':
        """
        Load a Qwen3-VL model from the specified path.

        Args:
            model_path: Path to the Qwen3-VL model
            config: Configuration for the model

        Returns:
            New instance of Qwen3VLAdapter with the loaded model
        """
        try:
            # Import here to avoid circular imports
            from ..models.base_model import create_model_from_pretrained
            from ..config.base_config import Qwen3VLConfig

            # Create config if not provided
            model_config = None
            if config:
                model_config = Qwen3VLConfig(**config)
            else:
                # Use default config if none provided
                model_config = Qwen3VLConfig()

            # Load the model
            model = create_model_from_pretrained(model_path, model_config)

            return Qwen3VLAdapter(model, config)

        except Exception as e:
            raise ModelLoadError(f"Failed to load Qwen3-VL model from {model_path}: {str(e)}")


class FlexibleModelManager:
    """
    Manager class that handles multiple models dynamically, supporting loading,
    unloading, and switching models at runtime.
    """

    def __init__(self):
        self.models: Dict[str, ModelAdapter] = {}
        self.adapters: Dict[str, Type[ModelAdapter]] = {
            'huggingface': HuggingFaceAdapter,
            'qwen3-vl': Qwen3VLAdapter,
            'default': BaseModelAdapter,
        }
        self.active_model: Optional[str] = None
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def register_adapter(self, name: str, adapter_class: Type[ModelAdapter]):
        """
        Register a new adapter type.

        Args:
            name: Name of the adapter type
            adapter_class: Class of the adapter to register
        """
        self.adapters[name.lower()] = adapter_class

    def get_registered_adapter(self, name: str) -> Optional[Type[ModelAdapter]]:
        """
        Get a registered adapter class by name.

        Args:
            name: Name of the adapter type

        Returns:
            Adapter class if found, None otherwise
        """
        return self.adapters.get(name.lower())

    def load_model(self, model_name: str, model_path: str, adapter_type: str = 'default',
                   config: Optional[Dict[str, Any]] = None, device: Optional[Union[str, torch.device]] = None):
        """
        Load a model with the specified adapter.

        Args:
            model_name: Name to assign to the loaded model
            model_path: Path to the model file/directory
            adapter_type: Type of adapter to use
            config: Configuration for the model
            device: Device to load the model on
        """
        # Get the adapter class
        adapter_class = self.get_registered_adapter(adapter_type)
        if adapter_class is None:
            raise ModelNotFoundError(f"Adapter type '{adapter_type}' not found")

        try:
            # Create an instance of the adapter and load the model
            temp_adapter = adapter_class(None)  # Create with None initially
            loaded_adapter = temp_adapter.load_model(model_path, config)

            # Move to specified device if provided
            if device:
                loaded_adapter.to_device(device)

            # Store the loaded model
            self.models[model_name] = loaded_adapter

            # If no active model is set, make this the active one
            if self.active_model is None:
                self.active_model = model_name

            self._logger.info(f"Successfully loaded model '{model_name}' using {adapter_type} adapter")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{model_name}': {str(e)}")

    def unload_model(self, model_name: str):
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            del self.models[model_name]

            # If this was the active model, reset active model
            if self.active_model == model_name:
                # Set to another available model or None
                available_models = list(self.models.keys())
                self.active_model = available_models[0] if available_models else None

            self._logger.info(f"Successfully unloaded model '{model_name}'")

    def switch_model(self, model_name: str):
        """
        Switch the active model to the specified model.

        Args:
            model_name: Name of the model to make active
        """
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        self.active_model = model_name
        self._logger.info(f"Switched active model to '{model_name}'")

    def get_active_model(self) -> Optional[ModelAdapter]:
        """
        Get the currently active model.

        Returns:
            Active model adapter or None if no model is active
        """
        if self.active_model and self.active_model in self.models:
            return self.models[self.active_model]
        return None

    def list_models(self) -> List[str]:
        """
        Get a list of all loaded model names.

        Returns:
            List of model names
        """
        return list(self.models.keys())

    def validate_config(self, config: Dict[str, Any]):
        """
        Validate a model configuration.

        Args:
            config: Configuration dictionary to validate
        """
        required_fields = ['model_path', 'adapter_type']

        for field in required_fields:
            if field not in config:
                raise ModelValidationError(f"Missing required configuration field: {field}")

    def invoke_active_model(self, *args, **kwargs):
        """
        Invoke the active model with the provided arguments.

        Args:
            *args: Positional arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model

        Returns:
            Model output
        """
        active_model = self.get_active_model()
        if not active_model:
            raise ModelNotFoundError("No active model available")

        return active_model(*args, **kwargs)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.

        Args:
            model_name: Name of the model to get info for

        Returns:
            Dictionary with model information or None if not found
        """
        if model_name not in self.models:
            return None

        adapter = self.models[model_name]
        return {
            'name': model_name,
            'device': str(adapter.get_device()),
            'config': adapter.get_config(),
            'type': type(adapter).__name__,
        }


def create_model_from_pretrained(model_path: str, config=None):
    """
    Create a model from a pretrained model path.

    Args:
        model_path: Path to the pretrained model
        config: Model configuration (optional)

    Returns:
        Model instance
    """
    # This is a placeholder implementation - in a real system,
    # this would load the actual model from the specified path
    import torch.nn as nn

    # Create a dummy model for testing purposes
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = nn.Parameter(torch.randn(1))

        def forward(self, x):
            return x

    return DummyModel()


# Global model manager instance
model_manager = FlexibleModelManager()


def get_model_manager() -> FlexibleModelManager:
    """
    Get the global model manager instance.

    Returns:
        FlexibleModelManager instance
    """
    return model_manager