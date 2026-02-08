from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    Module = torch.nn.Module
else:
    Module = Any

class PluginType(Enum):
    """Enumeration of plugin types."""
    MODEL_COMPONENT = "model_component"
    OPTIMIZATION = "optimization"
    SCHEDULING = "scheduling"
    HARDWARE = "hardware"
    UTILITY = "utility"

class PluginMetadata:
    """Metadata for a plugin."""
    def __init__(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        plugin_type: PluginType,
        dependencies: Optional[List[str]] = None,
        compatibility: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        **kwargs
    ):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.plugin_type = plugin_type
        self.dependencies = dependencies or []
        self.compatibility = compatibility or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

        # Store extra metadata
        for key, value in kwargs.items():
            setattr(self, key, value)

class BasePluginInterface(ABC):
    """
    Base interface for all plugins in the Inference-PIO system.
    """

    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.is_active = False
        self.is_loaded = False
        self.config = {}

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the plugin."""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up resources."""
        pass

class ModelPluginInterface(BasePluginInterface):
    """
    Interface for model plugins.
    """

    @abstractmethod
    def load_model(self, config: Any = None) -> Module:
        """Load the model."""
        pass

    @abstractmethod
    def infer(self, data: Any) -> Any:
        """Perform inference."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass

    @abstractmethod
    def get_model_config_template(self) -> Any:
        """Get model configuration template."""
        pass

    @abstractmethod
    def validate_model_compatibility(self, config: Any) -> bool:
        """Validate model compatibility."""
        pass

class TextModelPluginInterface(ModelPluginInterface):
    """
    Interface for text-based model plugins.
    """

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text."""
        pass

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> Any:
        """Tokenize text."""
        pass

    @abstractmethod
    def detokenize(self, token_ids: Any, **kwargs) -> str:
        """Detokenize text."""
        pass
