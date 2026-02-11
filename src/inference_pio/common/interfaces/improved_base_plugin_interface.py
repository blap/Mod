from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

class PluginType(Enum):
    MODEL_COMPONENT = "MODEL_COMPONENT"

@dataclass
class PluginMetadata:
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str]
    compatibility: Dict[str, str]
    model_architecture: str = ""
    model_size: str = ""
    required_memory_gb: float = 0.0
    supported_modalities: List[str] = None
    license: str = ""
    tags: List[str] = None
    model_family: str = ""
    num_parameters: int = 0
    test_coverage: float = 0.0
    validation_passed: bool = False
    created_at: Any = None
    updated_at: Any = None

class ModelPluginInterface:
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self._initialized = False
    def initialize(self, **kwargs) -> bool:
        self._initialized = True
        return True
    def cleanup(self): pass

class TextModelPluginInterface(ModelPluginInterface):
    def infer(self, data): raise NotImplementedError
    def generate_text(self, prompt, **kwargs): raise NotImplementedError
    def load_model(self, config=None): raise NotImplementedError

    # Optional Batching Interface
    def infer_batch(self, requests: List[Any]) -> List[Any]:
        """
        Process a batch of requests.
        Default implementation iterates serially.
        Plugins can override this to use BatchManager if configured.
        """
        results = []
        for req in requests:
            results.append(self.generate_text(req))
        return results
