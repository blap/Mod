"""
Qwen3-0.6B Plugin Implementation
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface,
    PluginMetadata,
    PluginType,
    TextModelPluginInterface,
)
from .config import Qwen3_0_6B_Config, Qwen3_0_6B_DynamicConfig
from .model import Qwen3_0_6B_Model

logger = logging.getLogger(__name__)

class Qwen3_0_6B_Plugin(TextModelPluginInterface):
    """
    Qwen3-0.6B Plugin - Backend Agnostic (C-Engine)
    """

    def __init__(self):
        # Create plugin metadata
        metadata = PluginMetadata(
            name="Qwen3-0.6B",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-0.6B small language model",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["safetensors"],
            compatibility={
                "python_version": ">=3.8",
                "min_memory_gb": 2.0,
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Qwen3 Transformer",
            model_size="0.6B",
            required_memory_gb=2.0,
            supported_modalities=["text"],
            license="MIT",
            tags=["language-model", "qwen", "0.6b"],
            model_family="Qwen",
            num_parameters=600000000,
        )
        super().__init__(metadata)
        self._model = None
        self._config = Qwen3_0_6B_Config()

    def initialize(self, **kwargs) -> bool:
        try:
            # Update config
            for k, v in kwargs.items():
                if hasattr(self._config, k):
                    setattr(self._config, k, v)

            logger.info("Initializing Qwen3-0.6B Plugin")
            self.load_model()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def load_model(self, config=None):
        if config:
            self._config = config

        self._model = Qwen3_0_6B_Model(self._config)
        return self._model

    def infer(self, data: Any) -> Any:
        if isinstance(data, str):
            return self.generate_text(data)
        return ""

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._model:
            self.load_model()

        # Tokenize
        if not self._model._tokenizer:
             # Basic fallback if tokenizer missing
             logger.warning("Tokenizer missing")
             return ""

        # Encode
        inputs = self._model._tokenizer.encode(prompt)

        # Pass to C Engine (requires simple list or pointer wrapper)
        from ...core.engine.backend import Tensor
        # Create tensor from list
        # Assuming batch size 1
        data = [float(x) for x in inputs]
        input_ids = Tensor([1, len(inputs)], data)

        # Generate
        output_ids_tensor = self._model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

        # Decode (Tensor -> List)
        output_ids_list = [int(x) for x in output_ids_tensor.to_list()]

        output_text = self._model._tokenizer.decode(output_ids_list)
        return output_text

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": "Qwen3-0.6B"}

    def get_model_parameters(self) -> Dict[str, Any]:
        return {"total_parameters": 600000000}

    def get_model_config_template(self) -> Any:
        return Qwen3_0_6B_Config()

    def validate_model_compatibility(self, config: Any) -> bool:
        return isinstance(config, Qwen3_0_6B_Config)

    # Implement required abstract methods
    def tokenize(self, text: str, **kwargs) -> Any:
        if self._model and self._model._tokenizer:
            return self._model._tokenizer.encode(text)
        return []

    def detokenize(self, token_ids: Any, **kwargs) -> str:
        if self._model and self._model._tokenizer:
            return self._model._tokenizer.decode(token_ids)
        return ""

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_qwen3_0_6b_plugin():
    return Qwen3_0_6B_Plugin()
