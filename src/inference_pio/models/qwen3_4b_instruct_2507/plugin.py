"""
Qwen3-4B-Instruct-2507 Plugin Implementation
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
from .config import Qwen3_4B_Instruct_2507_Config, Qwen3_4B_Instruct_2507_DynamicConfig
from .model import Qwen3_4B_Instruct_2507_Model

logger = logging.getLogger(__name__)

class Qwen3_4B_Instruct_2507_Plugin(TextModelPluginInterface):
    """
    Qwen3-4B-Instruct-2507 Plugin - Backend Agnostic (C-Engine)
    """

    def __init__(self):
        # Create plugin metadata
        metadata = PluginMetadata(
            name="Qwen3-4B-Instruct-2507",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-4B-Instruct-2507 language model",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["safetensors"],
            compatibility={
                "python_version": ">=3.8",
                "min_memory_gb": 8.0,
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Qwen3 Transformer",
            model_size="4B",
            required_memory_gb=8.0,
            supported_modalities=["text"],
            license="MIT",
            tags=["language-model", "qwen", "4b", "instruct"],
            model_family="Qwen",
            num_parameters=4000000000,
        )
        super().__init__(metadata)
        self._model = None
        self._config = Qwen3_4B_Instruct_2507_Config()

    def initialize(self, **kwargs) -> bool:
        try:
            # Update config
            for k, v in kwargs.items():
                if hasattr(self._config, k):
                    setattr(self._config, k, v)

            logger.info("Initializing Qwen3-4B-Instruct-2507 Plugin")
            self.load_model()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def load_model(self, config=None):
        if config:
            self._config = config

        self._model = Qwen3_4B_Instruct_2507_Model(self._config)
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
             return "Error: Tokenizer not available."

        # Encode
        try:
            inputs = self._model._tokenizer.encode(prompt)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return "Error during tokenization."

        # Pass to C Engine (requires simple list or pointer wrapper)
        from ...core.engine.backend import Tensor
        # Create tensor from list
        # Assuming batch size 1
        try:
            data = [float(x) for x in inputs]
            input_ids = Tensor([1, len(inputs)], data)

            # Generate
            output_ids_tensor = self._model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

            # Decode (Tensor -> List)
            output_ids_list = [int(x) for x in output_ids_tensor.to_list()]

            output_text = self._model._tokenizer.decode(output_ids_list)
            return output_text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Error during text generation."

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Qwen3-4B-Instruct-2507",
            "architecture": "Qwen3 Transformer",
            "parameters": 4000000000,
            "hidden_size": self._config.hidden_size,
            "num_attention_heads": self._config.num_attention_heads,
            "num_hidden_layers": self._config.num_hidden_layers,
        }

    def get_model_parameters(self) -> Dict[str, Any]:
        return {"total_parameters": 4000000000}

    def get_model_config_template(self) -> Any:
        return Qwen3_4B_Instruct_2507_Config()

    def validate_model_compatibility(self, config: Any) -> bool:
        return isinstance(config, Qwen3_4B_Instruct_2507_Config)

    # Implement required abstract methods
    def tokenize(self, text: str, **kwargs) -> Any:
        if self._model and self._model._tokenizer:
            try:
                return self._model._tokenizer.encode(text)
            except Exception:
                return []
        return []

    def detokenize(self, token_ids: Any, **kwargs) -> str:
        if self._model and self._model._tokenizer:
            try:
                return self._model._tokenizer.decode(token_ids)
            except Exception:
                return ""
        return ""

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_qwen3_4b_instruct_2507_plugin():
    return Qwen3_4B_Instruct_2507_Plugin()