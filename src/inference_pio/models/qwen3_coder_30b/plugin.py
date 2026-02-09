"""
Qwen3-Coder-30B Plugin - Self-Contained Implementation
"""

import logging
from typing import Any, Dict, List

from ...common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface,
    PluginMetadata,
    PluginType,
    TextModelPluginInterface,
)
from .config import Qwen3Coder30BConfig
from .model import Qwen3Coder30BModel

logger = logging.getLogger(__name__)

class Qwen3_Coder_30B_Plugin(TextModelPluginInterface):
    """
    Qwen3-Coder-30B model plugin.
    """

    def __init__(self):
        metadata = PluginMetadata(
            name="Qwen3-Coder-30B",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-Coder-30B specialized model",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[], # Removed torch
            compatibility={
                "python_version": ">=3.8",
                "min_memory_gb": 64.0,
            },
            model_architecture="Qwen3-Coder-30B",
            model_size="30B",
            required_memory_gb=64.0,
            supported_modalities=["text"],
            license="MIT",
            tags=["language-model", "code-generation", "30b", "qwen3"],
            model_family="Qwen3",
            num_parameters=30000000000,
        )
        super().__init__(metadata)
        self._model = None
        self._config = Qwen3Coder30BConfig()

    def load_model(self, config: Qwen3Coder30BConfig = None):
        if config: self._config = config
        logger.info(f"Loading Qwen3-Coder-30B model")
        self._model = Qwen3Coder30BModel(self._config)
        return self._model

    def infer(self, data: Any) -> Any:
        from ...core.engine.backend import Tensor

        # Determine input type
        input_ids = None
        tokenizer = self._model.get_tokenizer() if self._model else None

        if isinstance(data, str) and tokenizer:
            # Tokenize
            ids = tokenizer.encode(data)
            input_ids = Tensor([1, len(ids)])
            # Assuming backend uses floats for now
            input_ids.load([float(i) for i in ids])
        elif isinstance(data, list):
            # Raw tokens
            input_ids = Tensor([1, len(data)])
            input_ids.load([float(i) for i in data])

        if input_ids:
            output_tensor = self._model.generate(input_ids)
            output_ids = [int(x) for x in output_tensor.to_list()]

            if isinstance(data, str) and tokenizer:
                return tokenizer.decode(output_ids)
            return output_ids

        return "Error: Invalid input or tokenizer missing"

def create_qwen3_coder_30b_plugin() -> Qwen3_Coder_30B_Plugin:
    return Qwen3_Coder_30B_Plugin()

__all__ = ["Qwen3_Coder_30B_Plugin", "create_qwen3_coder_30b_plugin"]
