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
from .config import Qwen3_0_6bConfig as Qwen3_0_6B_Config, Qwen3_0_6B_DynamicConfig
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

            # Standardize Tokenizer Loading
            from ...common.custom_components.tokenizer import load_custom_tokenizer
            self.tokenizer = load_custom_tokenizer(getattr(self._config, 'model_path', None))

            self.load_model()

            from ...common.managers.batch_manager import BatchManager
            self.batch_manager = BatchManager(self._model)

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

    def tokenize(self, text: str, **kwargs) -> List[float]:
        tokenizer = getattr(self, 'tokenizer', None)
        if not tokenizer and self._model:
            tokenizer = getattr(self._model, 'tokenizer', getattr(self._model, '_tokenizer', None))

        if tokenizer:
            try:
                if hasattr(tokenizer, 'encode'):
                    return [float(x) for x in tokenizer.encode(text)]
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
        return [1.0] * 5

    def detokenize(self, token_ids: List[int], **kwargs) -> str:
        tokenizer = getattr(self, 'tokenizer', None)
        if not tokenizer and self._model:
            tokenizer = getattr(self._model, 'tokenizer', getattr(self._model, '_tokenizer', None))

        if tokenizer:
            try:
                if hasattr(tokenizer, 'decode'):
                    return tokenizer.decode(token_ids)
            except Exception as e:
                logger.warning(f"Detokenization error: {e}")
        return f"Generated {len(token_ids)} tokens"

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        results = []
        if not self.batch_manager: return super().infer_batch(requests)

        start_id = 5000
        req_ids = []
        for i, prompt in enumerate(requests):
            ids = self.tokenize(prompt)
            rid = start_id + i
            self.batch_manager.add_request(rid, ids)
            req_ids.append(rid)

        for _ in req_ids:
            out_tensor = self.batch_manager.step()
            if out_tensor:
                try:
                    res = self.detokenize([int(x) for x in out_tensor.to_list()])
                    results.append(res)
                except Exception as e:
                    logger.error(f"Batch decoding error: {e}")
                    results.append("Error decoding")
            else:
                results.append("Error in batch processing")
        return results

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._model: self.load_model()

        try:
            ids = self.tokenize(prompt)
            from ...core.engine.backend import Tensor
            t = Tensor([1, len(ids)])
            t.load(ids)

            out = self._model.generate(t, max_new_tokens=max_new_tokens, **kwargs)
            return self.detokenize([int(x) for x in out.to_list()])
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Error during text generation"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Qwen3-0.6B",
            "architecture": "Qwen3 Transformer",
            "parameters": 600000000,
            "hidden_size": self._config.hidden_size,
            "num_attention_heads": self._config.num_attention_heads,
            "num_hidden_layers": self._config.num_hidden_layers,
        }

    def get_model_parameters(self) -> Dict[str, Any]:
        return {"total_parameters": 600000000}

    def get_model_config_template(self) -> Any:
        return Qwen3_0_6B_Config()

    def validate_model_compatibility(self, config: Any) -> bool:
        return isinstance(config, Qwen3_0_6B_Config)

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_qwen3_0_6b_plugin():
    return Qwen3_0_6B_Plugin()