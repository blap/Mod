"""
Qwen3-VL-2B Plugin
"""
import logging
from typing import Any, List
from ...common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface, PluginMetadata, PluginType, TextModelPluginInterface
)
from ...core.engine.backend import Tensor
from .model import Qwen3VL2BModel, Qwen3VL2BConfig

logger = logging.getLogger(__name__)

class Qwen3_VL_2B_Plugin(TextModelPluginInterface):
    def __init__(self):
        meta = PluginMetadata(
            name="Qwen3-VL-2B", version="1.0", author="Alibaba",
            description="Qwen3-VL-2B",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[], compatibility={},
            model_architecture="Qwen3-VL", model_size="2B", required_memory_gb=10,
            supported_modalities=["text", "image"]
        )
        super().__init__(meta)
        self._model = None
        self._config = Qwen3VL2BConfig()

    def initialize(self, **kwargs) -> bool:
        for k, v in kwargs.items():
            if hasattr(self._config, k): setattr(self._config, k, v)

        # Standardize Tokenizer Loading
        from ...common.custom_components.tokenizer import load_custom_tokenizer
        self.tokenizer = load_custom_tokenizer(getattr(self._config, 'model_path', None))

        self.load_model()

        from ...common.managers.batch_manager import BatchManager
        self.batch_manager = BatchManager(self._model)

        return True

    def load_model(self, config=None):
        if config: self._config = config
        self._model = Qwen3VL2BModel(self._config)
        return self._model

    def infer(self, data: Any) -> Any:
        if isinstance(data, str):
            return self.generate_text(data)
        if isinstance(data, tuple) and len(data) == 2:
            return self._model.generate(data[0], pixel_values=data[1])
        if isinstance(data, Tensor):
            return self._model.generate(data)
        return None

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

        start_id = 3000
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
            t = Tensor([1, len(ids)])
            t.load(ids)

            out = self._model.generate(t, max_new_tokens=max_new_tokens)
            return self.detokenize([int(x) for x in out.to_list()])
        except Exception as e:
            logger.error(f"VL Generation failed: {e}")
            return "Error during generation"

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_qwen3_vl_2b_plugin(): return Qwen3_VL_2B_Plugin()
__all__ = ["Qwen3_VL_2B_Plugin", "create_qwen3_vl_2b_plugin"]
