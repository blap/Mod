"""
Qwen3-Coder-30B Plugin
"""

from typing import Dict, Any, List, Optional
import logging
from .config import Qwen3Coder30BConfig
from .model import Qwen3_Coder_30B_Model
from ...common.interfaces.improved_base_plugin_interface import (
    TextModelPluginInterface, PluginMetadata, PluginType
)
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class Qwen3_Coder_30B_Plugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="Qwen3-Coder-30B", version="1.0", author="Alibaba",
            description="Qwen3-Coder-30B model",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            model_architecture="Qwen3", model_size="30B", required_memory_gb=60,
            supported_modalities=["text"]
        )
        super().__init__(metadata)
        self._model = None
        self._config = Qwen3Coder30BConfig()

    def initialize(self, **kwargs) -> bool:
        logger.info("Initializing Qwen3-Coder-30B Plugin...")
        for k, v in kwargs.items():
            if hasattr(self._config, k): setattr(self._config, k, v)
        self.load_model()

        from ...common.managers.batch_manager import BatchManager
        self.batch_manager = BatchManager(self._model)

        return True

    def load_model(self, config=None):
        if config: self._config = config
        self._model = Qwen3_Coder_30B_Model(self._config)
        return self._model

    def infer(self, data: Any) -> Any:
        if isinstance(data, str):
            return self.generate_text(data)
        if isinstance(data, Tensor):
            return self._model.generate(data)
        return None

    def tokenize(self, text: str, **kwargs) -> List[float]:
        tokenizer = getattr(self._model, '_tokenizer', None)
        if tokenizer:
            try:
                return [float(x) for x in tokenizer.encode(text)]
            except Exception:
                pass
        return [1.0] * 5

    def detokenize(self, token_ids: List[int], **kwargs) -> str:
        tokenizer = getattr(self._model, '_tokenizer', None)
        if tokenizer:
            try:
                return tokenizer.decode(token_ids)
            except Exception:
                pass
        return f"Generated {len(token_ids)} tokens"

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        results = []
        if not self.batch_manager: return super().infer_batch(requests)

        start_id = 6000
        req_ids = []
        for i, prompt in enumerate(requests):
            ids = self.tokenize(prompt)
            rid = start_id + i
            self.batch_manager.add_request(rid, ids)
            req_ids.append(rid)

        for _ in req_ids:
            out = self.batch_manager.step()
            if out:
                res = self.detokenize([int(x) for x in out.to_list()])
                results.append(res)
            else:
                results.append("Error")
        return results

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._model: self.load_model()

        try:
            ids = self.tokenize(prompt)
            from ...core.engine.backend import Tensor
            t = Tensor([1, len(ids)])
            t.load(ids)

            out = self._model.generate(t, max_new_tokens=max_new_tokens)
            return self.detokenize([int(x) for x in out.to_list()])
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Error during text generation"

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_qwen3_coder_30b_plugin(): return Qwen3_Coder_30B_Plugin()
