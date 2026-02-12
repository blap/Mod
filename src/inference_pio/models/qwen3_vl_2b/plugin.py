"""
Qwen3-VL-2B Plugin
"""
from typing import Any, List
from ...common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface, PluginMetadata, PluginType, TextModelPluginInterface
)
from ...core.engine.backend import Tensor
from .model import Qwen3VL2BModel, Qwen3VL2BConfig

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

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        results = []
        if not self.batch_manager: return super().infer_batch(requests)

        start_id = 3000
        req_ids = []
        for i, prompt in enumerate(requests):
            # VL Model only supports text batching here via this interface
            # Image batching would require tuple inputs in requests list
            tokenizer = getattr(self._model, '_tokenizer', None)
            if tokenizer: ids = tokenizer.encode(prompt)
            else: ids = [1.0]*5

            rid = start_id + i
            self.batch_manager.add_request(rid, [float(x) for x in ids])
            req_ids.append(rid)

        for _ in req_ids:
            out = self.batch_manager.step()
            if out:
                tokenizer = getattr(self._model, '_tokenizer', None)
                if tokenizer: res = tokenizer.decode(out.to_list())
                else: res = f"Generated {out.shape[1]} tokens"
                results.append(res)
            else:
                results.append("Error")
        return results

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        # Simplified text-only generation wrapper
        if not self._model: self.load_model()

        try:
            # Tokenize (Mock if missing, but support real flow)
            if hasattr(self._model, '_tokenizer') and self._model._tokenizer:
                ids = self._model._tokenizer.encode(prompt)
            else:
                ids = [1.0] * 5

            t = Tensor([1, len(ids)])
            t.load([float(x) for x in ids])

            out = self._model.generate(t, max_new_tokens=max_new_tokens)

            if hasattr(self._model, '_tokenizer') and self._model._tokenizer:
                return self._model._tokenizer.decode(out.to_list())

            return f"Generated {out.shape[1]} tokens"
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"VL Generation failed: {e}")
            return "Error during generation"

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_qwen3_vl_2b_plugin(): return Qwen3_VL_2B_Plugin()
__all__ = ["Qwen3_VL_2B_Plugin", "create_qwen3_vl_2b_plugin"]
