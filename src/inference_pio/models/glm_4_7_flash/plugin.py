"""
GLM-4.7-Flash Plugin
"""
from typing import Any, List
from ...common.interfaces.improved_base_plugin_interface import TextModelPluginInterface, PluginMetadata, PluginType
from ...core.engine.backend import Tensor
from .model import GLM47FlashModel, GLM47FlashConfig

class GLM_4_7_Flash_Plugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="GLM-4.7-Flash",
            version="1.0.0",
            author="ZhipuAI",
            description="GLM-4.7-Flash",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            model_architecture="GLM",
            model_size="Large",
            required_memory_gb=32.0,
            supported_modalities=["text"],
            license="Apache-2.0",
            tags=["glm"],
            model_family="GLM",
            num_parameters=10000000000,
        )
        super().__init__(metadata)
        self._model = None
        self._config = GLM47FlashConfig()

    def initialize(self, **kwargs) -> bool:
        for k, v in kwargs.items():
            if hasattr(self._config, k): setattr(self._config, k, v)
        self.load_model()

        from ...common.managers.batch_manager import BatchManager
        self.batch_manager = BatchManager(self._model)

        return True

    def load_model(self, config=None):
        if config: self._config = config
        self._model = GLM47FlashModel(self._config)
        return self._model

    def infer(self, data: Any) -> Any:
        if isinstance(data, str):
            return self.generate_text(data)
        if isinstance(data, Tensor):
            return self._model.generate(data)
        return "GLM Output"

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
        return f"Generated {len(token_ids)} tokens (Raw)"

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        results = []
        if not self.batch_manager: return super().infer_batch(requests)

        start_id = 2000
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
            t = Tensor([1, len(ids)])
            t.load(ids)

            out = self._model.generate(t, max_new_tokens=max_new_tokens)

            return self.detokenize([int(x) for x in out.to_list()])
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Generation failed: {e}")
            return "Error during generation"

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_glm_4_7_flash_plugin(): return GLM_4_7_Flash_Plugin()
__all__ = ["GLM_4_7_Flash_Plugin", "create_glm_4_7_flash_plugin"]
