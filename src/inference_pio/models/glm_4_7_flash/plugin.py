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

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        results = []
        if not self.batch_manager: return super().infer_batch(requests)

        start_id = 2000
        req_ids = []
        for i, prompt in enumerate(requests):
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
        if not self._model: self.load_model()

        # Standardized generation flow
        try:
            # 1. Tokenize (Mock if missing, but support real flow)
            # Assuming model has _tokenizer or we use a fallback
            if hasattr(self._model, '_tokenizer') and self._model._tokenizer:
                ids = self._model._tokenizer.encode(prompt)
            else:
                # Fallback / Mock
                ids = [1.0] * 5

            # 2. Tensor
            t = Tensor([1, len(ids)])
            t.load([float(x) for x in ids])

            # 3. Generate
            out = self._model.generate(t, max_new_tokens=max_new_tokens)

            # 4. Decode
            if hasattr(self._model, '_tokenizer') and self._model._tokenizer:
                return self._model._tokenizer.decode(out.to_list())

            return f"Generated {out.shape[1]} tokens (Raw)"
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Generation failed: {e}")
            return "Error during generation"

    def cleanup(self) -> bool:
        self._model = None
        return True

def create_glm_4_7_flash_plugin(): return GLM_4_7_Flash_Plugin()
__all__ = ["GLM_4_7_Flash_Plugin", "create_glm_4_7_flash_plugin"]
