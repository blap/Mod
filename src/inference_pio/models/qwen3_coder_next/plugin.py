"""
Qwen3-Coder-Next Plugin
"""

from typing import Dict, Any, List, Optional
import logging
from .config import Qwen3CoderNextConfig, create_qwen3_coder_next_config
from .model import Qwen3CoderNextForCausalLM
from ...common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface,
    TextModelPluginInterface,
    PluginMetadata,
    PluginType
)
from ...core.model_loader import ModelLoader
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class Qwen3CoderNextPlugin(TextModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="Qwen3-Coder-Next",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-Coder-Next Hybrid MoE model",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            model_architecture="Qwen3 Hybrid MoE",
            model_size="Large",
            required_memory_gb=16.0,
            supported_modalities=["text"],
            license="MIT",
            tags=["language-model", "qwen", "coder", "moe"],
            model_family="Qwen",
            num_parameters=30000000000,
        )
        super().__init__(metadata)
        self.config: Optional[Qwen3CoderNextConfig] = None
        self._model: Optional[Qwen3CoderNextForCausalLM] = None
        self.tokenizer = None

    def initialize(self, **kwargs: Any) -> bool:
        logger.info("Initializing Qwen3-Coder-Next Plugin...")
        self.config = create_qwen3_coder_next_config(**kwargs)
        self.load_model()

        # Initialize Batch Manager
        from ...common.managers.batch_manager import BatchManager
        self.batch_manager = BatchManager(self._model)

        return True

    def load_model(self, config=None) -> None:
        if config: self.config = config
        if not self.config:
            raise ValueError("Plugin not initialized.")

        logger.info(f"Loading Qwen3-Coder-Next model from {self.config.model_path}")
        self._model = Qwen3CoderNextForCausalLM(self.config)

        try:
            loader = ModelLoader(self.config.model_path)
            loader.load_into_module(self._model)
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")

    def infer(self, input_data: Any) -> Any:
        if isinstance(input_data, str):
            return self.generate_text(input_data)
        if isinstance(input_data, Tensor):
            return self._model.generate(input_data)
        return None

    def infer_batch(self, requests: List[Any]) -> List[Any]:
        """
        Process batch using Serial Batch Manager.
        """
        results = []
        if not self.batch_manager:
            # Fallback
            return super().infer_batch(requests)

        # Add all to queue
        start_id = 1000
        req_ids = []
        for i, prompt in enumerate(requests):
            # Tokenize first
            if self.tokenizer:
                ids = self.tokenizer.encode(prompt)
            else:
                ids = [1.0] * 5

            rid = start_id + i
            self.batch_manager.add_request(rid, [float(x) for x in ids])
            req_ids.append(rid)

        # Process queue
        # Since step() processes one, we loop
        for _ in req_ids:
            out_tensor = self.batch_manager.step()
            if out_tensor:
                # Detokenize
                if self.tokenizer:
                    res = self.tokenizer.decode(out_tensor.to_list())
                else:
                    res = f"Generated {out_tensor.shape[1]} tokens"
                results.append(res)
            else:
                results.append("Error in batch processing")

        return results

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._model: self.load_model()

        # Standardized generation flow
        # 1. Tokenize (Mock if missing)
        if self.tokenizer:
            ids = self.tokenizer.encode(prompt)
        else:
            # Fallback
            ids = [1.0] * 5 # Dummy
            logger.warning("Tokenizer missing, using dummy input")

        # 2. Tensor
        t = Tensor([1, len(ids)])
        t.load([float(x) for x in ids])

        # 3. Generate
        out = self._model.generate(t, max_new_tokens=max_new_tokens)

        # 4. Decode
        out_list = out.to_list()
        if self.tokenizer:
            return self.tokenizer.decode([int(x) for x in out_list])
        return f"Generated {len(out_list)} tokens (Raw)"

    def cleanup(self) -> bool:
        self._model = None
        self.config = None
        return True

def create_qwen3_coder_next_plugin() -> Qwen3CoderNextPlugin:
    return Qwen3CoderNextPlugin()
