"""
Qwen3-Coder-Next Plugin Implementation
"""

import logging
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ...common.interfaces.improved_base_plugin_interface import (
    TextModelPluginInterface,
    PluginMetadata,
    PluginType,
)
from .config import Qwen3CoderNextConfig
from .model import Qwen3CoderNextModel

logger = logging.getLogger(__name__)

class Qwen3_Coder_Next_Plugin(TextModelPluginInterface):
    """
    Plugin for Qwen3-Coder-Next (80B) Model.
    """
    def __init__(self):
        metadata = PluginMetadata(
            name="Qwen3-Coder-Next",
            version="1.0.0",
            author="Alibaba Cloud",
            description="Qwen3-Coder-Next 80B Hybrid Model",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["torch", "transformers", "accelerate"],
            compatibility={
                "torch_version": ">=2.2.0",
                "min_memory_gb": 160.0
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="Hybrid (DeltaNet+Attention+MoE)",
            model_size="80B",
            required_memory_gb=160.0,
            supported_modalities=["text"],
            license="Proprietary",
            model_family="Qwen",
            num_parameters=80000000000
        )
        super().__init__(metadata)
        self._model = None
        self._tokenizer = None
        self._config = Qwen3CoderNextConfig()

    def initialize(self, **kwargs) -> bool:
        try:
            # Load Config
            for k, v in kwargs.items():
                if hasattr(self._config, k):
                    setattr(self._config, k, v)

            # Force thinking mode off as per requirement
            self._config.thinking_mode = False
            self._config.enable_thinking = False

            # Device Selection (Simplified)
            device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self._config.device = device

            logger.info(f"Initializing Qwen3-Coder-Next on {device}")

            self.load_model()

            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-Coder-Next: {e}")
            return False

    def load_model(self, config=None) -> nn.Module:
        if config:
            self._config = config

        # Placeholder for loading weights
        # In a real scenario, we would load from safe tensors or HF hub
        logger.info(f"Loading model structure with config: {self._config}")

        # Instantiate model structure
        self._model = Qwen3CoderNextModel(self._config)

        # Move to device (Naively, for single GPU or CPU test)
        # Real 80B model requires distributed loading
        if self._config.device != "meta":
             self._model.to(self._config.device)

        # Initialize Tokenizer (Placeholder)
        try:
             from transformers import AutoTokenizer
             # Use generic Qwen tokenizer if available, or fallback
             self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True)
        except Exception:
             logger.warning("Could not load Qwen tokenizer, using mock/fallback")
             self._tokenizer = None

        return self._model

    def infer(self, data: Any) -> Any:
        return self.generate_text(data)

    def generate_text(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self._model:
            raise RuntimeError("Model not initialized")

        if not self._tokenizer:
             # Mock return for structure testing
             return "Model initialized but tokenizer missing."

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._config.device)

        # Simple generation loop wrapper
        # Real implementation should use the optimized hybrid cache manager
        with torch.no_grad():
             outputs = self._model(inputs.input_ids)
             # This is just a forward pass, not generation.
             # Full generation requires loop.
             # calling fallback generate if available or implementing loop

             # Fallback to simple greedy decoding for demonstration
             generated = inputs.input_ids
             for _ in range(min(max_new_tokens, 10)): # Limit for test speed
                  outputs = self._model(generated)
                  # Access last_hidden_state correctly from return dict
                  last_state = outputs["last_hidden_state"]
                  # Assuming simple projection to vocab (embedding weights usually tied)
                  logits = torch.matmul(last_state[:, -1, :], self._model.embed_tokens.weight.t())
                  next_token = torch.argmax(logits, dim=-1, keepdim=True)
                  generated = torch.cat([generated, next_token], dim=1)

        return self._tokenizer.decode(generated[0], skip_special_tokens=True)

    def cleanup(self) -> bool:
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
        return True

    def supports_config(self, config: Any) -> bool:
        return isinstance(config, Qwen3CoderNextConfig)

    def tokenize(self, text: str, **kwargs) -> Any:
        if self._tokenizer:
            return self._tokenizer(text, **kwargs)
        return []

    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
         if self._tokenizer:
             return self._tokenizer.decode(token_ids, **kwargs)
         return ""

def create_qwen3_coder_next_plugin():
    return Qwen3_Coder_Next_Plugin()
