"""
Qwen3-4B-Instruct-2507 Model Implementation
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ...core.engine.backend import Module
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer

from .config import Qwen3_4B_Instruct_2507_Config
from .architecture import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

class Qwen3_4B_Instruct_2507_Model(Module):
    def __init__(self, config: Qwen3_4B_Instruct_2507_Config):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None

        self._initialize_model()

    def _initialize_model(self):
        logger.info("Initializing Qwen3-4B-Instruct-2507 model...")
        self._model = Qwen3ForCausalLM(self.config)
        model_path = getattr(self.config, "model_path", "H:/Qwen3-4B-Instruct-2507")

        try:
            # Attempt to load weights using the custom loader
            CustomModelLoader.load_weights(self._model, model_path, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Model will use random initialization.")

        try:
            # Attempt to load tokenizer
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Text processing will be limited.")

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

def create_qwen3_4b_instruct_2507_model(config: Qwen3_4B_Instruct_2507_Config) -> Qwen3_4B_Instruct_2507_Model:
    return Qwen3_4B_Instruct_2507_Model(config)