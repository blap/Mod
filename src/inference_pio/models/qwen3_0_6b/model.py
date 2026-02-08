"""
Qwen3-0.6B Model Implementation
"""

import logging
from typing import Any, Dict, List, Optional, Union

# Use Core Engine Layers instead of torch.nn
from ...core.engine.layers import Module
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer

from .config import Qwen3_0_6B_Config
from .architecture import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

class Qwen3_0_6B_Model(Module):
    def __init__(self, config: Qwen3_0_6B_Config):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None

        self._initialize_model()

    def _initialize_model(self):
        logger.info("Initializing Qwen3-0.6B model...")

        # Initialize Architecture
        self._model = Qwen3ForCausalLM(self.config)

        # Load Weights
        # Assuming model_path is in config
        model_path = getattr(self.config, "model_path", "H:/Qwen3-0.6B")

        # Use CustomModelLoader with numpy support
        try:
            CustomModelLoader.load_weights(self._model, model_path, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")

        # Load Tokenizer
        try:
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

def create_qwen3_0_6b_model(config: Qwen3_0_6B_Config) -> Qwen3_0_6B_Model:
    return Qwen3_0_6B_Model(config)
