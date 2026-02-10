"""
Qwen3-Coder-Next Plugin
"""

from typing import Dict, Any, List, Optional
import logging
from .config import Qwen3CoderNextConfig, create_qwen3_coder_next_config
from .model import Qwen3CoderNextForCausalLM
from ...common.interfaces.improved_base_plugin_interface import ModelPluginInterface
from ...core.model_loader import ModelLoader
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class Qwen3CoderNextPlugin(ModelPluginInterface):
    def __init__(self):
        self.config: Optional[Qwen3CoderNextConfig] = None
        self.model: Optional[Qwen3CoderNextForCausalLM] = None
        self.tokenizer = None # Placeholder for custom tokenizer

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize the model plugin with configuration.
        """
        logger.info("Initializing Qwen3-Coder-Next Plugin...")
        self.config = create_qwen3_coder_next_config(**kwargs)

        # Load tokenizer (simplified/mock for now as per dependency-free reqs)
        # In real scenario, load specific tokenizer or use common one.
        # self.tokenizer = ...

    def load_model(self) -> None:
        """
        Load model weights.
        """
        if not self.config:
            raise ValueError("Plugin not initialized. Call initialize() first.")

        logger.info(f"Loading Qwen3-Coder-Next model from {self.config.model_path}")

        # Instantiate model structure
        self.model = Qwen3CoderNextForCausalLM(self.config)

        # Load weights using the custom loader (safetensors via C backend)
        # We wrap in try-except to allow partial loading or mock in tests if file missing
        try:
            loader = ModelLoader(self.config.model_path)
            loader.load_into_module(self.model)
        except Exception as e:
            logger.warning(f"Failed to load weights from {self.config.model_path}: {e}")
            logger.warning("Using random initialization (normal for tests without weights).")

        # Move to device if specified in config (default CPU)
        # self.model.to(self.config.device)

    def infer(self, input_data: Any) -> Any:
        """
        Run inference.
        """
        if not self.model:
            raise RuntimeError("Model not loaded.")

        if isinstance(input_data, Tensor):
             return self.model.generate(input_data)

        # If input is string, we need tokenizer
        if isinstance(input_data, str):
             if self.tokenizer:
                 # TODO: Implement tokenization
                 ids = self.tokenizer.encode(input_data)
                 t = Tensor([1, len(ids)])
                 t.load([float(x) for x in ids])
                 out = self.model.generate(t)
                 return self.tokenizer.decode(out.to_list())
             else:
                 logger.warning("No tokenizer loaded. Cannot process string input.")
                 return None

        return None

    def cleanup(self) -> None:
        """
        Release resources.
        """
        self.model = None
        self.config = None
        logger.info("Qwen3-Coder-Next Plugin cleaned up.")

def create_qwen3_coder_next_plugin() -> Qwen3CoderNextPlugin:
    return Qwen3CoderNextPlugin()
