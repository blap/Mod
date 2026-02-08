"""
Custom Model Loader - Dependency Free (except C-Engine)
Replacing transformers.AutoModel.from_pretrained with efficient direct loading using C Engine.
"""

import json
import logging
import os
import importlib
from typing import Dict, List, Optional, Any

from ...core.engine.backend import load_safetensors

logger = logging.getLogger(__name__)

class CustomModelLoader:
    """
    Efficiently loads model weights from disk (safetensors or bin) directly into C Engine tensors.
    """

    def load_model(self, model_path: str, device: str = "cpu", dtype: str = "float32", **kwargs):
        """
        Load a full model from a path (config + weights).
        """
        logger.info(f"Loading model from {model_path}...")

        # 1. Load Config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            config_dict = {}
        else:
            with open(config_path, "r") as f:
                config_dict = json.load(f)

        # 2. Determine Architecture
        architectures = config_dict.get("architectures", [])
        model_type = config_dict.get("model_type", "")

        model_class = self._get_model_class(architectures, model_type)
        if not model_class:
            try:
                module = importlib.import_module("src.inference_pio.models.qwen3_0_6b.architecture")
                model_class = getattr(module, "Qwen3ForCausalLM")
            except:
                raise ValueError("Unsupported model architecture")

        # 3. Instantiate Model
        from argparse import Namespace
        config_obj = Namespace(**config_dict)

        try:
            model = model_class(config_obj)
        except Exception as e:
            try:
                model = model_class(**config_dict)
            except Exception as e2:
                logger.error(f"Failed to instantiate model: {e}, {e2}")
                raise e

        # 4. Load Weights using C Loader
        self.load_weights(model, model_path)

        return model

    def _get_model_class(self, architectures: List[str], model_type: str) -> Optional[Any]:
        """Resolve model class based on architecture name."""
        arch = architectures[0] if architectures else ""
        if "Qwen2" in arch or "Qwen2" in model_type or "qwen" in model_type:
            try:
                module = importlib.import_module("src.inference_pio.models.qwen3_0_6b.architecture")
                return getattr(module, "Qwen3ForCausalLM")
            except ImportError:
                pass
        return None

    @staticmethod
    def load_weights(model, model_path: str, device: str = "cpu", strict: bool = False):
        """
        Load weights from safetensors using C loader.
        """
        logger.info(f"Loading weights from {model_path} (C-Engine)...")

        # 1. Collect all model tensors into a map {name: Tensor}
        state_dict_map = {}
        # We need to traverse the model to find parameters
        # Assuming model has state_dict-like method or we implement a traversal
        # Since our Module implements state_dict(), we use it.

        if hasattr(model, "state_dict"):
            state_dict_map = model.state_dict()
        else:
            logger.warning("Model does not support state_dict(). Cannot load weights.")
            return

        # 2. Check for safetensors files
        files_to_load = []

        # Index file
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            unique_files = set(weight_map.values())
            for fname in unique_files:
                files_to_load.append(os.path.join(model_path, fname))
        else:
            # Single file
            single_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single_file):
                files_to_load.append(single_file)

        if not files_to_load:
            logger.warning("No .safetensors files found.")
            return

        # 3. Load each file
        for filepath in files_to_load:
            success = load_safetensors(filepath, state_dict_map)
            if success:
                logger.info(f"Loaded {filepath}")
            else:
                logger.warning(f"Failed to load {filepath}")
