"""
Custom Model Loader - Dependency Free (except safetensors)
Replacing transformers.AutoModel.from_pretrained with efficient direct loading using Numpy backend.
"""

import json
import logging
import os
import importlib
from typing import Dict, List, Optional, Any

import numpy as np

# Try to import safetensors for numpy
try:
    from safetensors.numpy import load_file as safe_load_file
except ImportError:
    safe_load_file = None

# Torch fallback (optional, for partial migration or if safetensors fails)
try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class CustomModelLoader:
    """
    Efficiently loads model weights from disk (safetensors or bin) directly into Numpy models.
    Supports sharded checkpoints and automatic architecture selection.
    """

    def load_model(self, model_path: str, device: str = "cpu", dtype: str = "float32", **kwargs):
        """
        Load a full model from a path (config + weights).
        """
        logger.info(f"Loading model from {model_path}...")

        # 1. Load Config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # 2. Determine Architecture
        architectures = config_dict.get("architectures", [])
        model_type = config_dict.get("model_type", "")

        model_class = self._get_model_class(architectures, model_type)
        if not model_class:
            logger.warning(f"Could not determine model class for architectures: {architectures}, type: {model_type}.")
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

        # 4. Load Weights
        self.load_weights(model, model_path, device=device)

        return model

    def _get_model_class(self, architectures: List[str], model_type: str) -> Optional[Any]:
        """Resolve model class based on architecture name."""
        arch = architectures[0] if architectures else ""

        # Qwen mapping
        if "Qwen2" in arch or "Qwen2" in model_type or "qwen" in model_type:
            try:
                module = importlib.import_module("src.inference_pio.models.qwen3_0_6b.architecture")
                return getattr(module, "Qwen3ForCausalLM")
            except ImportError:
                pass

        # GLM mapping
        if "GLM" in arch or "ChatGLM" in arch or "glm" in model_type:
            try:
                module = importlib.import_module("src.inference_pio.models.glm_4_7_flash.architecture")
                return getattr(module, "GLMForCausalLM")
            except ImportError:
                pass

        return None

    @staticmethod
    def load_weights(model, model_path: str, device: str = "cpu", strict: bool = False):
        """
        Load weights from model_path into model (Numpy backend).
        """
        logger.info(f"Loading weights from {model_path}...")

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            index_file = os.path.join(model_path, "pytorch_model.bin.index.json")

        if os.path.exists(index_file):
            CustomModelLoader._load_sharded_weights(model, model_path, index_file, device, strict)
        else:
            single_file = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(single_file):
                single_file = os.path.join(model_path, "pytorch_model.bin")

            if os.path.exists(single_file):
                CustomModelLoader._load_single_file(model, single_file, device, strict)
            else:
                logger.warning(f"No weights found in {model_path}. Model initialized with random weights.")

    @staticmethod
    def _load_single_file(model, filepath: str, device: str, strict: bool):
        logger.info(f"Loading single weight file: {filepath}")
        state_dict = {}

        if filepath.endswith(".safetensors"):
            if safe_load_file:
                state_dict = safe_load_file(filepath)
            else:
                logger.error("safetensors.numpy not available. Cannot load safetensors.")
                return
        else:
            # Fallback to torch.load if available, then convert to numpy
            if torch:
                torch_state_dict = torch.load(filepath, map_location="cpu")
                for k, v in torch_state_dict.items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v.detach().cpu().numpy()
                    else:
                        state_dict[k] = v
            else:
                 logger.error("PyTorch not installed. Cannot load .bin files.")
                 return

        if hasattr(model, "load_state_dict"):
             model.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _load_sharded_weights(model, model_path: str, index_file: str, device: str, strict: bool):
        logger.info(f"Loading sharded weights using index: {index_file}")
        with open(index_file, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        unique_files = set(weight_map.values())

        full_state_dict = {}

        for filename in unique_files:
            filepath = os.path.join(model_path, filename)
            logger.info(f"Loading shard: {filename}")

            if filename.endswith(".safetensors"):
                if safe_load_file:
                    shard_state = safe_load_file(filepath)
                    full_state_dict.update(shard_state)
            else:
                if torch:
                    shard_state_torch = torch.load(filepath, map_location="cpu")
                    for k, v in shard_state_torch.items():
                         full_state_dict[k] = v.detach().cpu().numpy()

        if hasattr(model, "load_state_dict"):
             model.load_state_dict(full_state_dict, strict=strict)
