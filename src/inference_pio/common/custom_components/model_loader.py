"""
Custom Model Loader - Dependency Free (except safetensors/torch)
Replacing transformers.AutoModel.from_pretrained with efficient direct loading.
"""

import json
import logging
import os
import re
import importlib
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

logger = logging.getLogger(__name__)

class CustomModelLoader:
    """
    Efficiently loads model weights from disk (safetensors or bin) directly into PyTorch models.
    Supports sharded checkpoints and automatic architecture selection.
    """

    def load_model(self, model_path: str, device: str = "cpu", dtype: torch.dtype = torch.float16, **kwargs) -> nn.Module:
        """
        Load a full model from a path (config + weights).

        Args:
            model_path: Path to the model directory.
            device: Device to load model onto.
            dtype: Data type for the model.

        Returns:
            Instantiated and loaded PyTorch model.
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
            logger.warning(f"Could not determine model class for architectures: {architectures}, type: {model_type}. Using generic fallback if available.")
            # Fallback logic or raise error
            raise ValueError("Unsupported model architecture")

        # 3. Create Configuration Object
        # Assuming the model class takes a config object or dict
        # We try to find a matching config class or use a generic one
        # For this refactor, we pass the dict or try to convert to the specific config object

        # Instantiate model
        # Note: This assumes the model class accepts a config dict or object
        # We might need to wrap config_dict in a namespace object
        from argparse import Namespace
        config_obj = Namespace(**config_dict)

        # Some custom models might expect specific Config objects
        # Ideally we'd map this too, but for now passing Namespace might work if attributes align

        try:
            model = model_class(config_obj)
        except Exception as e:
            # Try initializing with dict directly
            try:
                model = model_class(**config_dict)
            except Exception as e2:
                logger.error(f"Failed to instantiate model: {e}, {e2}")
                raise e

        # 4. Load Weights
        self.load_weights(model, model_path, device=device)

        # 5. Move to device/dtype
        model.to(device=device, dtype=dtype)

        return model

    def _get_model_class(self, architectures: List[str], model_type: str) -> Optional[Any]:
        """Resolve model class based on architecture name."""
        arch = architectures[0] if architectures else ""

        # Qwen mapping
        if "Qwen2" in arch or "Qwen2" in model_type or "qwen" in model_type:
            try:
                # Try importing Qwen3 architecture (assuming compatibility)
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
    def load_weights(model: nn.Module, model_path: str, device: str = "cpu", strict: bool = False):
        """
        Load weights from model_path into model.

        Args:
            model: The PyTorch model instance.
            model_path: Directory containing weight files.
            device: Device to load weights onto.
            strict: Whether to enforce strict state_dict matching.
        """
        logger.info(f"Loading weights from {model_path}...")

        # 1. Check for index file (sharded checkpoint)
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            index_file = os.path.join(model_path, "pytorch_model.bin.index.json")

        if os.path.exists(index_file):
            CustomModelLoader._load_sharded_weights(model, model_path, index_file, device, strict)
        else:
            # 2. Check for single file
            single_file = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(single_file):
                single_file = os.path.join(model_path, "pytorch_model.bin")

            if os.path.exists(single_file):
                CustomModelLoader._load_single_file(model, single_file, device, strict)
            else:
                logger.warning(f"No weights found in {model_path}. Model initialized with random weights.")

    @staticmethod
    def _load_single_file(model: nn.Module, filepath: str, device: str, strict: bool):
        logger.info(f"Loading single weight file: {filepath}")
        if filepath.endswith(".safetensors"):
            state_dict = safe_load_file(filepath, device=device)
        else:
            state_dict = torch.load(filepath, map_location=device)

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        CustomModelLoader._log_load_results(missing, unexpected)

    @staticmethod
    def _load_sharded_weights(model: nn.Module, model_path: str, index_file: str, device: str, strict: bool):
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
                shard_state = safe_load_file(filepath, device=device)
            else:
                shard_state = torch.load(filepath, map_location=device)
            full_state_dict.update(shard_state)

        missing, unexpected = model.load_state_dict(full_state_dict, strict=strict)
        CustomModelLoader._log_load_results(missing, unexpected)

    @staticmethod
    def _log_load_results(missing: List[str], unexpected: List[str]):
        if missing:
            logger.warning(f"Missing keys: {len(missing)}")
            logger.debug(f"Missing: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}")
            logger.debug(f"Unexpected: {unexpected}")
        if not missing and not unexpected:
            logger.info("All weights loaded successfully.")
