"""
Custom Model Loader - Dependency Free (except safetensors/torch)
Replacing transformers.AutoModel.from_pretrained with efficient direct loading.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

logger = logging.getLogger(__name__)

class CustomModelLoader:
    """
    Efficiently loads model weights from disk (safetensors or bin) directly into PyTorch models.
    Supports sharded checkpoints.
    """

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
