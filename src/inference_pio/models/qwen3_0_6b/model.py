"""
Qwen3-0.6B Model Implementation
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

from ...core.engine.backend import Module
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer

from .config import Qwen3_0_6bConfig as Qwen3_0_6B_Config
from .architecture import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

class Qwen3_0_6B_Model(Module):
    def __init__(self, config: Qwen3_0_6B_Config):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None

        self._initialize_model()

    def _resolve_model_path(self) -> str:
        """
        Resolves the model path with the following priority:
        1. H:/Qwen3-0.6B
        2. Local Cache (src/inference_pio/models/qwen3_0_6b/_model_cache/Qwen3-0.6B)
        3. Download from HuggingFace
        """
        model_name = "Qwen3-0.6B"
        hf_repo = "Qwen/Qwen3-0.6B"

        # 1. Check H: Drive
        h_drive_path = os.path.join("H:/", model_name)
        if os.path.exists(h_drive_path):
            logger.info(f"Found model on H: drive: {h_drive_path}")
            return h_drive_path

        # 2. Check Local Cache
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "_model_cache")
        local_path = os.path.join(cache_dir, model_name)

        if os.path.exists(local_path):
            if os.listdir(local_path):
                logger.info(f"Found model in local cache: {local_path}")
                return local_path

        # 3. Download
        logger.info(f"Model not found. Attempting to download {hf_repo} to {local_path}...")

        os.makedirs(cache_dir, exist_ok=True)

        # Check Disk Space
        total, used, free = shutil.disk_usage(cache_dir)
        required_space = 2 * 1024 * 1024 * 1024 # 2GB

        if free < required_space:
            raise RuntimeError(f"Insufficient disk space. Required: 2GB, Available: {free/1024/1024/1024:.2f}GB")

        try:
            logger.info("Cloning from HuggingFace...")
            subprocess.run(
                ["git", "clone", f"https://huggingface.co/{hf_repo}", local_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Download complete.")
            return local_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download model: {e}")
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            raise RuntimeError(f"Failed to download model from {hf_repo}")

    def _initialize_model(self):
        logger.info("Initializing Qwen3-0.6B model...")
        self._model = Qwen3ForCausalLM(self.config)

        try:
            model_path = self._resolve_model_path()
            # Attempt to load weights using the custom loader
            CustomModelLoader.load_weights(self._model, model_path, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Model will use random initialization.")

        try:
            # Attempt to load tokenizer
            model_path = self._resolve_model_path()
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Text processing will be limited.")

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

def create_qwen3_0_6b_model(config: Qwen3_0_6B_Config) -> Qwen3_0_6B_Model:
    return Qwen3_0_6B_Model(config)
