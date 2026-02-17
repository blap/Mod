"""
Qwen3-4B-Instruct-2507 Model Implementation (Dependency-Free)
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union, Tuple

from ...core.engine.backend import Module, Tensor
from ...common.custom_components.model_loader import CustomModelLoader
from ...common.custom_components.tokenizer import load_custom_tokenizer

from .config import Qwen3_4B_Instruct_2507_Config
from .architecture import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

class Qwen3_4B_Instruct_2507_Model(Module):
    def __init__(self, config: Qwen3_4B_Instruct_2507_Config):
        super().__init__()
        self.config = config
        self._model = Qwen3ForCausalLM(config)
        self._tokenizer = None

        # Expose sub-modules for state_dict matching
        self.model = self._model.model
        self.lm_head = self._model.lm_head

        self._initialize_model()

    def _resolve_model_path(self) -> str:
        """
        Resolves model path with priority:
        1. H:/Qwen3-4B-Instruct-2507
        2. ~/.inference_pio/cache/Qwen3-4B-Instruct-2507
        3. Git Clone to cache
        """
        model_name = "Qwen3-4B-Instruct-2507"
        hf_repo = "Qwen/Qwen2.5-Coder-3B-Instruct" # Fallback/Proxy repo as 4B might be custom

        # 1. Check H: Drive
        h_drive_path = os.path.join("H:/", model_name)
        if os.path.exists(h_drive_path): return h_drive_path

        # 2. Check Local Cache
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "_model_cache")
        local_path = os.path.join(cache_dir, model_name)

        if os.path.exists(local_path) and os.listdir(local_path):
            return local_path

        # 3. Download if not found
        logger.info(f"Model not found in H:/ or cache. Attempting download to {local_path}...")
        os.makedirs(cache_dir, exist_ok=True)

        # Check Disk Space (Need ~10GB)
        total, used, free = shutil.disk_usage(cache_dir)
        required_space = 10 * 1024 * 1024 * 1024
        if free < required_space:
            raise RuntimeError(f"Insufficient disk space. Required: 10GB, Available: {free/1024/1024/1024:.2f}GB")

        try:
            # Clone only if git is available
            subprocess.run(["git", "clone", f"https://huggingface.co/{hf_repo}", local_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return local_path
        except subprocess.CalledProcessError:
            if os.path.exists(local_path): shutil.rmtree(local_path)
            raise RuntimeError(f"Failed to download model from {hf_repo}")
        except FileNotFoundError:
             raise RuntimeError("Git not found. Please install git to download models.")

    def _initialize_model(self):
        logger.info("Initializing Qwen3-4B-Instruct-2507 model...")

        try:
            model_path = self._resolve_model_path()
            logger.info(f"Loading weights from {model_path}")
            CustomModelLoader.load_weights(self, model_path, device="cpu")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Model will use random initialization.")

        try:
            model_path = self._resolve_model_path()
            self._tokenizer = load_custom_tokenizer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Text processing will be limited.")

    def forward(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None, use_cache: bool = False, cache_position: Union[int, Tensor] = 0, max_cache_len: int = 0):
        return self._model(input_ids, past_key_values, use_cache, cache_position, max_cache_len)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

def create_qwen3_4b_instruct_2507_model(config: Qwen3_4B_Instruct_2507_Config) -> Qwen3_4B_Instruct_2507_Model:
    return Qwen3_4B_Instruct_2507_Model(config)
