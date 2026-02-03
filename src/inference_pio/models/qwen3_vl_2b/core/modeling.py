"""
Qwen3-VL-2B Modeling Logic
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Union

try:
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
    )
    from transformers import Qwen2VLForConditionalGeneration as AutoModelForVision2Seq
except ImportError:
    try:
        from transformers import (
            AutoImageProcessor,
        )
        from transformers import AutoModel as AutoModelForVision2Seq
        from transformers import (
            AutoTokenizer,
        )
    except ImportError:
        AutoModelForVision2Seq = None
        AutoTokenizer = None
        AutoImageProcessor = None

logger = logging.getLogger(__name__)

class Qwen3VL2BModeling(nn.Module):
    def __init__(self, config, system_profile):
        super().__init__()
        self.config = config
        self._system_profile = system_profile
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._model_name = config.model_path

        # Memory optimization settings
        self._memory_config = {
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_cache": config.use_cache,
            "torch_dtype": self._get_torch_dtype(config.torch_dtype),
            "device_map": config.device_map,
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
            "max_memory": config.max_memory,
        }

        self._initialize_model()

    def _get_torch_dtype(self, dtype_str: str):
        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float": torch.float32,
            "double": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        if isinstance(dtype_str, torch.dtype):
            return dtype_str

        if dtype_str in dtype_mapping:
            return dtype_mapping[dtype_str]

        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            logger.warning(f"Unknown dtype '{dtype_str}', defaulting to float16")
            return torch.float16

    def _initialize_model(self):
        try:
            logger.info(f"Loading Qwen3-VL-2B model from: {self._model_name}")

            load_kwargs = {
                "torch_dtype": self._memory_config.get("torch_dtype", torch.float16),
                "device_map": self._memory_config.get("device_map", "auto"),
                "trust_remote_code": True,
                "low_cpu_mem_usage": self._memory_config.get("low_cpu_mem_usage", True),
            }

            if self._system_profile.is_weak_hardware:
                if self._system_profile.safe_vram_limit_gb > 2.0:
                    load_kwargs["device_map"] = "auto"
                    max_mem = {
                        0: f"{int(self._system_profile.safe_vram_limit_gb * 1024)}MB"
                    }
                    load_kwargs["max_memory"] = max_mem
                    logger.info(
                        f"Applying safe VRAM limit: {self._system_profile.safe_vram_limit_gb:.2f} GB"
                    )
                else:
                    load_kwargs["device_map"] = "auto"
                    safe_limit_mb = int(self._system_profile.safe_vram_limit_gb * 1024)
                    if safe_limit_mb > 0:
                        load_kwargs["max_memory"] = {0: f"{safe_limit_mb}MB"}
                        logger.info(f"Applying minimal VRAM limit: {safe_limit_mb} MB")
                    else:
                        load_kwargs["device_map"] = "cpu"
                        logger.info("VRAM too low or unavailable. Forcing CPU.")

            if self._memory_config.get("max_memory"):
                load_kwargs["max_memory"] = self._memory_config["max_memory"]

            import os
            drive_exists = True
            path_sep = "\\" if "\\" in self._model_name else "/"
            if ":" in self._model_name:
                drive_letter = self._model_name.split(":")[0] + ":"
                if not os.path.exists(drive_letter):
                    logger.warning(
                        f"Drive {drive_letter} does not exist, using HuggingFace model..."
                    )
                    drive_exists = False

            if drive_exists and os.path.exists(self._model_name):
                model_path = self._model_name
                logger.info(f"Using local model from: {model_path}")
            else:
                logger.warning(
                    f"Local model path {self._model_name} not found, trying HuggingFace..."
                )
                model_path = "Qwen/Qwen2-VL-2B-Instruct"

            self._model = AutoModelForVision2Seq.from_pretrained(
                model_path, **load_kwargs
            )

            if self._memory_config.get("gradient_checkpointing", True):
                self._model.gradient_checkpointing_enable()

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            self._image_processor = AutoImageProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-VL-2B model: {e}")
            raise
