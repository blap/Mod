"""
PaddleOCR-VL-1.5 Plugin

This module implements the plugin interface for PaddleOCR-VL-1.5,
exposing standard inference methods and task-specific capabilities.
"""

import logging
from typing import Any, Dict, Optional, Union, List
from PIL import Image
from datetime import datetime

from ...common.standard_plugin_interface import (
    PluginMetadata,
    ModelPluginInterface,
    PluginType,
)
from .config import PaddleOCRVL15Config
from .model import PaddleOCRVL15Model

logger = logging.getLogger(__name__)

class PaddleOCRVL15Plugin(ModelPluginInterface):
    def __init__(self):
        metadata = PluginMetadata(
            name="PaddleOCR-VL-1.5",
            version="1.5.0",
            author="PaddlePaddle",
            description="Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=["transformers>=5.0.0", "torch", "pillow"],
            compatibility={"python_version": ">=3.8"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_architecture="PaddleOCR-VL",
            model_size="0.9B",
            required_memory_gb=4.0,
            supported_modalities=["image", "text"],
            license="Apache 2.0",
            tags=["ocr", "document-parsing", "vlm"],
            model_family="PaddleOCR",
            num_parameters=900000000,
            test_coverage=0.0, # To be updated
            validation_passed=False
        )
        super().__init__(metadata)
        self.model_wrapper = None
        self.config = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize with optional config dict."""
        try:
            # Create config object
            if config:
                self.config = PaddleOCRVL15Config(**config)
            else:
                self.config = PaddleOCRVL15Config()

            self.model_wrapper = PaddleOCRVL15Model(self.config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plugin: {e}")
            return False

    def load_model(self) -> Any:
        """Loads the model resources."""
        if not self.model_wrapper:
            self.initialize()

        self.model_wrapper.load_model()
        self.is_loaded = True
        return self.model_wrapper.model

    def infer(self, inputs: Dict[str, Any]) -> Any:
        """
        Standard inference method.
        inputs: {
            "image": PIL.Image,
            "task": str (optional, default 'ocr'),
            "prompt": str (optional),
            ...
        }
        """
        if not self.is_loaded:
            self.load_model()

        image = inputs.get("image")
        if not image:
            raise ValueError("Image input is required for PaddleOCR-VL")

        task = inputs.get("task", self.config.default_task)

        # Prepare standard message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._get_prompt_text(task)},
                ]
            }
        ]

        result = self.model_wrapper.generate(messages, task=task)
        return result

    def _get_prompt_text(self, task: str) -> str:
        PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }
        return PROMPTS.get(task, "OCR:")

    def cleanup(self):
        if self.model_wrapper:
            del self.model_wrapper
            self.model_wrapper = None
        self.is_loaded = False
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generates text from text prompt.
        """
        # For PaddleOCR-VL, this is typically part of standard generation.
        # We can treat this as an "infer" call with text only, but the model usually expects an image.
        # If no image is provided, we might fail or support text-only if underlying model supports it.
        # However, checking the model architecture (AutoModelForImageTextToText), it likely expects pixel_values.
        # We will attempt to run it or return empty string with warning if image is strictly required.
        logger.warning("generate_text called without image. This model expects images.")
        # Try to infer with dummy image or just text if supported?
        # Let's assume we can't do text-only generation easily without an image context in this specific architecture.
        return ""

    def tokenize(self, text: str) -> Any:
        if not self.is_loaded:
            self.load_model()
        return self.model_wrapper.processor.tokenizer(text)

    def detokenize(self, tokens: Any) -> str:
        if not self.is_loaded:
            self.load_model()
        return self.model_wrapper.processor.tokenizer.decode(tokens, skip_special_tokens=True)

    def supports_config(self, config: Dict[str, Any]) -> bool:
        return True

def create_paddleocr_vl_1_5_plugin() -> PaddleOCRVL15Plugin:
    return PaddleOCRVL15Plugin()
