"""
Unimodal Preprocessing Pipeline for Language Models
Dependency-Free
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ...core.engine.backend import Tensor
from ...common.custom_components.tokenizer import CustomBPETokenizer

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, tokenizer: Any, max_length: int = 32768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Assume tokenizer has pad/eos properties
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token'):
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, text: str, return_tensors: str = "pt", **kwargs) -> Dict[str, Tensor]:
        start = time.time()
        encoded = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        # Ensure returned types are backend Tensors if 'pt' requested (CustomTokenizer usually does this)
        # If CustomTokenizer returns dict of Tensors, we are good.
        logger.debug(f"Preprocessing took {time.time()-start:.4f}s")
        return encoded

    def batch_preprocess(self, texts: List[str], return_tensors: str = "pt", **kwargs) -> Dict[str, Tensor]:
        start = time.time()
        if not texts:
            # Return empty tensors
            return {
                "input_ids": Tensor([0, 0]),
                "attention_mask": Tensor([0, 0])
            }
        encoded = self.tokenizer(texts, return_tensors=return_tensors, **kwargs)
        logger.debug(f"Batch preprocessing took {time.time()-start:.4f}s")
        return encoded

class UnimodalPreprocessor:
    def __init__(self, tokenizer: Any, max_text_length: int = 32768):
        self.text_preprocessor = TextPreprocessor(tokenizer, max_text_length)
        self.total_time = 0.0
        self.calls = 0
        self.optimizations = {}

    def preprocess(self, text: str, return_tensors: str = "pt", model_type: Optional[str] = None) -> Dict[str, Tensor]:
        start = time.time()
        result = self.text_preprocessor.preprocess(text, return_tensors)
        if model_type and model_type in self.optimizations:
            result = self.optimizations[model_type](result)

        self.total_time += time.time() - start
        self.calls += 1
        return result

    def register_model_optimization(self, model_type, func):
        self.optimizations[model_type] = func

    def batch_preprocess(self, texts: List[str], return_tensors="pt", model_type=None):
        start = time.time()
        result = self.text_preprocessor.batch_preprocess(texts, return_tensors)
        if model_type and model_type in self.optimizations:
            result = self.optimizations[model_type](result)
        self.total_time += time.time() - start
        self.calls += 1
        return result

    def get_performance_stats(self):
        return {
            "total_preprocessing_time": self.total_time,
            "num_preprocessing_calls": self.calls,
            "average_preprocessing_time": self.total_time / self.calls if self.calls > 0 else 0.0
        }

def create_unimodal_preprocessor(model_path: str, max_text_length: int = 32768) -> UnimodalPreprocessor:
    logger.info(f"Creating preprocessor for {model_path}")
    # Always use CustomBPETokenizer for dependency-free behavior
    tokenizer = CustomBPETokenizer(f"{model_path}/vocab.json", f"{model_path}/merges.txt")
    return UnimodalPreprocessor(tokenizer, max_text_length)

def apply_unimodal_preprocessing_to_model(model, preprocessor):
    model.preprocessor = preprocessor
    return model

__all__ = ["TextPreprocessor", "UnimodalPreprocessor", "create_unimodal_preprocessor", "apply_unimodal_preprocessing_to_model"]
