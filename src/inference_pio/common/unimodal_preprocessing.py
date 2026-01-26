"""
Unimodal Preprocessing Pipeline for Language Models

This module implements a comprehensive preprocessing pipeline for unimodal text data
designed to work with various language models. The pipeline handles efficient
preprocessing of text inputs with a generic interface that can be extended by
model-specific implementations.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing component for the unimodal pipeline.
    Handles tokenization, normalization, and text-specific optimizations.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 32768):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, text: str, return_tensors: str = "pt",
                   add_special_tokens: bool = True,
                   truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Preprocess text input for the model.

        Args:
            text: Input text to preprocess
            return_tensors: Format for returned tensors ("pt", "np", etc.)
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate sequences

        Returns:
            Dictionary containing preprocessed text tensors
        """
        start_time = time.time()

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=True,
            truncation=truncation,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens
        )

        # Log preprocessing time
        elapsed_time = time.time() - start_time
        logger.debug(f"Text preprocessing took {elapsed_time:.4f}s for input length {len(text)}")

        return encoded

    def batch_preprocess(self, texts: List[str], return_tensors: str = "pt",
                         add_special_tokens: bool = True,
                         truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of text inputs.

        Args:
            texts: List of input texts to preprocess
            return_tensors: Format for returned tensors
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate sequences

        Returns:
            Dictionary containing preprocessed text tensors
        """
        start_time = time.time()

        # Handle empty list case
        if not texts:
            # Return empty tensors with appropriate shapes
            if return_tensors == "pt":
                return {
                    'input_ids': torch.empty((0, 0), dtype=torch.long),
                    'attention_mask': torch.empty((0, 0), dtype=torch.long)
                }
            else:
                # For other tensor formats, return empty arrays
                return {
                    'input_ids': np.array([]),
                    'attention_mask': np.array([])
                }

        # Tokenize the batch of texts
        encoded = self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=True,
            truncation=truncation,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens
        )

        # Log preprocessing time
        elapsed_time = time.time() - start_time
        logger.debug(f"Batch text preprocessing took {elapsed_time:.4f}s for {len(texts)} texts")

        return encoded


class UnimodalPreprocessor:
    """
    Main unimodal preprocessing pipeline that focuses on text preprocessing.
    Provides a generic interface that can be extended with model-specific optimizations.
    """

    def __init__(self, tokenizer: AutoTokenizer,
                 max_text_length: int = 32768):
        self.text_preprocessor = TextPreprocessor(tokenizer, max_text_length)

        # Performance metrics
        self.total_preprocessing_time = 0.0
        self.num_preprocessing_calls = 0

        # Model-specific optimizations - initially empty, can be registered externally
        self.model_specific_optimizations = {}

    def preprocess(self, text: str,
                   return_tensors: str = "pt",
                   model_type: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess unimodal text input for the model.

        Args:
            text: Input text
            return_tensors: Format for returned tensors ("pt", "np", etc.)
            model_type: Type of model for specific optimizations ("glm47", "qwen3_4b", "qwen3_coder")

        Returns:
            Dictionary containing preprocessed text tensors
        """
        start_time = time.time()

        # Preprocess text
        result = self.text_preprocessor.preprocess(text, return_tensors)

        # Apply model-specific optimizations if specified
        if model_type and model_type in self.model_specific_optimizations:
            result = self.model_specific_optimizations[model_type](result)

        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_preprocessing_time += elapsed_time
        self.num_preprocessing_calls += 1

        logger.debug(f"Unimodal preprocessing took {elapsed_time:.4f}s")

        return result

    def register_model_optimization(self, model_type: str, optimization_func):
        """
        Register a model-specific optimization function.

        Args:
            model_type: String identifier for the model type
            optimization_func: Function that takes a result dict and returns an optimized result dict
        """
        self.model_specific_optimizations[model_type] = optimization_func

    def batch_preprocess(self, texts: List[str],
                         return_tensors: str = "pt",
                         model_type: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of text inputs.

        Args:
            texts: List of input texts
            return_tensors: Format for returned tensors
            model_type: Type of model for specific optimizations

        Returns:
            Dictionary containing preprocessed text tensors
        """
        start_time = time.time()

        # Process texts
        result = self.text_preprocessor.batch_preprocess(texts, return_tensors)

        # Apply model-specific optimizations if specified
        if model_type and model_type in self.model_specific_optimizations:
            result = self.model_specific_optimizations[model_type](result)

        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_preprocessing_time += elapsed_time
        self.num_preprocessing_calls += 1

        logger.debug(f"Batch unimodal preprocessing took {elapsed_time:.4f}s for {len(texts)} inputs")

        return result


    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the preprocessing pipeline.

        Returns:
            Dictionary containing performance metrics
        """
        if self.num_preprocessing_calls > 0:
            avg_time = self.total_preprocessing_time / self.num_preprocessing_calls
        else:
            avg_time = 0.0

        return {
            'total_preprocessing_time': self.total_preprocessing_time,
            'num_preprocessing_calls': self.num_preprocessing_calls,
            'average_preprocessing_time': avg_time
        }

    def reset_performance_stats(self):
        """
        Reset performance statistics.
        """
        self.total_preprocessing_time = 0.0
        self.num_preprocessing_calls = 0


def create_unimodal_preprocessor(model_path: str,
                                max_text_length: int = 32768) -> UnimodalPreprocessor:
    """
    Factory function to create a unimodal preprocessor for language models.

    Args:
        model_path: Path to the language model
        max_text_length: Maximum length for text sequences

    Returns:
        UnimodalPreprocessor instance
    """
    logger.info(f"Creating unimodal preprocessor for model: {model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Create and return the preprocessor
        preprocessor = UnimodalPreprocessor(
            tokenizer=tokenizer,
            max_text_length=max_text_length
        )

        logger.info("Unimodal preprocessor created successfully")
        return preprocessor

    except Exception as e:
        logger.error(f"Failed to create unimodal preprocessor: {e}")
        raise


def apply_unimodal_preprocessing_to_model(model: nn.Module,
                                        preprocessor: UnimodalPreprocessor) -> nn.Module:
    """
    Apply unimodal preprocessing optimizations to the model.

    Args:
        model: The model to optimize
        preprocessor: The unimodal preprocessor to attach

    Returns:
        Optimized model with preprocessing capabilities
    """
    logger.info("Applying unimodal preprocessing optimizations to model...")

    # Attach the preprocessor to the model
    model.preprocessor = preprocessor

    # Optionally, we could add preprocessing hooks here
    # For now, we just attach the preprocessor as an attribute

    logger.info("Unimodal preprocessing optimizations applied successfully")
    return model


__all__ = [
    "TextPreprocessor",
    "UnimodalPreprocessor",
    "create_unimodal_preprocessor",
    "apply_unimodal_preprocessing_to_model"
]