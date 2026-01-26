"""
Multimodal Preprocessing Pipeline for Qwen3-VL-2B Model

This module implements a comprehensive preprocessing pipeline for multimodal data
(text and images) specifically optimized for the Qwen3-VL-2B model. The pipeline
handles efficient preprocessing of both text and image inputs, with optimizations
for vision-language tasks.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing component for the multimodal pipeline.
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


class ImagePreprocessor:
    """
    Image preprocessing component for the multimodal pipeline.
    Handles image resizing, normalization, and vision-specific optimizations.
    """
    
    def __init__(self, image_processor: AutoImageProcessor, 
                 image_size: int = 448, patch_size: int = 14):
        self.image_processor = image_processor
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Validate image processor
        if self.image_processor is None:
            raise ValueError("Image processor cannot be None")
    
    def preprocess(self, image: Union[Image.Image, str], 
                   return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess image input for the model.
        
        Args:
            image: Input image (PIL Image or path to image)
            return_tensors: Format for returned tensors ("pt", "np", etc.)
            
        Returns:
            Dictionary containing preprocessed image tensors
        """
        start_time = time.time()
        
        # Load image if it's a path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Image must be PIL Image or path string, got {type(image)}")
        
        # Process the image
        processed = self.image_processor(
            images=image,
            return_tensors=return_tensors
        )
        
        # Log preprocessing time
        elapsed_time = time.time() - start_time
        logger.debug(f"Image preprocessing took {elapsed_time:.4f}s for image size {image.size}")
        
        return processed
    
    def batch_preprocess(self, images: List[Union[Image.Image, str]], 
                         return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of image inputs.
        
        Args:
            images: List of input images (PIL Images or paths)
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary containing preprocessed image tensors
        """
        start_time = time.time()
        
        # Process each image
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Image must be PIL Image or path string, got {type(img)}")
            
            processed = self.image_processor(
                images=img,
                return_tensors=return_tensors
            )
            processed_images.append(processed)
        
        # Stack the processed images
        if return_tensors == "pt":
            pixel_values = torch.stack([img['pixel_values'] for img in processed_images], dim=0)
        else:
            pixel_values = np.stack([img['pixel_values'] for img in processed_images], axis=0)
        
        result = {'pixel_values': pixel_values}
        
        # Log preprocessing time
        elapsed_time = time.time() - start_time
        logger.debug(f"Batch image preprocessing took {elapsed_time:.4f}s for {len(images)} images")
        
        return result


class MultimodalPreprocessor:
    """
    Main multimodal preprocessing pipeline that combines text and image preprocessing.
    Optimized specifically for the Qwen3-VL-2B model architecture.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, image_processor: AutoImageProcessor,
                 max_text_length: int = 32768, image_size: int = 448, patch_size: int = 14):
        self.text_preprocessor = TextPreprocessor(tokenizer, max_text_length)
        self.image_preprocessor = ImagePreprocessor(image_processor, image_size, patch_size)
        
        # Performance metrics
        self.total_preprocessing_time = 0.0
        self.num_preprocessing_calls = 0
    
    def preprocess(self, text: Optional[str] = None, 
                   image: Optional[Union[Image.Image, str]] = None,
                   return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess multimodal input (text and/or image) for the model.
        
        Args:
            text: Input text (optional)
            image: Input image (optional)
            return_tensors: Format for returned tensors ("pt", "np", etc.)
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        start_time = time.time()
        
        result = {}
        
        # Preprocess text if provided
        if text is not None:
            text_result = self.text_preprocessor.preprocess(text, return_tensors)
            result.update(text_result)
        
        # Preprocess image if provided
        if image is not None:
            image_result = self.image_preprocessor.preprocess(image, return_tensors)
            result.update(image_result)
        
        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_preprocessing_time += elapsed_time
        self.num_preprocessing_calls += 1
        
        logger.debug(f"Multimodal preprocessing took {elapsed_time:.4f}s")
        
        return result
    
    def batch_preprocess(self, inputs: List[Dict[str, Any]], 
                         return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of multimodal inputs.
        
        Args:
            inputs: List of dictionaries containing 'text' and/or 'image' keys
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        start_time = time.time()
        
        # Separate texts and images
        texts = []
        images = []
        text_indices = []  # Track which inputs had text
        image_indices = []  # Track which inputs had images
        
        for i, input_item in enumerate(inputs):
            if 'text' in input_item and input_item['text'] is not None:
                texts.append(input_item['text'])
                text_indices.append(i)
            if 'image' in input_item and input_item['image'] is not None:
                images.append(input_item['image'])
                image_indices.append(i)
        
        result = {}
        
        # Process texts if any
        if texts:
            text_result = self.text_preprocessor.batch_preprocess(texts, return_tensors)
            result.update(text_result)
        
        # Process images if any
        if images:
            image_result = self.image_preprocessor.batch_preprocess(images, return_tensors)
            result.update(image_result)
        
        # Update performance metrics
        elapsed_time = time.time() - start_time
        self.total_preprocessing_time += elapsed_time
        self.num_preprocessing_calls += 1
        
        logger.debug(f"Batch multimodal preprocessing took {elapsed_time:.4f}s for {len(inputs)} inputs")
        
        return result
    
    def preprocess_multimodal_pair(self, text: str, image: Union[Image.Image, str],
                                   return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess a text-image pair for the model.
        
        Args:
            text: Input text
            image: Input image
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        return self.preprocess(text=text, image=image, return_tensors=return_tensors)
    
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


def create_multimodal_preprocessor(model_path: str, 
                                  max_text_length: int = 32768,
                                  image_size: int = 448, 
                                  patch_size: int = 14) -> MultimodalPreprocessor:
    """
    Factory function to create a multimodal preprocessor for Qwen3-VL-2B.
    
    Args:
        model_path: Path to the Qwen3-VL-2B model
        max_text_length: Maximum length for text sequences
        image_size: Size of processed images
        patch_size: Size of image patches
        
    Returns:
        MultimodalPreprocessor instance
    """
    logger.info(f"Creating multimodal preprocessor for model: {model_path}")
    
    try:
        # Load tokenizer and image processor
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Create and return the preprocessor
        preprocessor = MultimodalPreprocessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_text_length=max_text_length,
            image_size=image_size,
            patch_size=patch_size
        )
        
        logger.info("Multimodal preprocessor created successfully")
        return preprocessor
    
    except Exception as e:
        logger.error(f"Failed to create multimodal preprocessor: {e}")
        raise


def apply_multimodal_preprocessing_to_model(model: nn.Module, 
                                          preprocessor: MultimodalPreprocessor) -> nn.Module:
    """
    Apply multimodal preprocessing optimizations to the model.
    
    Args:
        model: The model to optimize
        preprocessor: The multimodal preprocessor to attach
        
    Returns:
        Optimized model with preprocessing capabilities
    """
    logger.info("Applying multimodal preprocessing optimizations to model...")
    
    # Attach the preprocessor to the model
    model.preprocessor = preprocessor
    
    # Optionally, we could add preprocessing hooks here
    # For now, we just attach the preprocessor as an attribute
    
    logger.info("Multimodal preprocessing optimizations applied successfully")
    return model


__all__ = [
    "TextPreprocessor",
    "ImagePreprocessor", 
    "MultimodalPreprocessor",
    "create_multimodal_preprocessor",
    "apply_multimodal_preprocessing_to_model"
]