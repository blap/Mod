"""
Multimodal Preprocessing Pipeline System for Qwen3-VL-2B Model

This module implements a complete pipeline system for multimodal preprocessing
that integrates with the Qwen3-VL-2B model. It includes optimized preprocessing
stages for both text and image data, with performance monitoring and optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import numpy as np

from .multimodal_preprocessing import (
    MultimodalPreprocessor,
    create_multimodal_preprocessor
)

logger = logging.getLogger(__name__)


class MultimodalPipelineStage:
    """
    Represents a single stage in the multimodal preprocessing pipeline.
    Each stage can perform specific preprocessing operations.
    """
    
    def __init__(self, name: str, operation: callable, **kwargs):
        self.name = name
        self.operation = operation
        self.kwargs = kwargs
        self.execution_time = 0.0
        self.call_count = 0
    
    def execute(self, data: Any) -> Any:
        """
        Execute the pipeline stage operation on the input data.
        
        Args:
            data: Input data for the stage
            
        Returns:
            Processed data
        """
        start_time = time.time()
        
        try:
            result = self.operation(data, **self.kwargs)
            self.execution_time += time.time() - start_time
            self.call_count += 1
            return result
        except Exception as e:
            logger.error(f"Error in pipeline stage '{self.name}': {e}")
            raise
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get execution statistics for this stage.
        
        Returns:
            Dictionary containing execution statistics
        """
        avg_time = self.execution_time / self.call_count if self.call_count > 0 else 0.0
        return {
            'name': self.name,
            'execution_time': self.execution_time,
            'call_count': self.call_count,
            'average_time': avg_time
        }


class MultimodalPreprocessingPipeline:
    """
    Complete multimodal preprocessing pipeline for Qwen3-VL-2B model.
    Orchestrates multiple preprocessing stages for efficient multimodal data processing.
    """
    
    def __init__(self, model_path: str, 
                 max_text_length: int = 32768,
                 image_size: int = 448, 
                 patch_size: int = 14):
        self.model_path = model_path
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Create the multimodal preprocessor
        self.preprocessor = create_multimodal_preprocessor(
            model_path=model_path,
            max_text_length=max_text_length,
            image_size=image_size,
            patch_size=patch_size
        )
        
        # Initialize pipeline stages
        self.stages = []
        self._setup_default_stages()
        
        # Performance tracking
        self.pipeline_execution_time = 0.0
        self.pipeline_call_count = 0
    
    def _setup_default_stages(self):
        """
        Set up default preprocessing stages for the pipeline.
        """
        # Stage 1: Input validation
        self.add_stage(
            name="input_validation",
            operation=self._validate_input
        )
        
        # Stage 2: Text preprocessing (if text is provided)
        self.add_stage(
            name="text_preprocessing",
            operation=self._preprocess_text
        )
        
        # Stage 3: Image preprocessing (if image is provided)
        self.add_stage(
            name="image_preprocessing", 
            operation=self._preprocess_image
        )
        
        # Stage 4: Multimodal fusion (combine text and image features)
        self.add_stage(
            name="multimodal_fusion",
            operation=self._fuse_multimodal_features
        )
    
    def add_stage(self, name: str, operation: callable, **kwargs):
        """
        Add a new stage to the pipeline.
        
        Args:
            name: Name of the stage
            operation: Function to execute in this stage
            **kwargs: Arguments to pass to the operation
        """
        stage = MultimodalPipelineStage(name, operation, **kwargs)
        self.stages.append(stage)
        logger.debug(f"Added pipeline stage: {name}")
    
    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data for the pipeline.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Validated input data
        """
        if not isinstance(data, dict):
            raise ValueError(f"Input must be a dictionary, got {type(data)}")
        
        # Check for required keys
        if 'text' not in data and 'image' not in data:
            raise ValueError("Input must contain either 'text' or 'image' key")
        
        # Validate text if present
        if 'text' in data and data['text'] is not None:
            if not isinstance(data['text'], str):
                raise ValueError(f"Text must be a string, got {type(data['text'])}")
        
        # Validate image if present
        if 'image' in data and data['image'] is not None:
            if not isinstance(data['image'], (str, Image.Image)):
                raise ValueError(f"Image must be a string (path) or PIL Image, got {type(data['image'])}")
        
        return data
    
    def _preprocess_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess text data if present in the input.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Updated data dictionary with preprocessed text
        """
        if 'text' in data and data['text'] is not None:
            text_result = self.preprocessor.text_preprocessor.preprocess(data['text'])
            data.update(text_result)
        
        return data
    
    def _preprocess_image(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess image data if present in the input.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Updated data dictionary with preprocessed image
        """
        if 'image' in data and data['image'] is not None:
            image_result = self.preprocessor.image_preprocessor.preprocess(data['image'])
            data.update(image_result)
        
        return data
    
    def _fuse_multimodal_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse multimodal features (placeholder for actual fusion logic).
        
        Args:
            data: Input data dictionary with preprocessed text and/or image
            
        Returns:
            Fused multimodal features
        """
        # In a real implementation, this would combine text and image features
        # For now, we just return the data as is
        return data
    
    def execute(self, data: Dict[str, Any], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Execute the complete multimodal preprocessing pipeline.
        
        Args:
            data: Input data dictionary with 'text' and/or 'image' keys
            return_tensors: Format for returned tensors ("pt", "np", etc.)
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        start_time = time.time()
        
        # Initialize result with return_tensors
        result = {'return_tensors': return_tensors}
        
        # Execute each stage in the pipeline
        current_data = data.copy()
        for stage in self.stages:
            current_data = stage.execute(current_data)
            # Merge stage results into final result
            for key, value in current_data.items():
                if key not in ['text', 'image']:  # Don't overwrite original inputs
                    result[key] = value
        
        # Update pipeline performance metrics
        elapsed_time = time.time() - start_time
        self.pipeline_execution_time += elapsed_time
        self.pipeline_call_count += 1
        
        logger.debug(f"Pipeline execution took {elapsed_time:.4f}s")
        
        return result
    
    def execute_batch(self, batch_data: List[Dict[str, Any]], 
                     return_tensors: str = "pt") -> List[Dict[str, torch.Tensor]]:
        """
        Execute the pipeline on a batch of multimodal inputs.
        
        Args:
            batch_data: List of input data dictionaries
            return_tensors: Format for returned tensors
            
        Returns:
            List of dictionaries containing preprocessed multimodal tensors
        """
        start_time = time.time()
        
        results = []
        for data in batch_data:
            result = self.execute(data, return_tensors)
            results.append(result)
        
        # Update pipeline performance metrics
        elapsed_time = time.time() - start_time
        self.pipeline_execution_time += elapsed_time
        self.pipeline_call_count += len(batch_data)
        
        logger.debug(f"Batch pipeline execution took {elapsed_time:.4f}s for {len(batch_data)} items")
        
        return results
    
    def execute_multimodal_pair(self, text: str, image: Union[Image.Image, str],
                               return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Execute the pipeline on a text-image pair.
        
        Args:
            text: Input text
            image: Input image
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        data = {'text': text, 'image': image}
        return self.execute(data, return_tensors)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the entire pipeline.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        avg_time = self.pipeline_execution_time / self.pipeline_call_count if self.pipeline_call_count > 0 else 0.0
        
        stage_stats = [stage.get_stats() for stage in self.stages]
        
        return {
            'pipeline_execution_time': self.pipeline_execution_time,
            'pipeline_call_count': self.pipeline_call_count,
            'average_pipeline_time': avg_time,
            'stages': stage_stats
        }
    
    def reset_stats(self):
        """
        Reset all pipeline statistics.
        """
        self.pipeline_execution_time = 0.0
        self.pipeline_call_count = 0
        
        for stage in self.stages:
            stage.execution_time = 0.0
            stage.call_count = 0


def create_multimodal_pipeline(model_path: str,
                              max_text_length: int = 32768,
                              image_size: int = 448,
                              patch_size: int = 14) -> MultimodalPreprocessingPipeline:
    """
    Factory function to create a multimodal preprocessing pipeline.
    
    Args:
        model_path: Path to the Qwen3-VL-2B model
        max_text_length: Maximum length for text sequences
        image_size: Size of processed images
        patch_size: Size of image patches
        
    Returns:
        MultimodalPreprocessingPipeline instance
    """
    logger.info(f"Creating multimodal preprocessing pipeline for model: {model_path}")
    
    try:
        pipeline = MultimodalPreprocessingPipeline(
            model_path=model_path,
            max_text_length=max_text_length,
            image_size=image_size,
            patch_size=patch_size
        )
        
        logger.info("Multimodal preprocessing pipeline created successfully")
        return pipeline
    
    except Exception as e:
        logger.error(f"Failed to create multimodal preprocessing pipeline: {e}")
        raise


def apply_multimodal_pipeline_to_model(model: nn.Module, 
                                     pipeline: MultimodalPreprocessingPipeline) -> nn.Module:
    """
    Apply the multimodal preprocessing pipeline to the model.
    
    Args:
        model: The model to enhance with preprocessing pipeline
        pipeline: The multimodal preprocessing pipeline to attach
        
    Returns:
        Enhanced model with preprocessing pipeline
    """
    logger.info("Applying multimodal preprocessing pipeline to model...")
    
    # Attach the pipeline to the model
    model.multimodal_pipeline = pipeline
    
    # Add a convenience method for preprocessing
    def preprocess_multimodal(self, text: Optional[str] = None, 
                             image: Optional[Union[Image.Image, str]] = None,
                             return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess multimodal input using the attached pipeline.
        
        Args:
            text: Input text (optional)
            image: Input image (optional)
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        data = {}
        if text is not None:
            data['text'] = text
        if image is not None:
            data['image'] = image
            
        return self.multimodal_pipeline.execute(data, return_tensors)
    
    # Bind the method to the model
    model.preprocess_multimodal = preprocess_multimodal.__get__(model, model.__class__)
    
    logger.info("Multimodal preprocessing pipeline applied successfully")
    return model


class OptimizedMultimodalPipeline(MultimodalPreprocessingPipeline):
    """
    An optimized version of the multimodal preprocessing pipeline with additional
    performance enhancements for the Qwen3-VL-2B model.
    """
    
    def __init__(self, model_path: str,
                 max_text_length: int = 32768,
                 image_size: int = 448,
                 patch_size: int = 14,
                 enable_caching: bool = True,
                 cache_size: int = 1000):
        super().__init__(model_path, max_text_length, image_size, patch_size)
        
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.preprocessing_cache = {}
        
        if self.enable_caching:
            logger.info(f"Initialized preprocessing cache with size {self.cache_size}")
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """
        Generate a cache key for the given input data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a hash of the input data for caching
        text_hash = hashlib.md5(data.get('text', '').encode()).hexdigest() if 'text' in data and data['text'] else ''
        image_hash = hashlib.md5(str(data.get('image')).encode()).hexdigest() if 'image' in data and data['image'] else ''
        
        return f"{text_hash}_{image_hash}"
    
    def execute(self, data: Dict[str, Any], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Execute the optimized multimodal preprocessing pipeline with caching.
        
        Args:
            data: Input data dictionary with 'text' and/or 'image' keys
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary containing preprocessed multimodal tensors
        """
        if self.enable_caching:
            cache_key = self._get_cache_key(data)
            
            # Check if result is in cache
            if cache_key in self.preprocessing_cache:
                logger.debug(f"Cache hit for key: {cache_key[:10]}...")
                return self.preprocessing_cache[cache_key]
        
        # Execute the pipeline normally
        result = super().execute(data, return_tensors)
        
        # Store in cache if caching is enabled
        if self.enable_caching:
            cache_key = self._get_cache_key(data)
            
            # Implement simple LRU-like cache eviction
            if len(self.preprocessing_cache) >= self.cache_size:
                # Remove oldest entry (first in insertion order)
                oldest_key = next(iter(self.preprocessing_cache))
                del self.preprocessing_cache[oldest_key]
            
            self.preprocessing_cache[cache_key] = result
            logger.debug(f"Stored in cache: {cache_key[:10]}...")
        
        return result
    
    def clear_cache(self):
        """
        Clear the preprocessing cache.
        """
        self.preprocessing_cache.clear()
        logger.info("Preprocessing cache cleared")


def create_optimized_multimodal_pipeline(model_path: str,
                                       max_text_length: int = 32768,
                                       image_size: int = 448,
                                       patch_size: int = 14,
                                       enable_caching: bool = True,
                                       cache_size: int = 1000) -> OptimizedMultimodalPipeline:
    """
    Factory function to create an optimized multimodal preprocessing pipeline.
    
    Args:
        model_path: Path to the Qwen3-VL-2B model
        max_text_length: Maximum length for text sequences
        image_size: Size of processed images
        patch_size: Size of image patches
        enable_caching: Whether to enable preprocessing result caching
        cache_size: Size of the preprocessing cache
        
    Returns:
        OptimizedMultimodalPipeline instance
    """
    logger.info(f"Creating optimized multimodal preprocessing pipeline for model: {model_path}")
    
    try:
        pipeline = OptimizedMultimodalPipeline(
            model_path=model_path,
            max_text_length=max_text_length,
            image_size=image_size,
            patch_size=patch_size,
            enable_caching=enable_caching,
            cache_size=cache_size
        )
        
        logger.info("Optimized multimodal preprocessing pipeline created successfully")
        return pipeline
    
    except Exception as e:
        logger.error(f"Failed to create optimized multimodal preprocessing pipeline: {e}")
        raise


__all__ = [
    "MultimodalPipelineStage",
    "MultimodalPreprocessingPipeline", 
    "create_multimodal_pipeline",
    "apply_multimodal_pipeline_to_model",
    "OptimizedMultimodalPipeline",
    "create_optimized_multimodal_pipeline"
]