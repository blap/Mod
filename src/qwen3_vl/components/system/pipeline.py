"""
Intel optimized pipeline for Qwen3-VL with dependency injection support.
"""
from typing import Dict, Any, Optional, List
from PIL import Image
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn as nn
from .interfaces import Pipeline, Preprocessor
from ...config.config import Qwen3VLConfig


class IntelOptimizedPipeline(Pipeline):
    """
    Optimized inference pipeline with Intel i5-10210U-specific optimizations
    and dependency injection support.
    """
    
    def __init__(self, model: nn.Module, config: Qwen3VLConfig, preprocessor: Preprocessor):
        """
        Initialize the Intel-optimized pipeline.
        
        Args:
            model: The model to run inference on
            config: Configuration for optimizations
            preprocessor: Preprocessor component to use
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Apply Intel-specific optimizations based on configuration
        self._apply_intel_optimizations()
    
    def _apply_intel_optimizations(self):
        """Apply Intel i5-10210U-specific optimizations."""
        # Optimize for Intel CPU features
        if hasattr(self.config, 'enable_cpu_optimizations') and self.config.enable_cpu_optimizations:
            # Enable AVX2 optimizations
            torch.backends.mkldnn.enabled = True
            
            # Set number of threads based on CPU configuration
            if hasattr(self.config, 'num_threads'):
                torch.set_num_threads(self.config.num_threads)
    
    def preprocess(self, texts: List[str], images: Optional[List[Image.Image]] = None, 
                   tokenizer: Optional[PreTrainedTokenizerBase] = None):
        """
        Preprocess inputs with Intel-specific optimizations.
        
        Args:
            texts: List of text inputs
            images: List of image inputs (optional)
            tokenizer: Tokenizer to use (optional)
            
        Returns:
            Preprocessed inputs
        """
        # Use optimized preprocessor
        inputs = self.preprocessor(texts, images, tokenizer)
        
        # Apply Intel-specific optimizations to inputs
        if torch.is_tensor(inputs['input_ids']):
            # Ensure tensor is contiguous for optimal memory access
            inputs['input_ids'] = inputs['input_ids'].contiguous()
        
        if 'pixel_values' in inputs and torch.is_tensor(inputs['pixel_values']):
            inputs['pixel_values'] = inputs['pixel_values'].contiguous()
        
        return inputs
    
    def forward(self, **kwargs):
        """
        Forward pass with Intel-specific optimizations.
        
        Args:
            **kwargs: Model inputs
            
        Returns:
            Model outputs
        """
        # Move inputs to appropriate device
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                kwargs[key] = value.to(self.device)
        
        # Run model with optimizations
        with torch.no_grad():
            outputs = self.model(**kwargs)
        
        return outputs
    
    def generate(self, inputs: Dict[str, Any], 
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 do_sample: bool = False,
                 **kwargs):
        """
        Generate tokens with Intel-specific optimizations.
        
        Args:
            inputs: Model inputs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling instead of greedy decoding
            **kwargs: Additional generation arguments
            
        Returns:
            Generated outputs
        """
        # Apply Intel-specific optimizations to generation parameters
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            'num_beams': 1 if not do_sample else getattr(self.config, 'num_beams', 1),
            **kwargs
        }
        
        # Run generation with optimizations
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        return outputs