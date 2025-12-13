"""
Interfaces and abstract base classes for Qwen3-VL components to support dependency injection.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
import torch
import torch.nn as nn
from PIL import Image
from transformers import PreTrainedTokenizerBase


class ConfigurableComponent(ABC):
    """
    Base interface for all configurable components in the Qwen3-VL system.
    """
    
    @abstractmethod
    def get_config(self) -> Any:
        """
        Get the configuration object for this component.
        
        Returns:
            Configuration object
        """
        pass


class MemoryManager(ABC):
    """
    Interface for memory management components.
    """
    
    @abstractmethod
    def allocate(self, size: int, pool_type: str = "general") -> Optional[Tuple[int, int]]:
        """
        Allocate memory of specified size.
        
        Args:
            size: Size in bytes to allocate
            pool_type: Type of memory pool to use
            
        Returns:
            Tuple of (memory_address, actual_allocated_size) or None if allocation fails
        """
        pass
    
    @abstractmethod
    def deallocate(self, ptr: int) -> bool:
        """
        Deallocate memory at the specified pointer.
        
        Args:
            ptr: Pointer to deallocate
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        pass


class Optimizer(ABC):
    """
    Interface for optimization components.
    """
    
    @abstractmethod
    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Apply optimizations to the given model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        pass


class Preprocessor(ABC):
    """
    Interface for preprocessing components.
    """
    
    @abstractmethod
    def preprocess_batch(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        return_tensors: str = "pt",
        tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of texts and images.
        
        Args:
            texts: List of text strings to process
            images: List of PIL Image objects to process (optional)
            return_tensors: Format for returned tensors (default "pt" for PyTorch)
            tokenizer: Tokenizer to use (optional)
            
        Returns:
            Dictionary containing processed text and image tensors
        """
        pass


class Pipeline(ABC):
    """
    Interface for inference pipelines.
    """
    
    @abstractmethod
    def preprocess_and_infer(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Preprocess inputs and run inference.
        
        Args:
            texts: List of text strings to process
            images: List of PIL Image objects to process (optional)
            tokenizer: Tokenizer to use (optional)
            **generation_kwargs: Additional generation arguments
            
        Returns:
            List of generated responses
        """
        pass


class AttentionMechanism(ABC):
    """
    Interface for attention mechanisms.
    """
    
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of the attention mechanism.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask (optional)
            position_ids: Position IDs (optional)
            past_key_value: Past key-value states for caching (optional)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value caching
            cache_position: Cache position (optional)
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        pass


class MLP(ABC):
    """
    Interface for MLP (feed-forward) components.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass


class Layer(ABC):
    """
    Interface for transformer layers.
    """
    
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of the transformer layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask (optional)
            position_ids: Position IDs (optional)
            past_key_value: Past key-value states for caching (optional)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value caching
            cache_position: Cache position (optional)
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        pass