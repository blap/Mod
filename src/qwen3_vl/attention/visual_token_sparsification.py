"""
Visual Token Sparsification for Qwen3-VL Model
Implementation of SparseVLM-inspired visual token sparsification for CPU optimization on Intel i5-10210U

This module implements visual token sparsification techniques inspired by SparseVLM,
which reduces computational complexity by eliminating redundant visual tokens during inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import logging
from collections import OrderedDict


@dataclass
class SparsificationConfig:
    """Configuration for visual token sparsification."""
    # Sparsification parameters
    sparsity_ratio: float = 0.5  # Ratio of tokens to keep (0.0-1.0)
    sparsity_method: str = "top_k"  # 'top_k', 'magnitude', 'attention', 'random'
    sparsity_temperature: float = 1.0  # Temperature for soft sparsity
    enable_layerwise_sparsification: bool = True  # Apply sparsification at each layer
    
    # Token selection parameters
    similarity_threshold: float = 0.1  # Threshold for token similarity
    min_tokens_per_image: int = 16  # Minimum number of tokens to keep per image
    max_tokens_per_image: int = 256  # Maximum number of tokens to keep per image
    
    # Performance optimization
    enable_cache: bool = True  # Cache sparsification results
    cache_size: int = 1000  # Size of the sparsification cache
    enable_dynamic_sparsity: bool = True  # Adjust sparsity based on input complexity
    
    # Model-specific parameters
    apply_to_vision_encoder: bool = True  # Apply sparsification to vision encoder
    apply_to_cross_attention: bool = True  # Apply sparsification to cross-attention
    apply_to_self_attention: bool = False  # Apply sparsification to self-attention


class TokenSparsifier(nn.Module):
    """
    Module that performs token sparsification based on various criteria.
    """
    def __init__(self, config: SparsificationConfig):
        super().__init__()
        self.config = config
        self.cache = OrderedDict()  # For caching sparsification results
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Apply sparsification to input tokens.
        
        Args:
            tokens: Input tokens of shape (batch_size, seq_len, hidden_dim)
            attention_weights: Attention weights for attention-based sparsification
            layer_idx: Current layer index (for layer-specific sparsification)
            text_features: Text features for cross-modal sparsification
            
        Returns:
            Tuple of (sparsified_tokens, kept_indices, sparsification_info)
        """
        batch_size, seq_len, hidden_dim = tokens.shape
        
        # Calculate target number of tokens to keep
        if self.config.enable_dynamic_sparsity and text_features is not None:
            # Adjust sparsity based on input complexity
            text_complexity = self._calculate_text_complexity(text_features)
            adaptive_ratio = self.config.sparsity_ratio * (0.8 + 0.4 * text_complexity)
            adaptive_ratio = max(0.1, min(1.0, adaptive_ratio))
        else:
            adaptive_ratio = self.config.sparsity_ratio
            
        target_tokens = max(
            self.config.min_tokens_per_image,
            min(
                int(seq_len * adaptive_ratio),
                self.config.max_tokens_per_image
            )
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(tokens, target_tokens, layer_idx)
        if self.config.enable_cache and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            # Move cached result to end (LRU)
            self.cache.move_to_end(cache_key)
            return cached_result
        
        # Apply sparsification based on method
        if self.config.sparsity_method == "top_k":
            kept_indices = self._top_k_sparsification(tokens, target_tokens)
        elif self.config.sparsity_method == "magnitude":
            kept_indices = self._magnitude_sparsification(tokens, target_tokens)
        elif self.config.sparsity_method == "attention":
            if attention_weights is not None:
                kept_indices = self._attention_sparsification(attention_weights, target_tokens)
            else:
                # Fallback to magnitude if attention weights not provided
                kept_indices = self._magnitude_sparsification(tokens, target_tokens)
        elif self.config.sparsity_method == "random":
            kept_indices = self._random_sparsification(tokens, target_tokens)
        else:
            # Default to top_k
            kept_indices = self._top_k_sparsification(tokens, target_tokens)
        
        # Apply token selection
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1)
        sparsified_tokens = tokens[batch_indices, kept_indices]
        
        # Create sparsification info
        sparsification_info = {
            'kept_ratio': kept_indices.size(1) / seq_len,
            'target_tokens': target_tokens,
            'actual_tokens': kept_indices.size(1),
            'sparsity_method': self.config.sparsity_method,
            'layer_idx': layer_idx
        }
        
        # Add to cache if enabled
        if self.config.enable_cache:
            if len(self.cache) >= self.config.cache_size:
                self.cache.popitem(last=False)  # Remove oldest
            self.cache[cache_key] = (sparsified_tokens, kept_indices, sparsification_info)
        
        return sparsified_tokens, kept_indices, sparsification_info

    def _generate_cache_key(self, tokens: torch.Tensor, target_tokens: int, layer_idx: Optional[int]) -> str:
        """Generate a cache key for the current input."""
        # Use a simplified hash based on tensor properties and target tokens
        tensor_hash = hash((tokens.shape[1], tokens.dtype, target_tokens, layer_idx or 0))
        return f"{tensor_hash}"

    def _top_k_sparsification(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """Select top-k tokens based on their L2 norm."""
        # Calculate L2 norm for each token
        norms = torch.norm(tokens, p=2, dim=-1)  # Shape: (batch_size, seq_len)
        
        # Get top-k indices
        _, top_k_indices = torch.topk(norms, k=min(k, norms.size(1)), dim=-1, largest=True)
        
        return top_k_indices

    def _magnitude_sparsification(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """Select tokens based on the magnitude of their elements."""
        # Calculate magnitude as sum of absolute values
        magnitudes = torch.sum(torch.abs(tokens), dim=-1)  # Shape: (batch_size, seq_len)
        
        # Get top-k indices
        _, top_k_indices = torch.topk(magnitudes, k=min(k, magnitudes.size(1)), dim=-1, largest=True)
        
        return top_k_indices

    def _attention_sparsification(self, attention_weights: torch.Tensor, k: int) -> torch.Tensor:
        """Select tokens based on attention weights."""
        # Average attention weights across heads and normalize
        if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
            avg_attention = torch.mean(attention_weights, dim=1)  # Average across heads
        else:  # (batch, seq, seq)
            avg_attention = attention_weights
            
        # Sum attention weights for each token (importance score)
        importance_scores = torch.sum(avg_attention, dim=-1)  # Shape: (batch_size, seq_len)
        
        # Get top-k indices
        _, top_k_indices = torch.topk(importance_scores, k=min(k, importance_scores.size(1)), dim=-1, largest=True)
        
        return top_k_indices

    def _random_sparsification(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """Randomly select tokens."""
        batch_size, seq_len = tokens.shape[:2]
        
        # Generate random indices
        random_indices = torch.rand(batch_size, seq_len, device=tokens.device)
        _, top_k_indices = torch.topk(random_indices, k=min(k, seq_len), dim=-1, largest=True)
        
        return top_k_indices

    def _calculate_text_complexity(self, text_features: torch.Tensor) -> float:
        """Calculate text complexity as a normalized value between 0 and 1."""
        # This is a simplified approach - in practice, you might use more sophisticated metrics
        # like sequence length, vocabulary diversity, or semantic complexity
        if text_features.numel() == 0:
            return 0.5  # Default to medium complexity
        
        # Use the variance of text features as a proxy for complexity
        complexity = torch.var(text_features.float())
        complexity = torch.clamp(complexity, 0, 10.0)  # Clamp to reasonable range
        normalized_complexity = min(1.0, complexity / 5.0)  # Normalize to 0-1
        
        return normalized_complexity.item()


class SparseVisionEncoder(nn.Module):
    """
    Vision encoder with integrated sparsification for visual tokens.
    """
    def __init__(self, original_vision_encoder: nn.Module, config: SparsificationConfig):
        super().__init__()
        self.original_encoder = original_vision_encoder
        self.sparsifier = TokenSparsifier(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        pixel_values: torch.Tensor,
        sparsity_temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with visual token sparsification.
        
        Args:
            pixel_values: Input pixel values
            sparsity_temperature: Temperature for soft sparsification
            
        Returns:
            Tuple of (sparsified_features, sparsification_info)
        """
        # Pass through original encoder
        if hasattr(self.original_encoder, 'forward'):
            original_features = self.original_encoder(pixel_values, **kwargs)
        else:
            # If original encoder doesn't have a forward method, assume it's a callable
            original_features = self.original_encoder(pixel_values)
        
        # Apply sparsification if the output is a tensor of tokens
        if isinstance(original_features, torch.Tensor):
            if original_features.dim() == 3:  # (batch, seq_len, hidden_dim)
                # Apply sparsification
                sparsity_temp = sparsity_temperature or self.config.sparsity_temperature
                sparsified_features, kept_indices, sparsification_info = self.sparsifier(
                    original_features,
                    layer_idx=0  # For vision encoder, layer_idx is 0
                )
                
                return sparsified_features, sparsification_info
            else:
                # If not a sequence of tokens, return original features
                return original_features, {'sparsity_applied': False}
        else:
            # If original_features is not a tensor, return as is
            return original_features, {'sparsity_applied': False}


class SparseCrossAttention(nn.Module):
    """
    Cross-attention module with sparsification for visual tokens.
    """
    def __init__(self, original_attention: nn.Module, config: SparsificationConfig):
        super().__init__()
        self.original_attention = original_attention
        self.sparsifier = TokenSparsifier(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with visual token sparsification in cross-attention.
        """
        # Apply sparsification to encoder_hidden_states (visual tokens) if provided
        if encoder_hidden_states is not None and self.config.apply_to_cross_attention:
            # Apply sparsification to visual tokens
            sparsified_visual_tokens, kept_indices, sparsification_info = self.sparsifier(
                encoder_hidden_states,
                layer_idx=kwargs.get('layer_idx', 0),
                text_features=hidden_states
            )

            # Use sparsified visual tokens in attention
            encoder_hidden_states = sparsified_visual_tokens
        else:
            sparsification_info = {'sparsity_applied': False}
            kept_indices = None

        # Apply original attention with (potentially) sparsified visual tokens
        if hasattr(self.original_attention, 'forward'):
            output = self.original_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            # If original attention doesn't have a forward method, assume it's a callable
            output = self.original_attention(
                hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )

        # Ensure consistent output format regardless of original attention return type
        if isinstance(output, tuple):
            if len(output) >= 3:
                return output[0], output[1], output[2]  # Return first 3 elements
            elif len(output) == 2:
                return output[0], output[1], None  # Add None for third element
            else:
                return output[0], None, None  # Add None for second and third elements
        else:
            # If output is not a tuple, return it as first element and None for others
            return output, None, None


class VisualTokenSparsifier:
    """
    Main class for applying visual token sparsification to the Qwen3-VL model.
    """
    def __init__(self, sparsification_config: SparsificationConfig = None):
        self.config = sparsification_config or SparsificationConfig()
        self.logger = logging.getLogger(__name__)

    def apply_sparsification_to_model(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply visual token sparsification to the Qwen3-VL model.
        
        Args:
            model: The Qwen3-VL model to apply sparsification to
            
        Returns:
            Tuple of (sparsified_model, sparsification_components)
        """
        self.logger.info("Applying visual token sparsification to the model...")
        
        # Apply sparsification to vision encoder if specified
        if self.config.apply_to_vision_encoder:
            self._apply_to_vision_encoder(model)
        
        # Apply sparsification to cross-attention layers if specified
        if self.config.apply_to_cross_attention:
            self._apply_to_cross_attention(model)
        
        # Create sparsification components
        sparsification_components = {
            'config': self.config,
            'sparsity_ratio': self.config.sparsity_ratio,
            'sparsity_method': self.config.sparsity_method,
            'apply_to_vision_encoder': self.config.apply_to_vision_encoder,
            'apply_to_cross_attention': self.config.apply_to_cross_attention,
            'apply_to_self_attention': self.config.apply_to_self_attention
        }
        
        self.logger.info("Visual token sparsification applied successfully!")
        return model, sparsification_components

    def _apply_to_vision_encoder(self, model: nn.Module):
        """Apply sparsification to the vision encoder."""
        # Look for vision encoder in the model
        if hasattr(model, 'vision_model') or hasattr(model, 'visual'):
            # Try different possible attribute names for the vision encoder
            vision_encoder = None
            if hasattr(model, 'vision_model'):
                vision_encoder = model.vision_model
            elif hasattr(model, 'visual'):
                vision_encoder = model.visual
            elif hasattr(model, 'get_vision_encoder') and callable(getattr(model, 'get_vision_encoder')):
                vision_encoder = model.get_vision_encoder()
            
            if vision_encoder is not None:
                # Replace the original vision encoder with a sparse version
                sparse_vision_encoder = SparseVisionEncoder(vision_encoder, self.config)
                
                # Replace the encoder in the model
                if hasattr(model, 'vision_model'):
                    model.vision_model = sparse_vision_encoder
                elif hasattr(model, 'visual'):
                    model.visual = sparse_vision_encoder
                elif hasattr(model, 'get_vision_encoder'):
                    # For models with getter methods, we might need a different approach
                    setattr(model, 'vision_model', sparse_vision_encoder)
                
                self.logger.info("Sparsification applied to vision encoder")
            else:
                self.logger.warning("Vision encoder not found in the model")
        else:
            self.logger.warning("No vision encoder found in the model")

    def _apply_to_cross_attention(self, model: nn.Module):
        """Apply sparsification to cross-attention layers."""
        # This is a general approach - the exact implementation depends on the model structure
        # Look for layers that contain cross-attention mechanisms
        for name, module in model.named_modules():
            # Look for attention modules that might be cross-attention
            if any(attn_type in name.lower() for attn_type in ['attn', 'attention']) and not any(skip_type in name.lower() for skip_type in ['self', 'encoder']):
                # This is a heuristic - in practice, you'd need to identify actual cross-attention layers
                if hasattr(module, 'forward') and hasattr(module, 'q_proj'):
                    # This looks like a self-attention layer, skip if we're not applying to self-attention
                    if not self.config.apply_to_self_attention:
                        continue
                
                # Replace with sparse attention (this is a simplified approach)
                sparse_attention = SparseCrossAttention(module, self.config)
                
                # Find the parent module and replace the attention module
                *parent_names, module_name = name.split('.')
                parent_module = model
                for parent_name in parent_names:
                    parent_module = getattr(parent_module, parent_name)
                
                setattr(parent_module, module_name, sparse_attention)
                
                self.logger.info(f"Sparsification applied to attention module: {name}")

    def calculate_sparsity_metrics(
        self,
        original_tokens: torch.Tensor,
        sparsified_tokens: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate metrics for the sparsification effect.
        
        Args:
            original_tokens: Original tokens before sparsification
            sparsified_tokens: Tokens after sparsification
            
        Returns:
            Dictionary with sparsity metrics
        """
        original_size = original_tokens.numel() * original_tokens.element_size()
        sparsified_size = sparsified_tokens.numel() * sparsified_tokens.element_size()
        
        reduction_ratio = 1.0 - (sparsified_tokens.size(1) / original_tokens.size(1))
        memory_reduction = (original_size - sparsified_size) / original_size * 100
        
        return {
            'token_reduction_ratio': reduction_ratio,
            'memory_reduction_percent': memory_reduction,
            'original_token_count': original_tokens.size(1),
            'sparsified_token_count': sparsified_tokens.size(1),
            'compression_ratio': original_tokens.size(1) / sparsified_tokens.size(1)
        }

    def benchmark_sparsification_impact(
        self,
        original_model: nn.Module,
        sparsified_model: nn.Module,
        test_data_loader
    ) -> Dict[str, Any]:
        """
        Benchmark the impact of sparsification on model performance.
        
        Args:
            original_model: The original model
            sparsified_model: The sparsified model
            test_data_loader: DataLoader with test data
            
        Returns:
            Dictionary with performance metrics
        """
        # Set both models to evaluation mode
        original_model.eval()
        sparsified_model.eval()
        
        # Track performance metrics
        original_times = []
        sparsified_times = []
        original_memory = []
        sparsified_memory = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                if i >= 10:  # Limit to 10 batches for quick benchmarking
                    break
                
                # Benchmark original model
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                else:
                    import time
                    start_time_cpu = time.time()
                
                try:
                    if isinstance(batch, dict):
                        _ = original_model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = original_model(*batch)
                    else:
                        _ = original_model(batch)
                except Exception as e:
                    self.logger.warning(f"Original model benchmark failed for batch {i}: {e}")
                    continue
                
                if start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    original_time = start_time.elapsed_time(end_time)
                else:
                    end_time_cpu = time.time()
                    original_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to milliseconds
                
                original_times.append(original_time)
                
                # Benchmark sparsified model
                if start_time:
                    start_time.record()
                
                try:
                    if isinstance(batch, dict):
                        _ = sparsified_model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        _ = sparsified_model(*batch)
                    else:
                        _ = sparsified_model(batch)
                except Exception as e:
                    self.logger.warning(f"Sparsified model benchmark failed for batch {i}: {e}")
                    continue
                
                if start_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    sparsified_time = start_time.elapsed_time(end_time)
                else:
                    start_time_cpu = time.time()
                    _ = sparsified_model(batch)
                    end_time_cpu = time.time()
                    sparsified_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to milliseconds
                
                sparsified_times.append(sparsified_time)
        
        # Calculate metrics
        avg_original_time = np.mean(original_times) if original_times else 0
        avg_sparsified_time = np.mean(sparsified_times) if sparsified_times else 0
        speedup = avg_original_time / avg_sparsified_time if avg_sparsified_time > 0 else float('inf')
        
        # Calculate model sizes
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024**2)  # MB
        sparsified_size = sum(p.numel() * p.element_size() for p in sparsified_model.parameters()) / (1024**2)  # MB
        size_reduction = (original_size - sparsified_size) / original_size * 100 if original_size > 0 else 0
        
        return {
            'original_avg_time_ms': avg_original_time,
            'sparsified_avg_time_ms': avg_sparsified_time,
            'speedup': speedup,
            'original_model_size_mb': original_size,
            'sparsified_model_size_mb': sparsified_size,
            'size_reduction_percent': size_reduction,
            'num_test_batches': len(original_times)
        }


def apply_visual_token_sparsification(
    model: nn.Module,
    config: Optional[SparsificationConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply visual token sparsification to the Qwen3-VL model.

    Args:
        model: The Qwen3-VL model to apply sparsification to
        config: Configuration for sparsification (optional)

    Returns:
        Tuple of (sparsified_model, sparsification_info)
    """
    logger = logging.getLogger(__name__)
    logger.info("Applying visual token sparsification to the Qwen3-VL model...")

    # Use default config if none provided
    if config is None:
        config = SparsificationConfig()

    # Initialize the sparsifier
    sparsifier = VisualTokenSparsifier(config)

    # Apply sparsification
    sparsified_model, sparsification_components = sparsifier.apply_sparsification_to_model(model)

    logger.info("Visual token sparsification applied successfully!")
    return sparsified_model, sparsification_components


if __name__ == "__main__":
    print("Visual Token Sparsification for Qwen3-VL Model")
    print("=" * 60)
    print("This module implements SparseVLM-inspired visual token sparsification")
    print("for CPU optimization targeting Intel i5-10210U architecture")
    print("=" * 60)
    
    # Example usage
    config = SparsificationConfig()
    print(f"Default sparsification config: {config}")