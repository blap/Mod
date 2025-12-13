"""Rotary Embeddings for Qwen3-VL model with hardware-specific optimizations."""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class Qwen3VLRotaryEmbedding(nn.Module):
    """Rotary Embedding implementation for Qwen3-VL model."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        if position_ids is None:
            seq_len = x.shape[2]  # Assuming x is [batch, heads, seq_len, dim]
        else:
            seq_len = position_ids.shape[-1]
            
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len

        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, log is taken first then outer product is taken
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OptimizedRotaryEmbedding(nn.Module):
    """Hardware-optimized rotary embedding for faster computation."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, strategy="cached"):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.strategy = strategy
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cached embeddings if strategy is cached
        if strategy == "cached":
            self.register_buffer(
                "cached_cos", 
                torch.cos(torch.arange(max_position_embeddings).unsqueeze(1) * inv_freq).unsqueeze(0).unsqueeze(0)
            )
            self.register_buffer(
                "cached_sin", 
                torch.sin(torch.arange(max_position_embeddings).unsqueeze(1) * inv_freq).unsqueeze(0).unsqueeze(0)
            )
        else:
            self.cached_cos = None
            self.cached_sin = None

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.strategy == "cached" and position_ids is not None:
            # Use cached embeddings if available and position_ids are provided
            max_pos = torch.max(position_ids).item() + 1
            
            if max_pos > self.max_position_embeddings:
                # Extend cached embeddings if needed
                self._extend_cached_embeddings(max_pos)
                
            # Get embeddings for the required positions
            cos = self.cached_cos[:, :, :max_pos, :].expand(position_ids.shape[0], -1, -1, -1)
            sin = self.cached_sin[:, :, :max_pos, :].expand(position_ids.shape[0], -1, -1, -1)
            
            # Select embeddings for the specific positions
            batch_indices = torch.arange(position_ids.shape[0]).unsqueeze(1).expand(-1, position_ids.shape[1])
            pos_indices = position_ids
            cos = cos[batch_indices, :, pos_indices, :]
            sin = sin[batch_indices, :, pos_indices, :]
            
            return cos, sin
        else:
            # Standard computation
            seq_len = x.shape[2] if position_ids is None else position_ids.max().item() + 1
            
            if seq_len > self.max_position_embeddings:
                self._extend_cached_embeddings(seq_len)
            
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
            if position_ids is not None:
                # Select embeddings for the specific positions
                cos = cos[position_ids]
                sin = sin[position_ids]
            
            return cos, sin
    
    def _extend_cached_embeddings(self, new_max_pos):
        """Extend cached embeddings to accommodate larger position IDs."""
        if self.strategy == "cached":
            old_max = self.max_position_embeddings
            self.max_position_embeddings = new_max_pos
            
            # Compute new embeddings
            new_inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            new_t = torch.arange(old_max, new_max_pos, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            new_freqs = torch.outer(new_t, new_inv_freq)
            new_emb = torch.cat((new_freqs, new_freqs), dim=-1)
            new_cos = new_emb.cos()
            new_sin = new_emb.sin()
            
            # Extend cached embeddings
            self.cached_cos = torch.cat([self.cached_cos, new_cos.unsqueeze(0).unsqueeze(0)], dim=2)
            self.cached_sin = torch.cat([self.cached_sin, new_sin.unsqueeze(0).unsqueeze(0)], dim=2)
            
            # Update inv_freq if base changed
            self.inv_freq = new_inv_freq.to(self.inv_freq.device)


class ApproximatedRotaryEmbedding(nn.Module):
    """Approximated rotary embedding for faster computation on specific hardware."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, approximation_factor=0.9):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.approximation_factor = approximation_factor
        
        # Compute inverse frequencies with potential approximation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        seq_len = x.shape[2] if position_ids is None else position_ids.max().item() + 1
        
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len

        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # Apply approximation if needed
        if self.approximation_factor < 1.0:
            # Apply a simplification to reduce computation
            freqs = freqs * self.approximation_factor
        
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        if position_ids is not None:
            # Select embeddings for the specific positions
            cos = cos[position_ids]
            sin = sin[position_ids]
        
        return cos, sin


class CachedRotaryEmbedding(nn.Module):
    """Fully cached rotary embedding for maximum performance."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Pre-compute all embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(max_position_embeddings, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        self.register_buffer("cached_cos", cos.unsqueeze(0).unsqueeze(0))
        self.register_buffer("cached_sin", sin.unsqueeze(0).unsqueeze(0))

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids is None:
            seq_len = x.shape[2]
            cos = self.cached_cos[:, :, :seq_len, :]
            sin = self.cached_sin[:, :, :seq_len, :]
        else:
            # Select embeddings for the specific positions
            cos = self.cached_cos[:, :, position_ids, :]
            sin = self.cached_sin[:, :, position_ids, :]
        
        return cos, sin


class InterpolatedRotaryEmbedding(nn.Module):
    """Rotary embedding with position interpolation for extended sequence lengths."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Scale position_ids for extrapolation
        scaled_position_ids = position_ids.float() / self.scale
        
        seq_len = x.shape[2] if position_ids is None else scaled_position_ids.max().item() + 1
        
        if seq_len > self.max_position_embeddings:
            # Use extrapolation beyond max position
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        else:
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        if position_ids is not None:
            # Select embeddings for the specific positions
            cos = cos[scaled_position_ids.long()]
            sin = sin[scaled_position_ids.long()]
        
        return cos, sin


class RotaryEmbeddingOptimizer:
    """Optimizer for selecting the best rotary embedding strategy based on hardware."""
    
    @staticmethod
    def create_embedding(strategy: str, dim: int, max_position_embeddings: int = 2048, base: int = 10000, **kwargs):
        """
        Create the appropriate rotary embedding based on the strategy.
        
        Args:
            strategy: Strategy to use ('standard', 'approximated', 'cached', 'interpolated', 'optimized')
            dim: Dimension of the embeddings
            max_position_embeddings: Maximum position embeddings
            base: Base value for frequency computation
            **kwargs: Additional parameters
            
        Returns:
            Appropriate rotary embedding instance
        """
        if strategy == 'standard':
            return Qwen3VLRotaryEmbedding(dim, max_position_embeddings, base)
        elif strategy == 'approximated':
            approximation_factor = kwargs.get('approximation_factor', 0.9)
            return ApproximatedRotaryEmbedding(dim, max_position_embeddings, base, approximation_factor)
        elif strategy == 'cached':
            return CachedRotaryEmbedding(dim, max_position_embeddings, base)
        elif strategy == 'interpolated':
            scale = kwargs.get('scale', 1.0)
            return InterpolatedRotaryEmbedding(dim, max_position_embeddings, base, scale)
        elif strategy == 'optimized':
            return OptimizedRotaryEmbedding(dim, max_position_embeddings, base, kwargs.get('optimization_strategy', 'cached'))
        else:
            # Default to standard implementation
            logger.warning(f"Unknown rotary embedding strategy '{strategy}', using standard implementation")
            return Qwen3VLRotaryEmbedding(dim, max_position_embeddings, base)
    
    @staticmethod
    def get_best_strategy_for_hardware(hardware_info: dict) -> str:
        """
        Determine the best rotary embedding strategy based on hardware capabilities.
        
        Args:
            hardware_info: Dictionary containing hardware information
            
        Returns:
            Best strategy for the given hardware
        """
        # Default to optimized strategy
        strategy = "optimized"
        
        # Check for specific hardware capabilities
        if 'gpu' in hardware_info:
            gpu_info = hardware_info['gpu']
            if gpu_info.get('compute_capability', (0, 0))[0] >= 8:  # Modern GPU
                strategy = "cached"  # Can afford the memory for caching
            elif gpu_info.get('memory_gb', 0) < 8:  # Limited memory
                strategy = "approximated"  # Use approximation to save memory
            else:
                strategy = "optimized"  # Use optimized implementation
        
        return strategy


# Export the main classes and functions
__all__ = [
    "Qwen3VLRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "OptimizedRotaryEmbedding",
    "ApproximatedRotaryEmbedding", 
    "CachedRotaryEmbedding",
    "InterpolatedRotaryEmbedding",
    "RotaryEmbeddingOptimizer"
]