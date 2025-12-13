"""
KV cache optimization system for Qwen3-VL model.

This module implements various KV cache optimization strategies including
low-rank approximation, sliding window, and hybrid approaches.
"""
import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class KVCacheState:
    """State of the KV cache for a specific layer."""
    key_states: torch.Tensor
    value_states: torch.Tensor
    compressed: bool = False
    compression_info: Optional[Dict[str, Any]] = None


class KVCacheOptimizer:
    """
    KV cache optimization system with multiple strategies.
    """
    
    def __init__(self, strategy: str = "hybrid", use_low_rank: bool = True,
                 window_size: int = 1024, low_rank_dim: int = 64, 
                 max_length: int = 32768):
        """
        Initialize the KV cache optimizer.
        
        Args:
            strategy: Strategy to use ("low_rank", "sliding_window", "hybrid")
            use_low_rank: Whether to use low-rank approximation
            window_size: Size of the sliding window
            low_rank_dim: Dimension for low-rank approximation
            max_length: Maximum sequence length
        """
        self.strategy = strategy
        self.use_low_rank = use_low_rank
        self.window_size = window_size
        self.low_rank_dim = low_rank_dim
        self.max_length = max_length
        
        # Cache states for each layer
        self.layer_caches: Dict[int, KVCacheState] = {}
    
    def optimize(self, key_states: torch.Tensor, value_states: torch.Tensor, 
                 layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize KV cache storage using the configured strategy.
        
        Args:
            key_states: Key states to optimize
            value_states: Value states to optimize
            layer_idx: Index of the transformer layer
            
        Returns:
            Optimized key and value states
        """
        if self.strategy == "low_rank" or (self.strategy == "hybrid" and self.use_low_rank):
            return self._apply_low_rank_optimization(key_states, value_states, layer_idx)
        elif self.strategy == "sliding_window":
            return self._apply_sliding_window_optimization(key_states, value_states, layer_idx)
        elif self.strategy == "hybrid":
            # Use a combination of both strategies based on sequence length
            seq_len = key_states.size(-2)
            if seq_len > self.window_size:
                # Use sliding window for long sequences
                k_opt, v_opt = self._apply_sliding_window_optimization(key_states, value_states, layer_idx)
                # Then apply low-rank if beneficial
                if self.use_low_rank and k_opt.size(-1) > self.low_rank_dim:
                    return self._apply_low_rank_optimization(k_opt, v_opt, layer_idx)
                return k_opt, v_opt
            else:
                # Use low-rank for shorter sequences if enabled
                if self.use_low_rank:
                    return self._apply_low_rank_optimization(key_states, value_states, layer_idx)
                else:
                    return key_states, value_states
        else:
            # No optimization
            return key_states, value_states
    
    def _apply_low_rank_optimization(self, key_states: torch.Tensor, 
                                   value_states: torch.Tensor, 
                                   layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply low-rank approximation to KV cache states.
        
        Args:
            key_states: Original key states
            value_states: Original value states
            layer_idx: Index of the transformer layer
            
        Returns:
            Low-rank approximated key and value states
        """
        if key_states.size(-1) <= self.low_rank_dim:
            # Dimension is already small enough
            return key_states, value_states
        
        # Apply SVD-based low-rank approximation
        # This is a simplified implementation - in practice, you'd want to use 
        # more sophisticated techniques like incremental SVD
        try:
            # Reshape to 3D: (batch, seq_len, embed_dim) -> (batch*seq_len, embed_dim)
            batch_size, num_heads, seq_len, embed_dim = key_states.shape
            k_reshaped = key_states.view(-1, embed_dim)
            v_reshaped = value_states.view(-1, embed_dim)
            
            # Apply SVD for low-rank approximation
            U_k, S_k, Vh_k = torch.svd(k_reshaped)
            U_v, S_v, Vh_v = torch.svd(v_reshaped)
            
            # Keep only the top low_rank_dim components
            U_k_low = U_k[:, :self.low_rank_dim]
            S_k_low = S_k[:self.low_rank_dim]
            Vh_k_low = Vh_k[:self.low_rank_dim, :]
            
            U_v_low = U_v[:, :self.low_rank_dim]
            S_v_low = S_v[:self.low_rank_dim]
            Vh_v_low = Vh_v[:self.low_rank_dim, :]
            
            # Reconstruct low-rank approximations
            k_low_rank = torch.mm(U_k_low * S_k_low.unsqueeze(0), Vh_k_low)
            v_low_rank = torch.mm(U_v_low * S_v_low.unsqueeze(0), Vh_v_low)
            
            # Reshape back to original shape
            k_opt = k_low_rank.view(batch_size, num_heads, seq_len, self.low_rank_dim)
            v_opt = v_low_rank.view(batch_size, num_heads, seq_len, self.low_rank_dim)
            
            # Store compression info
            compression_info = {
                "original_dim": embed_dim,
                "compressed_dim": self.low_rank_dim,
                "compression_ratio": self.low_rank_dim / embed_dim
            }
            
            # Update cache state
            self.layer_caches[layer_idx] = KVCacheState(
                key_states=k_opt, 
                value_states=v_opt, 
                compressed=True,
                compression_info=compression_info
            )
            
            return k_opt, v_opt
        except:
            # If SVD fails, return original tensors
            return key_states, value_states
    
    def _apply_sliding_window_optimization(self, key_states: torch.Tensor, 
                                         value_states: torch.Tensor, 
                                         layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sliding window optimization to KV cache states.
        
        Args:
            key_states: Original key states
            value_states: Original value states
            layer_idx: Index of the transformer layer
            
        Returns:
            Sliding window optimized key and value states
        """
        seq_len = key_states.size(-2)
        
        if seq_len <= self.window_size:
            # No need for sliding window if sequence is short enough
            return key_states, value_states
        
        # Keep only the most recent window_size tokens
        k_opt = key_states[..., -self.window_size:, :]
        v_opt = value_states[..., -self.window_size:, :]
        
        # Store compression info
        compression_info = {
            "original_seq_len": seq_len,
            "compressed_seq_len": self.window_size,
            "compression_ratio": self.window_size / seq_len
        }
        
        # Update cache state
        self.layer_caches[layer_idx] = KVCacheState(
            key_states=k_opt, 
            value_states=v_opt, 
            compressed=True,
            compression_info=compression_info
        )
        
        return k_opt, v_opt
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get KV cache optimization statistics.
        
        Returns:
            Dictionary containing optimization statistics
        """
        total_layers = len(self.layer_caches)
        compressed_layers = sum(1 for cache in self.layer_caches.values() if cache.compressed)
        
        compression_ratios = []
        for cache in self.layer_caches.values():
            if cache.compression_info:
                compression_ratios.append(cache.compression_info.get("compression_ratio", 1.0))
        
        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
        
        return {
            "strategy": self.strategy,
            "total_layers_cached": total_layers,
            "compressed_layers": compressed_layers,
            "average_compression_ratio": avg_compression_ratio,
            "window_size": self.window_size,
            "low_rank_dimension": self.low_rank_dim
        }
    
    def clear_cache(self, layer_idx: Optional[int] = None):
        """
        Clear the KV cache for a specific layer or all layers.
        
        Args:
            layer_idx: Index of the layer to clear, or None to clear all
        """
        if layer_idx is not None:
            if layer_idx in self.layer_caches:
                del self.layer_caches[layer_idx]
        else:
            self.layer_caches.clear()