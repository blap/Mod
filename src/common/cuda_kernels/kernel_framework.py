"""
Standardized CUDA Kernel Framework

This module provides a standardized framework for implementing CUDA kernels across different models.
It defines common interfaces, utilities, and base classes for CUDA kernel implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseCUDAKernel(nn.Module, ABC):
    """
    Abstract base class for all CUDA kernels.
    Defines the standard interface that all kernels must implement.
    """
    
    def __init__(self):
        super().__init__()
        self.kernel_name = self.__class__.__name__
        self.hardware_optimizer = CUDAHardwareOptimizer()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass implementation for the kernel.
        Must be implemented by subclasses.
        """
        pass
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get optimization report for this kernel.
        """
        return {
            "kernel_name": self.kernel_name,
            "hardware_info": self.hardware_optimizer.get_hardware_info(),
            "optimization_level": self.hardware_optimizer.get_optimization_level(),
            "tensor_cores_supported": self.hardware_optimizer.tensor_cores_supported,
        }


class CUDAHardwareOptimizer:
    """
    Hardware-specific optimizer that detects capabilities and applies appropriate optimizations.
    """
    
    def __init__(self):
        self.compute_capability = self._get_compute_capability()
        self.tensor_cores_supported = self._check_tensor_cores_support()
        self.optimization_level = self._determine_optimization_level()
        
    def _get_compute_capability(self) -> Tuple[int, int]:
        """Get the compute capability of the current GPU."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return torch.cuda.get_device_capability(device)
        return (0, 0)  # CPU fallback
        
    def _check_tensor_cores_support(self) -> bool:
        """Check if Tensor Cores are supported."""
        major, minor = self.compute_capability
        return major >= 7  # Tensor Cores available from Volta (7.0) onwards
        
    def _determine_optimization_level(self) -> str:
        """Determine the optimization level based on hardware."""
        major, minor = self.compute_capability
        if major >= 8:  # Ampere and later
            return "high"
        elif major >= 7:  # Volta, Turing
            return "medium"
        else:  # Older architectures
            return "basic"
            
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get detailed hardware information."""
        return {
            "compute_capability": self.compute_capability,
            "tensor_cores_supported": self.tensor_cores_supported,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "total_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        }
        
    def get_optimization_level(self) -> str:
        """Get the optimization level."""
        return self.optimization_level


class AttentionKernel(BaseCUDAKernel):
    """
    Standardized attention kernel interface.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        causal: bool = True,
        kv_groups: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_rate = dropout
        self.use_flash_attention = use_flash_attention
        self.causal = causal
        self.kv_groups = kv_groups  # For Grouped Query Attention
        
        # Initialize dropout if needed
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Create causal mask if needed
        if causal:
            self.register_buffer(
                "causal_mask", 
                torch.triu(torch.ones(1, 1, 1024, 1024, dtype=torch.bool), diagonal=1),
                persistent=False
            )
        
        # Scaling factor
        self.scaling = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for attention kernel.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Reshape to multi-head format
        q = query.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scale queries
        q = q * self.scaling
        
        # Compute attention scores
        if self.use_flash_attention and torch.cuda.is_available():
            # Use efficient attention computation (would use actual FlashAttention in real implementation)
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply causal mask if needed
            if self.causal and seq_len <= 1024:  # Use precomputed mask if sequence is not too long
                causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
                attn_weights.masked_fill_(causal_mask, float("-inf"))
                
            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand attention mask to match attention weights dimensions
                if attention_mask.dim() == 2:
                    # Shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    # Shape: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.dim() == 4:
                    # Already in the right format (batch, nhead, seq_len, seq_len)
                    pass
                else:
                    raise ValueError(
                        f"Invalid attention mask dimension: {attention_mask.dim()}"
                    )
                    
                # Expand to match number of heads if needed
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.expand(-1, self.nhead, -1, -1)
                    
                attn_weights = attn_weights + attention_mask
                
            # Apply softmax
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            
            # Apply dropout if configured
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)
                
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Standard attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply causal mask if needed
            if self.causal and seq_len <= 1024:
                causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
                attn_weights.masked_fill_(causal_mask, float("-inf"))
                
            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand attention mask to match attention weights dimensions
                if attention_mask.dim() == 2:
                    # Shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    # Shape: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.dim() == 4:
                    # Already in the right format (batch, nhead, seq_len, seq_len)
                    pass
                else:
                    raise ValueError(
                        f"Invalid attention mask dimension: {attention_mask.dim()}"
                    )
                    
                # Expand to match number of heads if needed
                if attention_mask.size(1) == 1:
                    attention_mask = attention_mask.expand(-1, self.nhead, -1, -1)
                    
                attn_weights = attn_weights + attention_mask
                
            # Apply softmax
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            
            # Apply dropout if configured
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)
                
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
        # Reshape to combine heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        
        # Return output and attention weights if needed
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None


class LinearProjectionKernel(BaseCUDAKernel):
    """
    Standardized linear projection kernel.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for linear projection.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return self.linear(x)
        
    def get_weight(self) -> torch.Tensor:
        """Get the weight tensor."""
        return self.linear.weight
        
    def get_bias(self) -> Optional[torch.Tensor]:
        """Get the bias tensor."""
        return self.linear.bias


class ActivationKernel(BaseCUDAKernel):
    """
    Standardized activation kernel interface.
    """
    
    def __init__(self, activation_type: str = "silu"):
        super().__init__()
        self.activation_type = activation_type
        self.activation_fn = self._get_activation_function(activation_type)
        
    def _get_activation_function(self, activation_type: str):
        """Get the activation function based on type."""
        if activation_type.lower() == "silu":
            return torch.nn.functional.silu
        elif activation_type.lower() == "gelu":
            return torch.nn.functional.gelu
        elif activation_type.lower() == "relu":
            return torch.nn.functional.relu
        elif activation_type.lower() == "swiglu":
            # SwiGLU is handled separately as it involves two projections
            return lambda x: torch.nn.functional.silu(x) * x
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for activation kernel.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return self.activation_fn(x)


class KVCacheKernel(BaseCUDAKernel):
    """
    Standardized KV cache kernel for managing key-value caches efficiently.
    """
    
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate cache tensors
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        self.value_cache = torch.zeros(cache_shape, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Track current sequence length per batch
        self.current_seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
        
    def update_cache(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        batch_idx: int, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache with new keys and values.
        
        Args:
            keys: New key tensors of shape (seq_len, num_heads, head_dim)
            values: New value tensors of shape (seq_len, num_heads, head_dim)
            batch_idx: Index of the batch to update
            position_ids: Position IDs indicating where to store the new values
            
        Returns:
            Updated key and value tensors including cached values
        """
        # Update cache with new values
        self.key_cache[batch_idx, :, position_ids, :] = keys.transpose(0, 1)
        self.value_cache[batch_idx, :, position_ids, :] = values.transpose(0, 1)
        
        # Update current sequence length
        self.current_seq_lens[batch_idx] = position_ids.max() + 1
        
        # Return the full cached keys and values up to current position
        current_len = self.current_seq_lens[batch_idx].item()
        full_keys = self.key_cache[batch_idx, :, :current_len, :]
        full_values = self.value_cache[batch_idx, :, :current_len, :]
        
        return full_keys.transpose(0, 1), full_values.transpose(0, 1)
        
    def reset_cache(self, batch_idx: Optional[int] = None):
        """
        Reset the KV cache for a specific batch or all batches.
        
        Args:
            batch_idx: Index of the batch to reset. If None, reset all batches.
        """
        if batch_idx is not None:
            self.key_cache[batch_idx] = 0
            self.value_cache[batch_idx] = 0
            self.current_seq_lens[batch_idx] = 0
        else:
            self.key_cache.zero_()
            self.value_cache.zero_()
            self.current_seq_lens.zero_()


class NormalizationKernel(BaseCUDAKernel):
    """
    Standardized normalization kernel interface.
    Supports both LayerNorm and RMSNorm.
    """
    
    def __init__(self, normalized_shape: int, norm_type: str = "rms", eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.norm_type = norm_type.lower()
        self.eps = eps
        
        if self.norm_type == "rms":
            # RMSNorm parameters
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        elif self.norm_type == "layer":
            # LayerNorm parameters
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for normalization kernel.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape as input
        """
        if self.norm_type == "rms":
            # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return x * self.weight
        elif self.norm_type == "layer":
            # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
            mean = x.mean(dim=-1, keepdim=True)
            variance = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) * torch.rsqrt(variance + self.eps)
            return x * self.weight + self.bias
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")


class MLPLayerKernel(BaseCUDAKernel):
    """
    Standardized MLP (Feed-Forward) layer kernel.
    Supports both standard FFN and SwiGLU variants.
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        activation_type: str = "silu",
        use_swiglu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.use_swiglu = use_swiglu
        self.dropout_rate = dropout
        
        if use_swiglu:
            # SwiGLU: gate_proj -> activation -> up_proj -> multiply -> down_proj
            self.gate_proj = LinearProjectionKernel(d_model, intermediate_size, bias=False)
            self.up_proj = LinearProjectionKernel(d_model, intermediate_size, bias=False)
            self.down_proj = LinearProjectionKernel(intermediate_size, d_model, bias=False)
            self.activation = ActivationKernel("silu")  # SwiGLU always uses SiLU
        else:
            # Standard FFN: up_proj -> activation -> down_proj
            self.up_proj = LinearProjectionKernel(d_model, intermediate_size)
            self.down_proj = LinearProjectionKernel(intermediate_size, d_model)
            self.activation = ActivationKernel(activation_type)
            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP layer kernel.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        if self.use_swiglu:
            # SwiGLU: SiLU(gate_proj(x)) * up_proj(x) -> down_proj
            gate = self.activation(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
        else:
            # Standard FFN: activation(up_proj(x)) -> down_proj
            x = self.activation(self.up_proj(x))
            
        x = self.down_proj(x)
        
        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


def create_standardized_cuda_kernels(
    d_model: int,
    nhead: int,
    intermediate_size: int,
    max_batch_size: int = 1,
    max_seq_len: int = 2048,
    use_flash_attention: bool = True,
    use_swiglu: bool = False,
    norm_type: str = "rms",
    activation_type: str = "silu",
) -> Dict[str, BaseCUDAKernel]:
    """
    Factory function to create standardized CUDA kernels.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        max_batch_size: Maximum batch size for KV cache
        max_seq_len: Maximum sequence length for KV cache
        use_flash_attention: Whether to use flash attention
        use_swiglu: Whether to use SwiGLU in MLP
        norm_type: Type of normalization ('rms' or 'layer')
        activation_type: Type of activation function
        
    Returns:
        Dictionary of created kernels
    """
    head_dim = d_model // nhead
    kernels = {}
    
    # Create attention kernel
    kernels["attention"] = AttentionKernel(
        d_model=d_model,
        nhead=nhead,
        use_flash_attention=use_flash_attention,
    )
    
    # Create MLP kernel
    kernels["mlp"] = MLPLayerKernel(
        d_model=d_model,
        intermediate_size=intermediate_size,
        use_swiglu=use_swiglu,
        activation_type=activation_type,
    )
    
    # Create normalization kernel
    kernels["norm"] = NormalizationKernel(
        normalized_shape=d_model,
        norm_type=norm_type,
    )
    
    # Create KV cache kernel
    kernels["kv_cache"] = KVCacheKernel(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_heads=nhead,
        head_dim=head_dim,
    )
    
    return kernels


def apply_standardized_cuda_optimizations_to_model(
    model: nn.Module,
    d_model: int,
    nhead: int,
    intermediate_size: int,
    config,
    model_type: str = "generic",
) -> nn.Module:
    """
    Apply standardized CUDA optimizations to the model.
    
    Args:
        model: The model to optimize
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        config: Model configuration
        model_type: Type of model for specific optimizations
        
    Returns:
        Optimized model
    """
    logger.info(f"Applying standardized CUDA optimizations for {model_type}...")
    
    # Create standardized kernels
    kernels = create_standardized_cuda_kernels(
        d_model=d_model,
        nhead=nhead,
        intermediate_size=intermediate_size,
        use_flash_attention=getattr(config, 'cuda_kernel_attention_enabled', True),
        use_swiglu=getattr(config, 'use_swiglu', False),
        norm_type=getattr(config, 'norm_type', 'rms'),
        activation_type=getattr(config, 'activation_type', 'silu'),
    )
    
    # Apply optimizations based on model architecture
    for name, module in model.named_modules():
        # Replace attention modules with standardized kernels
        if isinstance(module, nn.MultiheadAttention):
            parent_module = _get_parent_module(model, name.rsplit(".", 1)[0]) if "." in name else model
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            
            # Create attention kernel that mimics the original interface
            attn_kernel = AttentionKernel(
                d_model=d_model,
                nhead=nhead,
                use_flash_attention=getattr(config, 'cuda_kernel_attention_enabled', True),
            )
            
            setattr(parent_module, child_name, attn_kernel)
            logger.info(f"Replaced attention module {name} with standardized kernel")
            
        # Replace normalization modules with standardized kernels
        elif isinstance(module, nn.LayerNorm):
            parent_module = _get_parent_module(model, name.rsplit(".", 1)[0]) if "." in name else model
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            
            norm_kernel = NormalizationKernel(
                normalized_shape=module.normalized_shape[0],
                norm_type=getattr(config, 'norm_type', 'rms'),
                eps=module.eps,
            )
            
            # Copy parameters if available
            with torch.no_grad():
                norm_kernel.weight.copy_(module.weight)
                if hasattr(norm_kernel, 'bias') and hasattr(module, 'bias') and module.bias is not None:
                    norm_kernel.bias.copy_(module.bias)
                    
            setattr(parent_module, child_name, norm_kernel)
            logger.info(f"Replaced normalization module {name} with standardized kernel")
            
        # Replace MLP/feed-forward modules with standardized kernels
        elif hasattr(module, "forward") and any(
            name_part in name.lower()
            for name_part in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            parent_module = _get_parent_module(model, name.rsplit(".", 1)[0]) if "." in name else model
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            
            mlp_kernel = MLPLayerKernel(
                d_model=d_model,
                intermediate_size=intermediate_size,
                use_swiglu=getattr(config, 'use_swiglu', False),
                activation_type=getattr(config, 'activation_type', 'silu'),
            )
            
            setattr(parent_module, child_name, mlp_kernel)
            logger.info(f"Replaced MLP module {name} with standardized kernel")
    
    logger.info(f"Standardized CUDA optimizations applied to {model_type} successfully")
    return model


def _get_parent_module(model: nn.Module, parent_name: str) -> nn.Module:
    """
    Get parent module by name.
    
    Args:
        model: The model
        parent_name: Name of the parent module
        
    Returns:
        Parent module
    """
    parent_module = model
    for n in parent_name.split("."):
        if n:  # Skip empty strings
            parent_module = getattr(parent_module, n)
    return parent_module


__all__ = [
    "BaseCUDAKernel",
    "CUDAHardwareOptimizer",
    "AttentionKernel",
    "LinearProjectionKernel",
    "ActivationKernel",
    "KVCacheKernel",
    "NormalizationKernel",
    "MLPLayerKernel",
    "create_standardized_cuda_kernels",
    "apply_standardized_cuda_optimizations_to_model",
]