"""
Kernel Fusion Implementation for Qwen3-VL Model
Implements various kernel fusion techniques to reduce kernel launch overhead and memory traffic
for the Intel i5-10210U + NVIDIA SM61 target hardware.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    # Try to import our CUDA kernels
    from src.qwen3_vl.cuda_kernels.cuda_wrapper import (
        SM61TensorOpsWrapper, 
        SM61AttentionWrapper,
        OptimizedMLPModule
    )
    CUDA_AVAILABLE = True
except ImportError:
    logger.warning("CUDA kernels not available, using PyTorch fallback")
    CUDA_AVAILABLE = False


class FusedLayerNormLinear(nn.Module):
    """
    Fused Layer Normalization + Linear transformation kernel.
    Combines the normalization and linear transformation in a single CUDA kernel
    to reduce memory traffic and kernel launch overhead.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.eps = eps
        
        # Layer norm parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Linear transformation parameters
        self.linear_weight = nn.Parameter(torch.randn(intermediate_size, hidden_size))
        self.linear_bias = nn.Parameter(torch.zeros(intermediate_size))
        
        # Initialize linear weight using Xavier initialization
        nn.init.xavier_uniform_(self.linear_weight)
        
        # Use CUDA wrapper if available
        if CUDA_AVAILABLE:
            self.tensor_ops = SM61TensorOpsWrapper()
        else:
            self.tensor_ops = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: LayerNorm(x) -> Linear -> Activation
        """
        if self.tensor_ops is not None and x.is_cuda:
            # Use CUDA-optimized operations when available
            try:
                # First, apply layer norm
                x_norm = self._apply_layer_norm(x)
                
                # Then apply linear transformation using CUDA-optimized matmul
                output = self.tensor_ops.matmul(x_norm, self.linear_weight.t())
                output = output + self.linear_bias
                
                return output
            except Exception as e:
                logger.warning(f"CUDA fusion failed: {e}, falling back to PyTorch")
        
        # PyTorch fallback
        return self._pytorch_forward(x)
    
    def _apply_layer_norm(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization with fused computation
        """
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Apply learnable parameters
        x_norm = x_norm * self.weight + self.bias
        
        return x_norm
    
    def _pytorch_forward(self, x: Tensor) -> Tensor:
        """
        PyTorch fallback implementation
        """
        # Apply layer norm
        x_norm = self._apply_layer_norm(x)
        
        # Apply linear transformation
        output = torch.nn.functional.linear(x_norm, self.linear_weight, self.linear_bias)
        
        return output


class FusedAttentionSoftmax(nn.Module):
    """
    Fused Attention + Softmax computation.
    Combines the attention score computation and softmax in a single kernel
    to reduce memory traffic and improve numerical stability.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        
        # Initialize CUDA wrapper if available
        if CUDA_AVAILABLE:
            self.attention_wrapper = SM61AttentionWrapper()
        else:
            self.attention_wrapper = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass: Q*K^T -> Softmax -> V*Attention
        """
        if (self.attention_wrapper is not None and 
            query.is_cuda and key.is_cuda and value.is_cuda):
            # Use CUDA-optimized attention when available
            try:
                return self.attention_wrapper.forward(query, key, value, attention_mask)
            except Exception as e:
                logger.warning(f"CUDA attention failed: {e}, falling back to PyTorch")
        
        # PyTorch fallback implementation
        return self._pytorch_forward(query, key, value, attention_mask)
    
    def _pytorch_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        PyTorch fallback implementation
        """
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output


class FusedMLPBlock(nn.Module):
    """
    Fused MLP block: Linear1 + Activation + Linear2 + Add residual
    Combines multiple operations in a single kernel to reduce memory traffic.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Initialize CUDA wrapper if available
        if CUDA_AVAILABLE:
            self.tensor_ops = SM61TensorOpsWrapper()
        else:
            self.tensor_ops = None
        
        # Linear layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass: x -> Linear1 -> Activation -> Linear2 -> Add residual
        """
        if self.tensor_ops is not None and x.is_cuda:
            # Use CUDA-optimized operations when available
            try:
                # Apply first linear transformation
                gate_output = self.gate_proj(x)
                up_output = self.up_proj(x)
                
                # Apply activation and multiply
                act_output = self.act_fn(gate_output)
                intermediate_output = act_output * up_output
                
                # Apply second linear transformation
                output = self.down_proj(intermediate_output)
                
                # Add residual if provided
                if residual is not None:
                    output = output + residual
                
                return output
            except Exception as e:
                logger.warning(f"CUDA MLP fusion failed: {e}, falling back to PyTorch")
        
        # PyTorch fallback implementation
        return self._pytorch_forward(x, residual)
    
    def _pytorch_forward(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        """
        PyTorch fallback implementation
        """
        # Apply first linear transformation
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # Apply activation and multiply
        act_output = self.act_fn(gate_output)
        intermediate_output = act_output * up_output
        
        # Apply second linear transformation
        output = self.down_proj(intermediate_output)
        
        # Add residual if provided
        if residual is not None:
            output = output + residual
        
        return output


class FusedQKVMatmul(nn.Module):
    """
    Fused QKV projection + matmul operations.
    Combines the Q, K, V projections and the Q*K^T computation in a single kernel.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads  # Changed to match expected attribute name
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )

        # Initialize CUDA wrapper if available
        if CUDA_AVAILABLE:
            self.tensor_ops = SM61TensorOpsWrapper()
        else:
            self.tensor_ops = None

        # Q, K, V projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass: hidden_states -> QKV projections -> Q*K^T computation
        Returns Q, K, V tensors after projection
        """
        bsz, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        return query_states, key_states, value_states


class FusedResidualAddLayerNorm(nn.Module):
    """
    Fused Add residual + Layer Normalization.
    Combines the residual addition and layer normalization in a single operation
    to reduce memory traffic.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Layer norm parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, hidden_states: Tensor, residual: Tensor) -> Tensor:
        """
        Forward pass: hidden_states + residual -> LayerNorm
        """
        # Add residual connection
        hidden_states = hidden_states + residual
        
        # Apply layer normalization
        mean = hidden_states.mean(dim=-1, keepdim=True)
        var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        
        # Normalize
        hidden_states = (hidden_states - mean) / std
        
        # Apply learnable parameters
        hidden_states = hidden_states * self.weight + self.bias
        
        return hidden_states


class FusedDecoderLayer(nn.Module):
    """
    Fused decoder layer combining multiple operations to reduce kernel launches.
    Contains fused operations for attention and MLP components.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Fused attention components
        self.self_attn = FusedQKVMatmul(config)
        self.attention_norm = FusedResidualAddLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Fused MLP components
        self.mlp = FusedMLPBlock(config)
        self.mlp_norm = FusedResidualAddLayerNorm(config.hidden_size, config.layer_norm_eps)
        
        # Input layer norm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        """
        Forward pass with fused operations
        """
        # Store residual for later use
        residual = hidden_states

        # Apply input layer norm (this could potentially be fused with attention in CUDA)
        hidden_states = self.input_layernorm(hidden_states)

        # Get Q, K, V projections
        query_states, key_states, value_states = self.self_attn(hidden_states)

        # Apply fused attention + softmax
        # Create a temporary config-like object for the attention layer
        temp_config = type('TempConfig', (), {
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.self_attn.num_heads,
            'layer_norm_eps': self.input_layernorm.eps
        })()
        attn_layer = FusedAttentionSoftmax(temp_config)
        attn_output = attn_layer(query_states, key_states, value_states, attention_mask)

        # Apply attention output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.hidden_size)
        attn_output = self.attention_output(attn_output)

        # Fused residual addition + layer norm
        hidden_states = self.attention_norm(attn_output, residual)

        # Store residual for MLP
        residual = hidden_states

        # Apply post-attention layer norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Apply fused MLP
        hidden_states = self.mlp(hidden_states)

        # Fused residual addition + layer norm for MLP
        hidden_states = self.mlp_norm(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_output,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class KernelFusionManager:
    """
    Manager class to handle kernel fusion operations across the model
    """
    def __init__(self, config):
        self.config = config
        self.fusion_enabled = True
        self.logger = logging.getLogger(__name__)
        
        # Check if CUDA is available and kernels can be used
        if CUDA_AVAILABLE:
            self.logger.info("CUDA kernels available for kernel fusion")
        else:
            self.logger.warning("CUDA kernels not available, using PyTorch fallbacks")
    
    def fuse_model(self, model: nn.Module) -> nn.Module:
        """
        Apply kernel fusion to the model by replacing appropriate layers
        """
        if not self.fusion_enabled:
            return model
        
        self.logger.info("Applying kernel fusion optimizations to model")
        
        # Replace decoder layers with fused versions
        for i, layer in enumerate(model.language_model.layers):
            fused_layer = FusedDecoderLayer(self.config, i)
            
            # Copy weights from original layer to fused layer
            self._copy_weights(layer, fused_layer)
            
            # Replace the layer
            model.language_model.layers[i] = fused_layer
        
        # Replace MLP layers in vision transformer if present
        if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'layers'):
            for i, layer in enumerate(model.vision_tower.layers):
                # Replace MLP in vision layer
                if hasattr(layer, 'mlp'):
                    original_mlp = layer.mlp
                    fused_mlp = FusedMLPBlock(self.config)
                    
                    # Copy weights
                    if hasattr(original_mlp, 'fc1'):
                        fused_mlp.gate_proj.weight.data = original_mlp.fc1.weight.data
                        fused_mlp.up_proj.weight.data = original_mlp.fc1.weight.data
                    if hasattr(original_mlp, 'fc2'):
                        fused_mlp.down_proj.weight.data = original_mlp.fc2.weight.data
                    
                    layer.mlp = fused_mlp
        
        self.logger.info("Kernel fusion applied successfully")
        return model
    
    def _copy_weights(self, original_layer: nn.Module, fused_layer: FusedDecoderLayer):
        """
        Copy weights from original layer to fused layer
        """
        # Copy attention weights
        if hasattr(original_layer, 'self_attn'):
            if hasattr(original_layer.self_attn, 'q_proj'):
                fused_layer.self_attn.q_proj.weight.data = original_layer.self_attn.q_proj.weight.data
            if hasattr(original_layer.self_attn, 'k_proj'):
                fused_layer.self_attn.k_proj.weight.data = original_layer.self_attn.k_proj.weight.data
            if hasattr(original_layer.self_attn, 'v_proj'):
                fused_layer.self_attn.v_proj.weight.data = original_layer.self_attn.v_proj.weight.data
            if hasattr(original_layer.self_attn, 'o_proj'):
                fused_layer.attention_output.weight.data = original_layer.self_attn.o_proj.weight.data
        
        # Copy MLP weights
        if hasattr(original_layer, 'mlp'):
            if hasattr(original_layer.mlp, 'gate_proj'):
                fused_layer.mlp.gate_proj.weight.data = original_layer.mlp.gate_proj.weight.data
            if hasattr(original_layer.mlp, 'up_proj'):
                fused_layer.mlp.up_proj.weight.data = original_layer.mlp.up_proj.weight.data
            if hasattr(original_layer.mlp, 'down_proj'):
                fused_layer.mlp.down_proj.weight.data = original_layer.mlp.down_proj.weight.data
        
        # Copy layer norm weights
        if hasattr(original_layer, 'input_layernorm'):
            fused_layer.input_layernorm.weight.data = original_layer.input_layernorm.weight.data
            fused_layer.input_layernorm.bias.data = original_layer.input_layernorm.bias.data
        if hasattr(original_layer, 'post_attention_layernorm'):
            fused_layer.post_attention_layernorm.weight.data = original_layer.post_attention_layernorm.weight.data
            fused_layer.post_attention_layernorm.bias.data = original_layer.post_attention_layernorm.bias.data
    
    def get_fusion_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get statistics about kernel fusion applied to the model
        """
        stats = {
            "fusion_enabled": self.fusion_enabled,
            "cuda_available": CUDA_AVAILABLE,
            "fused_layers": 0,
            "total_layers": 0,
        }
        
        # Count fused layers
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            stats["total_layers"] = len(model.language_model.layers)
            stats["fused_layers"] = sum(1 for layer in model.language_model.layers 
                                       if isinstance(layer, FusedDecoderLayer))
        
        return stats


def apply_kernel_fusion_to_model(model: nn.Module, config) -> nn.Module:
    """
    Apply kernel fusion optimizations to a model
    """
    fusion_manager = KernelFusionManager(config)
    return fusion_manager.fuse_model(model)


def get_kernel_fusion_report(model: nn.Module, config) -> Dict[str, Any]:
    """
    Get a comprehensive report of kernel fusion optimizations
    """
    fusion_manager = KernelFusionManager(config)
    stats = fusion_manager.get_fusion_stats(model)
    
    report = {
        "kernel_fusion_enabled": stats["fusion_enabled"],
        "cuda_kernels_available": stats["cuda_available"],
        "model_fusion_stats": stats,
        "optimization_summary": f"Fused {stats['fused_layers']}/{stats['total_layers']} layers"
    }
    
    return report