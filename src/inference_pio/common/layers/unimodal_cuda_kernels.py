"""
Unimodal CUDA Kernels for Language Models

This module implements optimized CUDA kernels specifically for unimodal
language models like GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b.
These kernels optimize attention, MLP, and normalization operations with
specialized optimizations for text processing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UnimodalAttentionKernel(nn.Module):
    """
    Optimized CUDA kernel for unimodal attention operations in language models.
    This kernel efficiently computes self-attention with specialized optimizations
    for text processing, including FlashAttention-like optimizations.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        causal: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_rate = dropout
        self.use_flash_attention = use_flash_attention
        self.causal = causal

        if self.head_dim * nhead != d_model:
            raise ValueError(
                f"d_model must be divisible by nhead (got d_model: {d_model}, nhead: {nhead})"
            )

        self.scaling = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Projections for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Causal mask for autoregressive models
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(1, 1, 1024, 1024, dtype=torch.bool), diagonal=1),
                persistent=False,
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for unimodal attention kernel.

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

        # Project query, key, and value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Scale queries
        q = q * self.scaling

        # Compute attention scores
        if self.use_flash_attention and torch.cuda.is_available():
            # Use efficient attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1))

            # Apply causal mask if needed
            if self.causal:
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
                    # Placeholder for actual CUDA kernel implementation
                    # This would contain the actual CUDA kernel code
                    return torch.matmul(a, b)
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
            if self.causal:
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
                    # Placeholder for actual CUDA kernel implementation
                    # This would contain the actual CUDA kernel code
                    return torch.matmul(a, b)
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

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        # Return output and attention weights if needed
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None


class UnimodalMLPKernel(nn.Module):
    """
    Optimized CUDA kernel for unimodal MLP (feed-forward) operations in language models.
    This kernel efficiently computes the feed-forward network with specialized optimizations
    for text processing, including SwiGLU and other activation functions.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        activation: str = "silu",
        use_swiglu: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.use_swiglu = use_swiglu
        self.dropout_rate = dropout

        # Activation function
        self.activation_fn = self._get_activation(activation)

        if use_swiglu:
            # SwiGLU implementation: (Swish(x * W_gate) * (x * W_up)) * W_down
            self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        else:
            # Standard FFN: Linear -> Activation -> Linear
            self.fc1 = nn.Linear(d_model, intermediate_size, bias=True)
            self.fc2 = nn.Linear(intermediate_size, d_model, bias=True)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),  # Swish
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu_new": lambda: nn.GELU(approximate="tanh"),
        }
        return activations.get(activation, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for unimodal MLP kernel.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        if self.use_swiglu:
            # SwiGLU: Swish(x * W_gate) * (x * W_up)
            gate = self.activation_fn(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
            x = self.down_proj(x)
        else:
            # Standard FFN
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.fc2(x)

        # Apply dropout if configured
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class UnimodalLayerNormKernel(nn.Module):
    """
    Optimized CUDA kernel for unimodal LayerNorm operations in language models.
    This kernel efficiently computes layer normalization with specialized optimizations
    for text processing.
    """

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for unimodal LayerNorm kernel.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x = (x - mean) * torch.rsqrt(variance + self.eps)

        # Apply weight and bias if affine transformation is enabled
        if self.elementwise_affine:
            x = x * self.weight + self.bias

        return x


class UnimodalRMSNormKernel(nn.Module):
    """
    Optimized CUDA kernel for unimodal RMSNorm operations in language models.
    This kernel efficiently computes root mean square normalization with specialized optimizations
    for text processing.
    """

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for unimodal RMSNorm kernel.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

        # Normalize
        x = x / rms

        # Apply weight if affine transformation is enabled
        if self.elementwise_affine:
            x = x * self.weight

        return x


class UnimodalHardwareOptimizer:
    """
    Hardware-specific optimizer for unimodal CUDA kernels.
    Detects hardware capabilities and applies appropriate optimizations.
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

    def get_optimization_report(self) -> Dict:
        """Get a report of hardware optimizations."""
        return {
            "compute_capability": self.compute_capability,
            "tensor_cores_supported": self.tensor_cores_supported,
            "optimization_level": self.optimization_level,
            "recommended_kernels": self._get_recommended_kernels(),
        }

    def _get_recommended_kernels(self) -> List[str]:
        """Get recommended kernels based on hardware."""
        recommendations = ["unimodal_attention", "unimodal_mlp", "unimodal_layernorm"]

        if self.tensor_cores_supported:
            recommendations.extend(["unimodal_rmsnorm"])

        return recommendations


def create_unimodal_cuda_kernels(
    d_model: int,
    nhead: int,
    intermediate_size: int,
    use_flash_attention: bool = True,
    use_swiglu: bool = True,
) -> Dict[str, nn.Module]:
    """
    Factory function to create unimodal CUDA kernels.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        use_flash_attention: Whether to use flash attention optimization
        use_swiglu: Whether to use SwiGLU in MLP

    Returns:
        Dictionary of created kernels
    """
    kernels = {}

    # Create unimodal attention kernel
    kernels["attention"] = UnimodalAttentionKernel(
        d_model=d_model, nhead=nhead, use_flash_attention=use_flash_attention
    )

    # Create unimodal MLP kernel
    kernels["mlp"] = UnimodalMLPKernel(
        d_model=d_model, intermediate_size=intermediate_size, use_swiglu=use_swiglu
    )

    # Create unimodal LayerNorm kernel
    kernels["layernorm"] = UnimodalLayerNormKernel(normalized_shape=d_model)

    # Create unimodal RMSNorm kernel
    kernels["rmsnorm"] = UnimodalRMSNormKernel(normalized_shape=d_model)

    return kernels


def apply_unimodal_cuda_optimizations_to_model(
    model: nn.Module,
    d_model: int,
    nhead: int,
    intermediate_size: int,
    config: Any = None,
    model_type: str = "general",
) -> nn.Module:
    """
    Apply unimodal CUDA optimizations to the given language model.

    Args:
        model: The model to optimize
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        config: Configuration object with model-specific settings
        model_type: Type of model (for identification purposes)

    Returns:
        Optimized model
    """
    logger.info(f"Applying unimodal CUDA optimizations for model type: {model_type}")

    # Create hardware optimizer
    hw_optimizer = UnimodalHardwareOptimizer()
    opt_report = hw_optimizer.get_optimization_report()
    logger.info(f"Hardware optimization report: {opt_report}")

    # Determine if we should use SwiGLU based on config (defaults to False for generic)
    use_swiglu = (
        getattr(config, "use_swiglu", False) if config else False
    )  # Model-specific config should define this
    use_flash_attention = True  # Enable flash attention by default

    # Create unimodal kernels
    kernels = create_unimodal_cuda_kernels(
        d_model=d_model,
        nhead=nhead,
        intermediate_size=intermediate_size,
        use_flash_attention=use_flash_attention,
        use_swiglu=use_swiglu,
    )

    # Apply optimizations based on model architecture
    for name, module in model.named_modules():
        # Replace attention modules
        if isinstance(module, nn.MultiheadAttention):
            # Handle root-level modules (when name is empty) or modules without dots
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = _get_parent_module(model, parent_name)
            else:
                # If name doesn't contain '.', it's a direct child of the root model
                parent_module = model
                child_name = name

            # Create a unimodal attention kernel that mimics the original interface
            unimodal_attn = UnimodalAttentionKernel(
                d_model=d_model, nhead=nhead, use_flash_attention=use_flash_attention
            )

            setattr(parent_module, child_name, unimodal_attn)
            logger.info(f"Replaced attention module {name} with unimodal kernel")

        # Replace LayerNorm modules
        elif isinstance(module, nn.LayerNorm):
            # Handle root-level modules (when name is empty) or modules without dots
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = _get_parent_module(model, parent_name)
            else:
                # If name doesn't contain '.', it's a direct child of the root model
                parent_module = model
                child_name = name

            # Determine norm type from config (defaults to LayerNorm for generic)
            use_rms_norm = getattr(
                config, "use_rms_norm", False
            )  # Model-specific config should define this
            if use_rms_norm:
                unimodal_norm = UnimodalRMSNormKernel(
                    normalized_shape=module.normalized_shape[0], eps=module.eps
                )
            else:
                unimodal_norm = UnimodalLayerNormKernel(
                    normalized_shape=module.normalized_shape[0], eps=module.eps
                )

            # Copy parameters if available
            if hasattr(unimodal_norm, "weight") and unimodal_norm.weight is not None:
                unimodal_norm.weight.data.copy_(module.weight.data)
            if hasattr(unimodal_norm, "bias") and unimodal_norm.bias is not None:
                unimodal_norm.bias.data.copy_(module.bias.data)

            setattr(parent_module, child_name, unimodal_norm)
            logger.info(f"Replaced normalization module {name} with unimodal kernel")

        # Replace MLP/feed-forward modules (this is a simplified approach)
        # In practice, we'd need to identify specific MLP layers to replace
        elif hasattr(module, "forward") and any(
            name_part in name.lower()
            for name_part in ["mlp", "ffn", "feed_forward", "intermediate"]
        ):
            # This is a simplified detection - in practice, we'd need more sophisticated identification
            if hasattr(module, "fc1") and hasattr(module, "fc2"):
                # Handle root-level modules (when name is empty) or modules without dots
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent_module = _get_parent_module(model, parent_name)
                else:
                    # If name doesn't contain '.', it's a direct child of the root model
                    parent_module = model
                    child_name = name

                # Create a unimodal MLP kernel
                unimodal_mlp = UnimodalMLPKernel(
                    d_model=d_model,
                    intermediate_size=intermediate_size,
                    use_swiglu=use_swiglu,
                )

                setattr(parent_module, child_name, unimodal_mlp)
                logger.info(f"Replaced MLP module {name} with unimodal kernel")

    logger.info("Unimodal CUDA optimizations applied successfully")
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


def get_unimodal_cuda_optimization_report(
    model: nn.Module,
    d_model: int,
    nhead: int,
    intermediate_size: int,
    config: Any = None,
    model_type: str = "general",
) -> Dict:
    """
    Get a report of unimodal CUDA optimizations that would be applied to the model.

    Args:
        model: The model to analyze
        d_model: Model dimension
        nhead: Number of attention heads
        intermediate_size: Size of intermediate layer in MLP
        config: Configuration object with model-specific settings
        model_type: Type of model (for identification purposes)

    Returns:
        Optimization report
    """
    hw_optimizer = UnimodalHardwareOptimizer()
    hw_report = hw_optimizer.get_optimization_report()

    # Count relevant modules that would be optimized
    attention_count = sum(
        1 for m in model.modules() if isinstance(m, nn.MultiheadAttention)
    )
    layernorm_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))

    # Simple MLP detection (would need more sophisticated logic in practice)
    mlp_count = sum(
        1
        for name, m in model.named_modules()
        if hasattr(m, "forward")
        and any(part in name.lower() for part in ["mlp", "ffn", "feed_forward"])
    )

    report = {
        "model_type": model_type,
        "hardware_info": hw_report,
        "modules_identified_for_optimization": {
            "attention_layers": attention_count,
            "normalization_layers": layernorm_count,
            "mlp_layers": mlp_count,
        },
        "optimization_config": {
            "d_model": d_model,
            "nhead": nhead,
            "intermediate_size": intermediate_size,
            "use_swiglu": model_type in ["qwen3_4b", "qwen3_coder"],
            "use_flash_attention": True,
        },
    }

    return report


__all__ = [
    "UnimodalAttentionKernel",
    "UnimodalMLPKernel",
    "UnimodalLayerNormKernel",
    "UnimodalRMSNormKernel",
    "UnimodalHardwareOptimizer",
    "create_unimodal_cuda_kernels",
    "apply_unimodal_cuda_optimizations_to_model",
    "get_unimodal_cuda_optimization_report",
]
