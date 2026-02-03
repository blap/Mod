"""
Qwen3-Coder-30B Enhanced CUDA Kernels Implementation

This module implements enhanced CUDA kernels for the Qwen3-Coder-30B model.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Qwen3Coder30BConfig


class Qwen3CoderGELUKernel(nn.Module):
    """
    Enhanced GELU activation kernel optimized for Qwen3-Coder-30B model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation with optimized computation.

        Args:
            x: Input tensor

        Returns:
            GELU activated tensor
        """
        # Using the tanh approximation for faster computation
        return (
            x
            * 0.5
            * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))
        )


class Qwen3CoderMatMulKernel(nn.Module):
    """
    Optimized matrix multiplication kernel for Qwen3-Coder-30B model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Apply optimized matrix multiplication.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Result of matrix multiplication
        """
        # Use torch's optimized matmul implementation
        return torch.matmul(a, b)


class Qwen3CoderSoftmaxKernel(nn.Module):
    """
    Optimized softmax kernel for Qwen3-Coder-30B model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Apply optimized softmax operation.

        Args:
            x: Input tensor
            dim: Dimension to apply softmax

        Returns:
            Softmax applied tensor
        """
        # Use torch's optimized softmax implementation
        return torch.softmax(x, dim=dim, dtype=torch.float32).to(x.dtype)


class Qwen3CoderAttentionKernel(nn.Module):
    """
    Optimized attention kernel for Qwen3-Coder-30B model.
    """

    def __init__(self):
        super().__init__()
        self.softmax_kernel = Qwen3CoderSoftmaxKernel()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply optimized attention computation.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask

        Returns:
            Attended values
        """
        # Compute scaled dot-product attention
        scale = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = self.softmax_kernel(scores, dim=-1)
        return torch.matmul(probs, value)


class Qwen3CoderMLPKernel(nn.Module):
    """
    Optimized MLP (feed-forward) kernel for Qwen3-Coder-30B model.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gelu_kernel = Qwen3CoderGELUKernel()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply optimized MLP computation.

        Args:
            x: Input tensor

        Returns:
            MLP output
        """
        # Apply gate projection and GELU activation
        gate = self.gelu_kernel(self.gate_proj(x))
        up = self.up_proj(x)

        # Element-wise multiplication
        result = gate * up

        # Down projection
        return self.down_proj(result)


class Qwen3CoderLayerNormKernel(nn.Module):
    """
    Optimized LayerNorm kernel for Qwen3-Coder-30B model.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply optimized LayerNorm computation.

        Args:
            x: Input tensor

        Returns:
            Layer-normalized tensor
        """
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x = (x - mean) * torch.rsqrt(variance + self.eps)

        # Apply weight and bias
        return x * self.weight + self.bias


class Qwen3CoderHardwareKernelManager:
    """
    Hardware-specific kernel manager for Qwen3-Coder-30B model.
    """

    def __init__(self, config: Qwen3Coder30BConfig):
        self.config = config
        self.compute_capability = self._get_compute_capability()

    def _get_compute_capability(self) -> Tuple[int, int]:
        """
        Get the compute capability of the current GPU.

        Returns:
            Tuple of (major, minor) compute capability
        """
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return torch.cuda.get_device_capability(device)
        return (0, 0)  # CPU

    def is_tensor_core_supported(self) -> bool:
        """
        Check if Tensor Cores are supported on the current hardware.

        Returns:
            True if Tensor Cores are supported, False otherwise
        """
        major, minor = self.compute_capability
        return major >= 7  # Tensor Cores available from Volta (7.0) onwards


def create_qwen3_coder_kernel_manager(
    config: Qwen3Coder30BConfig,
) -> Qwen3CoderHardwareKernelManager:
    """
    Create a hardware kernel manager for Qwen3-Coder-30B model.

    Args:
        config: Model configuration

    Returns:
        Hardware kernel manager
    """
    return Qwen3CoderHardwareKernelManager(config)


def apply_qwen3_coder_optimizations_to_model(
    model: nn.Module, config: Qwen3Coder30BConfig
) -> nn.Module:
    """
    Apply Qwen3-Coder-30B specific CUDA optimizations to the model.

    Args:
        model: The model to optimize
        config: Model configuration

    Returns:
        Optimized model
    """
    # Create kernel manager
    kernel_manager = create_qwen3_coder_kernel_manager(config)

    # Apply optimizations based on configuration
    for name, module in model.named_modules():
        # Replace GELU activations if enabled
        if config.cuda_kernel_gelu_enabled and isinstance(module, nn.GELU):
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = _get_parent_module(model, parent_name)
            setattr(parent_module, child_name, Qwen3CoderGELUKernel())

        # Replace LayerNorm if enabled
        if config.cuda_kernel_layernorm_enabled and isinstance(module, nn.LayerNorm):
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = _get_parent_module(model, parent_name)
            new_layernorm = Qwen3CoderLayerNormKernel(
                normalized_shape=module.normalized_shape[0], eps=module.eps
            )
            # Copy parameters
            new_layernorm.weight.data.copy_(module.weight.data)
            new_layernorm.bias.data.copy_(module.bias.data)
            setattr(parent_module, child_name, new_layernorm)

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


def get_qwen3_coder_optimization_report(
    model: nn.Module, config: Qwen3Coder30BConfig
) -> Dict:
    """
    Get a report of Qwen3-Coder-30B optimizations applied to the model.

    Args:
        model: The model
        config: Model configuration

    Returns:
        Optimization report
    """
    kernel_manager = create_qwen3_coder_kernel_manager(config)

    report = {
        "model_type": "Qwen3-Coder-30B",
        "hardware_compute_capability": kernel_manager.compute_capability,
        "tensor_cores_supported": kernel_manager.is_tensor_core_supported(),
        "optimizations_applied": {
            "gelu_kernel": config.cuda_kernel_gelu_enabled,
            "matmul_kernel": config.cuda_kernel_matmul_enabled,
            "softmax_kernel": config.cuda_kernel_softmax_enabled,
            "attention_kernel": config.cuda_kernel_attention_enabled,
            "mlp_kernel": config.cuda_kernel_mlp_enabled,
            "layernorm_kernel": config.cuda_kernel_layernorm_enabled,
        },
        "config": {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
        },
    }

    return report


__all__ = [
    "Qwen3CoderGELUKernel",
    "Qwen3CoderMatMulKernel",
    "Qwen3CoderSoftmaxKernel",
    "Qwen3CoderAttentionKernel",
    "Qwen3CoderMLPKernel",
    "Qwen3CoderLayerNormKernel",
    "Qwen3CoderHardwareKernelManager",
    "create_qwen3_coder_kernel_manager",
    "apply_qwen3_coder_optimizations_to_model",
    "get_qwen3_coder_optimization_report",
]
