"""
Generic CPU Processor Plugin

This plugin implements standard PyTorch/NumPy operations for generic CPUs.
It serves as a robust fallback for weak hardware or unsupported architectures.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..base.plugin_interface import HardwareProcessorPluginInterface

logger = logging.getLogger(__name__)


class GenericCPUPlugin(HardwareProcessorPluginInterface):
    """
    Generic CPU implementation using standard PyTorch operations.
    Capable of handling low-memory situations by relying on system paging if needed,
    though purely relying on OS paging is slow, this plugin focuses on correctness.
    """

    def __init__(self):
        self._config = {}
        self._num_threads = 1

    @property
    def name(self) -> str:
        return "GenericCPU"

    @property
    def supported_architectures(self) -> list[str]:
        return ["x86_64", "arm64", "amd64", "i386"]

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._config = config
        # Default thread setting
        self._num_threads = config.get("num_threads", torch.get_num_threads())
        self.manage_threads(self._num_threads)
        logger.info(f"GenericCPUPlugin initialized with {self._num_threads} threads.")
        return True

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Standard matmul.
        Note: If tensors are on different devices (e.g. one on disk/cpu, one on gpu),
        this handles moving them to a common device (CPU) if necessary for 'weak hardware' logic.
        """
        # Ensure compatibility
        device = a.device
        if b.device != device:
            # For weak hardware, we might prefer CPU execution if VRAM is tight
            # forcing b to a's device or both to CPU
            b = b.to(device)

        return torch.matmul(a, b)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Uses PyTorch's optimized SDPA which often dispatches to efficient CPU kernels (like FlashAttention on CPU if available in newer torch versions or standard math).
        """
        # For CPU, native SDPA is decent.
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def apply_activation(self, x: torch.Tensor, activation_type: str) -> torch.Tensor:
        if activation_type == "silu" or activation_type == "swish":
            return F.silu(x)
        elif activation_type == "gelu":
            return F.gelu(x)
        elif activation_type == "relu":
            return F.relu(x)
        else:
            logger.warning(f"Unknown activation {activation_type}, returning identity.")
            return x

    def manage_threads(self, num_threads: int):
        self._num_threads = num_threads
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)


def create_generic_cpu_plugin() -> GenericCPUPlugin:
    return GenericCPUPlugin()
