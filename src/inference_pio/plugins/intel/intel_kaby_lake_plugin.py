"""
Intel Kaby Lake Processor Plugin

This plugin provides optimized execution strategies for Intel Kaby Lake
processors, specifically targeting the i5-7500 (4 cores / 4 threads).
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ...common.hardware.hardware_analyzer import SystemProfile
from ..base.plugin_interface import HardwareProcessorPluginInterface

logger = logging.getLogger(__name__)

# Try to import IPEX
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401

    _IPEX_AVAILABLE = True
except ImportError:
    _IPEX_AVAILABLE = False


class IntelKabyLakePlugin(HardwareProcessorPluginInterface):
    """
    Optimized plugin for Intel Kaby Lake CPUs (e.g., i5-7500).
    Focuses on AVX2 utilization and optimal threading for 4-core desktop chips.
    """

    def __init__(self, profile: Optional[SystemProfile] = None):
        self._config = {}
        # Default to physical cores (i5-7500 has 4 cores/4 threads)
        self._num_threads = 4
        self._ipex_enabled = False
        self._profile = profile

    @property
    def plugin_name(self) -> str:
        return "IntelKabyLake"

    @property
    def name(self) -> str:
        return "IntelKabyLake"

    @property
    def supported_architectures(self) -> list[str]:
        return ["x86_64", "KabyLake", "i5-7500"]

    @property
    def supported_hardware_architectures(self) -> list[str]:
        return ["x86_64", "KabyLake", "i5-7500"]

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._config = config

        # Determine thread count
        # i5-7500 has 4 physical cores and 4 threads.
        # Using all 4 is optimal for dedicated inference tasks.
        self._num_threads = config.get("num_threads", 4)

        self.configure_thread_management(self._num_threads)

        # IPEX Initialization
        if config.get("enable_ipex", False) and _IPEX_AVAILABLE:
            self._ipex_enabled = True
            logger.info("Intel Extension for PyTorch (IPEX) enabled.")
        elif config.get("enable_ipex", False) and not _IPEX_AVAILABLE:
            logger.warning(
                "IPEX enabled but not installed. Falling back to MKL."
            )

        logger.info(
            f"IntelKabyLakePlugin initialized. Threads: {self._num_threads}. "
            f"Optimized for AVX2."
        )
        return True

    def optimized_matmul(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """
        Matrix multiplication optimized for Intel Kaby Lake.
        Utilizes AVX2 via PyTorch's backend (MKL/OneDNN).
        """
        device = a.device
        if b.device != device:
            b = b.to(device)
        return torch.matmul(a, b)

    # Legacy alias
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.optimized_matmul(a, b)

    def optimized_scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Standard SDPA. Kaby Lake supports AVX2 which PyTorch uses efficiently.
        """
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    # Legacy alias
    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        return self.optimized_scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal
        )

    def optimized_apply_activation(
        self, x: torch.Tensor, activation_type: str
    ) -> torch.Tensor:
        if activation_type == "silu" or activation_type == "swish":
            return F.silu(x)
        elif activation_type == "gelu":
            return F.gelu(x)
        elif activation_type == "relu":
            return F.relu(x)
        else:
            return x

    # Legacy alias
    def apply_activation(
        self, x: torch.Tensor, activation_type: str
    ) -> torch.Tensor:
        return self.optimized_apply_activation(x, activation_type)

    def configure_thread_management(self, num_threads: int):
        self._num_threads = num_threads
        torch.set_num_threads(num_threads)
        # Kaby Lake i5-7500 is 4C/4T. Inter-op threads can be minimal.
        torch.set_num_interop_threads(min(2, num_threads))

        # Set environment variables for MKL
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_DYNAMIC"] = "FALSE"

    # Legacy alias
    def manage_threads(self, num_threads: int):
        self.configure_thread_management(num_threads)


def create_intel_kaby_lake_plugin(
    profile: Optional[SystemProfile] = None,
) -> IntelKabyLakePlugin:
    return IntelKabyLakePlugin(profile)
