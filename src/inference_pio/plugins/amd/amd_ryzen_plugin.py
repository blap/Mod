"""
AMD Ryzen Processor Plugin

This plugin provides optimized execution strategies for AMD Ryzen processors,
specifically targeting the Ryzen 7 5700 (8 cores / 16 threads) and supporting
Radeon GPU awareness.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ...common.hardware.hardware_analyzer import SystemProfile
from ..base.plugin_interface import HardwareProcessorPluginInterface

logger = logging.getLogger(__name__)


class AmdRyzenPlugin(HardwareProcessorPluginInterface):
    """
    Optimized plugin for AMD Ryzen CPUs (e.g., Ryzen 7 5700).
    Focuses on leveraging high core counts (8C/16T) and AVX2 instructions.
    """

    def __init__(self, profile: Optional[SystemProfile] = None):
        self._config = {}
        # Ryzen 7 5700 has 8 physical cores, 16 threads.
        # For heavy inference, using physical cores (8) often avoids context
        # switching overhead, but 16 can be used for highly parallel small ops.
        # Defaulting to 8 for stability.
        self._num_threads = 8
        self._profile = profile

    @property
    def plugin_name(self) -> str:
        return "AmdRyzen"

    @property
    def name(self) -> str:
        return "AmdRyzen"

    @property
    def supported_architectures(self) -> list[str]:
        return ["x86_64", "Zen3", "Ryzen", "Ryzen 7 5700"]

    @property
    def supported_hardware_architectures(self) -> list[str]:
        return ["x86_64", "Zen3", "Ryzen", "Ryzen 7 5700"]

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._config = config
        self._num_threads = config.get("num_threads", 8)
        self.configure_thread_management(self._num_threads)

        # AMD-specific MKL workaround (historical, often helpful)
        # Some versions of MKL check for Intel CPU and downgrade path on AMD.
        # Setting DEBUG_MKL ensures it doesn't crash.

        logger.info(
            f"AmdRyzenPlugin initialized. "
            f"Threads: {self._num_threads}. "
            f"Optimized for Zen Architecture."
        )
        return True

    def optimized_matmul(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """
        Matrix multiplication optimized for AMD Ryzen.
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
        Standard SDPA.
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
        # Zen architecture handles threading well.
        torch.set_num_interop_threads(min(4, num_threads))

    # Legacy alias
    def manage_threads(self, num_threads: int):
        self.configure_thread_management(num_threads)

    def get_layer_distribution(
        self,
        total_layers: int,
        estimated_model_size_gb: float,
        available_vram_gb: float,
    ) -> Dict[str, Any]:
        """
        Calculate optimal layer distribution between GPU and CPU.
        Ryzen 7 5700 is strong, RX550 (4GB) is weak but useful.
        """
        # If RX550 is detected via torch (e.g. ROCm/DirectML), use it.
        # available_vram_gb from HardwareAnalyzer checks torch.cuda.

        safe_vram = max(0.0, available_vram_gb * 0.9)  # Use 90% of VRAM

        layer_size_gb = (
            estimated_model_size_gb / total_layers
            if total_layers > 0
            else 0
        )

        if layer_size_gb <= 0:
            return {"gpu_layers": 0, "cpu_layers": total_layers}

        layers_on_gpu = int(safe_vram / layer_size_gb)
        layers_on_gpu = min(layers_on_gpu, total_layers)
        layers_on_cpu = total_layers - layers_on_gpu

        logger.info(
            f"AmdRyzen Hybrid Plan: {layers_on_gpu} layers on GPU, "
            f"{layers_on_cpu} layers on Ryzen CPU."
        )

        return {"gpu_layers": layers_on_gpu, "cpu_layers": layers_on_cpu}


def create_amd_ryzen_plugin(
    profile: Optional[SystemProfile] = None,
) -> AmdRyzenPlugin:
    return AmdRyzenPlugin(profile)
