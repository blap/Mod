"""
Intel Comet Lake Processor Plugin

This plugin provides optimized execution strategies for Intel Comet Lake processors,
specifically targeting the i5-10210U (4 cores / 8 threads).
"""

import os
import torch
import torch.nn.functional as F
import logging
from typing import Any, Dict, Optional

from .processor_interface import ProcessorPluginInterface

logger = logging.getLogger(__name__)

class IntelCometLakePlugin(ProcessorPluginInterface):
    """
    Optimized plugin for Intel Comet Lake CPUs (e.g., i5-10210U).
    Focuses on MKL-DNN utilization and optimal threading for 4-core mobile chips.
    """

    def __init__(self):
        self._config = {}
        self._num_threads = 4  # Default to physical cores for stability on mobile chips

    @property
    def name(self) -> str:
        return "IntelCometLake"

    @property
    def supported_architectures(self) -> list[str]:
        return ["x86_64", "CometLake", "i5-10210U"]

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._config = config

        # Determine thread count
        # For i5-10210U, sticking to 4 physical cores is often better for sustained
        # FP32 inference to manage thermals and cache thrashing, but 8 is valid for bursty IO.
        # We default to 4 unless overridden.
        self._num_threads = config.get("num_threads", 4)

        # Intel MKL Optimizations
        # Setting these environment variables helps PyTorch use MKL efficiently
        os.environ["MKL_NUM_THREADS"] = str(self._num_threads)
        os.environ["OMP_NUM_THREADS"] = str(self._num_threads)
        os.environ["MKL_DYNAMIC"] = "FALSE"

        self.manage_threads(self._num_threads)

        logger.info(f"IntelCometLakePlugin initialized. Threads: {self._num_threads}. "
                    "Optimized for AVX2 and MKL-DNN.")
        return True

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication optimized for Intel CPUs.
        PyTorch uses MKL by default on x86, checking ensure we are on CPU.
        """
        device = a.device
        if b.device != device:
            b = b.to(device)

        # Optional: IPEX (Intel Extension for PyTorch) integration could go here
        # if installed. For now, standard MKL-backed torch.matmul is sufficient
        # and highly optimized for AVX2.
        return torch.matmul(a, b)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Standard SDPA. Intel CPUs benefit significantly from the 'math' or 'flash'
        backends if available in newer torch versions.
        """
        # We can try to force a specific backend if needed, but default is usually best.
        # torch.backends.cuda.sdp_kernel does not apply to CPU.
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    def apply_activation(self, x: torch.Tensor, activation_type: str) -> torch.Tensor:
        # standard activations are well optimized in MKL
        if activation_type == "silu" or activation_type == "swish":
            return F.silu(x)
        elif activation_type == "gelu":
            return F.gelu(x)
        elif activation_type == "relu":
            return F.relu(x)
        else:
            return x

    def manage_threads(self, num_threads: int):
        self._num_threads = num_threads
        torch.set_num_threads(num_threads)
        # Inter-op threads can add overhead on low-core count CPUs, keep low
        torch.set_num_interop_threads(min(2, num_threads))

def create_intel_comet_lake_plugin() -> IntelCometLakePlugin:
    return IntelCometLakePlugin()
