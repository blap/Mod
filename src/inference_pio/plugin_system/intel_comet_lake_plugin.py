"""
Intel Comet Lake Processor Plugin

This plugin provides optimized execution strategies for Intel Comet Lake processors,
specifically targeting the i5-10210U (4 cores / 8 threads).
"""

import os
import torch
import torch.nn.functional as F
import logging
from typing import Any, Dict, Optional, Tuple

from .processor_interface import ProcessorPluginInterface
from ..common.hardware_analyzer import SystemProfile

logger = logging.getLogger(__name__)

# Try to import IPEX
try:
    import intel_extension_for_pytorch as ipex
    _IPEX_AVAILABLE = True
except ImportError:
    _IPEX_AVAILABLE = False

class IntelCometLakePlugin(ProcessorPluginInterface):
    """
    Optimized plugin for Intel Comet Lake CPUs (e.g., i5-10210U).
    Focuses on MKL-DNN utilization and optimal threading for 4-core mobile chips.
    Also supports Hybrid Offloading (NVIDIA dGPU + Intel CPU) and IPEX acceleration.
    """

    def __init__(self, profile: Optional[SystemProfile] = None):
        self._config = {}
        self._num_threads = 4  # Default to physical cores for stability on mobile chips
        self._ipex_enabled = False
        # Store profile, accepting it as an argument.
        # Note: factory.py passes it as the first argument, so we need to ensure correct handling.
        self._profile = profile

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

        # IPEX Initialization
        if config.get("enable_ipex", False) and _IPEX_AVAILABLE:
            self._ipex_enabled = True
            # IPEX specific optimizations usually happen at the model level (ipex.optimize),
            # but we can set flags or prepare the environment here.
            logger.info("Intel Extension for PyTorch (IPEX) enabled.")
        elif config.get("enable_ipex", False) and not _IPEX_AVAILABLE:
            logger.warning("IPEX enabled in config but not installed. Falling back to MKL.")

        # Determine hybrid capability from profile if not in config
        # Prioritize profile if available
        if self._profile and self._profile.hybrid_capability:
            hybrid = True
        else:
            hybrid = config.get('hybrid_capability', False)

        # Ensure we don't accidentally rely on stale config if profile says otherwise
        if self._profile and not self._profile.hybrid_capability and not hybrid:
             pass # Stay false

        logger.info(f"IntelCometLakePlugin initialized. Threads: {self._num_threads}. "
                    f"Optimized for AVX2 and MKL-DNN. Hybrid Support: {hybrid}")
        return True

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication optimized for Intel CPUs.
        PyTorch uses MKL by default on x86, checking ensure we are on CPU.
        """
        device = a.device
        if b.device != device:
            b = b.to(device)

        # If IPEX is enabled and we are on CPU, it usually hooks into torch.matmul automatically.
        # Explicit IPEX calls are rarely needed for simple matmul unless using XPU.
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

    def get_layer_distribution(self, total_layers: int, estimated_model_size_gb: float, available_vram_gb: float) -> Dict[str, Any]:
        """
        Calculate optimal layer distribution between NVIDIA GPU and Intel CPU.

        Args:
            total_layers: Total number of transformer layers in the model.
            estimated_model_size_gb: Estimated size of the model weights in GB.
            available_vram_gb: Available VRAM on the NVIDIA GPU (e.g., MX330).

        Returns:
            Dict containing 'gpu_layers' (int) and 'cpu_layers' (int).
        """
        # Safety margin for VRAM (activations, KV cache, system overhead)
        # MX330 has 2GB, very tight. We leave ~0.5GB buffer ideally, or proportional.
        safe_vram = max(0.0, available_vram_gb * 0.8) # 80% utilization

        # Estimate size per layer (naive)
        # Assuming model size is mostly layers (ignoring embeddings for a moment)
        layer_size_gb = estimated_model_size_gb / total_layers if total_layers > 0 else 0

        if layer_size_gb <= 0:
             return {"gpu_layers": 0, "cpu_layers": total_layers}

        # How many layers fit in safe VRAM?
        layers_on_gpu = int(safe_vram / layer_size_gb)
        layers_on_gpu = min(layers_on_gpu, total_layers)

        # Ensure we don't put 0 if we have some space, unless it's really tiny
        if layers_on_gpu == 0 and safe_vram > 0.3: # If we have at least 300MB, maybe fit 1 layer?
             # But overhead might kill us. Let's stick to calculated.
             pass

        layers_on_cpu = total_layers - layers_on_gpu

        logger.info(f"Hybrid Plan: {layers_on_gpu} layers on GPU, {layers_on_cpu} layers on CPU "
                    f"(VRAM: {available_vram_gb:.2f}GB, Safe: {safe_vram:.2f}GB)")

        return {
            "gpu_layers": layers_on_gpu,
            "cpu_layers": layers_on_cpu
        }

def create_intel_comet_lake_plugin(profile: Optional[SystemProfile] = None) -> IntelCometLakePlugin:
    return IntelCometLakePlugin(profile)
