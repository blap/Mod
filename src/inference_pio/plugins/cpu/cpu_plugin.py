"""
Native CPU Processor Plugin (C-Engine Backend)

This plugin implements hardware operations using the custom C-based Tensor Engine.
It replaces the legacy PyTorch-based GenericCPUPlugin.
"""

import logging
import ctypes
import os
from typing import Any, Dict, Optional, List

from ...common.interfaces.improved_base_plugin_interface import BasePluginInterface, PluginMetadata, PluginType
from ...core.engine.backend import Tensor, scaled_dot_product_attention
from ...common.utils.lib_loader import load_backend_lib

logger = logging.getLogger(__name__)

class NativeCPUPlugin(BasePluginInterface):
    """
    Native CPU implementation using custom C-based tensor operations.
    Zero external dependencies (no Torch, no Numpy).
    """

    def __init__(self):
        metadata = PluginMetadata(
            name="NativeCPU",
            version="2.0.0",
            author="System",
            description="High-performance native C tensor engine for CPU.",
            plugin_type=PluginType.HARDWARE,
            dependencies=[],
            compatibility={"os": ["linux", "windows"], "arch": ["x86_64", "arm64"]},
        )
        super().__init__(metadata)
        self._config = {}
        self._num_threads = 1
        self.lib = None

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        self._config = config or {}

        # Load backend library
        try:
            self.lib = load_backend_lib("cpu")
        except ImportError as e:
            logger.error(f"Failed to load CPU backend: {e}")
            return False

        # Determine optimal thread count (physical cores)
        from ...common.hardware.hardware_analyzer import get_system_profile
        try:
            profile = get_system_profile()
            physical_cores = profile.cpu_cores if profile.cpu_cores else 1
        except:
            physical_cores = 1

        target_threads = self._config.get("num_threads", physical_cores)
        self._num_threads = target_threads

        # Set threads dynamically via backend handle
        try:
            if hasattr(self.lib, 'omp_set_num_threads'):
                self.lib.omp_set_num_threads(ctypes.c_int(target_threads))
                logger.info(f"Set OpenMP threads to {target_threads}")
            else:
                logger.info("Dynamic thread setting not exposed by libtensor_ops.so")
        except Exception as e:
            logger.warning(f"Failed to set CPU threads: {e}")

        logger.info(f"NativeCPUPlugin initialized (Threads: {self._num_threads})")

        # CPU Affinity (Optim. 5)
        self._set_affinity()

        return True

    def _set_affinity(self):
        try:
            # Platform specific affinity
            if os.name == 'posix':
                if hasattr(os, "sched_setaffinity"):
                    cpu_ids = list(range(self._num_threads))
                    os.sched_setaffinity(0, cpu_ids)
                    logger.info(f"Thread Affinity Set: Pinned to CPUs {cpu_ids}")
        except Exception as e:
            logger.debug(f"Affinity setting skipped: {e}")

    def cleanup(self) -> bool:
        return True

    # --- Hardware Operations Interface ---

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return a.matmul(b)

    def optimized_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return self.matmul(a, b)

    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> Tensor:
        # Native C implementation of SDPA (Fused)
        return scaled_dot_product_attention(query, key, value)

    def apply_activation(self, x: Tensor, activation_type: str) -> Tensor:
        if activation_type == "silu" or activation_type == "swish":
            return x.silu()
        elif activation_type == "gelu":
            return x.gelu()
        return x

    def manage_threads(self, num_threads: int):
        # Set env var for OpenMP
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        if self.lib and hasattr(self.lib, 'omp_set_num_threads'):
             self.lib.omp_set_num_threads(ctypes.c_int(num_threads))

def create_generic_cpu_plugin() -> NativeCPUPlugin:
    """
    Factory that replaces the old GenericCPUPlugin with NativeCPUPlugin.
    Keeps function name for compatibility.
    """
    return NativeCPUPlugin()
